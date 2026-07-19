"""Profile-aware durable Batch CLI command boundaries."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from captioner.adapters.persistence.batch_lease import inspect_batch_lease
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.bootstrap import (
    DurableServiceBundle,
    build_durable_service,
    create_batch_lease,
    create_job_config,
    create_llm_job_snapshot,
)
from captioner.core.application.durable_pipeline import BatchStatus, write_cancel_marker
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobState, validate_identifier
from captioner.core.domain.journal import replay
from captioner.core.domain.llm_job_config import LLMJobSnapshot
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value, thaw_json_value
from captioner.core.domain.stage import (
    PipelineProfile,
    StageName,
    stage_plan_for,
    stage_versions_for,
)
from captioner.infrastructure.app_paths import AppPaths, resolve_safe_child
from captioner.infrastructure.ids import new_id


class _LanguageUnset:
    __slots__ = ()


LANGUAGE_UNSET = _LanguageUnset()


@dataclass(frozen=True, slots=True)
class BatchRunOptions:
    inputs: tuple[Path, ...]
    output_dir: Path
    model_ref: str
    device: str
    compute_type: str
    language: str | None
    ffmpeg_bin: str
    ffprobe_bin: str
    overwrite: bool
    pipeline_profile: PipelineProfile = PipelineProfile.DETERMINISTIC
    target_language: str | None = None
    llm_provider_profile: str = "default"


@dataclass(frozen=True, slots=True)
class ResumeOverrides:
    model_ref: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str | None | _LanguageUnset = LANGUAGE_UNSET
    output_dir: Path | None = None
    pipeline_profile: PipelineProfile | None = None
    llm: Mapping[str, object] | None = None
    target_language: str | None = None
    llm_provider_profile: str | None = None

    @property
    def has_language_override(self) -> bool:
        return not isinstance(self.language, _LanguageUnset)


def run(options: BatchRunOptions, *, paths: AppPaths) -> BatchProjection:
    _validate_output_collisions(options.inputs, options.output_dir)
    batch_id = new_id("batch-")
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_snapshot = None
    if options.pipeline_profile in {PipelineProfile.FAST, PipelineProfile.QUALITY}:
        if options.target_language is None:
            raise AppError("llm.target_language_missing")
        llm_snapshot = create_llm_job_snapshot(
            target_language=options.target_language,
            provider_profile=options.llm_provider_profile,
            source_language=options.language,
            paths=paths,
            pipeline_profile=options.pipeline_profile,
        )
    config = create_job_config(
        model_ref=options.model_ref,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language,
        ffmpeg_bin=options.ffmpeg_bin,
        ffprobe_bin=options.ffprobe_bin,
        output_dir=output_dir,
        overwrite=options.overwrite,
        paths=paths,
        pipeline_profile=options.pipeline_profile,
        llm=llm_snapshot,
    )
    bundle = build_durable_service(
        batch_id,
        model_ref=options.model_ref,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language,
        ffmpeg_bin=options.ffmpeg_bin,
        ffprobe_bin=options.ffprobe_bin,
        paths=paths,
        pipeline_profile=options.pipeline_profile,
        llm=config.llm,
    )
    lease = create_batch_lease(bundle.batch_dir)
    lease.acquire()
    try:
        jobs = tuple(
            (f"job-{index:06d}", source.expanduser().resolve(), config)
            for index, source in enumerate(options.inputs, 1)
        )
        projection = bundle.service.create(batch_id, jobs)
        return asyncio.run(_run_and_close(bundle, lambda: bundle.service.run(projection)))
    finally:
        lease.release()


def status(batch_id: str, *, paths: AppPaths) -> BatchStatus:
    projection = _read_projection(batch_id, paths=paths, repair=False)
    config = _common_config(projection)
    bundle = _bundle(batch_id, config, paths, initialize_runtime=False)
    return bundle.service.read_status()


def resume(
    batch_id: str, *, paths: AppPaths, overrides: ResumeOverrides | None = None
) -> BatchProjection:
    batch_dir = resolve_safe_child(paths.batches_dir, batch_id, field="batch_id")
    lease = create_batch_lease(batch_dir)
    lease.acquire()
    try:
        # Preview the complete prefix without repair before creating an output
        # override.  A directory failure must not truncate an incomplete tail
        # or append any configuration event.
        preview = _read_projection(batch_id, paths=paths, repair=False)
        _common_config(preview)
        if overrides is not None and overrides.output_dir is not None:
            output_dir = _prepare_output_directory(overrides.output_dir)
            overrides = replace(overrides, output_dir=output_dir)
        projection = _read_projection(batch_id, paths=paths, repair=True)
        config = _common_config(projection)
        selected = config if overrides is None else _apply_overrides(config, overrides, paths)
        bundle = _bundle(batch_id, selected, paths)
        if selected != config:
            earliest = min(
                (_earliest_change(job.config, selected) for job in projection.jobs),
                key=lambda stage: stage_plan_for(selected.pipeline_profile).index(stage),
            )
            projection = bundle.service.update_config(
                projection,
                config=selected,
                earliest_stage=earliest,
            )
        return asyncio.run(_run_and_close(bundle, bundle.service.resume))
    finally:
        lease.release()


def retry(batch_id: str, job_id: str, stage: StageName, *, paths: AppPaths) -> BatchProjection:
    batch_dir = resolve_safe_child(paths.batches_dir, batch_id, field="batch_id")
    lease = create_batch_lease(batch_dir)
    lease.acquire()
    try:
        projection = _read_projection(batch_id, paths=paths, repair=True)
        config = _common_config(projection)
        bundle = _bundle(batch_id, config, paths)
        return asyncio.run(_run_and_close(bundle, lambda: bundle.service.retry(job_id, stage)))
    finally:
        lease.release()


def cancel(batch_id: str, job_id: str | None, *, paths: AppPaths) -> Path:
    projection = _read_projection(batch_id, paths=paths, repair=False)
    if job_id is not None:
        validate_identifier(job_id, field="job_id")
        job = projection.job(job_id)
        if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            raise AppError("batch.cancel_invalid", {"reason": "terminal"})
    elif all(
        job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
        for job in projection.jobs
    ):
        raise AppError("batch.cancel_invalid", {"reason": "terminal"})
    batch_dir = resolve_safe_child(paths.batches_dir, batch_id, field="batch_id")
    return write_cancel_marker(batch_dir / "control", job_id=job_id)


def projection_payload(
    projection: BatchProjection | BatchStatus, *, paths: AppPaths
) -> dict[str, object]:
    if isinstance(projection, BatchStatus):
        status_result: BatchStatus | None = projection
        current = projection.projection
    else:
        status_result = None
        current = projection
    control = paths.batches_dir / current.batch_id / "control"
    stale_execution = _lease_is_stale(paths.batches_dir / current.batch_id / "lease.json")
    payload: dict[str, object] = {
        "schema_version": 1,
        "batch_id": current.batch_id,
        "state": "interrupted"
        if stale_execution and current.state.value == "running"
        else current.state.value,
        "last_event_seq": current.last_event_seq,
        "manifest_status": JsonManifestStore(
            resolve_safe_child(paths.batches_dir, current.batch_id, field="batch_id")
            / "manifest.json"
        ).inspect(current),
        "cancel_requested": (control / "cancel-batch").exists()
        or any(control.glob("cancel-job-*")),
        "jobs": [
            {
                "job_id": job.job_id,
                "state": "interrupted"
                if stale_execution and job.state.value == "running"
                else job.state.value,
                "input_path": job.input_path,
                "output_dir": job.config.output_dir,
                "current_stage": next(
                    (stage.name.value for stage in job.stages if stage.state.value != "committed"),
                    None,
                ),
                "stages": {
                    stage.name.value: {
                        "state": "interrupted"
                        if stale_execution and stage.state.value == "running"
                        else stage.state.value,
                        "attempt": stage.attempt,
                        "cache_key": stage.cache_key,
                    }
                    for stage in job.stages
                },
                **(
                    {}
                    if status_result is not None and status_result.integrity != "valid"
                    else _success_fields(job.input_path, job.config.output_dir)
                ),
            }
            for job in current.jobs
        ],
    }
    if status_result is not None:
        payload["journal_tail_status"] = status_result.journal_tail_status
        payload["manifest_status"] = status_result.manifest_status
        payload["integrity"] = status_result.integrity
        payload["integrity_errors"] = [
            {
                "job_id": issue.job_id,
                "stage_name": issue.stage_name,
                "code": issue.code,
                "logical_name": issue.logical_name,
                "sha256": issue.sha256,
            }
            for issue in status_result.integrity_errors
        ]
    return payload


def _lease_is_stale(path: Path) -> bool:
    return inspect_batch_lease(path) in {
        "missing",
        "stale",
        "invalid",
    }


def _success_fields(input_path: str, output_dir: str) -> dict[str, object]:
    stem = Path(input_path).stem
    transcript_path = Path(output_dir) / f"{stem}.transcript.json"
    subtitle_json_path = Path(output_dir) / f"{stem}.subtitle.json"
    subtitle_path = Path(output_dir) / f"{stem}.srt"
    vtt_path = Path(output_dir) / f"{stem}.vtt"
    ass_path = Path(output_dir) / f"{stem}.ass"
    if not all(
        path.is_file()
        for path in (transcript_path, subtitle_json_path, subtitle_path, vtt_path, ass_path)
    ):
        return {}
    try:
        root = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = root["transcript"]
        return {
            "transcript_id": transcript["id"],
            "transcript_path": str(transcript_path),
            "subtitle_json_path": str(subtitle_json_path),
            "subtitle_path": str(subtitle_path),
            "vtt_path": str(vtt_path),
            "ass_path": str(ass_path),
            "detected_language": transcript["language"],
            "word_count": len(transcript["words"]),
            "cue_count": len(
                [
                    block
                    for block in subtitle_path.read_text(encoding="utf-8").split("\n\n")
                    if block.strip()
                ]
            ),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AppError(
            "output.publication_invalid", {"logical_name": transcript_path.name}
        ) from exc


def _bundle(
    batch_id: str,
    config: JobConfig,
    paths: AppPaths,
    *,
    initialize_runtime: bool = True,
) -> DurableServiceBundle:
    return build_durable_service(
        batch_id,
        model_ref=config.model_ref,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        ffmpeg_bin=config.ffmpeg_bin,
        ffprobe_bin=config.ffprobe_bin,
        paths=paths,
        segmentation=config.segmentation,
        pipeline_profile=config.pipeline_profile,
        llm=config.llm,
        initialize_runtime=initialize_runtime,
    )


def _common_config(projection: BatchProjection) -> JobConfig:
    if not projection.jobs:
        raise AppError("batch.config_inconsistent", {"reason": "no_jobs"})
    config = projection.jobs[0].config
    if any(job.config.runtime_signature != config.runtime_signature for job in projection.jobs[1:]):
        raise AppError("batch.config_inconsistent", {"reason": "runtime"})
    return config


def _validate_output_collisions(inputs: tuple[Path, ...], output_dir: Path) -> None:
    normalized: dict[str, Path] = {}
    target_root = output_dir.expanduser().resolve()
    for source in inputs:
        stem = source.expanduser().resolve().stem
        for suffix in (
            ".transcript.json",
            ".subtitle.json",
            ".srt",
            ".vtt",
            ".ass",
        ):
            target = target_root / f"{stem}{suffix}"
            key = os.path.normcase(str(target))
            previous = normalized.get(key)
            if previous is not None:
                raise AppError(
                    "batch.output_collision",
                    {"logical_name": target.name},
                )
            normalized[key] = source


def _prepare_output_directory(path: Path) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise AppError("output.directory_invalid", {"path": str(requested)})
    try:
        requested.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppError("output.directory_failed", {"path": str(requested)}) from exc
    if requested.is_symlink() or not requested.is_dir():
        raise AppError("output.directory_invalid", {"path": str(requested)})
    return requested.resolve()


def _read_projection(batch_id: str, *, paths: AppPaths, repair: bool) -> BatchProjection:
    batch_dir = resolve_safe_child(paths.batches_dir, batch_id, field="batch_id")
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    events = journal.repair_and_read() if repair else journal.read_snapshot().events
    if not events:
        raise AppError("batch.not_found", {"batch_id": batch_id})
    return replay(events)


def _apply_overrides(
    config: JobConfig,
    overrides: ResumeOverrides,
    paths: AppPaths | None = None,
) -> JobConfig:
    selected_profile = (
        config.pipeline_profile
        if overrides.pipeline_profile is None
        else PipelineProfile(overrides.pipeline_profile)
    )
    llm = _llm_for_resume(config, overrides, selected_profile, paths)
    effective_language = _effective_language(config, overrides)
    if overrides.model_ref is not None:
        candidate = create_job_config(
            model_ref=overrides.model_ref,
            device=overrides.device or config.device,
            compute_type=overrides.compute_type or config.compute_type,
            language=effective_language,
            ffmpeg_bin=config.ffmpeg_bin,
            ffprobe_bin=config.ffprobe_bin,
            output_dir=Path(config.output_dir)
            if overrides.output_dir is None
            else overrides.output_dir,
            overwrite=config.overwrite,
            paths=paths,
            pipeline_profile=selected_profile,
            llm=llm,
        )
        return replace(
            candidate,
            vad_filter=config.vad_filter,
            ffmpeg_bin=config.ffmpeg_bin,
            ffprobe_bin=config.ffprobe_bin,
            normalization=config.normalization,
            segmentation=config.segmentation,
            stage_versions=stage_versions_for(selected_profile),
            pipeline_profile=candidate.pipeline_profile,
            llm=candidate.llm,
        )
    return replace(
        config,
        device=overrides.device or config.device,
        compute_type=overrides.compute_type or config.compute_type,
        language=effective_language,
        output_dir=config.output_dir
        if overrides.output_dir is None
        else str(overrides.output_dir.resolve()),
        pipeline_profile=selected_profile,
        stage_versions=stage_versions_for(selected_profile),
        llm=llm,
    )


def _llm_for_resume(
    config: JobConfig,
    overrides: ResumeOverrides,
    selected_profile: PipelineProfile,
    paths: AppPaths | None,
) -> Mapping[str, FrozenJsonValue] | None:
    if selected_profile is PipelineProfile.DETERMINISTIC:
        if overrides.llm is not None:
            raise AppError("llm.snapshot_invalid", {"reason": "profile"})
        return None
    # Compute effective values first so snapshot identity never reads stale config.
    effective_language = _effective_language(config, overrides)
    effective_target_language = (
        config.target_language if overrides.target_language is None else overrides.target_language
    )
    if effective_target_language is None:
        raise AppError("llm.target_language_missing")
    effective_provider_profile = (
        config.provider_profile or "default"
        if overrides.llm_provider_profile is None
        else overrides.llm_provider_profile
    )
    if overrides.llm is not None:
        snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(_frozen_llm(overrides.llm)))
        if snapshot.profile is not selected_profile:
            raise AppError("llm.snapshot_invalid", {"reason": "profile"})
        if snapshot.source_language != effective_language:
            raise AppError("llm.snapshot_invalid", {"reason": "source_language"})
        if snapshot.target_language != effective_target_language:
            raise AppError("llm.snapshot_invalid", {"reason": "target_language"})
        if snapshot.provider.provider_profile != effective_provider_profile:
            raise AppError("llm.snapshot_invalid", {"reason": "provider_profile"})
        return snapshot.to_mapping()
    profile_changed = selected_profile is not config.pipeline_profile
    snapshot_source = None if config.llm is None else config.llm.get("source_language")
    if snapshot_source is not None and not isinstance(snapshot_source, str):
        raise AppError("llm.snapshot_invalid", {"reason": "source_language"})
    identity_changed = (
        overrides.target_language is not None
        or overrides.llm_provider_profile is not None
        or overrides.has_language_override
        or profile_changed
        or effective_language != snapshot_source
    )
    if not identity_changed:
        if config.llm is None:
            raise AppError("llm.config_missing", {"reason": "job_snapshot"})
        return config.llm
    if paths is None:
        raise AppError("llm.config_missing", {"reason": "paths"})
    snapshot = create_llm_job_snapshot(
        target_language=effective_target_language,
        provider_profile=effective_provider_profile,
        source_language=effective_language,
        paths=paths,
        pipeline_profile=selected_profile,
    )
    return snapshot


def _effective_language(config: JobConfig, overrides: ResumeOverrides) -> str | None:
    if not overrides.has_language_override:
        return config.language
    language = overrides.language
    if isinstance(language, _LanguageUnset):
        raise AppError("job.config_invalid", {"field": "language"})
    return language


async def _run_and_close(
    bundle: DurableServiceBundle,
    operation: Callable[[], Awaitable[BatchProjection]],
) -> BatchProjection:
    try:
        return await operation()
    finally:
        await bundle.close()


def _earliest_change(old: JobConfig, new: JobConfig) -> StageName:
    plan = stage_plan_for(new.pipeline_profile)
    if old.ffprobe_bin != new.ffprobe_bin:
        return StageName.INSPECT
    if old.ffmpeg_bin != new.ffmpeg_bin or old.normalization != new.normalization:
        return StageName.NORMALIZE
    if (
        old.model_ref,
        old.model_identity,
        old.device,
        old.compute_type,
        old.language,
        old.vad_filter,
    ) != (
        new.model_ref,
        new.model_identity,
        new.device,
        new.compute_type,
        new.language,
        new.vad_filter,
    ):
        return StageName.TRANSCRIBE
    candidates: list[StageName] = []
    if old.segmentation != new.segmentation:
        candidates.append(StageName.SEGMENT)
    if old.pipeline_profile != new.pipeline_profile:
        if new.pipeline_profile is PipelineProfile.QUALITY:
            candidates.append(StageName.CORRECT_SOURCE)
        elif old.pipeline_profile is PipelineProfile.QUALITY:
            candidates.append(StageName.SEGMENT)
        elif new.pipeline_profile is PipelineProfile.FAST:
            candidates.append(StageName.TRANSLATE)
        else:
            candidates.append(StageName.SEGMENT)
    if old.llm != new.llm:
        candidates.extend(_changed_llm_stages(old, new, plan))
    if old.stage_versions != new.stage_versions:
        for stage in plan:
            if old.stage_versions.get(stage.value) != new.stage_versions.get(stage.value):
                candidates.append(stage)
                break
        else:
            if old.pipeline_profile is new.pipeline_profile:
                raise AppError("batch.config_inconsistent", {"reason": "stage_versions"})
    if old.output_dir != new.output_dir or old.overwrite != new.overwrite:
        candidates.append(StageName.PUBLISH)
    if candidates:
        return min(candidates, key=plan.index)
    if old != new:
        raise AppError("batch.config_inconsistent", {"reason": "unknown_config_change"})
    return StageName.PUBLISH


def _available_stage(
    preferred: StageName,
    plan: tuple[StageName, ...],
    *,
    fallback: StageName | None = None,
) -> StageName:
    if preferred in plan:
        return preferred
    if fallback is not None and fallback in plan:
        return fallback
    for candidate in (StageName.SEGMENT, StageName.EXPORT, StageName.PUBLISH):
        if candidate in plan:
            return candidate
    raise AppError("batch.config_inconsistent", {"reason": "empty_stage_plan"})


def _changed_prompt_stages(
    old_snapshot: LLMJobSnapshot,
    new_snapshot: LLMJobSnapshot,
    plan: tuple[StageName, ...],
) -> tuple[StageName, ...]:
    changed_ids = {
        prompt_id
        for prompt_id in set(old_snapshot.prompts) | set(new_snapshot.prompts)
        if old_snapshot.prompts.get(prompt_id) != new_snapshot.prompts.get(prompt_id)
    }
    stages: list[StageName] = []
    if changed_ids & {"terminology", "correct_source"}:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    if changed_ids & {"translate_fast", "translate_quality"}:
        stages.append(_available_stage(StageName.TRANSLATE, plan))
    if "review_anomalies" in changed_ids and StageName.REVIEW in plan:
        stages.append(StageName.REVIEW)
    known_ids = {
        "terminology",
        "correct_source",
        "translate_fast",
        "translate_quality",
        "review_anomalies",
    }
    if changed_ids - known_ids:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    return tuple(stages)


def _changed_llm_stages(
    old: JobConfig,
    new: JobConfig,
    plan: tuple[StageName, ...],
) -> tuple[StageName, ...]:
    if old.llm is None or new.llm is None:
        return ()
    old_snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(old.llm))
    new_snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(new.llm))
    stages: list[StageName] = []
    if old_snapshot.provider != new_snapshot.provider:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    if old_snapshot.source_language != new_snapshot.source_language:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    if old_snapshot.target_language != new_snapshot.target_language:
        stages.append(_available_stage(StageName.TRANSLATE, plan))
    if old_snapshot.chunk != new_snapshot.chunk:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    if old_snapshot.response_schema_version != new_snapshot.response_schema_version:
        stages.append(
            _available_stage(StageName.CORRECT_SOURCE, plan, fallback=StageName.TRANSLATE)
        )
    stages.extend(_changed_prompt_stages(old_snapshot, new_snapshot, plan))
    return tuple(stages)


def _frozen_llm(value: Mapping[str, object]) -> Mapping[str, FrozenJsonValue]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, Mapping):
        raise AppError("job.config_invalid", {"field": "llm"})
    return cast(Mapping[str, FrozenJsonValue], frozen)
