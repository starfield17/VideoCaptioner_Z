"""Durable Phase 2 Batch CLI command boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass, replace
from pathlib import Path

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.bootstrap import (
    DurableServiceBundle,
    build_durable_service,
    create_batch_lease,
    create_job_config,
)
from captioner.core.application.durable_pipeline import BatchStatus, write_cancel_marker
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobState, validate_identifier
from captioner.core.domain.journal import replay
from captioner.core.domain.stage import STAGE_PLAN, StageName
from captioner.infrastructure.app_paths import AppPaths, resolve_safe_child
from captioner.infrastructure.ids import new_id


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


@dataclass(frozen=True, slots=True)
class ResumeOverrides:
    model_ref: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    output_dir: Path | None = None


def run(options: BatchRunOptions, *, paths: AppPaths) -> BatchProjection:
    _validate_output_collisions(options.inputs, options.output_dir)
    batch_id = new_id("batch-")
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = create_job_config(
        model_ref=options.model_ref,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language,
        ffmpeg_bin=options.ffmpeg_bin,
        ffprobe_bin=options.ffprobe_bin,
        output_dir=output_dir,
        overwrite=options.overwrite,
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
    )
    lease = create_batch_lease(bundle.batch_dir)
    lease.acquire()
    try:
        jobs = tuple(
            (f"job-{index:06d}", source.expanduser().resolve(), config)
            for index, source in enumerate(options.inputs, 1)
        )
        projection = bundle.service.create(batch_id, jobs)
        return asyncio.run(bundle.service.run(projection))
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
        selected = config if overrides is None else _apply_overrides(config, overrides)
        bundle = _bundle(batch_id, selected, paths)
        if selected != config:
            earliest = min(
                (_earliest_change(job.config, selected) for job in projection.jobs),
                key=STAGE_PLAN.index,
            )
            projection = bundle.service.update_config(
                projection,
                config=selected,
                earliest_stage=earliest,
            )
        return asyncio.run(bundle.service.resume())
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
        return asyncio.run(bundle.service.retry(job_id, stage))
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
                    if status_result is not None
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
    if not path.is_file():
        return True
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = value["pid"]
        hostname = value["hostname"]
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or not isinstance(hostname, str)
            or not hostname
        ):
            return True
        if hostname != socket.gethostname():
            return False
        os.kill(pid, 0)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return True
    return False


def _success_fields(input_path: str, output_dir: str) -> dict[str, object]:
    stem = Path(input_path).stem
    transcript_path = Path(output_dir) / f"{stem}.transcript.json"
    subtitle_path = Path(output_dir) / f"{stem}.srt"
    if not transcript_path.is_file() or not subtitle_path.is_file():
        return {}
    try:
        root = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = root["transcript"]
        return {
            "transcript_id": transcript["id"],
            "transcript_path": str(transcript_path),
            "subtitle_path": str(subtitle_path),
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
        for suffix in (".transcript.json", ".srt"):
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


def _apply_overrides(config: JobConfig, overrides: ResumeOverrides) -> JobConfig:
    if overrides.model_ref is not None:
        candidate = create_job_config(
            model_ref=overrides.model_ref,
            device=overrides.device or config.device,
            compute_type=overrides.compute_type or config.compute_type,
            language=config.language if overrides.language is None else overrides.language,
            ffmpeg_bin=config.ffmpeg_bin,
            ffprobe_bin=config.ffprobe_bin,
            output_dir=Path(config.output_dir)
            if overrides.output_dir is None
            else overrides.output_dir,
            overwrite=config.overwrite,
        )
        return replace(
            candidate,
            vad_filter=config.vad_filter,
            ffmpeg_bin=config.ffmpeg_bin,
            ffprobe_bin=config.ffprobe_bin,
            normalization=config.normalization,
            segmentation=config.segmentation,
            stage_versions=config.stage_versions,
        )
    return replace(
        config,
        device=overrides.device or config.device,
        compute_type=overrides.compute_type or config.compute_type,
        language=config.language if overrides.language is None else overrides.language,
        output_dir=config.output_dir
        if overrides.output_dir is None
        else str(overrides.output_dir.resolve()),
    )


def _earliest_change(old: JobConfig, new: JobConfig) -> StageName:
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
    if old.segmentation != new.segmentation:
        return StageName.SEGMENT
    if old.stage_versions != new.stage_versions:
        for stage in STAGE_PLAN:
            if old.stage_versions.get(stage.value) != new.stage_versions.get(stage.value):
                return stage
        raise AppError("batch.config_inconsistent", {"reason": "stage_versions"})
    if old.output_dir != new.output_dir or old.overwrite != new.overwrite:
        return StageName.PUBLISH
    if old != new:
        raise AppError("batch.config_inconsistent", {"reason": "unknown_config_change"})
    return StageName.PUBLISH
