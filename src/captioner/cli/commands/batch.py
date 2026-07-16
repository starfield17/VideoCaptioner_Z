"""Durable Phase 2 Batch CLI command boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass, replace
from pathlib import Path

from captioner.bootstrap import (
    DurableServiceBundle,
    build_durable_service,
    create_batch_lease,
    create_job_config,
    load_batch_config,
)
from captioner.core.application.durable_pipeline import write_cancel_marker
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig
from captioner.core.domain.stage import StageName
from captioner.infrastructure.app_paths import AppPaths
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


def status(batch_id: str, *, paths: AppPaths) -> BatchProjection:
    config = load_batch_config(batch_id, paths=paths)
    return _bundle(batch_id, config, paths).service.status()


def resume(
    batch_id: str, *, paths: AppPaths, overrides: ResumeOverrides | None = None
) -> BatchProjection:
    config = load_batch_config(batch_id, paths=paths)
    selected = config if overrides is None else _apply_overrides(config, overrides)
    bundle = _bundle(batch_id, selected, paths)
    lease = create_batch_lease(bundle.batch_dir)
    lease.acquire()
    try:
        if selected != config:
            projection = bundle.service.status()
            bundle.service.update_config(
                projection,
                job_id=projection.jobs[0].job_id,
                config=selected,
                earliest_stage=_earliest_change(config, selected),
            )
        return asyncio.run(bundle.service.resume())
    finally:
        lease.release()


def retry(batch_id: str, job_id: str, stage: StageName, *, paths: AppPaths) -> BatchProjection:
    config = load_batch_config(batch_id, paths=paths)
    bundle = _bundle(batch_id, config, paths)
    lease = create_batch_lease(bundle.batch_dir)
    lease.acquire()
    try:
        return asyncio.run(bundle.service.retry(job_id, stage))
    finally:
        lease.release()


def cancel(batch_id: str, job_id: str | None, *, paths: AppPaths) -> Path:
    return write_cancel_marker(paths.batches_dir / batch_id / "control", job_id=job_id)


def projection_payload(projection: BatchProjection, *, paths: AppPaths) -> dict[str, object]:
    control = paths.batches_dir / projection.batch_id / "control"
    stale_execution = _lease_is_stale(paths.batches_dir / projection.batch_id / "lease.json")
    return {
        "schema_version": 1,
        "batch_id": projection.batch_id,
        "state": "interrupted"
        if stale_execution and projection.state.value == "running"
        else projection.state.value,
        "last_event_seq": projection.last_event_seq,
        "manifest_status": "current",
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
                **_success_fields(job.input_path, job.config.output_dir),
            }
            for job in projection.jobs
        ],
    }


def _lease_is_stale(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = value["pid"]
        hostname = value["hostname"]
        if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(hostname, str):
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


def _bundle(batch_id: str, config: JobConfig, paths: AppPaths) -> DurableServiceBundle:
    return build_durable_service(
        batch_id,
        model_ref=config.model_ref,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        ffmpeg_bin=config.ffmpeg_bin,
        ffprobe_bin=config.ffprobe_bin,
        paths=paths,
    )


def _apply_overrides(config: JobConfig, overrides: ResumeOverrides) -> JobConfig:
    if overrides.model_ref is not None:
        return create_job_config(
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
        config,
        device=overrides.device or config.device,
        compute_type=overrides.compute_type or config.compute_type,
        language=config.language if overrides.language is None else overrides.language,
        output_dir=config.output_dir
        if overrides.output_dir is None
        else str(overrides.output_dir.resolve()),
    )


def _earliest_change(old: JobConfig, new: JobConfig) -> StageName:
    if (
        old.model_identity,
        old.device,
        old.compute_type,
        old.language,
        old.vad_filter,
    ) != (
        new.model_identity,
        new.device,
        new.compute_type,
        new.language,
        new.vad_filter,
    ):
        return StageName.TRANSCRIBE
    if old.segmentation != new.segmentation:
        return StageName.SEGMENT
    return StageName.PUBLISH
