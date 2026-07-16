"""Durable Phase 2 Batch CLI command boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


def resume(batch_id: str, *, paths: AppPaths) -> BatchProjection:
    config = load_batch_config(batch_id, paths=paths)
    bundle = _bundle(batch_id, config, paths)
    lease = create_batch_lease(bundle.batch_dir)
    lease.acquire()
    try:
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
    return {
        "schema_version": 1,
        "batch_id": projection.batch_id,
        "state": projection.state.value,
        "last_event_seq": projection.last_event_seq,
        "cancel_requested": (control / "cancel-batch").exists(),
        "jobs": [
            {
                "job_id": job.job_id,
                "state": job.state.value,
                "input_path": job.input_path,
                "output_dir": job.config.output_dir,
                "current_stage": next(
                    (stage.name.value for stage in job.stages if stage.state.value != "committed"),
                    None,
                ),
                "stages": {
                    stage.name.value: {
                        "state": stage.state.value,
                        "attempt": stage.attempt,
                        "cache_key": stage.cache_key,
                    }
                    for stage in job.stages
                },
            }
            for job in projection.jobs
        ],
    }


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
