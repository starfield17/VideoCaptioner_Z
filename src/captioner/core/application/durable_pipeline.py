"""Batch creation, sequential execution, recovery, retry, and cancellation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import CancellationToken, ExecutionContext
from captioner.core.domain.job import JobConfig, JobState
from captioner.core.domain.journal import JournalEvent, apply_event, replay
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageState
from captioner.core.ports.journal import JournalPort
from captioner.core.ports.manifest import ManifestStorePort
from captioner.core.ports.stage_runner import StageRunner


class MarkerCancellationToken(CancellationToken):
    def __init__(self, markers: tuple[Path, ...]) -> None:
        super().__init__()
        self._markers = markers

    @property
    def is_cancelled(self) -> bool:
        return super().is_cancelled or any(path.exists() for path in self._markers)


@dataclass(slots=True)
class DurablePipelineService:
    journal: JournalPort
    manifest: ManifestStorePort
    executor: StageExecutor
    event_factory: EventFactory
    runners: Mapping[StageName, StageRunner]
    control_dir: Path

    def create(self, batch_id: str, jobs: Sequence[tuple[str, Path, JobConfig]]) -> BatchProjection:
        if self.journal.read():
            raise AppError("batch.exists", {"batch_id": batch_id})
        first = JournalEvent(
            1,
            self.event_factory.next_id(),
            self.event_factory.now_utc(),
            batch_id,
            "batch.created",
            {},
        )
        self.journal.append(first)
        projection = replay((first,))
        for job_id, input_path, config in jobs:
            projection = self._append(
                projection,
                "job.created",
                cast(
                    Mapping[str, FrozenJsonValue],
                    freeze_json_value(
                        {
                            "job_id": job_id,
                            "input_path": str(input_path.resolve()),
                            "config": config.to_dict(),
                        }
                    ),
                ),
            )
        self.manifest.write(projection)
        return projection

    async def run(self, projection: BatchProjection) -> BatchProjection:
        for job in projection.jobs:
            if job.state in {JobState.SUCCEEDED, JobState.CANCELLED}:
                continue
            projection = await self._run_job(projection, job.job_id)
        return projection

    async def resume(self) -> BatchProjection:
        events = self.journal.read()
        if not events:
            raise AppError("batch.not_found")
        projection = replay(events)
        self.manifest.reconcile(projection)
        projection = self._interrupt_open_attempts(projection)
        projection = self._invalidate_corrupt_artifacts(projection)
        self._remove_stale_workspaces()
        return await self.run(projection)

    def status(self) -> BatchProjection:
        events = self.journal.read()
        if not events:
            raise AppError("batch.not_found")
        projection = replay(events)
        self.manifest.reconcile(projection)
        return projection

    async def retry(self, job_id: str, stage: StageName) -> BatchProjection:
        projection = self.status()
        job = projection.job(job_id)
        current = job.stage(stage)
        if current.state is not StageState.COMMITTED:
            raise AppError("retry.stage_invalid", {"stage_name": stage.value})
        projection = self._append(
            projection,
            "stage.invalidated",
            {
                "job_id": job_id,
                "stage_name": stage.value,
                "attempt": current.attempt,
            },
        )
        self.manifest.write(projection)
        return await self.run(projection)

    async def _run_job(self, projection: BatchProjection, job_id: str) -> BatchProjection:
        job = projection.job(job_id)
        token = MarkerCancellationToken(
            (self.control_dir / "cancel-batch", self.control_dir / f"cancel-{job_id}")
        )
        context = ExecutionContext(token)
        try:
            for name in STAGE_PLAN:
                context.raise_if_cancelled()
                job = projection.job(job_id)
                stage = job.stage(name)
                if stage.state is StageState.COMMITTED:
                    continue
                inputs = tuple(
                    artifact
                    for prior in job.stages[: STAGE_PLAN.index(name)]
                    for artifact in prior.artifacts
                    if prior.state is StageState.COMMITTED
                )
                projection = await self.executor.execute(
                    projection,
                    job_id=job_id,
                    runner=self.runners[name],
                    input_artifacts=inputs,
                    cache_config=_cache_config(job.config, name),
                    context=context,
                )
        except AppError as exc:
            current = replay(self.journal.read())
            if (
                exc.code == "operation.cancelled"
                and current.job(job_id).state is not JobState.CANCELLED
            ):
                current = self._append(current, "job.cancelled", {"job_id": job_id})
                self.manifest.write(current)
                self._clear_markers(job_id)
            raise
        projection = self._append(projection, "job.succeeded", {"job_id": job_id})
        self.manifest.write(projection)
        self._clear_markers(job_id)
        return projection

    def _interrupt_open_attempts(self, projection: BatchProjection) -> BatchProjection:
        for job in projection.jobs:
            for stage in job.stages:
                if stage.state is StageState.RUNNING:
                    projection = self._append(
                        projection,
                        "stage.interrupted",
                        {
                            "job_id": job.job_id,
                            "stage_name": stage.name.value,
                            "attempt": stage.attempt,
                        },
                    )
        if projection.last_event_seq != replay(self.journal.read()).last_event_seq:
            raise AppError("journal.corrupt")
        self.manifest.write(projection)
        return projection

    def _invalidate_corrupt_artifacts(self, projection: BatchProjection) -> BatchProjection:
        for job in projection.jobs:
            for stage in job.stages:
                if stage.state is not StageState.COMMITTED:
                    continue
                try:
                    for artifact in stage.artifacts:
                        self.executor.artifact_store.verify(artifact)
                except AppError:
                    projection = self._append(
                        projection,
                        "stage.invalidated",
                        {
                            "job_id": job.job_id,
                            "stage_name": stage.name.value,
                            "attempt": stage.attempt,
                        },
                    )
                    break
        self.manifest.write(projection)
        return projection

    def _append(
        self, projection: BatchProjection, event_type: str, payload: Mapping[str, FrozenJsonValue]
    ) -> BatchProjection:
        event = self.event_factory.create(projection, event_type, payload)
        apply_event(projection, event)
        self.journal.append(event)
        return replay(self.journal.read())

    def _remove_stale_workspaces(self) -> None:
        import shutil

        if self.executor.work_root.exists():
            shutil.rmtree(self.executor.work_root)

    def _clear_markers(self, job_id: str) -> None:
        (self.control_dir / f"cancel-{job_id}").unlink(missing_ok=True)
        (self.control_dir / "cancel-batch").unlink(missing_ok=True)


def write_cancel_marker(control_dir: Path, *, job_id: str | None) -> Path:
    control_dir.mkdir(parents=True, exist_ok=True)
    target = control_dir / ("cancel-batch" if job_id is None else f"cancel-{job_id}")
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(b"cancel\n")
    temporary.replace(target)
    return target


def _cache_config(config: JobConfig, stage: StageName) -> Mapping[str, FrozenJsonValue]:
    values: dict[str, object]
    if stage is StageName.INSPECT:
        values = {"ffprobe_bin": config.ffprobe_bin}
    elif stage is StageName.NORMALIZE:
        values = {"ffmpeg_bin": config.ffmpeg_bin, "normalization": config.normalization}
    elif stage is StageName.TRANSCRIBE:
        values = {
            "model_identity": config.model_identity,
            "language": config.language,
            "vad_filter": config.vad_filter,
            "device": config.device,
            "compute_type": config.compute_type,
        }
    elif stage is StageName.SEGMENT:
        values = {"segmentation": config.segmentation}
    elif stage is StageName.EXPORT:
        values = {"schema_version": 1}
    else:
        values = {"output_dir": config.output_dir, "overwrite": config.overwrite}
    return cast(Mapping[str, FrozenJsonValue], freeze_json_value(values))
