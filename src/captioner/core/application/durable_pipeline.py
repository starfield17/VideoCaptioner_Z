"""Batch creation, sequential execution, recovery, retry, and cancellation."""

from __future__ import annotations

import hashlib
import os
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
        if not jobs:
            raise AppError("batch.config_inconsistent", {"reason": "no_jobs"})
        first_config = jobs[0][2]
        if any(
            config.runtime_signature != first_config.runtime_signature for _, _, config in jobs[1:]
        ):
            raise AppError("batch.config_inconsistent", {"reason": "runtime"})
        targets: set[str] = set()
        for _, input_path, config in jobs:
            for suffix in (".transcript.json", ".srt"):
                target = os.path.normcase(
                    str(Path(config.output_dir) / f"{input_path.stem}{suffix}")
                )
                if target in targets:
                    raise AppError("batch.output_collision", {"logical_name": Path(target).name})
                targets.add(target)
        if self.journal.read_snapshot().events:
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
            if job.state in {JobState.CANCELLED, JobState.FAILED}:
                continue
            try:
                projection = await self._run_job(projection, job.job_id)
            except AppError as exc:
                if (
                    exc.code == "operation.cancelled"
                    and len(projection.jobs) > 1
                    and exc.params.get("scope") != "batch"
                ):
                    projection = replay(self.journal.repair_and_read())
                    continue
                raise
            if (self.control_dir / "cancel-batch").exists():
                projection = self._cancel_pending_jobs(projection)
                break
        return projection

    async def resume(self) -> BatchProjection:
        events = self.journal.repair_and_read()
        if not events:
            raise AppError("batch.not_found")
        projection = replay(events)
        self.manifest.reconcile(projection)
        projection = self._interrupt_open_attempts(projection)
        projection = self._invalidate_corrupt_artifacts(projection)
        self._remove_stale_workspaces()
        return await self.run(projection)

    def status(self) -> BatchProjection:
        snapshot = self.journal.read_snapshot()
        events = snapshot.events
        if not events:
            raise AppError("batch.not_found")
        projection = replay(events)
        if projection.jobs:
            first_config = projection.jobs[0].config
            if any(
                job.config.runtime_signature != first_config.runtime_signature
                for job in projection.jobs[1:]
            ):
                raise AppError("batch.config_inconsistent", {"reason": "runtime"})
        for job in projection.jobs:
            for stage in job.stages:
                if stage.state is StageState.COMMITTED:
                    self.executor.verify_committed(job, stage)
        return projection

    def update_config(
        self,
        projection: BatchProjection,
        *,
        job_id: str,
        config: JobConfig,
        earliest_stage: StageName,
    ) -> BatchProjection:
        projection = self._append(
            projection,
            "job.config_updated",
            cast(
                Mapping[str, FrozenJsonValue],
                freeze_json_value({"job_id": job_id, "config": config.to_dict()}),
            ),
        )
        current = projection.job(job_id).stage(earliest_stage)
        if current.state is StageState.COMMITTED:
            projection = self._append(
                projection,
                "stage.invalidated",
                {
                    "job_id": job_id,
                    "stage_name": earliest_stage.value,
                    "attempt": current.attempt,
                },
            )
        self.manifest.write(projection)
        return projection

    async def retry(self, job_id: str, stage: StageName) -> BatchProjection:
        projection = self.status()
        projection = self._append(
            projection,
            "job.retry_requested",
            {
                "job_id": job_id,
                "stage_name": stage.value,
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
                    cache_config=_cache_config(job.config, name, Path(job.input_path)),
                    context=context,
                )
        except AppError as exc:
            current = replay(self.journal.repair_and_read())
            batch_requested = (self.control_dir / "cancel-batch").exists()
            if (
                exc.code == "operation.cancelled"
                and current.job(job_id).state is not JobState.CANCELLED
            ):
                current = self._append(current, "job.cancelled", {"job_id": job_id})
                self.manifest.write(current)
                self._clear_job_marker(job_id)
                if (self.control_dir / "cancel-batch").exists():
                    self._cancel_pending_jobs(current)
            elif (
                exc.code not in {"operation.cancelled", "stage.post_commit_failed"}
                and current.job(job_id).state is not JobState.FAILED
            ):
                current = self._append(current, "job.failed", {"job_id": job_id})
                self.manifest.write(current)
            if exc.code == "operation.cancelled" and batch_requested:
                raise AppError("operation.cancelled", {"scope": "batch"}) from exc
            raise
        if projection.job(job_id).state is JobState.SUCCEEDED:
            return projection
        try:
            context.raise_if_cancelled()
        except AppError as exc:
            if exc.code == "operation.cancelled":
                batch_requested = (self.control_dir / "cancel-batch").exists()
                projection = self._append(projection, "job.cancelled", {"job_id": job_id})
                self.manifest.write(projection)
                self._clear_job_marker(job_id)
                if (self.control_dir / "cancel-batch").exists():
                    projection = self._cancel_pending_jobs(projection)
                if batch_requested:
                    raise AppError("operation.cancelled", {"scope": "batch"}) from exc
                raise
            raise
        projection = self._append(projection, "job.succeeded", {"job_id": job_id})
        self.manifest.write(projection)
        self._clear_job_marker(job_id)
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
        if projection.last_event_seq != replay(self.journal.repair_and_read()).last_event_seq:
            raise AppError("journal.corrupt")
        self.manifest.write(projection)
        return projection

    def _invalidate_corrupt_artifacts(self, projection: BatchProjection) -> BatchProjection:
        for job in projection.jobs:
            for stage in job.stages:
                if stage.state is not StageState.COMMITTED:
                    continue
                for artifact in stage.artifacts:
                    try:
                        self.executor.verify_committed(job, stage)
                        break
                    except AppError:
                        self.executor.artifact_store.resolve(artifact).unlink(missing_ok=True)
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
        updated = apply_event(projection, event)
        self.journal.append(event)
        return updated

    def _remove_stale_workspaces(self) -> None:
        import shutil

        if self.executor.work_root.exists():
            shutil.rmtree(self.executor.work_root)

    def _clear_markers(self, job_id: str) -> None:
        self._clear_job_marker(job_id)
        (self.control_dir / "cancel-batch").unlink(missing_ok=True)

    def _clear_job_marker(self, job_id: str) -> None:
        (self.control_dir / f"cancel-{job_id}").unlink(missing_ok=True)

    def _cancel_pending_jobs(self, projection: BatchProjection) -> BatchProjection:
        for job in projection.jobs:
            if job.state in {JobState.PENDING, JobState.RUNNING, JobState.INTERRUPTED}:
                projection = self._append(projection, "job.cancelled", {"job_id": job.job_id})
        self.manifest.write(projection)
        (self.control_dir / "cancel-batch").unlink(missing_ok=True)
        return projection


def write_cancel_marker(control_dir: Path, *, job_id: str | None) -> Path:
    import os
    import tempfile

    control_dir.mkdir(parents=True, exist_ok=True)
    target = control_dir / ("cancel-batch" if job_id is None else f"cancel-{job_id}")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=control_dir)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"cancel\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if os.name != "nt":
            directory = os.open(control_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as exc:
        raise AppError("batch.cancel_marker_failed") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _cache_config(
    config: JobConfig, stage: StageName, input_path: Path
) -> Mapping[str, FrozenJsonValue]:
    values: dict[str, object]
    if stage is StageName.INSPECT:
        values = {
            "source_sha256": _source_sha256(input_path),
            "ffprobe_bin": config.ffprobe_bin,
        }
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


def _source_sha256(path: Path) -> str:
    if not path.is_file():
        raise AppError("media.input_missing", {"path": str(path)})
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise AppError("media.input_read_failed", {"path": str(path)}) from exc
    return digest.hexdigest()
