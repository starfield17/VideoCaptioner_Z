"""Generic six-stage execution and durable commit protocol."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.cache_key import derive_stage_cache_key
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobProjection
from captioner.core.domain.journal import JournalEvent, apply_event
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value
from captioner.core.domain.stage import StageName, StageProjection, StageState
from captioner.core.ports.durable_artifact_store import DurableArtifactStorePort
from captioner.core.ports.fault_injector import FaultInjector, NoOpFaultInjector
from captioner.core.ports.journal import JournalPort
from captioner.core.ports.manifest import ManifestStorePort
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
    StageRunner,
)


@dataclass(frozen=True, slots=True)
class EventFactory:
    next_id: Callable[[], str]
    now_utc: Callable[[], str]

    def create(
        self,
        projection: BatchProjection,
        event_type: str,
        payload: Mapping[str, FrozenJsonValue],
    ) -> JournalEvent:
        return JournalEvent(
            projection.last_event_seq + 1,
            self.next_id(),
            self.now_utc(),
            projection.batch_id,
            event_type,
            payload,
        )


@dataclass(slots=True)
class StageExecutor:
    journal: JournalPort
    manifest: ManifestStorePort
    artifact_store: DurableArtifactStorePort
    event_factory: EventFactory
    work_root: Path
    fault_injector: FaultInjector = field(default_factory=NoOpFaultInjector)
    committed_verifier: Callable[[JobProjection, StageProjection], None] | None = None

    def verify_artifact(self, ref: ArtifactRef) -> None:
        """Verify one durable blob without evaluating Stage-specific state."""
        self.artifact_store.verify(ref)

    def verify_stage_external_state(self, job: JobProjection, stage: StageProjection) -> None:
        """Verify external state associated with a committed Stage."""
        if self.committed_verifier is not None:
            self.committed_verifier(job, stage)

    def verify_committed(self, job: JobProjection, stage: StageProjection) -> None:
        for artifact in stage.artifacts:
            self.verify_artifact(artifact)
        self.verify_stage_external_state(job, stage)

    async def execute(
        self,
        projection: BatchProjection,
        *,
        job_id: str,
        runner: StageRunner,
        input_artifacts: tuple[ArtifactRef, ...],
        cache_config: Mapping[str, FrozenJsonValue],
        context: ExecutionContext,
    ) -> BatchProjection:
        job = projection.job(job_id)
        current = job.stage(runner.name)
        cache_key = derive_stage_cache_key(
            stage_name=runner.name.value,
            stage_version=runner.version,
            input_artifacts=input_artifacts,
            config=cache_config,
        )
        if (
            current.state is StageState.COMMITTED
            and current.cache_key == cache_key
            and current.artifacts
        ):
            self.verify_committed(job, current)
            return projection
        if current.state is StageState.COMMITTED:
            projection = self._append(
                projection,
                "stage.invalidated",
                _stage_payload(job_id, runner.name, current.attempt),
            )
            current = projection.job(job_id).stage(runner.name)
        recovery = current.attempt > 0
        attempt = current.attempt + 1
        workspace = self.work_root / job_id / runner.name.value / f"attempt-{attempt}"
        workspace.mkdir(parents=True, exist_ok=False)

        def checkpoint(point: str) -> None:
            self._hit(projection.batch_id, job_id, runner.name, attempt, point)

        committed = False
        cancelled = False
        post_commit_reason = "recovery"
        try:
            projection = self._append(
                projection,
                "stage.started",
                _stage_payload(job_id, runner.name, attempt),
            )
            self.manifest.write(projection)
            self._hit(projection.batch_id, job_id, runner.name, attempt, "before_execute")
            context.raise_if_cancelled()
            produced = await runner.execute(
                StageExecutionRequest(
                    projection.batch_id,
                    job_id,
                    Path(job.input_path),
                    job.config,
                    input_artifacts,
                    recovery,
                ),
                StageExecutionContext(
                    replace(
                        context,
                        checkpoint_hook=checkpoint,
                    ),
                    workspace,
                ),
            )
            context.raise_if_cancelled()
            refs = tuple(self._import(item) for item in produced)
            _require_outputs(refs, runner.name)
            for ref in refs:
                self.artifact_store.verify(ref)
            self._hit(projection.batch_id, job_id, runner.name, attempt, "after_artifact_write")
            self._hit(projection.batch_id, job_id, runner.name, attempt, "before_journal_commit")
            projection = self._append(
                projection,
                "stage.committed",
                {
                    **_stage_payload(job_id, runner.name, attempt),
                    "cache_key": cache_key,
                    "artifacts": tuple(freeze_json_value(ref.to_dict()) for ref in refs),
                },
            )
            committed = True
            post_commit_reason = "after_commit_hook"
            self._hit(projection.batch_id, job_id, runner.name, attempt, "after_journal_commit")
            post_commit_reason = "artifact_verification"
            for ref in refs:
                self.artifact_store.verify(ref)
            self._hit(
                projection.batch_id, job_id, runner.name, attempt, "before_manifest_projection"
            )
            post_commit_reason = "manifest_projection"
            self.manifest.write(projection)
        except AppError as exc:
            if exc.code == "operation.cancelled":
                projection = self._append(
                    projection,
                    "stage.cancelled",
                    {
                        **_stage_payload(job_id, runner.name, attempt),
                        "error_code": exc.code,
                    },
                )
                projection = self._append(
                    projection,
                    "job.cancelled",
                    {"job_id": job_id},
                )
                try:
                    self.manifest.write(projection)
                except AppError as manifest_error:
                    raise AppError(
                        "stage.cancellation_projection_failed",
                        {"job_id": job_id, "stage_name": runner.name.value},
                    ) from manifest_error
                cancelled = True
                raise
            if committed:
                raise AppError(
                    "stage.post_commit_failed",
                    {
                        "job_id": job_id,
                        "stage_name": runner.name.value,
                        "attempt": attempt,
                        "reason": post_commit_reason,
                    },
                ) from exc
            event_type = "stage.failed"
            projection = self._append(
                projection,
                event_type,
                {
                    **_stage_payload(job_id, runner.name, attempt),
                    "error_code": exc.code,
                },
            )
            projection = self._append(projection, "job.failed", {"job_id": job_id})
            self.manifest.write(projection)
            raise
        else:
            return projection
        finally:
            try:
                shutil.rmtree(workspace)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if committed:
                    raise AppError(
                        "stage.post_commit_failed",
                        {
                            "job_id": job_id,
                            "stage_name": runner.name.value,
                            "attempt": attempt,
                            "reason": "workspace_cleanup",
                        },
                    ) from exc
                if cancelled:
                    raise AppError("operation.cancelled") from exc
                raise AppError("stage.workspace_cleanup_failed") from exc

    def _append(
        self,
        projection: BatchProjection,
        event_type: str,
        payload: Mapping[str, FrozenJsonValue],
    ) -> BatchProjection:
        event = self.event_factory.create(projection, event_type, payload)
        updated = apply_event(projection, event)
        self.journal.append(event)
        return updated

    def _import(self, produced: ProducedArtifact) -> ArtifactRef:
        if produced.data is not None:
            return self.artifact_store.put_bytes(
                produced.data,
                kind=produced.kind,
                media_type=produced.media_type,
                logical_name=produced.logical_name,
            )
        if produced.source_path is None:
            raise AppError("stage.output_invalid")
        return self.artifact_store.put_file(
            produced.source_path,
            kind=produced.kind,
            media_type=produced.media_type,
            logical_name=produced.logical_name,
        )

    def _hit(self, batch_id: str, job_id: str, stage: StageName, attempt: int, point: str) -> None:
        self.fault_injector.hit(
            batch_id=batch_id,
            job_id=job_id,
            stage_name=stage.value,
            attempt=attempt,
            point=point,
        )


def _stage_payload(job_id: str, stage: StageName, attempt: int) -> dict[str, FrozenJsonValue]:
    return {"job_id": job_id, "stage_name": stage.value, "attempt": attempt}


def _require_outputs(refs: tuple[ArtifactRef, ...], stage: StageName) -> None:
    if not refs:
        raise AppError("stage.output_invalid", {"stage_name": stage.value})
    logical_names = [ref.logical_name for ref in refs]
    if len(set(logical_names)) != len(logical_names):
        raise AppError("stage.output_invalid", {"stage_name": stage.value})
