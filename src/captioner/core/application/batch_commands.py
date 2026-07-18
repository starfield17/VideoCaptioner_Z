"""Batch command DTOs and Application command service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from captioner.core.application.input_selection import BatchDraft
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier

if TYPE_CHECKING:
    from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
    from captioner.core.ports.batch_gateway import BatchGatewayPort


class BatchCommandKind(StrEnum):
    SUBMIT = "submit"
    RESUME_BATCH = "resume_batch"
    RETRY_JOB = "retry_job"
    CANCEL_JOB = "cancel_job"
    CANCEL_BATCH = "cancel_batch"
    PAUSE_BATCH = "pause_batch"
    RUN_AGAIN = "run_again"
    CANCEL_LOCAL_WORK = "cancel_local_work"


@dataclass(frozen=True, slots=True)
class SubmitBatchRequest:
    request_id: str
    draft: BatchDraft

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")


@dataclass(frozen=True, slots=True)
class BatchActionRequest:
    request_id: str
    kind: Literal[
        BatchCommandKind.RESUME_BATCH,
        BatchCommandKind.CANCEL_BATCH,
        BatchCommandKind.PAUSE_BATCH,
    ]
    batch_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")
        validate_identifier(self.batch_id, field="batch_id")
        if self.kind not in {
            BatchCommandKind.RESUME_BATCH,
            BatchCommandKind.CANCEL_BATCH,
            BatchCommandKind.PAUSE_BATCH,
        }:
            raise AppError("batch.command_invalid", {"field": "kind"})


@dataclass(frozen=True, slots=True)
class JobActionRequest:
    request_id: str
    kind: Literal[
        BatchCommandKind.RETRY_JOB,
        BatchCommandKind.CANCEL_JOB,
        BatchCommandKind.RUN_AGAIN,
    ]
    batch_id: str
    job_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")
        validate_identifier(self.batch_id, field="batch_id")
        validate_identifier(self.job_id, field="job_id")
        if self.kind not in {
            BatchCommandKind.RETRY_JOB,
            BatchCommandKind.CANCEL_JOB,
            BatchCommandKind.RUN_AGAIN,
        }:
            raise AppError("batch.command_invalid", {"field": "kind"})


@dataclass(frozen=True, slots=True)
class CancelLocalWorkRequest:
    request_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")


@dataclass(frozen=True, slots=True)
class BatchCommandAck:
    request_id: str
    kind: BatchCommandKind
    batch_id: str | None
    job_id: str | None
    accepted_at_utc: str
    scheduled: bool
    created_batch_id: str | None = None
    affected_batch_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchCommandFailure:
    request_id: str
    kind: BatchCommandKind
    code: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class LocalExecutionSnapshot:
    active_batch_id: str | None
    queued_batch_ids: tuple[str, ...]

    @property
    def has_work(self) -> bool:
        return self.active_batch_id is not None or bool(self.queued_batch_ids)


@dataclass(frozen=True, slots=True)
class ExecutionCompletion:
    batch_id: str
    kind: BatchCommandKind
    job_id: str | None
    ok: bool
    code: str


@dataclass(slots=True)
class BatchCommandService:
    gateway: BatchGatewayPort
    coordinator: SerialExecutionCoordinator
    now_utc: Callable[[], str]

    def submit(self, request: SubmitBatchRequest) -> BatchCommandAck:
        created = self.gateway.create_batch(request.draft)
        batch_id = created.batch_id
        self.coordinator.schedule(
            batch_id=batch_id,
            kind=BatchCommandKind.SUBMIT,
            job_id=None,
            operation=lambda: self.gateway.execute_created_batch(batch_id),
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.SUBMIT,
            batch_id=batch_id,
            job_id=None,
            accepted_at_utc=self.now_utc(),
            scheduled=True,
            created_batch_id=batch_id,
        )

    def perform_batch_action(self, request: BatchActionRequest) -> BatchCommandAck:
        if request.kind is BatchCommandKind.RESUME_BATCH:
            return self._resume_batch(request)
        if request.kind is BatchCommandKind.PAUSE_BATCH:
            return self._pause_batch(request)
        if request.kind is BatchCommandKind.CANCEL_BATCH:
            return self._cancel_batch(request)
        raise AppError("batch.command_invalid", {"field": "kind"})

    def perform_job_action(self, request: JobActionRequest) -> BatchCommandAck:
        if request.kind is BatchCommandKind.RETRY_JOB:
            return self._retry_job(request)
        if request.kind is BatchCommandKind.CANCEL_JOB:
            return self._cancel_job(request)
        if request.kind is BatchCommandKind.RUN_AGAIN:
            return self._run_again(request)
        raise AppError("batch.command_invalid", {"field": "kind"})

    def cancel_local_work(self, request: CancelLocalWorkRequest) -> BatchCommandAck:
        snapshot = self.coordinator.snapshot()
        affected: list[str] = []
        for batch_id in snapshot.queued_batch_ids:
            cancelled = self.coordinator.cancel_queued(batch_id)
            if cancelled:
                self.gateway.request_cancel(
                    batch_id,
                    job_id=None,
                    execution_scheduled=False,
                )
                affected.append(batch_id)
        if snapshot.active_batch_id is not None:
            active = snapshot.active_batch_id
            self.gateway.request_cancel(
                active,
                job_id=None,
                execution_scheduled=True,
            )
            if active not in affected:
                affected.append(active)
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.CANCEL_LOCAL_WORK,
            batch_id=None,
            job_id=None,
            accepted_at_utc=self.now_utc(),
            scheduled=False,
            affected_batch_ids=tuple(affected),
        )

    def _resume_batch(self, request: BatchActionRequest) -> BatchCommandAck:
        batch_id = request.batch_id
        if batch_id in self.coordinator.scheduled_batch_ids():
            raise AppError("batch.operation_conflict", {"batch_id": batch_id})
        self.coordinator.schedule(
            batch_id=batch_id,
            kind=BatchCommandKind.RESUME_BATCH,
            job_id=None,
            operation=lambda: self.gateway.resume_batch(batch_id),
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.RESUME_BATCH,
            batch_id=batch_id,
            job_id=None,
            accepted_at_utc=self.now_utc(),
            scheduled=True,
        )

    def _pause_batch(self, request: BatchActionRequest) -> BatchCommandAck:
        batch_id = request.batch_id
        scheduled = batch_id in self.coordinator.scheduled_batch_ids()
        self.gateway.request_pause(batch_id, execution_scheduled=scheduled)
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.PAUSE_BATCH,
            batch_id=batch_id,
            job_id=None,
            accepted_at_utc=self.now_utc(),
            scheduled=False,
        )

    def _cancel_batch(self, request: BatchActionRequest) -> BatchCommandAck:
        batch_id = request.batch_id
        scheduled = batch_id in self.coordinator.scheduled_batch_ids()
        if scheduled:
            self.coordinator.cancel_queued(batch_id)
        still_scheduled = batch_id in self.coordinator.scheduled_batch_ids()
        self.gateway.request_cancel(
            batch_id,
            job_id=None,
            execution_scheduled=still_scheduled,
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.CANCEL_BATCH,
            batch_id=batch_id,
            job_id=None,
            accepted_at_utc=self.now_utc(),
            scheduled=False,
        )

    def _cancel_job(self, request: JobActionRequest) -> BatchCommandAck:
        batch_id = request.batch_id
        job_id = request.job_id
        scheduled = batch_id in self.coordinator.scheduled_batch_ids()
        self.gateway.request_cancel(
            batch_id,
            job_id=job_id,
            execution_scheduled=scheduled,
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.CANCEL_JOB,
            batch_id=batch_id,
            job_id=job_id,
            accepted_at_utc=self.now_utc(),
            scheduled=False,
        )

    def _retry_job(self, request: JobActionRequest) -> BatchCommandAck:
        batch_id = request.batch_id
        job_id = request.job_id
        if batch_id in self.coordinator.scheduled_batch_ids():
            raise AppError("batch.operation_conflict", {"batch_id": batch_id})

        # Resolve earliest stage under gateway validation during schedule.
        def _retry() -> None:
            self.gateway.retry_job(batch_id, job_id)

        self.coordinator.schedule(
            batch_id=batch_id,
            kind=BatchCommandKind.RETRY_JOB,
            job_id=job_id,
            operation=_retry,
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.RETRY_JOB,
            batch_id=batch_id,
            job_id=job_id,
            accepted_at_utc=self.now_utc(),
            scheduled=True,
        )

    def _run_again(self, request: JobActionRequest) -> BatchCommandAck:
        created = self.gateway.create_run_again(request.batch_id, request.job_id)
        new_batch_id = created.batch_id
        self.coordinator.schedule(
            batch_id=new_batch_id,
            kind=BatchCommandKind.RUN_AGAIN,
            job_id=None,
            operation=lambda: self.gateway.execute_created_batch(new_batch_id),
        )
        return BatchCommandAck(
            request_id=request.request_id,
            kind=BatchCommandKind.RUN_AGAIN,
            batch_id=request.batch_id,
            job_id=request.job_id,
            accepted_at_utc=self.now_utc(),
            scheduled=True,
            created_batch_id=new_batch_id,
        )


__all__ = [
    "BatchActionRequest",
    "BatchCommandAck",
    "BatchCommandFailure",
    "BatchCommandKind",
    "BatchCommandService",
    "CancelLocalWorkRequest",
    "ExecutionCompletion",
    "JobActionRequest",
    "LocalExecutionSnapshot",
    "SubmitBatchRequest",
]
