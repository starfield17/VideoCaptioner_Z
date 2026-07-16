"""Durable Batch projection and deterministic aggregate state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobProjection, JobState, validate_identifier


class BatchState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class BatchProjection:
    batch_id: str
    jobs: tuple[JobProjection, ...] = ()
    last_event_seq: int = 0
    event_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_identifier(self.batch_id, field="batch_id")

    @property
    def state(self) -> BatchState:
        if not self.jobs or all(job.state is JobState.PENDING for job in self.jobs):
            return BatchState.PENDING
        states = {job.state for job in self.jobs}
        if JobState.RUNNING in states:
            return BatchState.RUNNING
        if JobState.INTERRUPTED in states:
            return BatchState.INTERRUPTED
        if JobState.FAILED in states:
            return BatchState.FAILED
        if states == {JobState.SUCCEEDED}:
            return BatchState.SUCCEEDED
        if states == {JobState.CANCELLED}:
            return BatchState.CANCELLED
        if states <= {JobState.SUCCEEDED, JobState.CANCELLED}:
            return BatchState.PARTIAL
        return BatchState.PARTIAL

    def job(self, job_id: str) -> JobProjection:
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise AppError("batch.job_not_found", {"job_id": job_id})
