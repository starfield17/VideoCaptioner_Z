"""Application port for durable Batch creation, execution, and recovery reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from captioner.core.application.input_selection import BatchDraft
from captioner.core.domain.batch import BatchProjection, BatchState
from captioner.core.domain.job import JobState
from captioner.core.domain.journal import JournalEvent
from captioner.core.domain.stage import StageName, StageState
from captioner.core.ports.batch_catalog import LeaseExecutionState
from captioner.core.ports.manifest import ManifestStatus


@dataclass(frozen=True, slots=True)
class CreatedBatch:
    batch_id: str
    job_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JobDetailSource:
    batch_id: str
    job_id: str
    input_path: str
    output_dir: str
    state: JobState
    active_stage: StageName | None
    active_stage_state: StageState | None
    active_stage_attempt: int
    lease_state: LeaseExecutionState
    cancel_requested: bool
    pause_requested: bool
    input_exists: bool
    batch_inputs_available: bool
    batch_has_nonterminal: bool
    batch_cancel_requested: bool
    job_cancel_requested: bool
    events: tuple[JournalEvent, ...]
    journal_tail_status: Literal["clean", "incomplete"]
    manifest_status: ManifestStatus
    stage_states: tuple[tuple[StageName, StageState], ...]
    pipeline_profile: str


@dataclass(frozen=True, slots=True)
class RecoverySource:
    batch_id: str
    created_at_utc: str
    state: BatchState
    job_count: int
    pause_requested: bool
    missing_input_paths: tuple[str, ...]
    last_event_seq: int
    lease_state: LeaseExecutionState
    projection: BatchProjection


@dataclass(frozen=True, slots=True)
class RecoverySourceIssue:
    batch_name: str
    code: str


@dataclass(frozen=True, slots=True)
class RecoveryReadResult:
    sources: tuple[RecoverySource, ...]
    issues: tuple[RecoverySourceIssue, ...]


class BatchGatewayPort(Protocol):
    def create_batch(self, draft: BatchDraft) -> CreatedBatch: ...

    def execute_created_batch(self, batch_id: str) -> None: ...

    def validate_resume(self, batch_id: str) -> None: ...

    def resume_batch(self, batch_id: str) -> None: ...

    def resolve_retry_stage(self, batch_id: str, job_id: str) -> StageName: ...

    def retry_job(self, batch_id: str, job_id: str, stage: StageName) -> None: ...

    def request_cancel(
        self,
        batch_id: str,
        *,
        job_id: str | None,
        execution_scheduled: bool,
    ) -> None: ...

    def request_pause(
        self,
        batch_id: str,
        *,
        execution_scheduled: bool,
    ) -> None: ...

    def create_run_again(
        self,
        batch_id: str,
        job_id: str,
    ) -> CreatedBatch: ...

    def read_job_detail_source(
        self,
        batch_id: str,
        job_id: str,
    ) -> JobDetailSource: ...

    def read_recovery_sources(self) -> RecoveryReadResult: ...

    def close_shared_runtime(self) -> None: ...


__all__ = [
    "BatchGatewayPort",
    "CreatedBatch",
    "JobDetailSource",
    "RecoveryReadResult",
    "RecoverySource",
    "RecoverySourceIssue",
]
