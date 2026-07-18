"""Startup recovery discovery projections over durable Batch sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from captioner.core.domain.batch import BatchState
from captioner.core.domain.job import JobState, validate_identifier
from captioner.core.ports.batch_catalog import LeaseExecutionState

if TYPE_CHECKING:
    from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
    from captioner.core.ports.batch_gateway import BatchGatewayPort

RECOVERY_SCHEMA_VERSION = 1
_STALE_LEASE_STATES: frozenset[LeaseExecutionState] = frozenset({"missing", "stale", "invalid"})
_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryIssue:
    batch_name: str
    code: str


@dataclass(frozen=True, slots=True)
class RecoveryItem:
    batch_id: str
    created_at_utc: str
    state: BatchState
    job_count: int
    pause_requested: bool
    missing_input_paths: tuple[str, ...]
    last_event_seq: int
    blocked_code: str | None

    @property
    def can_resume(self) -> bool:
        return self.blocked_code is None


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    schema_version: int
    request_id: str
    items: tuple[RecoveryItem, ...]
    issues: tuple[RecoveryIssue, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_SCHEMA_VERSION:
            raise ValueError("recovery.schema_version_invalid")


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    request_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")


@dataclass(slots=True)
class RecoveryService:
    gateway: BatchGatewayPort
    coordinator: SerialExecutionCoordinator

    def scan(self, request: RecoveryRequest) -> RecoverySnapshot:
        read = self.gateway.read_recovery_sources()
        scheduled = self.coordinator.scheduled_batch_ids()
        items: list[RecoveryItem] = []
        for source in read.sources:
            if source.batch_id in scheduled:
                continue
            if source.lease_state not in _STALE_LEASE_STATES:
                continue
            has_nonterminal = any(
                job.state not in _TERMINAL_JOB_STATES for job in source.projection.jobs
            )
            if not has_nonterminal:
                continue
            # Pending / interrupted / pause-batch candidates only.
            if not (
                source.pause_requested
                or source.state
                in {
                    BatchState.PENDING,
                    BatchState.INTERRUPTED,
                    BatchState.RUNNING,
                    BatchState.PARTIAL,
                    BatchState.FAILED,
                }
            ):
                continue
            blocked = "recovery.input_missing" if source.missing_input_paths else None
            items.append(
                RecoveryItem(
                    batch_id=source.batch_id,
                    created_at_utc=source.created_at_utc,
                    state=source.state,
                    job_count=source.job_count,
                    pause_requested=source.pause_requested,
                    missing_input_paths=source.missing_input_paths,
                    last_event_seq=source.last_event_seq,
                    blocked_code=blocked,
                )
            )
        items.sort(key=lambda item: (item.created_at_utc, item.batch_id))
        # Propagate catalog issues; keep stable ordering by batch_name then code.
        issues = tuple(
            sorted(
                (
                    RecoveryIssue(batch_name=issue.batch_name, code=issue.code)
                    for issue in read.issues
                ),
                key=lambda issue: (issue.batch_name, issue.code),
            )
        )
        return RecoverySnapshot(
            schema_version=RECOVERY_SCHEMA_VERSION,
            request_id=request.request_id,
            items=tuple(items),
            issues=issues,
        )


__all__ = [
    "RECOVERY_SCHEMA_VERSION",
    "RecoveryIssue",
    "RecoveryItem",
    "RecoveryRequest",
    "RecoveryService",
    "RecoverySnapshot",
]
