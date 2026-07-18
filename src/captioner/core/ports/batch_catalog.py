"""Read-only Batch catalog discovery boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from captioner.core.domain.batch import BatchProjection
from captioner.core.ports.manifest import ManifestStatus

LeaseExecutionState = Literal[
    "missing",
    "active_local",
    "active_remote",
    "stale",
    "invalid",
]


@dataclass(frozen=True, slots=True)
class BatchCatalogEntry:
    batch_id: str
    created_at_utc: str
    projection: BatchProjection
    journal_tail_status: Literal["clean", "incomplete"]
    manifest_status: ManifestStatus
    lease_state: LeaseExecutionState
    batch_cancel_requested: bool
    job_cancel_requests: frozenset[str]
    batch_pause_requested: bool


@dataclass(frozen=True, slots=True)
class BatchCatalogIssue:
    batch_name: str
    code: str


@dataclass(frozen=True, slots=True)
class BatchCatalogSnapshot:
    batches: tuple[BatchCatalogEntry, ...]
    issues: tuple[BatchCatalogIssue, ...]


class BatchCatalogPort(Protocol):
    def read_snapshot(self) -> BatchCatalogSnapshot: ...


__all__ = [
    "BatchCatalogEntry",
    "BatchCatalogIssue",
    "BatchCatalogPort",
    "BatchCatalogSnapshot",
    "LeaseExecutionState",
]
