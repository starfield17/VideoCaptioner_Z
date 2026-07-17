"""Immutable GUI-neutral Queue projections over durable Batch catalog state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from captioner.core.domain.job import JobProjection, JobState
from captioner.core.domain.stage import PipelineProfile, StageName, StageState
from captioner.core.ports.batch_catalog import (
    BatchCatalogEntry,
    BatchCatalogPort,
    BatchCatalogSnapshot,
    LeaseExecutionState,
)
from captioner.core.ports.manifest import ManifestStatus

QUEUE_SCHEMA_VERSION = 1
_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)
_ACTIVE_LEASE_STATES: frozenset[LeaseExecutionState] = frozenset(
    {
        "active_local",
        "active_remote",
    }
)
_STALE_RUNNING_LEASE_STATES: frozenset[LeaseExecutionState] = frozenset(
    {
        "missing",
        "stale",
        "invalid",
    }
)


@dataclass(frozen=True, slots=True)
class QueueLoadIssue:
    batch_name: str
    code: str


@dataclass(frozen=True, slots=True)
class JobQueueItem:
    batch_id: str
    job_id: str
    batch_created_at_utc: str
    job_order: int
    input_path: str
    output_dir: str
    pipeline_profile: PipelineProfile
    state: JobState
    active_stage: StageName | None
    active_stage_state: StageState | None
    active_stage_attempt: int
    cancel_requested: bool
    last_event_seq: int
    journal_tail_status: Literal["clean", "incomplete"]
    manifest_status: ManifestStatus

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_JOB_STATES


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    schema_version: int
    revision: int
    items: tuple[JobQueueItem, ...]
    issues: tuple[QueueLoadIssue, ...]
    omitted_terminal_jobs: int

    def __post_init__(self) -> None:
        if self.schema_version != QUEUE_SCHEMA_VERSION:
            raise ValueError("queue.schema_version_invalid")
        if self.revision < 1:
            raise ValueError("queue.revision_invalid")
        if self.omitted_terminal_jobs < 0:
            raise ValueError("queue.omitted_terminal_jobs_invalid")

    @property
    def active_count(self) -> int:
        return sum(1 for item in self.items if not item.terminal)

    @property
    def terminal_count(self) -> int:
        return sum(1 for item in self.items if item.terminal)


@dataclass(slots=True)
class QueueProjectionService:
    catalog: BatchCatalogPort
    recent_terminal_limit: int = 100
    _revision: int = field(default=0, init=False, repr=False)
    _last_signature: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _subscribers: list[Callable[[QueueSnapshot], None]] = field(
        default_factory=lambda: [],
        init=False,
        repr=False,
    )
    _current_snapshot: QueueSnapshot | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.recent_terminal_limit < 0:
            raise ValueError("queue.recent_terminal_limit_invalid")

    def get_queue_snapshot(self) -> QueueSnapshot:
        return self.refresh_queue()

    def refresh_queue(self) -> QueueSnapshot:
        catalog = self.catalog.read_snapshot()
        items, omitted = _select_items(catalog, self.recent_terminal_limit)
        issues = tuple(QueueLoadIssue(issue.batch_name, issue.code) for issue in catalog.issues)
        signature = _snapshot_signature(items, issues, omitted)
        if self._last_signature == signature and self._current_snapshot is not None:
            return self._current_snapshot
        revision = self._revision + 1
        snapshot = QueueSnapshot(
            schema_version=QUEUE_SCHEMA_VERSION,
            revision=revision,
            items=items,
            issues=issues,
            omitted_terminal_jobs=omitted,
        )
        self._revision = revision
        self._last_signature = signature
        self._current_snapshot = snapshot
        for callback in list(self._subscribers):
            callback(snapshot)
        return snapshot

    def subscribe_queue(
        self,
        callback: Callable[[QueueSnapshot], None],
    ) -> Callable[[], None]:
        self._subscribers.append(callback)
        active = True

        def unsubscribe() -> None:
            nonlocal active
            if not active:
                return
            active = False
            try:
                self._subscribers.remove(callback)
            except ValueError:
                return

        return unsubscribe


def _select_items(
    catalog: BatchCatalogSnapshot,
    recent_terminal_limit: int,
) -> tuple[tuple[JobQueueItem, ...], int]:
    all_items = tuple(item for entry in catalog.batches for item in _items_for_batch(entry))
    active = [item for item in all_items if not item.terminal]
    terminal = sorted(
        (item for item in all_items if item.terminal),
        key=_submission_key,
    )
    if recent_terminal_limit == 0:
        retained_terminal: list[JobQueueItem] = []
        omitted = len(terminal)
    elif len(terminal) <= recent_terminal_limit:
        retained_terminal = terminal
        omitted = 0
    else:
        retained_terminal = terminal[-recent_terminal_limit:]
        omitted = len(terminal) - recent_terminal_limit
    selected = sorted([*active, *retained_terminal], key=_submission_key)
    return tuple(selected), omitted


def _items_for_batch(entry: BatchCatalogEntry) -> tuple[JobQueueItem, ...]:
    return tuple(
        _item_for_job(entry, job, job_order) for job_order, job in enumerate(entry.projection.jobs)
    )


def _item_for_job(
    entry: BatchCatalogEntry,
    job: JobProjection,
    job_order: int,
) -> JobQueueItem:
    state = _projected_job_state(job.state, entry.lease_state)
    active_stage, active_stage_state, active_stage_attempt = _active_stage(job)
    # Stale/missing/invalid leases project RUNNING Jobs as INTERRUPTED. The
    # active Stage must match so Queue rows never show "Interrupted / Running".
    if state is JobState.INTERRUPTED and active_stage_state is StageState.RUNNING:
        active_stage_state = StageState.INTERRUPTED
    cancel_requested = entry.batch_cancel_requested or job.job_id in entry.job_cancel_requests
    return JobQueueItem(
        batch_id=entry.batch_id,
        job_id=job.job_id,
        batch_created_at_utc=entry.created_at_utc,
        job_order=job_order,
        input_path=job.input_path,
        output_dir=job.config.output_dir,
        pipeline_profile=job.config.pipeline_profile,
        state=state,
        active_stage=active_stage,
        active_stage_state=active_stage_state,
        active_stage_attempt=active_stage_attempt,
        cancel_requested=cancel_requested,
        last_event_seq=entry.projection.last_event_seq,
        journal_tail_status=entry.journal_tail_status,
        manifest_status=entry.manifest_status,
    )


def _projected_job_state(
    state: JobState,
    lease_state: LeaseExecutionState,
) -> JobState:
    if state is JobState.RUNNING and lease_state in _STALE_RUNNING_LEASE_STATES:
        return JobState.INTERRUPTED
    if state is JobState.RUNNING and lease_state in _ACTIVE_LEASE_STATES:
        return JobState.RUNNING
    return state


def _active_stage(
    job: JobProjection,
) -> tuple[StageName | None, StageState | None, int]:
    for stage in job.stages:
        if stage.state is StageState.RUNNING:
            return stage.name, stage.state, stage.attempt
    for stage in job.stages:
        if stage.state in {
            StageState.INTERRUPTED,
            StageState.FAILED,
            StageState.CANCELLED,
        }:
            return stage.name, stage.state, stage.attempt
    for stage in job.stages:
        if stage.state is not StageState.COMMITTED:
            return stage.name, stage.state, stage.attempt
    return None, None, 0


def _submission_key(item: JobQueueItem) -> tuple[str, str, int, str]:
    return (
        item.batch_created_at_utc,
        item.batch_id,
        item.job_order,
        item.job_id,
    )


def _snapshot_signature(
    items: tuple[JobQueueItem, ...],
    issues: tuple[QueueLoadIssue, ...],
    omitted_terminal_jobs: int,
) -> tuple[object, ...]:
    return (
        tuple(
            (
                item.batch_id,
                item.job_id,
                item.batch_created_at_utc,
                item.job_order,
                item.input_path,
                item.output_dir,
                item.pipeline_profile.value,
                item.state.value,
                None if item.active_stage is None else item.active_stage.value,
                None if item.active_stage_state is None else item.active_stage_state.value,
                item.active_stage_attempt,
                item.cancel_requested,
                item.last_event_seq,
                item.journal_tail_status,
                item.manifest_status,
            )
            for item in items
        ),
        tuple((issue.batch_name, issue.code) for issue in issues),
        omitted_terminal_jobs,
    )


__all__ = [
    "JobQueueItem",
    "QueueLoadIssue",
    "QueueProjectionService",
    "QueueSnapshot",
]
