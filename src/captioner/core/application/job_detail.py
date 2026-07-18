"""Job detail and Activity Log projections over durable Journal sources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState, validate_identifier
from captioner.core.domain.journal import JournalEvent
from captioner.core.domain.stage import StageName, StageState, stage_plan_for
from captioner.core.ports.batch_catalog import LeaseExecutionState
from captioner.core.ports.batch_gateway import JobDetailSource
from captioner.core.ports.manifest import ManifestStatus

if TYPE_CHECKING:
    from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
    from captioner.core.ports.batch_gateway import BatchGatewayPort

JOB_DETAIL_SCHEMA_VERSION = 1
_ACTIVITY_LIMIT = 200
_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)
_STALE_LEASE_STATES: frozenset[LeaseExecutionState] = frozenset({"missing", "stale", "invalid"})
_ACTIVE_LEASE_STATES: frozenset[LeaseExecutionState] = frozenset({"active_local", "active_remote"})


class JobAction(StrEnum):
    CANCEL_JOB = "cancel_job"
    CANCEL_BATCH = "cancel_batch"
    PAUSE_BATCH = "pause_batch"
    RESUME_BATCH = "resume_batch"
    RETRY_JOB = "retry_job"
    RUN_AGAIN = "run_again"


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    seq: int
    timestamp_utc: str
    event_type: str
    job_id: str | None
    stage_name: StageName | None
    attempt: int | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class JobDetailSnapshot:
    schema_version: int
    request_id: str
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
    paused: bool
    input_exists: bool
    retry_stage: StageName | None
    available_actions: tuple[JobAction, ...]
    activity: tuple[ActivityEntry, ...]
    omitted_activity_count: int
    journal_tail_status: Literal["clean", "incomplete"]
    manifest_status: ManifestStatus

    def __post_init__(self) -> None:
        if self.schema_version != JOB_DETAIL_SCHEMA_VERSION:
            raise ValueError("job_detail.schema_version_invalid")


@dataclass(frozen=True, slots=True)
class JobDetailRequest:
    request_id: str
    batch_id: str
    job_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")
        validate_identifier(self.batch_id, field="batch_id")
        validate_identifier(self.job_id, field="job_id")


@dataclass(slots=True)
class JobDetailService:
    gateway: BatchGatewayPort
    coordinator: SerialExecutionCoordinator

    def load(self, request: JobDetailRequest) -> JobDetailSnapshot:
        source = self.gateway.read_job_detail_source(request.batch_id, request.job_id)
        activity, omitted = _select_activity(source.events, request.job_id)
        terminal = source.state in _TERMINAL_JOB_STATES
        pause_requested = bool(source.pause_requested and not terminal)
        paused = pause_requested and source.lease_state in _STALE_LEASE_STATES
        scheduled = request.batch_id in self.coordinator.scheduled_batch_ids()
        retry_stage = _resolve_retry_stage(source)
        actions = _available_actions(
            source,
            pause_requested=pause_requested,
            paused=paused,
            scheduled=scheduled,
            retry_stage=retry_stage,
        )
        return JobDetailSnapshot(
            schema_version=JOB_DETAIL_SCHEMA_VERSION,
            request_id=request.request_id,
            batch_id=source.batch_id,
            job_id=source.job_id,
            input_path=source.input_path,
            output_dir=source.output_dir,
            state=source.state,
            active_stage=source.active_stage,
            active_stage_state=source.active_stage_state,
            active_stage_attempt=source.active_stage_attempt,
            lease_state=source.lease_state,
            cancel_requested=source.cancel_requested,
            pause_requested=pause_requested,
            paused=paused,
            input_exists=source.input_exists,
            retry_stage=retry_stage,
            available_actions=actions,
            activity=activity,
            omitted_activity_count=omitted,
            journal_tail_status=source.journal_tail_status,
            manifest_status=source.manifest_status,
        )


def _select_activity(
    events: tuple[JournalEvent, ...],
    job_id: str,
) -> tuple[tuple[ActivityEntry, ...], int]:
    selected: list[ActivityEntry] = []
    for event in events:
        event_job = event.payload.get("job_id")
        if event_job is not None and event_job != job_id:
            continue
        stage_raw = event.payload.get("stage_name")
        stage_name: StageName | None = None
        if isinstance(stage_raw, str):
            try:
                stage_name = StageName(stage_raw)
            except ValueError:
                stage_name = None
        attempt_raw = event.payload.get("attempt")
        attempt = attempt_raw if isinstance(attempt_raw, int) else None
        error_raw = event.payload.get("error_code")
        error_code = error_raw if isinstance(error_raw, str) else None
        selected.append(
            ActivityEntry(
                seq=event.seq,
                timestamp_utc=event.timestamp_utc,
                event_type=event.type,
                job_id=event_job if isinstance(event_job, str) else None,
                stage_name=stage_name,
                attempt=attempt,
                error_code=error_code,
            )
        )
    omitted = max(0, len(selected) - _ACTIVITY_LIMIT)
    retained = selected[-_ACTIVITY_LIMIT:] if omitted else selected
    return tuple(retained), omitted


def _resolve_retry_stage(source: JobDetailSource) -> StageName | None:
    if source.state not in {JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}:
        return None
    if not source.input_exists:
        return None
    plan = stage_plan_for(source.pipeline_profile)
    states = dict(source.stage_states)
    for index, name in enumerate(plan):
        state = states.get(name, StageState.PENDING)
        if state is StageState.COMMITTED:
            continue
        # First non-committed Stage is retryable only when priors are committed.
        priors = plan[:index]
        if any(
            states.get(prior, StageState.PENDING) is not StageState.COMMITTED for prior in priors
        ):
            return None
        return name
    return None


def _available_actions(
    source: JobDetailSource,
    *,
    pause_requested: bool,
    paused: bool,
    scheduled: bool,
    retry_stage: StageName | None,
) -> tuple[JobAction, ...]:
    actions: list[JobAction] = []
    terminal = source.state in _TERMINAL_JOB_STATES
    if not terminal and not source.job_cancel_requested and not source.batch_cancel_requested:
        actions.append(JobAction.CANCEL_JOB)
    if source.batch_has_nonterminal and not source.batch_cancel_requested:
        actions.append(JobAction.CANCEL_BATCH)
    if (
        scheduled
        and source.lease_state == "active_local"
        and not pause_requested
        and not source.batch_cancel_requested
    ):
        actions.append(JobAction.PAUSE_BATCH)
    elif (
        scheduled
        and not pause_requested
        and not source.batch_cancel_requested
        and source.lease_state not in _ACTIVE_LEASE_STATES
    ):
        # Locally queued (scheduled but lease not yet active).
        actions.append(JobAction.PAUSE_BATCH)

    if (
        not scheduled
        and source.lease_state in _STALE_LEASE_STATES
        and source.batch_has_nonterminal
        and source.batch_inputs_available
        and not source.batch_cancel_requested
        and (paused or pause_requested or source.state in {JobState.PENDING, JobState.INTERRUPTED})
    ):
        actions.append(JobAction.RESUME_BATCH)

    if (
        retry_stage is not None
        and not scheduled
        and source.input_exists
        and source.state in {JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}
    ):
        actions.append(JobAction.RETRY_JOB)
    if terminal and source.input_exists:
        actions.append(JobAction.RUN_AGAIN)
    return tuple(actions)


def resolve_earliest_retry_stage(
    stage_states: tuple[tuple[StageName, StageState], ...],
    pipeline_profile: str,
) -> StageName:
    """Return the first non-committed Stage with committed dependencies."""
    plan = stage_plan_for(pipeline_profile)
    states = dict(stage_states)
    for index, name in enumerate(plan):
        state = states.get(name, StageState.PENDING)
        if state is StageState.COMMITTED:
            continue
        priors = plan[:index]
        if any(
            states.get(prior, StageState.PENDING) is not StageState.COMMITTED for prior in priors
        ):
            raise AppError("batch.retry_invalid", {"reason": "dependencies"})
        return name
    raise AppError("batch.retry_invalid", {"reason": "no_retryable_stage"})


__all__ = [
    "JOB_DETAIL_SCHEMA_VERSION",
    "ActivityEntry",
    "JobAction",
    "JobDetailRequest",
    "JobDetailService",
    "JobDetailSnapshot",
    "resolve_earliest_retry_stage",
]
