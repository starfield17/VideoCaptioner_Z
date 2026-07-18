"""Unit tests for Job detail and Activity projections."""

from __future__ import annotations

from typing import Any, cast

from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
from captioner.core.application.job_detail import (
    JobAction,
    JobDetailRequest,
    JobDetailService,
)
from captioner.core.domain.job import JobState
from captioner.core.domain.journal import JournalEvent
from captioner.core.domain.stage import StageName, StageState
from captioner.core.ports.batch_gateway import JobDetailSource


class FakeGateway:
    def __init__(self, source: JobDetailSource) -> None:
        self.source = source

    def read_job_detail_source(self, batch_id: str, job_id: str) -> JobDetailSource:
        del batch_id, job_id
        return self.source

    def create_batch(self, draft: object) -> object:
        raise NotImplementedError

    def execute_created_batch(self, batch_id: str) -> None:
        raise NotImplementedError

    def validate_resume(self, batch_id: str) -> None:
        raise NotImplementedError

    def resume_batch(self, batch_id: str) -> None:
        raise NotImplementedError

    def resolve_retry_stage(self, batch_id: str, job_id: str) -> StageName:
        raise NotImplementedError

    def retry_job(self, batch_id: str, job_id: str, stage: StageName) -> None:
        raise NotImplementedError

    def request_cancel(
        self, batch_id: str, *, job_id: str | None, execution_scheduled: bool
    ) -> None:
        raise NotImplementedError

    def request_pause(self, batch_id: str, *, execution_scheduled: bool) -> None:
        raise NotImplementedError

    def create_run_again(self, batch_id: str, job_id: str) -> object:
        raise NotImplementedError

    def read_recovery_sources(self) -> object:
        from captioner.core.ports.batch_gateway import RecoveryReadResult

        return RecoveryReadResult(sources=(), issues=())

    def close_shared_runtime(self) -> None:
        return None


def _event(
    seq: int,
    event_type: str,
    *,
    job_id: str | None = None,
    stage: str | None = None,
    attempt: int | None = None,
    error_code: str | None = None,
) -> JournalEvent:
    payload: dict[str, object] = {}
    if job_id is not None:
        payload["job_id"] = job_id
    if stage is not None:
        payload["stage_name"] = stage
    if attempt is not None:
        payload["attempt"] = attempt
    if error_code is not None:
        payload["error_code"] = error_code
    return JournalEvent(
        seq,
        f"event-{seq}",
        "2026-01-01T00:00:00+00:00",
        "batch-a",
        event_type,
        payload,  # type: ignore[arg-type]
    )


def test_activity_filters_siblings_and_limits() -> None:
    events = (
        _event(1, "batch.created"),
        _event(2, "job.created", job_id="job-000001"),
        _event(3, "job.created", job_id="job-000002"),
        _event(4, "stage.started", job_id="job-000001", stage="inspect", attempt=1),
        _event(5, "stage.committed", job_id="job-000001", stage="inspect", attempt=1),
    )
    source = JobDetailSource(
        batch_id="batch-a",
        job_id="job-000001",
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        state=JobState.FAILED,
        active_stage=StageName.SEGMENT,
        active_stage_state=StageState.FAILED,
        active_stage_attempt=1,
        lease_state="missing",
        cancel_requested=False,
        pause_requested=False,
        input_exists=True,
        batch_inputs_available=True,
        batch_has_nonterminal=True,
        batch_cancel_requested=False,
        job_cancel_requested=False,
        events=events,
        journal_tail_status="clean",
        manifest_status="missing",
        stage_states=(
            (StageName.INSPECT, StageState.COMMITTED),
            (StageName.NORMALIZE, StageState.COMMITTED),
            (StageName.TRANSCRIBE, StageState.COMMITTED),
            (StageName.SEGMENT, StageState.FAILED),
            (StageName.EXPORT, StageState.PENDING),
            (StageName.PUBLISH, StageState.PENDING),
        ),
        pipeline_profile="deterministic",
    )
    service = JobDetailService(cast(Any, FakeGateway(source)), SerialExecutionCoordinator())
    detail = service.load(
        JobDetailRequest(request_id="req-1", batch_id="batch-a", job_id="job-000001")
    )
    types = [entry.event_type for entry in detail.activity]
    assert "batch.created" in types
    assert "job.created" in types
    assert all(entry.job_id in {None, "job-000001"} for entry in detail.activity)
    assert JobAction.RETRY_JOB in detail.available_actions
    assert detail.retry_stage is StageName.SEGMENT
    for entry in detail.activity:
        assert not hasattr(entry, "payload")


def test_action_matrix_pause_and_run_again() -> None:
    events = (_event(1, "batch.created"), _event(2, "job.created", job_id="job-000001"))
    source = JobDetailSource(
        batch_id="batch-a",
        job_id="job-000001",
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        state=JobState.SUCCEEDED,
        active_stage=None,
        active_stage_state=None,
        active_stage_attempt=0,
        lease_state="missing",
        cancel_requested=False,
        pause_requested=False,
        input_exists=True,
        batch_inputs_available=True,
        batch_has_nonterminal=False,
        batch_cancel_requested=False,
        job_cancel_requested=False,
        events=events,
        journal_tail_status="clean",
        manifest_status="current",
        stage_states=tuple(
            (name, StageState.COMMITTED)
            for name in (
                StageName.INSPECT,
                StageName.NORMALIZE,
                StageName.TRANSCRIBE,
                StageName.SEGMENT,
                StageName.EXPORT,
                StageName.PUBLISH,
            )
        ),
        pipeline_profile="deterministic",
    )
    service = JobDetailService(cast(Any, FakeGateway(source)), SerialExecutionCoordinator())
    detail = service.load(
        JobDetailRequest(request_id="req-2", batch_id="batch-a", job_id="job-000001")
    )
    assert JobAction.RUN_AGAIN in detail.available_actions
    assert JobAction.CANCEL_JOB not in detail.available_actions


def test_resume_uses_batch_inputs_available() -> None:
    events = (_event(1, "batch.created"), _event(2, "job.created", job_id="job-000001"))
    source = JobDetailSource(
        batch_id="batch-a",
        job_id="job-000001",
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        state=JobState.INTERRUPTED,
        active_stage=StageName.NORMALIZE,
        active_stage_state=StageState.INTERRUPTED,
        active_stage_attempt=1,
        lease_state="missing",
        cancel_requested=False,
        pause_requested=False,
        input_exists=True,
        batch_inputs_available=False,
        batch_has_nonterminal=True,
        batch_cancel_requested=False,
        job_cancel_requested=False,
        events=events,
        journal_tail_status="clean",
        manifest_status="missing",
        stage_states=(
            (StageName.INSPECT, StageState.COMMITTED),
            (StageName.NORMALIZE, StageState.INTERRUPTED),
            (StageName.TRANSCRIBE, StageState.PENDING),
            (StageName.SEGMENT, StageState.PENDING),
            (StageName.EXPORT, StageState.PENDING),
            (StageName.PUBLISH, StageState.PENDING),
        ),
        pipeline_profile="deterministic",
    )
    service = JobDetailService(cast(Any, FakeGateway(source)), SerialExecutionCoordinator())
    detail = service.load(
        JobDetailRequest(request_id="req-3", batch_id="batch-a", job_id="job-000001")
    )
    # Selected Job input exists so Retry may remain available; Resume uses Batch inputs.
    assert JobAction.RESUME_BATCH not in detail.available_actions
    assert JobAction.RETRY_JOB in detail.available_actions
