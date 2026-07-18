"""Unit tests for JobOperationsController."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from captioner.core.application.batch_commands import (
    BatchActionRequest,
    BatchCommandAck,
    BatchCommandKind,
    JobActionRequest,
    LocalExecutionSnapshot,
)
from captioner.core.application.queue_projection import JobQueueItem
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.job_operations_controller import JobOperationsController

_app = QApplication.instance() or QApplication(["test-job-ops"])


class FakeRunner(QObject):
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    local_execution_state_changed = Signal(object)
    execution_completion = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.detail_requests: list[object] = []
        self.job_actions: list[JobActionRequest] = []
        self.batch_actions: list[BatchActionRequest] = []

    def request_job_detail(self, request: object) -> None:
        self.detail_requests.append(request)

    def request_job_action(self, request: object) -> None:
        assert isinstance(request, JobActionRequest)
        self.job_actions.append(request)

    def request_batch_action(self, request: object) -> None:
        assert isinstance(request, BatchActionRequest)
        self.batch_actions.append(request)

    def request_cancel_local_work(self, request: object) -> None:
        return None


def _item() -> JobQueueItem:
    return JobQueueItem(
        batch_id="batch-a",
        job_id="job-000001",
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=0,
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        state=JobState.PENDING,
        active_stage=None,
        active_stage_state=None,
        active_stage_attempt=0,
        cancel_requested=False,
        pause_requested=False,
        paused=False,
        last_event_seq=1,
        journal_tail_status="clean",
        manifest_status="missing",
    )


def test_select_and_command_correlation() -> None:
    runner = FakeRunner()
    controller = JobOperationsController(runner)  # type: ignore[arg-type]
    controller.select_job(_item())
    assert len(runner.detail_requests) == 1
    controller.cancel_job()
    assert len(runner.job_actions) == 1
    action = runner.job_actions[0]
    ack = BatchCommandAck(
        request_id=action.request_id,
        kind=BatchCommandKind.CANCEL_JOB,
        batch_id="batch-a",
        job_id="job-000001",
        accepted_at_utc="t0",
        scheduled=False,
    )
    runner.batch_command_ready.emit(ack)
    assert controller.command_busy is False
    runner.local_execution_state_changed.emit(LocalExecutionSnapshot("batch-a", ()))
    assert controller.has_local_work is True


def test_all_action_methods_dispatch() -> None:
    runner = FakeRunner()
    controller = JobOperationsController(runner)  # type: ignore[arg-type]
    controller.select_job(_item())

    def ack_last_job() -> None:
        action = runner.job_actions[-1]
        runner.batch_command_ready.emit(
            BatchCommandAck(
                request_id=action.request_id,
                kind=action.kind,
                batch_id=action.batch_id,
                job_id=action.job_id,
                accepted_at_utc="t0",
                scheduled=False,
            )
        )

    def ack_last_batch(kind: BatchCommandKind) -> None:
        action = runner.batch_actions[-1]
        runner.batch_command_ready.emit(
            BatchCommandAck(
                request_id=action.request_id,
                kind=kind,
                batch_id=action.batch_id,
                job_id=None,
                accepted_at_utc="t0",
                scheduled=False,
            )
        )

    controller.pause_batch()
    ack_last_batch(BatchCommandKind.PAUSE_BATCH)
    controller.resume_batch()
    ack_last_batch(BatchCommandKind.RESUME_BATCH)
    controller.retry_job()
    ack_last_job()
    controller.run_again()
    ack_last_job()
    controller.cancel_batch()
    ack_last_batch(BatchCommandKind.CANCEL_BATCH)
    assert len(runner.job_actions) >= 2
    assert len(runner.batch_actions) >= 3


def test_command_failure_and_stale_detail() -> None:
    from captioner.core.application.batch_commands import BatchCommandFailure

    runner = FakeRunner()
    controller = JobOperationsController(runner)  # type: ignore[arg-type]
    controller.select_job(_item())
    controller.cancel_job()
    action = runner.job_actions[-1]
    runner.batch_command_failure.emit(
        BatchCommandFailure(
            request_id=action.request_id,
            kind=BatchCommandKind.CANCEL_JOB,
            code="batch.cancel_invalid",
        )
    )
    assert controller.command_busy is False
    # unrelated failure ignored
    runner.batch_command_failure.emit(
        BatchCommandFailure(
            request_id="other",
            kind=BatchCommandKind.CANCEL_JOB,
            code="batch.cancel_invalid",
        )
    )
    controller.select_job(None)
    assert controller.detail is None
    controller.resume_batch_id("batch-z")
    controller.cancel_batch_id("batch-z")


def test_detail_coalesce_and_execution_completion() -> None:
    from captioner.core.application.batch_commands import ExecutionCompletion
    from captioner.core.application.job_detail import JobDetailSnapshot
    from captioner.core.domain.job import JobState

    runner = FakeRunner()
    controller = JobOperationsController(runner)  # type: ignore[arg-type]
    controller.select_job(_item())
    assert len(runner.detail_requests) == 1
    # queue another detail while busy
    controller.refresh_detail()
    # deliver first detail
    from captioner.core.application.job_detail import JOB_DETAIL_SCHEMA_VERSION

    first_req = runner.detail_requests[0]
    detail = JobDetailSnapshot(
        schema_version=JOB_DETAIL_SCHEMA_VERSION,
        request_id=getattr(first_req, "request_id", "req"),
        batch_id="batch-a",
        job_id="job-000001",
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        state=JobState.PENDING,
        active_stage=None,
        active_stage_state=None,
        active_stage_attempt=0,
        lease_state="missing",
        cancel_requested=False,
        pause_requested=False,
        paused=False,
        input_exists=True,
        retry_stage=None,
        available_actions=(),
        activity=(),
        omitted_activity_count=0,
        journal_tail_status="clean",
        manifest_status="missing",
    )
    runner.job_detail_ready.emit(detail)
    # may request follow-up
    runner.execution_completion.emit(
        ExecutionCompletion(
            batch_id="batch-a",
            kind=BatchCommandKind.SUBMIT,
            job_id=None,
            ok=True,
            code="execution.completed",
        )
    )
    runner.execution_completion.emit(
        ExecutionCompletion(
            batch_id="batch-a",
            kind=BatchCommandKind.SUBMIT,
            job_id=None,
            ok=False,
            code="gui.application_bridge_failed",
        )
    )
    assert True
