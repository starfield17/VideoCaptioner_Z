"""Unit tests for JobDetailPanel widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.widgets.job_detail_panel import JobDetailPanel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-job-detail-panel"])


class FakeRunner(QObject):
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    local_execution_state_changed = Signal(object)
    execution_completion = Signal(object)

    def request_job_detail(self, request: object) -> None:
        return None


def test_required_object_names() -> None:
    service = I18nService("en")
    ops = JobOperationsController(FakeRunner())  # type: ignore[arg-type]
    panel = JobDetailPanel(service, ops)
    assert panel.objectName() == "jobDetailPanel"
    for name in (
        "jobDetailTitle",
        "jobDetailInputLabel",
        "jobDetailOutputLabel",
        "jobDetailStateLabel",
        "jobDetailStageLabel",
        "jobDetailAttemptLabel",
        "jobDetailFailureLabel",
        "jobCancelButton",
        "batchCancelButton",
        "batchPauseButton",
        "batchResumeButton",
        "jobRetryButton",
        "jobRunAgainButton",
        "jobActivitySummaryLabel",
        "jobActivityList",
        "jobActivityFailureLabel",
    ):
        assert panel.findChild(QObject, name) is not None, name


def test_render_detail_snapshot() -> None:
    from captioner.core.application.job_detail import (
        ActivityEntry,
        JobAction,
        JobDetailSnapshot,
    )
    from captioner.core.domain.job import JobState
    from captioner.core.domain.stage import StageName, StageState

    service = I18nService("en")
    ops = JobOperationsController(FakeRunner())  # type: ignore[arg-type]
    panel = JobDetailPanel(service, ops)
    detail = JobDetailSnapshot(
        schema_version=1,
        request_id="req",
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
        paused=False,
        input_exists=True,
        retry_stage=StageName.SEGMENT,
        available_actions=(
            JobAction.CANCEL_JOB,
            JobAction.CANCEL_BATCH,
            JobAction.RETRY_JOB,
            JobAction.RUN_AGAIN,
        ),
        activity=(
            ActivityEntry(
                seq=1,
                timestamp_utc="t0",
                event_type="job.failed",
                job_id="job-000001",
                stage_name=StageName.SEGMENT,
                attempt=1,
                error_code="stage.failed",
            ),
        ),
        omitted_activity_count=0,
        journal_tail_status="incomplete",
        manifest_status="current",
    )
    ops.detail_changed.emit(detail)
    assert panel.findChild(QObject, "jobRetryButton") is not None
