"""Unit tests for the Queue page presentation."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStackedWidget, QTableView

from captioner.core.application.queue_projection import (
    JobQueueItem,
    QueueLoadIssue,
    QueueSnapshot,
)
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName, StageState
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.batch_controller import BatchController
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.pages.queue_page import QueuePage
from captioner.gui.queue_table_model import QueueTableModel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-queue-page"])


def _item(
    job_id: str,
    *,
    batch_id: str = "batch-a",
    state: JobState = JobState.PENDING,
    job_order: int = 0,
) -> JobQueueItem:
    return JobQueueItem(
        batch_id=batch_id,
        job_id=job_id,
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=job_order,
        input_path=f"/media/{job_id}.wav",
        output_dir="/tmp/out",
        pipeline_profile=PipelineProfile.FAST,
        state=state,
        active_stage=StageName.TRANSCRIBE if state is JobState.RUNNING else None,
        active_stage_state=StageState.RUNNING if state is JobState.RUNNING else None,
        active_stage_attempt=1 if state is JobState.RUNNING else 0,
        cancel_requested=False,
        pause_requested=False,
        paused=False,
        last_event_seq=1,
        journal_tail_status="clean",
        manifest_status="missing",
    )


def _snapshot(
    items: tuple[JobQueueItem, ...] = (),
    *,
    revision: int = 1,
    issues: tuple[QueueLoadIssue, ...] = (),
    omitted: int = 0,
) -> QueueSnapshot:
    return QueueSnapshot(2, revision, items, issues, omitted)


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    execution_completion = Signal(object)
    local_execution_state_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.refresh_calls = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self.started.emit()

    def request_refresh(self) -> None:
        self.refresh_calls += 1

    def stop(self, timeout_ms: int = 5000) -> bool:
        del timeout_ms
        self._running = False
        self.stopped.emit()
        return True


def _build_page(locale: str = "en") -> tuple[QueuePage, BatchController, FakeRunner]:
    service = I18nService(locale)
    model = QueueTableModel(service)
    runner = FakeRunner()
    controller = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    operations = JobOperationsController(runner)  # type: ignore[arg-type]
    page = QueuePage(service, controller, operations)
    page.show()
    return page, controller, runner


def test_required_widgets_exist() -> None:
    page, controller, _runner = _build_page()
    for name in (
        "queueTitle",
        "queueRefreshButton",
        "queueBusyLabel",
        "queueSummaryLabel",
        "queueIssueLabel",
        "queueFailureLabel",
        "queueTable",
        "queueEmptyLabel",
    ):
        assert page.findChild(QObject, name) is not None, name
    controller.stop()


def test_empty_state() -> None:
    page, controller, runner = _build_page("zh-CN")
    controller.start()
    runner.snapshot_ready.emit(_snapshot(revision=1))
    empty = page.findChild(QLabel, "queueEmptyLabel")
    table = page.findChild(QTableView, "queueTable")
    stack = page.findChild(QStackedWidget)
    assert empty is not None and table is not None and stack is not None
    assert stack.currentWidget() is empty
    assert "任务" in empty.text() or "近期" in empty.text()
    controller.stop()


def test_populated_state_and_summary() -> None:
    page, controller, runner = _build_page()
    controller.start()
    items = (
        _item("job-1", state=JobState.RUNNING),
        _item("job-2", state=JobState.SUCCEEDED, job_order=1),
    )
    snapshot = _snapshot(items, revision=1, omitted=4)
    runner.snapshot_ready.emit(snapshot)
    table = page.findChild(QTableView, "queueTable")
    empty = page.findChild(QLabel, "queueEmptyLabel")
    summary = page.findChild(QLabel, "queueSummaryLabel")
    stack = page.findChild(QStackedWidget)
    assert table is not None and empty is not None and summary is not None and stack is not None
    assert stack.currentWidget() is table
    assert table.model().rowCount() == 2
    assert "1 active" in summary.text()
    assert "1 terminal" in summary.text()
    assert "4 older hidden" in summary.text()
    controller.stop()


def test_refresh_button_requests_controller_refresh() -> None:
    page, controller, runner = _build_page()
    controller.start()
    runner.snapshot_ready.emit(_snapshot(revision=1))
    before = runner.refresh_calls
    button = page.findChild(QPushButton, "queueRefreshButton")
    assert button is not None
    from PySide6.QtCore import Qt

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert runner.refresh_calls == before + 1
    controller.stop()


def test_busy_state_disables_refresh() -> None:
    page, controller, runner = _build_page()
    controller.start()
    button = page.findChild(QPushButton, "queueRefreshButton")
    busy = page.findChild(QLabel, "queueBusyLabel")
    assert button is not None and busy is not None
    assert not button.isEnabled()
    assert busy.isVisible()
    runner.snapshot_ready.emit(_snapshot(revision=1))
    assert button.isEnabled()
    assert not busy.isVisible()
    controller.stop()


def test_issue_state_shows_count_only() -> None:
    page, controller, runner = _build_page()
    controller.start()
    issues = (
        QueueLoadIssue("secret-batch-dir", "catalog.batch_corrupt"),
        QueueLoadIssue("another-secret", "catalog.batch_corrupt"),
    )
    runner.snapshot_ready.emit(_snapshot((_item("job-1"),), revision=1, issues=issues))
    issue_label = page.findChild(QLabel, "queueIssueLabel")
    assert issue_label is not None
    assert issue_label.isVisible()
    assert "2" in issue_label.text()
    assert "secret-batch-dir" not in issue_label.text()
    assert "another-secret" not in issue_label.text()
    controller.stop()


def test_failure_state_shows_stable_code() -> None:
    page, controller, runner = _build_page()
    controller.start()
    runner.snapshot_ready.emit(_snapshot((_item("job-1"),), revision=1))
    runner.failure.emit(RunnerFailure(code="queue.test_failure", retryable=False))
    failure_label = page.findChild(QLabel, "queueFailureLabel")
    table = page.findChild(QTableView, "queueTable")
    assert failure_label is not None and table is not None
    assert failure_label.isVisible()
    assert "queue.test_failure" in failure_label.text()
    assert "Traceback" not in failure_label.text()
    assert table.model().rowCount() == 1
    controller.stop()


def test_selection_preserved_across_structural_update() -> None:
    page, controller, runner = _build_page()
    controller.start()
    first = (
        _item("job-keep", state=JobState.RUNNING),
        _item("job-old", state=JobState.SUCCEEDED, job_order=1),
    )
    runner.snapshot_ready.emit(_snapshot(first, revision=1))
    table = page.findChild(QTableView, "queueTable")
    assert table is not None
    model = controller.model
    row = model.row_for_key(("batch-a", "job-keep"))
    assert row is not None
    index = model.index(row, 0)
    table.setCurrentIndex(index)
    table.selectRow(row)
    second = (
        _item("job-new", state=JobState.PENDING),
        _item("job-keep", state=JobState.RUNNING, job_order=1),
    )
    runner.snapshot_ready.emit(_snapshot(second, revision=2))
    selected = table.selectionModel().selectedRows()
    assert selected
    key = model.key_at(selected[0].row())
    assert key == ("batch-a", "job-keep")
    controller.stop()
