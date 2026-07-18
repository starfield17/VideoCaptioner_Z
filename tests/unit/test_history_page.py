"""Unit tests for History page."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from captioner.gui.batch_controller import BatchController
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.pages.history_page import HistoryPage
from captioner.gui.queue_table_model import QueueTableModel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-history-page"])


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    local_execution_state_changed = Signal(object)
    execution_completion = Signal(object)

    def start(self) -> None:
        return None

    def stop(self, timeout_ms: int = 5000) -> bool:
        return True

    def request_refresh(self) -> None:
        return None

    def request_job_detail(self, request: object) -> None:
        return None

    @property
    def running(self) -> bool:
        return False


def test_history_widgets() -> None:
    service = I18nService("en")
    runner = FakeRunner()
    model = QueueTableModel(service)
    queue = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    ops = JobOperationsController(runner)  # type: ignore[arg-type]
    page = HistoryPage(service, queue, ops)
    for name in (
        "historyPage",
        "historyTitle",
        "historySummaryLabel",
        "historyTable",
        "historyEmptyLabel",
        "historyDetailSplitter",
    ):
        if name == "historyPage":
            assert page.objectName() == name
        else:
            assert page.findChild(QObject, name) is not None, name


def test_history_selection_clears() -> None:
    service = I18nService("en")
    runner = FakeRunner()
    model = QueueTableModel(service)
    queue = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    ops = JobOperationsController(runner)  # type: ignore[arg-type]
    page = HistoryPage(service, queue, ops)
    page._on_selection()  # type: ignore[attr-defined]
    assert ops.selected is None
