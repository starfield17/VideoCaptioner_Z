"""GUI entry and MainWindow shell tests."""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget

from captioner.gui.batch_controller import BatchController
from captioner.gui.gui_entry import main
from captioner.gui.main_window import MainWindow
from captioner.gui.queue_table_model import QueueTableModel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-gui-entry"])


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.start_calls += 1
        self._running = True
        self.started.emit()

    def request_refresh(self) -> None:
        return None

    def stop(self, timeout_ms: int = 5000) -> bool:
        del timeout_ms
        self.stop_calls += 1
        self._running = False
        self.stopped.emit()
        return True


def _window(locale: str = "en") -> tuple[MainWindow, BatchController, FakeRunner]:
    service = I18nService(locale)
    model = QueueTableModel(service)
    runner = FakeRunner()
    controller = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    window = MainWindow(service, controller)
    return window, controller, runner


def test_navigation_shell_defaults_to_queue() -> None:
    window, controller, _runner = _window("en")
    assert window.windowTitle() == "Captioner"
    for name in (
        "navCreateButton",
        "navQueueButton",
        "navHistoryButton",
        "navSettingsButton",
        "navDiagnosticsButton",
    ):
        assert window.findChild(QPushButton, name) is not None
    stack = window.findChild(QStackedWidget, "mainPageStack")
    assert stack is not None
    for name in (
        "createPage",
        "queuePage",
        "historyPage",
        "settingsPage",
        "diagnosticsPage",
    ):
        assert window.findChild(QObject, name) is not None
    queue_button = window.findChild(QPushButton, "navQueueButton")
    assert queue_button is not None
    assert queue_button.isChecked()
    queue_page = window.findChild(QObject, "queuePage")
    assert stack.currentWidget() is queue_page
    controller.stop()
    window.close()


def test_chinese_navigation_labels() -> None:
    window, controller, _runner = _window("zh-CN")
    labels = {
        "navCreateButton": "创建",
        "navQueueButton": "队列",
        "navHistoryButton": "历史记录",
        "navSettingsButton": "设置",
        "navDiagnosticsButton": "诊断",
    }
    for name, text in labels.items():
        button = window.findChild(QPushButton, name)
        assert button is not None
        assert button.text() == text
    controller.stop()
    window.close()


def test_navigation_switches_pages() -> None:
    window, controller, _runner = _window()
    stack = window.findChild(QStackedWidget, "mainPageStack")
    assert stack is not None
    mapping = {
        "navCreateButton": "createPage",
        "navQueueButton": "queuePage",
        "navHistoryButton": "historyPage",
        "navSettingsButton": "settingsPage",
        "navDiagnosticsButton": "diagnosticsPage",
    }
    for button_name, page_name in mapping.items():
        button = window.findChild(QPushButton, button_name)
        page = window.findChild(QObject, page_name)
        assert button is not None and page is not None
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        assert stack.currentWidget() is page
    controller.stop()
    window.close()


def test_window_lifecycle_starts_and_stops_controller() -> None:
    window, _controller, runner = _window()
    window.start()
    window.start()
    assert runner.start_calls == 1
    window.close()
    assert runner.stop_calls == 1
    assert not runner.running


def test_gui_smoke_test_returns_zero_chinese() -> None:
    assert main(["--lang", "zh-CN", "--smoke-test"]) == 0


def test_gui_smoke_test_returns_zero_english() -> None:
    assert main(["--lang", "en", "--smoke-test"]) == 0


def test_gui_modules_do_not_import_heavy_sdks() -> None:
    script = """
import sys
modules = [
    "captioner.gui.queue_table_model",
    "captioner.gui.application_runner",
    "captioner.gui.batch_controller",
    "captioner.gui.pages.queue_page",
    "captioner.gui.composition",
]
for name in modules:
    __import__(name)
prohibited = {
    "faster_whisper",
    "ctranslate2",
    "torch",
    "transformers",
    "openai",
}
loaded = {name.split(".", 1)[0] for name in sys.modules}
assert not prohibited & loaded, sorted(prohibited & loaded)
assert any(name == "PySide6" or name.startswith("PySide6.") for name in sys.modules)
"""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
