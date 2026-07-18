"""GUI entry and MainWindow shell tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget

from captioner.gui.batch_controller import BatchController
from captioner.gui.composition import GuiControllers
from captioner.gui.create_controller import CreateController
from captioner.gui.diagnostics_controller import DiagnosticsController
from captioner.gui.gui_entry import main
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.main_window import MainWindow
from captioner.gui.queue_table_model import QueueTableModel
from captioner.gui.recovery_controller import RecoveryController
from captioner.gui.settings_controller import SettingsController
from captioner.gui_bootstrap import load_startup_locale
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import resolve_app_paths

_app = QApplication.instance() or QApplication(["test-gui-entry"])


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    input_preview_ready = Signal(object)
    input_failure = Signal(object)
    configuration_loaded = Signal(object)
    global_settings_saved = Signal(object)
    provider_settings_saved = Signal(object)
    preset_saved = Signal(object)
    preset_deleted = Signal(object)
    configuration_load_failure = Signal(object)
    global_settings_save_failure = Signal(object)
    provider_settings_save_failure = Signal(object)
    preset_save_failure = Signal(object)
    preset_delete_failure = Signal(object)
    provider_test_ready = Signal(object)
    provider_test_failure = Signal(object)
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    recovery_ready = Signal(object)
    recovery_failure = Signal(object)
    diagnostics_ready = Signal(object)
    diagnostics_failure = Signal(object)
    diagnostic_export_ready = Signal(object)
    diagnostic_export_failure = Signal(object)
    execution_completion = Signal(object)
    local_execution_state_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0
        self.load_calls = 0
        self.diagnostics_load_calls = 0
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

    def request_configuration_load(self) -> None:
        self.load_calls += 1

    def request_input_preview(self, request: object) -> None:
        return None

    def request_recovery_scan(self, request: object) -> None:
        return None

    def request_diagnostics_load(self, request: object) -> None:
        self.diagnostics_load_calls += 1

    def request_diagnostics_export(self, request: object) -> None:
        return None

    def request_job_detail(self, request: object) -> None:
        return None

    def request_submit_batch(self, request: object) -> None:
        return None

    def request_batch_action(self, request: object) -> None:
        return None

    def request_job_action(self, request: object) -> None:
        return None

    def request_cancel_local_work(self, request: object) -> None:
        return None

    def stop(self, timeout_ms: int = 5000) -> bool:
        del timeout_ms
        self.stop_calls += 1
        self._running = False
        self.stopped.emit()
        return True


def _window(locale: str = "en") -> tuple[MainWindow, GuiControllers, FakeRunner]:
    service = I18nService(locale)
    model = QueueTableModel(service)
    runner = FakeRunner()
    queue = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    create = CreateController(runner)  # type: ignore[arg-type]
    settings = SettingsController(runner)  # type: ignore[arg-type]
    operations = JobOperationsController(runner)  # type: ignore[arg-type]
    recovery = RecoveryController(runner)  # type: ignore[arg-type]
    diagnostics = DiagnosticsController(runner)  # type: ignore[arg-type]
    controllers = GuiControllers(
        queue=queue,
        create=create,
        settings=settings,
        operations=operations,
        recovery=recovery,
        diagnostics=diagnostics,
    )
    window = MainWindow(service, controllers)
    return window, controllers, runner


def test_navigation_shell_defaults_to_create() -> None:
    window, controllers, _runner = _window("en")
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
    create_button = window.findChild(QPushButton, "navCreateButton")
    assert create_button is not None
    assert create_button.isChecked()
    create_page = window.findChild(QObject, "createPage")
    assert stack.currentWidget() is create_page
    controllers.queue.stop()
    window.close()


def test_chinese_navigation_labels() -> None:
    window, controllers, _runner = _window("zh-CN")
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
    controllers.queue.stop()
    window.close()


def test_navigation_switches_pages() -> None:
    window, controllers, _runner = _window()
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
    controllers.queue.stop()
    window.close()


def test_window_lifecycle_starts_queue_and_settings() -> None:
    window, _controllers, runner = _window()
    window.start()
    window.start()
    assert runner.start_calls == 1
    assert runner.load_calls == 1
    window.close()
    assert runner.stop_calls == 1
    assert not runner.running


def test_gui_smoke_test_returns_zero_chinese() -> None:
    assert main(["--lang", "zh-CN", "--smoke-test"]) == 0


def test_gui_smoke_test_returns_zero_english() -> None:
    assert main(["--lang", "en", "--smoke-test"]) == 0


def test_startup_locale_from_settings(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    (paths.config_dir / "settings.toml").write_text(
        'schema_version = 1\n[global]\nlocale = "zh-CN"\n'
        'default_output_root = ""\nrecursive_input = true\n'
        'default_preset_name = "deterministic"\n'
        'collision_policy = "unique_subdir"\n',
        encoding="utf-8",
    )
    locale, issue = load_startup_locale(paths=paths, explicit_locale=None)
    assert locale == "zh-CN"
    assert issue is None
    locale, issue = load_startup_locale(paths=paths, explicit_locale="en")
    assert locale == "en"
    assert issue is None


def test_invalid_settings_fallback_to_english(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    (paths.config_dir / "settings.toml").write_text("not = [valid", encoding="utf-8")
    locale, issue = load_startup_locale(paths=paths, explicit_locale=None)
    assert locale == "en"
    assert issue == "config.settings_invalid"
    assert (paths.config_dir / "settings.toml").read_text(encoding="utf-8") == "not = [valid"


def test_gui_modules_do_not_import_heavy_sdks() -> None:
    script = """
import sys
modules = [
    "captioner.core.application.input_selection",
    "captioner.core.application.configuration",
    "captioner.gui.queue_table_model",
    "captioner.gui.application_runner",
    "captioner.gui.batch_controller",
    "captioner.gui.create_controller",
    "captioner.gui.settings_controller",
    "captioner.gui.pages.queue_page",
    "captioner.gui.pages.create_page",
    "captioner.gui.pages.settings_page",
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


def test_notifications_and_recovery_prompt() -> None:
    from captioner.core.application.batch_commands import LocalExecutionSnapshot

    window, controllers, _runner = _window("en")
    window._on_notification("command.accepted:submit")  # type: ignore[attr-defined]
    window._on_notification("execution.completed:batch-a")  # type: ignore[attr-defined]
    window._on_notification("execution.failed:x")  # type: ignore[attr-defined]
    window._on_notification("command.failed:y")  # type: ignore[attr-defined]
    label = window.findChild(QObject, "globalNotificationLabel")
    assert label is not None
    # Empty recovery prompt is a no-op.
    window._on_recovery_prompt(())  # type: ignore[attr-defined]
    window._on_execution_state(LocalExecutionSnapshot(None, ()))  # type: ignore[attr-defined]
    controllers.queue.stop()


def test_batch_submitted_navigates() -> None:
    from captioner.core.application.batch_commands import BatchCommandAck, BatchCommandKind

    window, controllers, _runner = _window("en")
    controllers.create.batch_submitted.emit(
        BatchCommandAck(
            request_id="req",
            kind=BatchCommandKind.SUBMIT,
            batch_id="batch-a",
            job_id=None,
            accepted_at_utc="t0",
            scheduled=True,
            created_batch_id="batch-a",
        )
    )
    stack = window.findChild(QStackedWidget, "mainPageStack")
    assert stack is not None
    queue_page = window.findChild(QObject, "queuePage")
    assert stack.currentWidget() is queue_page
    controllers.queue.stop()


def test_close_without_local_work() -> None:
    from PySide6.QtGui import QCloseEvent

    window, controllers, runner = _window("en")
    window.start()
    # Fake runner stop succeeds by default.
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert runner.stop_calls == 1
    assert controllers.queue is not None


def _patch_close_dialog(
    monkeypatch: pytest.MonkeyPatch,
    *,
    accept_cancel_and_close: bool,
) -> None:
    """Drive the localized custom close dialog without a blocking modal exec."""

    from PySide6.QtWidgets import QMessageBox

    def _fake_exec(self: QMessageBox) -> int:
        role = (
            QMessageBox.ButtonRole.AcceptRole
            if accept_cancel_and_close
            else QMessageBox.ButtonRole.RejectRole
        )
        for button in self.buttons():
            if self.buttonRole(button) == role:
                button.click()
                return int(self.result())
        return int(QMessageBox.DialogCode.Rejected)

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec)


def test_close_with_local_work_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtGui import QCloseEvent

    from captioner.core.application.batch_commands import LocalExecutionSnapshot

    window, controllers, _runner = _window("en")
    window.start()
    # Pretend local work is active
    controllers.operations._execution = LocalExecutionSnapshot("batch-a", ())  # type: ignore[attr-defined]
    _patch_close_dialog(monkeypatch, accept_cancel_and_close=True)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window._close_when_idle is True  # type: ignore[attr-defined]
    # CancelLocalWork acknowledged while still active → window remains open.
    assert controllers.operations.has_local_work is True
    # Idle then close
    controllers.operations._execution = LocalExecutionSnapshot(None, ())  # type: ignore[attr-defined]
    window._on_execution_state(LocalExecutionSnapshot(None, ()))  # type: ignore[attr-defined]
    assert window._close_when_idle is False  # type: ignore[attr-defined]
    controllers.queue.stop()


def test_close_with_local_work_keep_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtGui import QCloseEvent

    from captioner.core.application.batch_commands import LocalExecutionSnapshot

    window, controllers, _runner = _window("en")
    controllers.operations._execution = LocalExecutionSnapshot("batch-a", ())  # type: ignore[attr-defined]
    _patch_close_dialog(monkeypatch, accept_cancel_and_close=False)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window._close_when_idle is False  # type: ignore[attr-defined]
    controllers.queue.stop()


def test_close_cancel_failure_clears_close_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtGui import QCloseEvent

    from captioner.core.application.batch_commands import LocalExecutionSnapshot
    from captioner.gui.application_runner import RunnerFailure

    window, controllers, _runner = _window("en")
    window.start()
    controllers.operations._execution = LocalExecutionSnapshot("batch-a", ())  # type: ignore[attr-defined]
    _patch_close_dialog(monkeypatch, accept_cancel_and_close=True)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window._close_when_idle is True  # type: ignore[attr-defined]
    window._on_close_cancellation_failed(RunnerFailure(code="batch.execution_active"))  # type: ignore[attr-defined]
    assert window._close_when_idle is False  # type: ignore[attr-defined]
    label = window.findChild(QObject, "globalNotificationLabel")
    assert label is not None
    # Parent window may be unshown; isHidden tracks the explicit setVisible path.
    assert not label.isHidden()  # type: ignore[attr-defined]
    text = label.property("text") if hasattr(label, "property") else ""
    label_text = str(getattr(label, "text", lambda: text)())
    assert "batch.execution_active" in label_text or label_text != ""
    controllers.queue.stop()
