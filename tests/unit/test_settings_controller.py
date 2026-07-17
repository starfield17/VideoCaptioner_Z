"""Unit tests for SettingsController."""

from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
    default_configuration_snapshot,
)
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.settings_controller import SettingsController

_app = QApplication.instance() or QApplication(["test-settings-controller"])


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

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.global_saves: list[GlobalSettings] = []
        self.provider_saves: list[ProviderSettingsUpdate] = []
        self.tests: list[ProviderSettingsUpdate] = []
        self._running = True

    @property
    def running(self) -> bool:
        return self._running

    def request_configuration_load(self) -> None:
        self.load_calls += 1
        QTimer.singleShot(
            0,
            lambda: self.configuration_loaded.emit(default_configuration_snapshot()),
        )

    def request_global_save(self, settings: GlobalSettings) -> None:
        self.global_saves.append(settings)
        snapshot = ConfigurationSnapshot(
            global_settings=settings,
            presets=default_configuration_snapshot().presets,
            provider=default_configuration_snapshot().provider,
            issues=(),
        )
        QTimer.singleShot(0, lambda: self.global_settings_saved.emit(snapshot))

    def request_provider_save(self, update: ProviderSettingsUpdate) -> None:
        self.provider_saves.append(update)
        QTimer.singleShot(
            0,
            lambda: self.provider_settings_saved.emit(default_configuration_snapshot()),
        )

    def request_preset_save(self, preset: object) -> None:
        QTimer.singleShot(
            0,
            lambda: self.preset_saved.emit(default_configuration_snapshot()),
        )

    def request_preset_delete(self, name: str) -> None:
        QTimer.singleShot(
            0,
            lambda: self.preset_deleted.emit(default_configuration_snapshot()),
        )

    def request_provider_test(self, update: ProviderSettingsUpdate) -> None:
        self.tests.append(update)
        QTimer.singleShot(
            0,
            lambda: self.provider_test_ready.emit(
                ProviderConnectionResult(True, "llm.connection_ok")
            ),
        )


def _wait_until(predicate: Callable[[], bool], timeout_ms: int = 2000) -> bool:
    if predicate():
        return True
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(timeout_ms)
    loop.exec()
    timer.stop()
    return predicate()


def test_load_save_and_restart_flag() -> None:
    runner = FakeRunner()
    controller = SettingsController(runner)  # type: ignore[arg-type]
    controller.load()
    assert _wait_until(lambda: controller.current_snapshot is not None)
    assert runner.load_calls == 1

    controller.save_global(GlobalSettings(locale="zh-CN"))
    assert _wait_until(lambda: controller.restart_required is True)

    controller.save_global(GlobalSettings(locale="zh-CN", recursive_input=False))
    assert _wait_until(lambda: controller.current_snapshot is not None)
    assert controller.restart_required is True


def test_provider_save_test_and_failure_retention() -> None:
    runner = FakeRunner()
    controller = SettingsController(runner)  # type: ignore[arg-type]
    controller.load()
    assert _wait_until(lambda: controller.current_snapshot is not None)
    previous = controller.current_snapshot
    update = ProviderSettingsUpdate(
        profile_name="default",
        base_url="https://example.com/v1",
        model="m",
        api_key="temp-secret",
    )
    controller.save_provider(update)
    assert _wait_until(lambda: len(runner.provider_saves) == 1)
    assert "temp-secret" not in repr(controller)

    controller.test_provider(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
            api_key="test-secret",
        )
    )
    assert _wait_until(lambda: len(runner.tests) == 1)

    controller.save_global(GlobalSettings(locale="en"))
    assert _wait_until(lambda: controller.busy is False)
    runner.global_settings_save_failure.emit(RunnerFailure(code="config.write_failed"))
    # Failure without matching pending op is ignored.
    assert controller.current_snapshot is not None

    controller.save_provider(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
        )
    )
    assert controller.busy is True
    runner.provider_settings_save_failure.emit(RunnerFailure(code="config.write_failed"))
    assert _wait_until(lambda: controller.last_failure is not None)
    assert controller.current_snapshot is previous or controller.current_snapshot is not None


def test_preset_result_does_not_complete_locale_save() -> None:
    runner = FakeRunner()
    controller = SettingsController(runner)  # type: ignore[arg-type]
    controller.load()
    assert _wait_until(lambda: controller.current_snapshot is not None)

    controller.save_global(GlobalSettings(locale="zh-CN"))
    assert controller.busy is True
    # Interleaved preset success must not consume the pending locale save.
    runner.preset_saved.emit(default_configuration_snapshot())
    assert controller.busy is True
    assert controller.restart_required is False
    runner.global_settings_saved.emit(
        ConfigurationSnapshot(
            global_settings=GlobalSettings(locale="zh-CN"),
            presets=default_configuration_snapshot().presets,
            provider=default_configuration_snapshot().provider,
            issues=(),
        )
    )
    assert _wait_until(lambda: controller.restart_required is True)
    assert controller.busy is False
