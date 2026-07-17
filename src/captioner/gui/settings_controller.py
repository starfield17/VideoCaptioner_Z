"""Main-thread controller for Settings page configuration operations."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
)
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure


class SettingsController(QObject):
    """Coordinates configuration load/save and provider tests via the runner."""

    configuration_changed = Signal(object)
    busy_changed = Signal(bool)
    failure_changed = Signal(object)
    provider_test_changed = Signal(object)
    restart_required_changed = Signal(bool)

    def __init__(
        self,
        runner: ApplicationRunnerBridge,
        parent: QObject | None = None,
        *,
        startup_issue: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._current: ConfigurationSnapshot | None = None
        self._last_failure: RunnerFailure | None = None
        self._busy = False
        self._restart_required = False
        self._startup_issue = startup_issue
        self._pending_locale: str | None = None

        self._runner.configuration_ready.connect(self._on_configuration)
        self._runner.configuration_failure.connect(self._on_failure)
        self._runner.provider_test_ready.connect(self._on_provider_test)
        self._runner.provider_test_failure.connect(self._on_provider_test_failure)

    @property
    def current_snapshot(self) -> ConfigurationSnapshot | None:
        return self._current

    @property
    def last_failure(self) -> RunnerFailure | None:
        return self._last_failure

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def restart_required(self) -> bool:
        return self._restart_required

    @property
    def startup_issue(self) -> str | None:
        return self._startup_issue

    def load(self) -> None:
        self._set_busy(True)
        self._runner.request_configuration_load()

    def save_global(self, settings: GlobalSettings) -> None:
        previous_locale = None if self._current is None else self._current.global_settings.locale
        self._pending_locale = (
            settings.locale
            if previous_locale is not None and settings.locale != previous_locale
            else None
        )
        self._set_busy(True)
        self._runner.request_global_save(settings)

    def save_provider(self, update: ProviderSettingsUpdate) -> None:
        self._set_busy(True)
        self._runner.request_provider_save(update)
        # Do not retain the key-bearing update after dispatch.
        del update

    def save_preset(self, preset: ExecutionPreset) -> None:
        self._set_busy(True)
        self._runner.request_preset_save(preset)

    def delete_preset(self, name: str) -> None:
        self._set_busy(True)
        self._runner.request_preset_delete(name)

    def test_provider(self, update: ProviderSettingsUpdate) -> None:
        self._set_busy(True)
        self._runner.request_provider_test(update)
        del update

    def _on_configuration(self, snapshot: object) -> None:
        if not isinstance(snapshot, ConfigurationSnapshot):
            return
        self._current = snapshot
        if self._pending_locale is not None:
            self._restart_required = True
            self.restart_required_changed.emit(True)
            self._pending_locale = None
        self._last_failure = None
        self.configuration_changed.emit(snapshot)
        self.failure_changed.emit(None)
        self._set_busy(False)

    def _on_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        self._last_failure = failure
        self._pending_locale = None
        self.failure_changed.emit(failure)
        self._set_busy(False)

    def _on_provider_test(self, result: object) -> None:
        if not isinstance(result, ProviderConnectionResult):
            return
        self.provider_test_changed.emit(result)
        self._set_busy(False)

    def _on_provider_test_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        self.provider_test_changed.emit(ProviderConnectionResult(ok=False, code=failure.code))
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._busy is busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)


__all__ = ["SettingsController"]
