"""Dedicated Qt worker thread for Application boundary operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from captioner.core.application.configuration import (
    ExecutionPreset,
    GlobalSettings,
    ProviderSettingsUpdate,
)
from captioner.core.application.input_selection import InputSelectionRequest
from captioner.core.domain.errors import AppError
from captioner.gui.application_boundary import GuiApplicationBoundary

BoundaryFactory = Callable[[], GuiApplicationBoundary]

_UNEXPECTED_FAILURE_CODE = "gui.application_bridge_failed"


@dataclass(frozen=True, slots=True)
class RunnerFailure:
    code: str
    retryable: bool = False


def _failure_from_exception(exc: BaseException) -> RunnerFailure:
    if isinstance(exc, AppError):
        return RunnerFailure(code=exc.code, retryable=exc.retryable)
    return RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)


class _ApplicationRunnerWorker(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    initialized = Signal()
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

    def __init__(self, factory: BoundaryFactory) -> None:
        super().__init__()
        self._factory = factory
        self._boundary: GuiApplicationBoundary | None = None

    @Slot()
    def initialize(self) -> None:
        try:
            self._boundary = self._factory()
            snapshot = self._boundary.get_queue_snapshot()
            self.snapshot_ready.emit(snapshot)
        except AppError as exc:
            self.failure.emit(_failure_from_exception(exc))
        except Exception:
            self.failure.emit(RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False))
        finally:
            self.initialized.emit()

    @Slot()
    def refresh(self) -> None:
        boundary = self._boundary
        if boundary is None:
            self.failure.emit(RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False))
            return
        try:
            snapshot = boundary.refresh_queue()
            self.snapshot_ready.emit(snapshot)
        except Exception as exc:
            self.failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def preview_inputs(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.input_failure.emit(RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False))
            return
        if not isinstance(request, InputSelectionRequest):
            self.input_failure.emit(RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False))
            return
        try:
            preview = boundary.preview_inputs(request)
            self.input_preview_ready.emit(preview)
        except Exception as exc:
            self.input_failure.emit(_failure_from_exception(exc))

    @Slot()
    def load_configuration(self) -> None:
        boundary = self._boundary
        if boundary is None:
            self.configuration_load_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.load_configuration()
            self.configuration_loaded.emit(snapshot)
        except Exception as exc:
            self.configuration_load_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def save_global(self, settings: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.global_settings_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        if not isinstance(settings, GlobalSettings):
            self.global_settings_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.save_global_settings(settings)
            self.global_settings_saved.emit(snapshot)
        except Exception as exc:
            self.global_settings_save_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def save_provider(self, update: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.provider_settings_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        if not isinstance(update, ProviderSettingsUpdate):
            self.provider_settings_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.save_provider_settings(update)
            self.provider_settings_saved.emit(snapshot)
        except Exception as exc:
            self.provider_settings_save_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def save_preset(self, preset: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.preset_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        if not isinstance(preset, ExecutionPreset):
            self.preset_save_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.save_user_preset(preset)
            self.preset_saved.emit(snapshot)
        except Exception as exc:
            self.preset_save_failure.emit(_failure_from_exception(exc))

    @Slot(str)
    def delete_preset(self, name: str) -> None:
        boundary = self._boundary
        if boundary is None:
            self.preset_delete_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.delete_user_preset(name)
            self.preset_deleted.emit(snapshot)
        except Exception as exc:
            self.preset_delete_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def test_provider(self, update: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.provider_test_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        if not isinstance(update, ProviderSettingsUpdate):
            self.provider_test_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            result = boundary.test_provider_connection(update)
            self.provider_test_ready.emit(result)
        except Exception as exc:
            self.provider_test_failure.emit(_failure_from_exception(exc))


class ApplicationRunnerBridge(QObject):
    """Main-thread facade over one Application worker thread."""

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

    _refresh_requested = Signal()
    _preview_inputs_requested = Signal(object)
    _load_configuration_requested = Signal()
    _save_global_requested = Signal(object)
    _save_provider_requested = Signal(object)
    _save_preset_requested = Signal(object)
    _delete_preset_requested = Signal(str)
    _test_provider_requested = Signal(object)

    def __init__(
        self,
        factory: BoundaryFactory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = factory
        self._thread: QThread | None = None
        self._worker: _ApplicationRunnerWorker | None = None
        self._running = False
        self._stop_emitted = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        thread = QThread(self)
        worker = _ApplicationRunnerWorker(self._factory)
        worker.moveToThread(thread)

        thread.started.connect(worker.initialize)
        worker.snapshot_ready.connect(self.snapshot_ready)
        worker.failure.connect(self.failure)
        worker.input_preview_ready.connect(self.input_preview_ready)
        worker.input_failure.connect(self.input_failure)
        worker.configuration_loaded.connect(self.configuration_loaded)
        worker.global_settings_saved.connect(self.global_settings_saved)
        worker.provider_settings_saved.connect(self.provider_settings_saved)
        worker.preset_saved.connect(self.preset_saved)
        worker.preset_deleted.connect(self.preset_deleted)
        worker.configuration_load_failure.connect(self.configuration_load_failure)
        worker.global_settings_save_failure.connect(self.global_settings_save_failure)
        worker.provider_settings_save_failure.connect(self.provider_settings_save_failure)
        worker.preset_save_failure.connect(self.preset_save_failure)
        worker.preset_delete_failure.connect(self.preset_delete_failure)
        worker.provider_test_ready.connect(self.provider_test_ready)
        worker.provider_test_failure.connect(self.provider_test_failure)

        queued = Qt.ConnectionType.QueuedConnection
        self._refresh_requested.connect(worker.refresh, queued)
        self._preview_inputs_requested.connect(worker.preview_inputs, queued)
        self._load_configuration_requested.connect(worker.load_configuration, queued)
        self._save_global_requested.connect(worker.save_global, queued)
        self._save_provider_requested.connect(worker.save_provider, queued)
        self._save_preset_requested.connect(worker.save_preset, queued)
        self._delete_preset_requested.connect(worker.delete_preset, queued)
        self._test_provider_requested.connect(worker.test_provider, queued)

        self._thread = thread
        self._worker = worker
        self._running = True
        self._stop_emitted = False
        thread.start()
        self.started.emit()

    def request_refresh(self) -> None:
        if not self._running:
            return
        self._refresh_requested.emit()

    def request_input_preview(self, request: InputSelectionRequest) -> None:
        if not self._running:
            return
        self._preview_inputs_requested.emit(request)

    def request_configuration_load(self) -> None:
        if not self._running:
            return
        self._load_configuration_requested.emit()

    def request_global_save(self, settings: GlobalSettings) -> None:
        if not self._running:
            return
        self._save_global_requested.emit(settings)

    def request_provider_save(self, update: ProviderSettingsUpdate) -> None:
        if not self._running:
            return
        self._save_provider_requested.emit(update)

    def request_preset_save(self, preset: ExecutionPreset) -> None:
        if not self._running:
            return
        self._save_preset_requested.emit(preset)

    def request_preset_delete(self, name: str) -> None:
        if not self._running:
            return
        self._delete_preset_requested.emit(name)

    def request_provider_test(self, update: ProviderSettingsUpdate) -> None:
        if not self._running:
            return
        self._test_provider_requested.emit(update)

    def stop(self, timeout_ms: int = 5000) -> bool:
        if timeout_ms < 0:
            raise ValueError("gui.runner_timeout_invalid")
        if not self._running and self._thread is None:
            return True

        thread = self._thread
        if thread is None:
            self._running = False
            if not self._stop_emitted:
                self._stop_emitted = True
                self.stopped.emit()
            return True

        thread.quit()
        finished = thread.wait(timeout_ms)
        if not finished:
            return False

        worker = self._worker
        if worker is not None:
            worker.deleteLater()
        thread.deleteLater()
        self._worker = None
        self._thread = None
        self._running = False
        if not self._stop_emitted:
            self._stop_emitted = True
            self.stopped.emit()
        return True


__all__ = ["ApplicationRunnerBridge", "BoundaryFactory", "RunnerFailure"]
