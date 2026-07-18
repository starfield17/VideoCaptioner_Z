"""Dedicated Qt worker thread for Application boundary operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from captioner.core.application.batch_commands import (
    BatchActionRequest,
    CancelLocalWorkRequest,
    JobActionRequest,
    LocalExecutionSnapshot,
    SubmitBatchRequest,
)
from captioner.core.application.configuration import (
    ExecutionPreset,
    GlobalSettings,
    ProviderSettingsUpdate,
)
from captioner.core.application.diagnostics import (
    DiagnosticExportRequest,
    DiagnosticsRequest,
)
from captioner.core.application.input_selection import InputSelectionRequest
from captioner.core.application.job_detail import JobDetailRequest
from captioner.core.application.recovery import RecoveryRequest
from captioner.core.domain.errors import AppError
from captioner.gui.application_boundary import GuiApplicationBoundary

BoundaryFactory = Callable[[], GuiApplicationBoundary]

_UNEXPECTED_FAILURE_CODE = "gui.application_bridge_failed"
_EXECUTION_POLL_MS = 250


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
    shutdown_finished = Signal()

    def __init__(self, factory: BoundaryFactory) -> None:
        super().__init__()
        self._factory = factory
        self._boundary: GuiApplicationBoundary | None = None
        self._poll_timer: QTimer | None = None
        self._last_execution_state: LocalExecutionSnapshot | None = None

    @Slot()
    def initialize(self) -> None:
        try:
            self._boundary = self._factory()
            snapshot = self._boundary.get_queue_snapshot()
            self.snapshot_ready.emit(snapshot)
            timer = QTimer(self)
            timer.setInterval(_EXECUTION_POLL_MS)
            timer.timeout.connect(self.poll_execution)
            timer.start()
            self._poll_timer = timer
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

    @Slot(object)
    def submit_batch(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None:
            self.batch_command_failure.emit(_command_failure(request, _UNEXPECTED_FAILURE_CODE))
            return
        if not isinstance(request, SubmitBatchRequest):
            self.batch_command_failure.emit(_command_failure(request, _UNEXPECTED_FAILURE_CODE))
            return
        try:
            ack = boundary.submit_batch(request)
            self.batch_command_ready.emit(ack)
        except Exception as exc:
            failure = _failure_from_exception(exc)
            from captioner.core.application.batch_commands import (
                BatchCommandFailure,
                BatchCommandKind,
            )

            self.batch_command_failure.emit(
                BatchCommandFailure(
                    request_id=request.request_id,
                    kind=BatchCommandKind.SUBMIT,
                    code=failure.code,
                    retryable=failure.retryable,
                )
            )

    @Slot(object)
    def perform_batch_action(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, BatchActionRequest):
            self.batch_command_failure.emit(_command_failure(request, _UNEXPECTED_FAILURE_CODE))
            return
        try:
            ack = boundary.perform_batch_action(request)
            self.batch_command_ready.emit(ack)
        except Exception as exc:
            failure = _failure_from_exception(exc)
            from captioner.core.application.batch_commands import BatchCommandFailure

            self.batch_command_failure.emit(
                BatchCommandFailure(
                    request_id=request.request_id,
                    kind=request.kind,
                    code=failure.code,
                    retryable=failure.retryable,
                )
            )

    @Slot(object)
    def perform_job_action(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, JobActionRequest):
            self.batch_command_failure.emit(_command_failure(request, _UNEXPECTED_FAILURE_CODE))
            return
        try:
            ack = boundary.perform_job_action(request)
            self.batch_command_ready.emit(ack)
        except Exception as exc:
            failure = _failure_from_exception(exc)
            from captioner.core.application.batch_commands import BatchCommandFailure

            self.batch_command_failure.emit(
                BatchCommandFailure(
                    request_id=request.request_id,
                    kind=request.kind,
                    code=failure.code,
                    retryable=failure.retryable,
                )
            )

    @Slot(object)
    def cancel_local_work(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, CancelLocalWorkRequest):
            self.batch_command_failure.emit(_command_failure(request, _UNEXPECTED_FAILURE_CODE))
            return
        try:
            ack = boundary.cancel_local_work(request)
            self.batch_command_ready.emit(ack)
        except Exception as exc:
            failure = _failure_from_exception(exc)
            from captioner.core.application.batch_commands import (
                BatchCommandFailure,
                BatchCommandKind,
            )

            self.batch_command_failure.emit(
                BatchCommandFailure(
                    request_id=request.request_id,
                    kind=BatchCommandKind.CANCEL_LOCAL_WORK,
                    code=failure.code,
                    retryable=failure.retryable,
                )
            )

    @Slot(object)
    def load_job_detail(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, JobDetailRequest):
            self.job_detail_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            detail = boundary.load_job_detail(request)
            self.job_detail_ready.emit(detail)
        except Exception as exc:
            self.job_detail_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def scan_recovery(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, RecoveryRequest):
            self.recovery_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.scan_recovery(request)
            self.recovery_ready.emit(snapshot)
        except Exception as exc:
            self.recovery_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def load_diagnostics(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, DiagnosticsRequest):
            self.diagnostics_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            snapshot = boundary.load_diagnostics(request)
            self.diagnostics_ready.emit(snapshot)
        except Exception as exc:
            self.diagnostics_failure.emit(_failure_from_exception(exc))

    @Slot(object)
    def export_diagnostics(self, request: object) -> None:
        boundary = self._boundary
        if boundary is None or not isinstance(request, DiagnosticExportRequest):
            self.diagnostic_export_failure.emit(
                RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False)
            )
            return
        try:
            result = boundary.export_diagnostics(request)
            self.diagnostic_export_ready.emit(result)
        except Exception as exc:
            self.diagnostic_export_failure.emit(_failure_from_exception(exc))

    @Slot()
    def poll_execution(self) -> None:
        boundary = self._boundary
        if boundary is None:
            return
        try:
            poll = boundary.poll_execution()
        except Exception:
            return
        if self._last_execution_state != poll.state:
            self._last_execution_state = poll.state
            self.local_execution_state_changed.emit(poll.state)
        for completion in poll.completions:
            self.execution_completion.emit(completion)

    def _ensure_execution_polling(self) -> None:
        timer = self._poll_timer
        if timer is not None and timer.isActive():
            return
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(_EXECUTION_POLL_MS)
            timer.timeout.connect(self.poll_execution)
            self._poll_timer = timer
        timer.start()

    def _stop_execution_polling(self) -> None:
        timer = self._poll_timer
        if timer is not None:
            timer.stop()

    @Slot()
    def shutdown(self) -> None:
        boundary = self._boundary
        if boundary is None:
            self._stop_execution_polling()
            self.shutdown_finished.emit()
            self.thread().quit()
            return

        try:
            boundary.shutdown()
        except Exception as exc:
            # Keep boundary and polling so stop/cancel/refresh remain retryable.
            self.failure.emit(_failure_from_exception(exc))
            self._ensure_execution_polling()
            return

        self._stop_execution_polling()
        self._poll_timer = None
        self._boundary = None
        self.shutdown_finished.emit()
        self.thread().quit()


def _command_failure(request: object, code: str) -> object:
    from captioner.core.application.batch_commands import (
        BatchCommandFailure,
        BatchCommandKind,
    )

    request_id = getattr(request, "request_id", "unknown")
    kind = getattr(request, "kind", BatchCommandKind.SUBMIT)
    if not isinstance(kind, BatchCommandKind):
        kind = BatchCommandKind.SUBMIT
    if not isinstance(request_id, str):
        request_id = "unknown"
    return BatchCommandFailure(request_id=request_id, kind=kind, code=code, retryable=False)


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

    _refresh_requested = Signal()
    _preview_inputs_requested = Signal(object)
    _load_configuration_requested = Signal()
    _save_global_requested = Signal(object)
    _save_provider_requested = Signal(object)
    _save_preset_requested = Signal(object)
    _delete_preset_requested = Signal(str)
    _test_provider_requested = Signal(object)
    _submit_batch_requested = Signal(object)
    _batch_action_requested = Signal(object)
    _job_action_requested = Signal(object)
    _cancel_local_work_requested = Signal(object)
    _load_job_detail_requested = Signal(object)
    _scan_recovery_requested = Signal(object)
    _load_diagnostics_requested = Signal(object)
    _export_diagnostics_requested = Signal(object)
    _shutdown_requested = Signal()

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
        self._stopping = False

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
        worker.batch_command_ready.connect(self.batch_command_ready)
        worker.batch_command_failure.connect(self.batch_command_failure)
        worker.job_detail_ready.connect(self.job_detail_ready)
        worker.job_detail_failure.connect(self.job_detail_failure)
        worker.recovery_ready.connect(self.recovery_ready)
        worker.recovery_failure.connect(self.recovery_failure)
        worker.diagnostics_ready.connect(self.diagnostics_ready)
        worker.diagnostics_failure.connect(self.diagnostics_failure)
        worker.diagnostic_export_ready.connect(self.diagnostic_export_ready)
        worker.diagnostic_export_failure.connect(self.diagnostic_export_failure)
        worker.execution_completion.connect(self.execution_completion)
        worker.local_execution_state_changed.connect(self.local_execution_state_changed)

        queued = Qt.ConnectionType.QueuedConnection
        self._refresh_requested.connect(worker.refresh, queued)
        self._preview_inputs_requested.connect(worker.preview_inputs, queued)
        self._load_configuration_requested.connect(worker.load_configuration, queued)
        self._save_global_requested.connect(worker.save_global, queued)
        self._save_provider_requested.connect(worker.save_provider, queued)
        self._save_preset_requested.connect(worker.save_preset, queued)
        self._delete_preset_requested.connect(worker.delete_preset, queued)
        self._test_provider_requested.connect(worker.test_provider, queued)
        self._submit_batch_requested.connect(worker.submit_batch, queued)
        self._batch_action_requested.connect(worker.perform_batch_action, queued)
        self._job_action_requested.connect(worker.perform_job_action, queued)
        self._cancel_local_work_requested.connect(worker.cancel_local_work, queued)
        self._load_job_detail_requested.connect(worker.load_job_detail, queued)
        self._scan_recovery_requested.connect(worker.scan_recovery, queued)
        self._load_diagnostics_requested.connect(worker.load_diagnostics, queued)
        self._export_diagnostics_requested.connect(worker.export_diagnostics, queued)
        self._shutdown_requested.connect(worker.shutdown, queued)

        self._thread = thread
        self._worker = worker
        self._running = True
        self._stop_emitted = False
        self._stopping = False
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

    def request_submit_batch(self, request: SubmitBatchRequest) -> None:
        if not self._running:
            return
        self._submit_batch_requested.emit(request)

    def request_batch_action(self, request: BatchActionRequest) -> None:
        if not self._running:
            return
        self._batch_action_requested.emit(request)

    def request_job_action(self, request: JobActionRequest) -> None:
        if not self._running:
            return
        self._job_action_requested.emit(request)

    def request_cancel_local_work(self, request: CancelLocalWorkRequest) -> None:
        if not self._running:
            return
        self._cancel_local_work_requested.emit(request)

    def request_job_detail(self, request: JobDetailRequest) -> None:
        if not self._running:
            return
        self._load_job_detail_requested.emit(request)

    def request_recovery_scan(self, request: RecoveryRequest) -> None:
        if not self._running:
            return
        self._scan_recovery_requested.emit(request)

    def request_diagnostics_load(self, request: DiagnosticsRequest) -> None:
        if not self._running:
            return
        self._load_diagnostics_requested.emit(request)

    def request_diagnostics_export(self, request: DiagnosticExportRequest) -> None:
        if not self._running:
            return
        self._export_diagnostics_requested.emit(request)

    def stop(self, timeout_ms: int = 5000) -> bool:
        if timeout_ms < 0:
            raise ValueError("gui.runner_timeout_invalid")
        if not self._running and self._thread is None:
            return True
        if self._stopping:
            thread = self._thread
            if thread is None:
                return True
            return thread.wait(timeout_ms)

        thread = self._thread
        if thread is None:
            self._running = False
            if not self._stop_emitted:
                self._stop_emitted = True
                self.stopped.emit()
            return True

        self._stopping = True
        self._shutdown_requested.emit()
        finished = thread.wait(timeout_ms)
        if not finished:
            self._stopping = False
            return False

        worker = self._worker
        if worker is not None:
            worker.deleteLater()
        thread.deleteLater()
        self._worker = None
        self._thread = None
        self._running = False
        self._stopping = False
        if not self._stop_emitted:
            self._stop_emitted = True
            self.stopped.emit()
        return True


__all__ = ["ApplicationRunnerBridge", "BoundaryFactory", "RunnerFailure"]
