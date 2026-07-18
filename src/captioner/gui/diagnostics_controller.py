"""Main-thread controller for Diagnostics snapshot load and export."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from captioner.core.application.diagnostics import (
    DiagnosticExportRequest,
    DiagnosticExportResult,
    DiagnosticsRequest,
    DiagnosticsSnapshot,
)
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure
from captioner.infrastructure.ids import new_id


class DiagnosticsController(QObject):
    """Coordinates diagnostics refresh/export via the shared Application runner."""

    snapshot_changed = Signal(object)
    refresh_busy_changed = Signal(bool)
    refresh_failure_changed = Signal(object)
    export_busy_changed = Signal(bool)
    export_succeeded = Signal(object)
    export_failure_changed = Signal(object)

    def __init__(
        self,
        runner: ApplicationRunnerBridge,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._snapshot: DiagnosticsSnapshot | None = None
        self._refresh_generation = 0
        self._pending_refresh_generation = 0
        self._refresh_in_flight = False
        self._refresh_queued = False
        self._export_in_flight = False
        self._export_request_id: str | None = None
        self._refresh_failure: RunnerFailure | None = None
        self._export_failure: RunnerFailure | None = None

        self._runner.diagnostics_ready.connect(self._on_snapshot)
        self._runner.diagnostics_failure.connect(self._on_refresh_failure)
        self._runner.diagnostic_export_ready.connect(self._on_export_ready)
        self._runner.diagnostic_export_failure.connect(self._on_export_failure)

    @property
    def snapshot(self) -> DiagnosticsSnapshot | None:
        return self._snapshot

    @property
    def refresh_busy(self) -> bool:
        return self._refresh_in_flight

    @property
    def export_busy(self) -> bool:
        return self._export_in_flight

    def refresh(self) -> None:
        self._refresh_generation += 1
        if self._refresh_in_flight:
            self._refresh_queued = True
            return
        self._dispatch_refresh()

    def export(self, destination: str, *, overwrite: bool) -> None:
        if self._export_in_flight:
            return
        request_id = new_id("req-")
        self._export_request_id = request_id
        self._export_in_flight = True
        self._export_failure = None
        self.export_failure_changed.emit(None)
        self.export_busy_changed.emit(True)
        self._runner.request_diagnostics_export(
            DiagnosticExportRequest(
                request_id=request_id,
                destination=destination,
                overwrite=overwrite,
            )
        )

    def _dispatch_refresh(self) -> None:
        self._refresh_in_flight = True
        self._refresh_queued = False
        self._pending_refresh_generation = self._refresh_generation
        self.refresh_busy_changed.emit(True)
        self._runner.request_diagnostics_load(DiagnosticsRequest(request_id=new_id("req-")))

    @Slot(object)
    def _on_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, DiagnosticsSnapshot):
            return
        if self._pending_refresh_generation != self._refresh_generation:
            if self._refresh_queued:
                self._dispatch_refresh()
            else:
                self._refresh_in_flight = False
                self.refresh_busy_changed.emit(False)
            return
        self._snapshot = snapshot
        self._refresh_failure = None
        self.refresh_failure_changed.emit(None)
        self.snapshot_changed.emit(snapshot)
        if self._refresh_queued:
            self._dispatch_refresh()
            return
        self._refresh_in_flight = False
        self.refresh_busy_changed.emit(False)

    @Slot(object)
    def _on_refresh_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        if self._pending_refresh_generation != self._refresh_generation:
            if self._refresh_queued:
                self._dispatch_refresh()
            else:
                self._refresh_in_flight = False
                self.refresh_busy_changed.emit(False)
            return
        self._refresh_failure = failure
        self.refresh_failure_changed.emit(failure)
        if self._refresh_queued:
            self._dispatch_refresh()
            return
        self._refresh_in_flight = False
        self.refresh_busy_changed.emit(False)

    @Slot(object)
    def _on_export_ready(self, result: object) -> None:
        if not isinstance(result, DiagnosticExportResult):
            return
        if self._export_request_id is None or result.request_id != self._export_request_id:
            return
        self._export_request_id = None
        self._export_in_flight = False
        self._export_failure = None
        self.export_failure_changed.emit(None)
        self.export_busy_changed.emit(False)
        self.export_succeeded.emit(result)

    @Slot(object)
    def _on_export_failure(self, failure: object) -> None:
        if not self._export_in_flight:
            return
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        self._export_request_id = None
        self._export_in_flight = False
        self._export_failure = failure
        self.export_busy_changed.emit(False)
        self.export_failure_changed.emit(failure)


__all__ = ["DiagnosticsController"]
