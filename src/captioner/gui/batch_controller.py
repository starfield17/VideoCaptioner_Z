"""Main-thread coordinator for Queue presentation refresh."""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from captioner.core.application.queue_projection import QueueSnapshot
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure
from captioner.gui.queue_table_model import QueueTableModel


class BatchController(QObject):
    """Coordinates Queue model updates; does not submit or cancel Jobs."""

    snapshot_changed = Signal(object)
    failure_changed = Signal(object)
    busy_changed = Signal(bool)
    running_changed = Signal(bool)

    def __init__(
        self,
        model: QueueTableModel,
        runner: ApplicationRunnerBridge,
        *,
        refresh_interval_ms: int = 1000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if refresh_interval_ms <= 0:
            raise ValueError("gui.refresh_interval_invalid")
        self._model = model
        self._runner = runner
        self._timer = QTimer(self)
        self._timer.setInterval(refresh_interval_ms)
        self._timer.timeout.connect(self.refresh)
        self._running = False
        self._refresh_in_flight = False
        self._refresh_queued = False
        self._current_snapshot: QueueSnapshot | None = None
        self._last_failure: RunnerFailure | None = None

        self._runner.snapshot_ready.connect(self._on_snapshot)
        self._runner.failure.connect(self._on_failure)

    @property
    def model(self) -> QueueTableModel:
        return self._model

    @property
    def running(self) -> bool:
        return self._running

    @property
    def busy(self) -> bool:
        return self._refresh_in_flight

    @property
    def current_snapshot(self) -> QueueSnapshot | None:
        return self._current_snapshot

    @property
    def last_failure(self) -> RunnerFailure | None:
        return self._last_failure

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._refresh_in_flight = True
        self._refresh_queued = False
        self._runner.start()
        self._timer.start()
        self.running_changed.emit(True)
        self.busy_changed.emit(True)

    @Slot()
    def refresh(self) -> None:
        if not self._running:
            return
        if self._refresh_in_flight:
            self._refresh_queued = True
            return
        self._refresh_in_flight = True
        self.busy_changed.emit(True)
        self._runner.request_refresh()

    def stop(self, timeout_ms: int = 5000) -> bool:
        if not self._running and not self._runner.running:
            return True
        self._timer.stop()
        self._refresh_queued = False
        stopped = self._runner.stop(timeout_ms=timeout_ms)
        if not stopped:
            return False
        was_busy = self._refresh_in_flight
        self._refresh_in_flight = False
        if self._running:
            self._running = False
            self.running_changed.emit(False)
        if was_busy:
            self.busy_changed.emit(False)
        return True

    @Slot(object)
    def _on_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, QueueSnapshot):
            self._finish_refresh()
            return
        accepted = self._model.apply_snapshot(snapshot)
        if accepted:
            self._current_snapshot = snapshot
            if self._last_failure is not None:
                self._last_failure = None
                self.failure_changed.emit(None)
            self.snapshot_changed.emit(snapshot)
        self._finish_refresh()

    @Slot(object)
    def _on_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        self._last_failure = failure
        self.failure_changed.emit(failure)
        self._finish_refresh()

    def _finish_refresh(self) -> None:
        queued = self._refresh_queued
        self._refresh_queued = False
        self._refresh_in_flight = False
        self.busy_changed.emit(False)
        if queued and self._running:
            self.refresh()


__all__ = ["BatchController"]
