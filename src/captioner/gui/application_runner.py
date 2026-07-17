"""Dedicated Qt worker thread for Application Queue boundary calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from captioner.core.domain.errors import AppError
from captioner.gui.application_boundary import GuiApplicationBoundary

BoundaryFactory = Callable[[], GuiApplicationBoundary]

_UNEXPECTED_FAILURE_CODE = "gui.application_bridge_failed"


@dataclass(frozen=True, slots=True)
class RunnerFailure:
    code: str
    retryable: bool = False


class _ApplicationRunnerWorker(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    initialized = Signal()

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
            self.failure.emit(RunnerFailure(code=exc.code, retryable=exc.retryable))
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
        except AppError as exc:
            self.failure.emit(RunnerFailure(code=exc.code, retryable=exc.retryable))
        except Exception:
            self.failure.emit(RunnerFailure(code=_UNEXPECTED_FAILURE_CODE, retryable=False))


class ApplicationRunnerBridge(QObject):
    """Main-thread facade over one Application worker thread."""

    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    _refresh_requested = Signal()

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
        self._refresh_requested.connect(
            worker.refresh,
            Qt.ConnectionType.QueuedConnection,
        )

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
