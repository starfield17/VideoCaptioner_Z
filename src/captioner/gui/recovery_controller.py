"""Main-thread controller for explicit startup recovery discovery."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from captioner.core.application.recovery import RecoveryRequest, RecoverySnapshot
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure
from captioner.infrastructure.ids import new_id


class RecoveryController(QObject):
    snapshot_changed = Signal(object)
    prompt_requested = Signal(object)
    busy_changed = Signal(bool)
    failure_changed = Signal(object)

    def __init__(
        self,
        runner: ApplicationRunnerBridge,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._prompted_batch_ids: set[str] = set()
        self._request_generation = 0
        self._pending_generation = 0
        self._scan_in_flight = False
        self._scan_queued = False
        self._snapshot: RecoverySnapshot | None = None
        self._failure: RunnerFailure | None = None

        self._runner.recovery_ready.connect(self._on_ready)
        self._runner.recovery_failure.connect(self._on_failure)

    @property
    def snapshot(self) -> RecoverySnapshot | None:
        return self._snapshot

    @property
    def busy(self) -> bool:
        return self._scan_in_flight

    def scan(self) -> None:
        self._request_generation += 1
        if self._scan_in_flight:
            self._scan_queued = True
            return
        self._dispatch()

    def mark_handled(self, batch_id: str) -> None:
        self._prompted_batch_ids.add(batch_id)

    def _dispatch(self) -> None:
        self._scan_in_flight = True
        self._scan_queued = False
        self._pending_generation = self._request_generation
        self.busy_changed.emit(True)
        self._runner.request_recovery_scan(RecoveryRequest(request_id=new_id("req-")))

    @Slot(object)
    def _on_ready(self, snapshot: object) -> None:
        if not isinstance(snapshot, RecoverySnapshot):
            return
        if self._pending_generation != self._request_generation:
            if self._scan_queued:
                self._dispatch()
            else:
                self._scan_in_flight = False
                self.busy_changed.emit(False)
            return
        self._snapshot = snapshot
        self._failure = None
        self.failure_changed.emit(None)
        self.snapshot_changed.emit(snapshot)
        new_items = tuple(
            item for item in snapshot.items if item.batch_id not in self._prompted_batch_ids
        )
        if new_items:
            for item in new_items:
                self._prompted_batch_ids.add(item.batch_id)
            self.prompt_requested.emit(new_items)
        if self._scan_queued:
            self._dispatch()
            return
        self._scan_in_flight = False
        self.busy_changed.emit(False)

    @Slot(object)
    def _on_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        if self._pending_generation != self._request_generation:
            if self._scan_queued:
                self._dispatch()
            else:
                self._scan_in_flight = False
                self.busy_changed.emit(False)
            return
        self._failure = failure
        self.failure_changed.emit(failure)
        if self._scan_queued:
            self._dispatch()
            return
        self._scan_in_flight = False
        self.busy_changed.emit(False)


__all__ = ["RecoveryController"]
