"""Main-thread controller for Job actions, detail, and local execution state."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from captioner.core.application.batch_commands import (
    BatchActionRequest,
    BatchCommandAck,
    BatchCommandFailure,
    BatchCommandKind,
    CancelLocalWorkRequest,
    ExecutionCompletion,
    JobActionRequest,
    LocalExecutionSnapshot,
)
from captioner.core.application.job_detail import JobDetailRequest, JobDetailSnapshot
from captioner.core.application.queue_projection import JobQueueItem
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure
from captioner.infrastructure.ids import new_id


class JobOperationsController(QObject):
    selection_changed = Signal(object)
    detail_changed = Signal(object)
    detail_busy_changed = Signal(bool)
    command_busy_changed = Signal(bool)
    command_succeeded = Signal(object)
    command_failed = Signal(object)
    notification_changed = Signal(object)
    local_execution_state_changed = Signal(object)
    refresh_requested = Signal()
    # Emitted when CancelLocalWork fails while a close-when-idle workflow is active.
    close_cancellation_failed = Signal(object)

    def __init__(
        self,
        runner: ApplicationRunnerBridge,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._selected: JobQueueItem | None = None
        self._detail: JobDetailSnapshot | None = None
        self._detail_busy = False
        self._detail_generation = 0
        self._pending_detail_generation = 0
        self._detail_queued = False
        self._command_busy = False
        self._pending_request_id: str | None = None
        self._pending_kind: BatchCommandKind | None = None
        # Independent shutdown-cancellation state: never swallowed by ordinary commands.
        self._cancel_local_work_pending = False
        self._cancel_local_work_request_id: str | None = None
        self._close_cancellation_in_flight = False
        self._execution = LocalExecutionSnapshot(active_batch_id=None, queued_batch_ids=())
        self._last_notification: str | None = None
        self._last_command_failure: RunnerFailure | None = None

        self._runner.job_detail_ready.connect(self._on_detail)
        self._runner.job_detail_failure.connect(self._on_detail_failure)
        self._runner.batch_command_ready.connect(self._on_command)
        self._runner.batch_command_failure.connect(self._on_command_failure)
        self._runner.local_execution_state_changed.connect(self._on_execution_state)
        self._runner.execution_completion.connect(self._on_execution_completion)

    @property
    def selected(self) -> JobQueueItem | None:
        return self._selected

    @property
    def detail(self) -> JobDetailSnapshot | None:
        return self._detail

    @property
    def detail_busy(self) -> bool:
        return self._detail_busy

    @property
    def command_busy(self) -> bool:
        return self._command_busy

    @property
    def cancel_local_work_pending(self) -> bool:
        return self._cancel_local_work_pending

    @property
    def close_cancellation_in_flight(self) -> bool:
        return self._close_cancellation_in_flight

    @property
    def has_local_work(self) -> bool:
        return self._execution.has_work

    @property
    def execution_state(self) -> LocalExecutionSnapshot:
        return self._execution

    @property
    def last_notification(self) -> str | None:
        return self._last_notification

    def select_job(self, item: JobQueueItem | None) -> None:
        self._selected = item
        self.selection_changed.emit(item)
        if item is None:
            self._detail = None
            self.detail_changed.emit(None)
            return
        self.refresh_detail()

    def refresh_detail(self) -> None:
        item = self._selected
        if item is None:
            self._detail = None
            self.detail_changed.emit(None)
            return
        self._detail_generation += 1
        if self._detail_busy:
            self._detail_queued = True
            return
        self._dispatch_detail()

    def cancel_job(self) -> None:
        self._dispatch_job_action(BatchCommandKind.CANCEL_JOB)

    def cancel_batch(self) -> None:
        self._dispatch_batch_action(BatchCommandKind.CANCEL_BATCH)

    def pause_batch(self) -> None:
        self._dispatch_batch_action(BatchCommandKind.PAUSE_BATCH)

    def resume_batch(self) -> None:
        self._dispatch_batch_action(BatchCommandKind.RESUME_BATCH)

    def retry_job(self) -> None:
        self._dispatch_job_action(BatchCommandKind.RETRY_JOB)

    def run_again(self) -> None:
        self._dispatch_job_action(BatchCommandKind.RUN_AGAIN)

    def cancel_all_local_work(self) -> bool:
        """Queue or dispatch CancelLocalWork without dropping close-when-idle intent.

        Returns True when cancel-all is already pending, in flight, or newly dispatched.
        """
        if self._close_cancellation_in_flight or self._cancel_local_work_pending:
            return True
        if self._command_busy and self._pending_kind is not BatchCommandKind.CANCEL_LOCAL_WORK:
            # Ordinary command in flight: mark pending and dispatch after it settles.
            self._cancel_local_work_pending = True
            return True
        self._dispatch_cancel_local_work()
        return True

    def resume_batch_id(self, batch_id: str) -> None:
        if self._mutation_blocked():
            return
        request_id = new_id("req-")
        self._pending_request_id = request_id
        self._pending_kind = BatchCommandKind.RESUME_BATCH
        self._command_busy = True
        self.command_busy_changed.emit(True)
        self._runner.request_batch_action(
            BatchActionRequest(
                request_id=request_id,
                kind=BatchCommandKind.RESUME_BATCH,
                batch_id=batch_id,
            )
        )

    def cancel_batch_id(self, batch_id: str) -> None:
        if self._mutation_blocked():
            return
        request_id = new_id("req-")
        self._pending_request_id = request_id
        self._pending_kind = BatchCommandKind.CANCEL_BATCH
        self._command_busy = True
        self.command_busy_changed.emit(True)
        self._runner.request_batch_action(
            BatchActionRequest(
                request_id=request_id,
                kind=BatchCommandKind.CANCEL_BATCH,
                batch_id=batch_id,
            )
        )

    def _mutation_blocked(self) -> bool:
        return (
            self._command_busy
            or self._cancel_local_work_pending
            or self._close_cancellation_in_flight
        )

    def _dispatch_cancel_local_work(self) -> None:
        if self._close_cancellation_in_flight:
            return
        request_id = new_id("req-")
        self._cancel_local_work_request_id = request_id
        self._close_cancellation_in_flight = True
        self._cancel_local_work_pending = False
        self._runner.request_cancel_local_work(CancelLocalWorkRequest(request_id=request_id))

    def _maybe_dispatch_pending_cancel_local_work(self) -> None:
        if self._cancel_local_work_pending and not self._close_cancellation_in_flight:
            self._dispatch_cancel_local_work()

    def _dispatch_detail(self) -> None:
        item = self._selected
        if item is None:
            self._detail_busy = False
            self.detail_busy_changed.emit(False)
            return
        self._detail_busy = True
        self._detail_queued = False
        self._pending_detail_generation = self._detail_generation
        self.detail_busy_changed.emit(True)
        request = JobDetailRequest(
            request_id=new_id("req-"),
            batch_id=item.batch_id,
            job_id=item.job_id,
        )
        self._runner.request_job_detail(request)

    def _dispatch_job_action(self, kind: BatchCommandKind) -> None:
        item = self._selected
        if item is None or self._mutation_blocked():
            return
        request_id = new_id("req-")
        self._pending_request_id = request_id
        self._pending_kind = kind
        self._command_busy = True
        self.command_busy_changed.emit(True)
        self._runner.request_job_action(
            JobActionRequest(
                request_id=request_id,
                kind=kind,  # type: ignore[arg-type]
                batch_id=item.batch_id,
                job_id=item.job_id,
            )
        )

    def _dispatch_batch_action(self, kind: BatchCommandKind) -> None:
        item = self._selected
        if item is None or self._mutation_blocked():
            return
        request_id = new_id("req-")
        self._pending_request_id = request_id
        self._pending_kind = kind
        self._command_busy = True
        self.command_busy_changed.emit(True)
        self._runner.request_batch_action(
            BatchActionRequest(
                request_id=request_id,
                kind=kind,  # type: ignore[arg-type]
                batch_id=item.batch_id,
            )
        )

    @Slot(object)
    def _on_detail(self, detail: object) -> None:
        if not isinstance(detail, JobDetailSnapshot):
            return
        if self._pending_detail_generation != self._detail_generation:
            if self._detail_queued:
                self._dispatch_detail()
            else:
                self._detail_busy = False
                self.detail_busy_changed.emit(False)
            return
        selected = self._selected
        if selected is None or (
            detail.batch_id,
            detail.job_id,
        ) != (selected.batch_id, selected.job_id):
            if self._detail_queued:
                self._dispatch_detail()
            else:
                self._detail_busy = False
                self.detail_busy_changed.emit(False)
            return
        self._detail = detail
        self.detail_changed.emit(detail)
        if self._detail_queued:
            self._dispatch_detail()
            return
        self._detail_busy = False
        self.detail_busy_changed.emit(False)

    @Slot(object)
    def _on_detail_failure(self, failure: object) -> None:
        del failure
        if self._detail_queued:
            self._dispatch_detail()
            return
        self._detail_busy = False
        self.detail_busy_changed.emit(False)

    @Slot(object)
    def _on_command(self, ack: object) -> None:
        if not isinstance(ack, BatchCommandAck):
            return
        # Independent CancelLocalWork correlation (not ordinary selected-Job command).
        if (
            self._cancel_local_work_request_id is not None
            and ack.request_id == self._cancel_local_work_request_id
        ):
            self._cancel_local_work_request_id = None
            self._close_cancellation_in_flight = False
            self._cancel_local_work_pending = False
            self._last_notification = f"command.accepted:{ack.kind.value}"
            self.notification_changed.emit(self._last_notification)
            self.command_succeeded.emit(ack)
            self.refresh_requested.emit()
            return
        if ack.request_id != self._pending_request_id:
            return
        if self._pending_kind is not None and ack.kind is not self._pending_kind:
            return
        self._pending_request_id = None
        self._pending_kind = None
        self._command_busy = False
        self.command_busy_changed.emit(False)
        self._last_notification = f"command.accepted:{ack.kind.value}"
        self.notification_changed.emit(self._last_notification)
        self.command_succeeded.emit(ack)
        self.refresh_requested.emit()
        self.refresh_detail()
        self._maybe_dispatch_pending_cancel_local_work()

    @Slot(object)
    def _on_command_failure(self, failure: object) -> None:
        if not isinstance(failure, BatchCommandFailure):
            return
        if (
            self._cancel_local_work_request_id is not None
            and failure.request_id == self._cancel_local_work_request_id
        ):
            self._cancel_local_work_request_id = None
            self._close_cancellation_in_flight = False
            self._cancel_local_work_pending = False
            safe = RunnerFailure(code=failure.code, retryable=failure.retryable)
            self._last_command_failure = safe
            self.command_failed.emit(safe)
            self.close_cancellation_failed.emit(safe)
            self._last_notification = f"command.failed:{failure.code}"
            self.notification_changed.emit(self._last_notification)
            return
        if failure.request_id != self._pending_request_id:
            return
        self._pending_request_id = None
        self._pending_kind = None
        self._command_busy = False
        self.command_busy_changed.emit(False)
        self._last_command_failure = RunnerFailure(code=failure.code, retryable=failure.retryable)
        self.command_failed.emit(self._last_command_failure)
        self._last_notification = f"command.failed:{failure.code}"
        self.notification_changed.emit(self._last_notification)
        # Still dispatch pending cancel-all after ordinary command failure.
        self._maybe_dispatch_pending_cancel_local_work()

    @Slot(object)
    def _on_execution_state(self, state: object) -> None:
        if not isinstance(state, LocalExecutionSnapshot):
            return
        self._execution = state
        self.local_execution_state_changed.emit(state)

    @Slot(object)
    def _on_execution_completion(self, completion: object) -> None:
        if not isinstance(completion, ExecutionCompletion):
            return
        if completion.ok:
            self._last_notification = f"execution.completed:{completion.batch_id}"
        else:
            self._last_notification = f"execution.failed:{completion.code}"
        self.notification_changed.emit(self._last_notification)
        self.refresh_requested.emit()
        self.refresh_detail()


__all__ = ["JobOperationsController"]
