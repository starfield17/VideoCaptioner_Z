"""Job detail panel bound only to JobOperationsController."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.job_detail import JobAction, JobDetailSnapshot
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import StageName
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.i18n.service import I18nService

_EVENT_KEYS = {
    "batch.created": "gui.activity.event.batch_created",
    "batch.config_updated": "gui.activity.event.batch_config_updated",
    "job.created": "gui.activity.event.job_created",
    "job.config_updated": "gui.activity.event.job_config_updated",
    "job.retry_requested": "gui.activity.event.job_retry_requested",
    "stage.started": "gui.activity.event.stage_started",
    "stage.interrupted": "gui.activity.event.stage_interrupted",
    "stage.committed": "gui.activity.event.stage_committed",
    "stage.failed": "gui.activity.event.stage_failed",
    "stage.cancelled": "gui.activity.event.stage_cancelled",
    "stage.invalidated": "gui.activity.event.stage_invalidated",
    "job.succeeded": "gui.activity.event.job_succeeded",
    "job.failed": "gui.activity.event.job_failed",
    "job.cancelled": "gui.activity.event.job_cancelled",
}
_JOB_STATE_KEYS: dict[JobState, str] = {
    JobState.PENDING: "gui.queue.state.pending",
    JobState.RUNNING: "gui.queue.state.running",
    JobState.INTERRUPTED: "gui.queue.state.interrupted",
    JobState.FAILED: "gui.queue.state.failed",
    JobState.CANCELLED: "gui.queue.state.cancelled",
    JobState.SUCCEEDED: "gui.queue.state.succeeded",
}
_STAGE_KEYS: dict[StageName, str] = {
    StageName.INSPECT: "gui.queue.stage.inspect",
    StageName.NORMALIZE: "gui.queue.stage.normalize",
    StageName.TRANSCRIBE: "gui.queue.stage.transcribe",
    StageName.CORRECT_SOURCE: "gui.queue.stage.correct_source",
    StageName.SEGMENT: "gui.queue.stage.segment",
    StageName.TRANSLATE: "gui.queue.stage.translate",
    StageName.REVIEW: "gui.queue.stage.review",
    StageName.EXPORT: "gui.queue.stage.export",
    StageName.PUBLISH: "gui.queue.stage.publish",
}


class JobDetailPanel(QWidget):
    def __init__(
        self,
        service: I18nService,
        operations: JobOperationsController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("jobDetailPanel")
        self._service = service
        self._operations = operations

        layout = QVBoxLayout(self)
        self._title = QLabel(service.translate("gui.job.detail.title"))
        self._title.setObjectName("jobDetailTitle")
        layout.addWidget(self._title)

        self._input = QLabel(service.translate("gui.job.detail.none"))
        self._input.setObjectName("jobDetailInputLabel")
        self._input.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._input.setWordWrap(True)
        layout.addWidget(self._input)

        self._output = QLabel("")
        self._output.setObjectName("jobDetailOutputLabel")
        self._output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._output.setWordWrap(True)
        layout.addWidget(self._output)

        self._state = QLabel("")
        self._state.setObjectName("jobDetailStateLabel")
        layout.addWidget(self._state)

        self._stage = QLabel("")
        self._stage.setObjectName("jobDetailStageLabel")
        layout.addWidget(self._stage)

        self._attempt = QLabel("")
        self._attempt.setObjectName("jobDetailAttemptLabel")
        layout.addWidget(self._attempt)

        self._failure = QLabel("")
        self._failure.setObjectName("jobDetailFailureLabel")
        self._failure.setVisible(False)
        layout.addWidget(self._failure)

        buttons = QHBoxLayout()
        self._cancel_job = QPushButton(service.translate("gui.job.action.cancel_job"))
        self._cancel_job.setObjectName("jobCancelButton")
        self._cancel_job.clicked.connect(self._on_cancel_job)
        self._cancel_batch = QPushButton(service.translate("gui.job.action.cancel_batch"))
        self._cancel_batch.setObjectName("batchCancelButton")
        self._cancel_batch.clicked.connect(self._on_cancel_batch)
        self._pause = QPushButton(service.translate("gui.job.action.pause_batch"))
        self._pause.setObjectName("batchPauseButton")
        self._pause.clicked.connect(operations.pause_batch)
        self._resume = QPushButton(service.translate("gui.job.action.resume_batch"))
        self._resume.setObjectName("batchResumeButton")
        self._resume.clicked.connect(operations.resume_batch)
        self._retry = QPushButton(service.translate("gui.job.action.retry"))
        self._retry.setObjectName("jobRetryButton")
        self._retry.clicked.connect(operations.retry_job)
        self._run_again = QPushButton(service.translate("gui.job.action.run_again"))
        self._run_again.setObjectName("jobRunAgainButton")
        self._run_again.clicked.connect(operations.run_again)
        for button in (
            self._cancel_job,
            self._cancel_batch,
            self._pause,
            self._resume,
            self._retry,
            self._run_again,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._activity_summary = QLabel(service.translate("gui.activity.title"))
        self._activity_summary.setObjectName("jobActivitySummaryLabel")
        layout.addWidget(self._activity_summary)

        self._activity_list = QListWidget()
        self._activity_list.setObjectName("jobActivityList")
        layout.addWidget(self._activity_list, stretch=1)

        self._activity_failure = QLabel("")
        self._activity_failure.setObjectName("jobActivityFailureLabel")
        self._activity_failure.setVisible(False)
        layout.addWidget(self._activity_failure)

        operations.detail_changed.connect(self._on_detail)
        operations.command_busy_changed.connect(self._on_command_busy)
        operations.command_failed.connect(self._on_command_failed)
        self._render(None)
        self._set_command_enabled(False)

    def _on_detail(self, detail: object) -> None:
        if detail is None:
            self._render(None)
            return
        if isinstance(detail, JobDetailSnapshot):
            self._render(detail)

    def _on_command_busy(self, busy: bool) -> None:
        if busy:
            self._set_command_enabled(False)
        else:
            self._render(self._operations.detail)

    def _on_command_failed(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            return
        self._failure.setText(
            self._service.translate("gui.job.detail.failure", {"code": failure.code})
        )
        self._failure.setVisible(True)

    def _on_cancel_job(self) -> None:
        if not self._confirm(
            "gui.job.confirm.cancel_job.title",
            "gui.job.confirm.cancel_job.message",
        ):
            return
        self._operations.cancel_job()

    def _on_cancel_batch(self) -> None:
        if not self._confirm(
            "gui.job.confirm.cancel_batch.title",
            "gui.job.confirm.cancel_batch.message",
        ):
            return
        self._operations.cancel_batch()

    def _confirm(self, title_key: str, message_key: str) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle(self._service.translate(title_key))
        box.setText(self._service.translate(message_key))
        yes = box.addButton(
            self._service.translate("gui.value.yes"),
            QMessageBox.ButtonRole.AcceptRole,
        )
        no = box.addButton(
            self._service.translate("gui.value.no"),
            QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(no)
        box.exec()
        return box.clickedButton() is yes

    def _render(self, detail: JobDetailSnapshot | None) -> None:
        if detail is None:
            self._input.setText(self._service.translate("gui.job.detail.none"))
            self._input.setToolTip("")
            self._output.clear()
            self._state.clear()
            self._stage.clear()
            self._attempt.clear()
            self._failure.clear()
            self._failure.setVisible(False)
            self._activity_list.clear()
            self._activity_summary.setText(self._service.translate("gui.activity.empty"))
            self._activity_failure.clear()
            self._activity_failure.setVisible(False)
            self._set_command_enabled(False)
            return

        self._input.setText(
            self._service.translate(
                "gui.job.detail.input",
                {"path": detail.input_path},
            )
        )
        self._input.setToolTip(detail.input_path)
        self._output.setText(
            self._service.translate(
                "gui.job.detail.output",
                {"path": detail.output_dir},
            )
        )
        self._output.setToolTip(detail.output_dir)
        state_label = self._service.translate(_JOB_STATE_KEYS[detail.state])
        self._state.setText(
            self._service.translate(
                "gui.job.detail.state",
                {"state": state_label},
            )
        )
        if detail.active_stage is None:
            stage_text = "—"
        else:
            stage_text = self._service.translate(_STAGE_KEYS[detail.active_stage])
        self._stage.setText(self._service.translate("gui.job.detail.stage", {"stage": stage_text}))
        self._attempt.setText(
            self._service.translate(
                "gui.job.detail.attempt",
                {"attempt": str(detail.active_stage_attempt)},
            )
        )
        self._failure.clear()
        self._failure.setVisible(False)

        actions = set(detail.available_actions)
        busy = self._operations.command_busy
        self._cancel_job.setEnabled(not busy and JobAction.CANCEL_JOB in actions)
        self._cancel_batch.setEnabled(not busy and JobAction.CANCEL_BATCH in actions)
        self._pause.setEnabled(not busy and JobAction.PAUSE_BATCH in actions)
        self._resume.setEnabled(not busy and JobAction.RESUME_BATCH in actions)
        self._retry.setEnabled(not busy and JobAction.RETRY_JOB in actions)
        if detail.retry_stage is not None:
            retry_stage = self._service.translate(_STAGE_KEYS[detail.retry_stage])
            self._retry.setText(
                self._service.translate(
                    "gui.job.action.retry_stage",
                    {"stage": retry_stage},
                )
            )
        else:
            self._retry.setText(self._service.translate("gui.job.action.retry"))
        self._run_again.setEnabled(not busy and JobAction.RUN_AGAIN in actions)
        self._run_again.setToolTip(self._service.translate("gui.job.run_again.description"))

        self._activity_list.clear()
        # Newest first for usability.
        for entry in reversed(detail.activity):
            event_key = _EVENT_KEYS.get(entry.event_type, "gui.activity.event.unknown")
            if entry.event_type not in _EVENT_KEYS:
                event_label = self._service.translate(
                    "gui.activity.event.unknown",
                    {"code": entry.event_type},
                )
            else:
                event_label = self._service.translate(event_key)
            label = f"{entry.timestamp_utc} · {event_label} · seq={entry.seq}"
            if entry.stage_name is not None:
                stage_label = self._service.translate(_STAGE_KEYS[entry.stage_name])
                label += f" · {stage_label}"
            if entry.attempt is not None:
                label += f" · attempt={entry.attempt}"
            if entry.error_code is not None:
                label += f" · {entry.error_code}"
            item = QListWidgetItem(label)
            item.setToolTip(label)
            self._activity_list.addItem(item)

        summary = self._service.translate(
            "gui.activity.summary",
            {"count": str(len(detail.activity))},
        )
        if detail.omitted_activity_count:
            summary += " · " + self._service.translate(
                "gui.activity.omitted",
                {"count": str(detail.omitted_activity_count)},
            )
        self._activity_summary.setText(summary)
        if detail.journal_tail_status == "incomplete":
            self._activity_failure.setText(self._service.translate("gui.activity.incomplete_tail"))
            self._activity_failure.setVisible(True)
        else:
            self._activity_failure.clear()
            self._activity_failure.setVisible(False)

    def _set_command_enabled(self, enabled: bool) -> None:
        for button in (
            self._cancel_job,
            self._cancel_batch,
            self._pause,
            self._resume,
            self._retry,
            self._run_again,
        ):
            if not enabled:
                button.setEnabled(False)


__all__ = ["JobDetailPanel"]
