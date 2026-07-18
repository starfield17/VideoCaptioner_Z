"""Modal recovery prompt for recoverable Batches."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.recovery import RecoveryItem
from captioner.core.domain.batch import BatchState
from captioner.i18n.service import I18nService

_BATCH_STATE_KEYS: dict[BatchState, str] = {
    BatchState.PENDING: "gui.recovery.state.pending",
    BatchState.RUNNING: "gui.recovery.state.running",
    BatchState.PARTIAL: "gui.recovery.state.partial",
    BatchState.INTERRUPTED: "gui.recovery.state.interrupted",
    BatchState.FAILED: "gui.recovery.state.failed",
    BatchState.CANCELLED: "gui.recovery.state.failed",
    BatchState.SUCCEEDED: "gui.recovery.state.pending",
}


class RecoveryDialog(QDialog):
    def __init__(
        self,
        service: I18nService,
        items: tuple[RecoveryItem, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("recoveryDialog")
        self.setWindowTitle(service.translate("gui.recovery.title"))
        self._service = service
        self._items = items
        self._selected_batch_id: str | None = None
        self._action: str | None = None

        layout = QVBoxLayout(self)
        description = QLabel(service.translate("gui.recovery.description"))
        description.setWordWrap(True)
        layout.addWidget(description)

        self._table = QTableWidget(0, 5)
        self._table.setObjectName("recoveryTable")
        self._table.setHorizontalHeaderLabels(
            [
                service.translate("gui.recovery.batch"),
                service.translate("gui.recovery.state"),
                service.translate("gui.recovery.jobs"),
                service.translate("gui.recovery.reason"),
                service.translate("gui.recovery.missing"),
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self._table)

        self._failure = QLabel("")
        self._failure.setObjectName("recoveryFailureLabel")
        self._failure.setVisible(False)
        layout.addWidget(self._failure)

        buttons = QHBoxLayout()
        self._resume = QPushButton(service.translate("gui.recovery.resume"))
        self._resume.setObjectName("recoveryResumeButton")
        self._resume.clicked.connect(self._on_resume)
        self._cancel = QPushButton(service.translate("gui.recovery.cancel"))
        self._cancel.setObjectName("recoveryCancelButton")
        self._cancel.clicked.connect(self._on_cancel)
        self._later = QPushButton(service.translate("gui.recovery.later"))
        self._later.setObjectName("recoveryLaterButton")
        self._later.clicked.connect(self._on_later)
        buttons.addWidget(self._resume)
        buttons.addWidget(self._cancel)
        buttons.addStretch(1)
        buttons.addWidget(self._later)
        layout.addLayout(buttons)

        self._populate()
        self._update_buttons()

    @property
    def action(self) -> str | None:
        return self._action

    @property
    def selected_batch_id(self) -> str | None:
        return self._selected_batch_id

    def _populate(self) -> None:
        self._table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            reason = self._service.translate("gui.recovery.pending")
            if item.pause_requested:
                reason = self._service.translate("gui.recovery.paused")
            elif item.state is BatchState.INTERRUPTED:
                reason = self._service.translate("gui.recovery.interrupted")
            if item.blocked_code:
                reason = self._service.translate("gui.recovery.blocked")
            missing_text = "—"
            if item.missing_input_paths:
                basenames = [Path(path).name for path in item.missing_input_paths[:5]]
                missing_text = self._service.translate(
                    "gui.recovery.input_missing",
                    {
                        "count": str(len(item.missing_input_paths)),
                        "names": ", ".join(basenames),
                    },
                )
            state_key = _BATCH_STATE_KEYS.get(item.state, "gui.recovery.state.pending")
            if item.blocked_code is not None:
                state_key = "gui.recovery.state.blocked"
            elif item.pause_requested:
                state_key = "gui.recovery.state.paused"
            values = (
                item.batch_id,
                self._service.translate(state_key),
                str(item.job_count),
                reason,
                missing_text,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item.batch_id)
                if column == 4 and item.missing_input_paths:
                    cell.setToolTip("\n".join(item.missing_input_paths))
                self._table.setItem(row, column, cell)
        if self._items:
            self._table.selectRow(0)

    def _on_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._selected_batch_id = None
        else:
            item = self._table.item(rows[0].row(), 0)
            self._selected_batch_id = (
                None if item is None else str(item.data(Qt.ItemDataRole.UserRole))
            )
        self._update_buttons()

    def _selected_item(self) -> RecoveryItem | None:
        if self._selected_batch_id is None:
            return None
        for item in self._items:
            if item.batch_id == self._selected_batch_id:
                return item
        return None

    def _update_buttons(self) -> None:
        item = self._selected_item()
        self._resume.setEnabled(item is not None and item.can_resume)
        self._cancel.setEnabled(item is not None)

    def _on_resume(self) -> None:
        item = self._selected_item()
        if item is None or not item.can_resume:
            return
        self._action = "resume"
        self.accept()

    def _on_cancel(self) -> None:
        if self._selected_item() is None:
            return
        self._action = "cancel"
        self.accept()

    def _on_later(self) -> None:
        self._action = "later"
        self.reject()


__all__ = ["RecoveryDialog"]
