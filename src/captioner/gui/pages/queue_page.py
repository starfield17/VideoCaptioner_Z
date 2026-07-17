"""Queue presentation page bound to BatchController."""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.queue_projection import QueueSnapshot
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.batch_controller import BatchController
from captioner.gui.queue_table_model import QueueColumn
from captioner.i18n.service import I18nService


class QueuePage(QWidget):
    """Functional Queue surface for PR5.2."""

    def __init__(
        self,
        service: I18nService,
        controller: BatchController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("queuePage")
        self._service = service
        self._controller = controller
        self._selected_key: tuple[str, str] | None = None

        self._title = QLabel(service.translate("gui.queue.title"))
        self._title.setObjectName("queueTitle")

        self._busy_label = QLabel(service.translate("gui.queue.refreshing"))
        self._busy_label.setObjectName("queueBusyLabel")
        self._busy_label.setVisible(False)

        self._refresh_button = QPushButton(service.translate("gui.queue.refresh"))
        self._refresh_button.setObjectName("queueRefreshButton")
        self._refresh_button.clicked.connect(controller.refresh)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._title)
        toolbar.addStretch(1)
        toolbar.addWidget(self._busy_label)
        toolbar.addWidget(self._refresh_button)

        self._summary_label = QLabel("")
        self._summary_label.setObjectName("queueSummaryLabel")

        self._issue_label = QLabel("")
        self._issue_label.setObjectName("queueIssueLabel")
        self._issue_label.setVisible(False)

        self._failure_label = QLabel("")
        self._failure_label.setObjectName("queueFailureLabel")
        self._failure_label.setVisible(False)

        self._table = QTableView()
        self._table.setObjectName("queueTable")
        self._table.setModel(controller.model)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(int(QueueColumn.INPUT), QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(int(QueueColumn.OUTPUT), QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            int(QueueColumn.ATTEMPT),
            QHeaderView.ResizeMode.ResizeToContents,
        )
        header.setStretchLastSection(False)

        self._empty_label = QLabel(service.translate("gui.queue.empty"))
        self._empty_label.setObjectName("queueEmptyLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)

        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(self._empty_label)
        self._content_stack.addWidget(self._table)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self._summary_label)
        layout.addWidget(self._issue_label)
        layout.addWidget(self._failure_label)
        layout.addWidget(self._content_stack, stretch=1)

        model = controller.model
        model.modelAboutToBeReset.connect(self._capture_selection)
        model.modelReset.connect(self._restore_selection)
        model.rowsInserted.connect(self._on_rows_changed)
        model.rowsRemoved.connect(self._on_rows_changed)
        model.dataChanged.connect(self._on_data_changed)

        controller.snapshot_changed.connect(self._on_snapshot)
        controller.failure_changed.connect(self._on_failure)
        controller.busy_changed.connect(self._on_busy)

        self._apply_busy(controller.busy)
        self._render_snapshot(controller.current_snapshot)
        self._render_failure(controller.last_failure)

    def _capture_selection(self) -> None:
        self._selected_key = self._current_selection_key()

    def _restore_selection(self) -> None:
        key = self._selected_key
        if key is None:
            self._table.clearSelection()
            self._update_empty_state()
            return
        row = self._controller.model.row_for_key(key)
        if row is None:
            self._selected_key = None
            self._table.clearSelection()
        else:
            index = self._controller.model.index(row, 0)
            self._table.selectionModel().select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            self._table.setCurrentIndex(index)
        self._update_empty_state()

    def _current_selection_key(self) -> tuple[str, str] | None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return self._selected_key
        return self._controller.model.key_at(indexes[0].row())

    def _on_snapshot(self, snapshot: object) -> None:
        if isinstance(snapshot, QueueSnapshot):
            self._render_snapshot(snapshot)

    def _on_failure(self, failure: object) -> None:
        if failure is None:
            self._render_failure(None)
            return
        if isinstance(failure, RunnerFailure):
            self._render_failure(failure)

    def _on_busy(self, busy: bool) -> None:
        self._apply_busy(busy)

    def _on_rows_changed(self, *_args: object) -> None:
        self._update_empty_state()

    def _on_data_changed(
        self,
        _top_left: QModelIndex,
        _bottom_right: QModelIndex,
        _roles: object = None,
    ) -> None:
        # Keep selection when only cell values change.
        return

    def _apply_busy(self, busy: bool) -> None:
        self._refresh_button.setEnabled(not busy)
        self._busy_label.setVisible(busy)

    def _render_snapshot(self, snapshot: QueueSnapshot | None) -> None:
        if snapshot is None:
            self._summary_label.setText("")
            self._issue_label.clear()
            self._issue_label.setVisible(False)
            self._update_empty_state()
            return
        self._summary_label.setText(
            self._service.translate(
                "gui.queue.summary",
                {
                    "active": snapshot.active_count,
                    "terminal": snapshot.terminal_count,
                    "hidden": snapshot.omitted_terminal_jobs,
                },
            )
        )
        issue_count = len(snapshot.issues)
        if issue_count > 0:
            self._issue_label.setText(
                self._service.translate("gui.queue.issues", {"count": issue_count})
            )
            self._issue_label.setVisible(True)
        else:
            self._issue_label.clear()
            self._issue_label.setVisible(False)
        self._update_empty_state()

    def _render_failure(self, failure: RunnerFailure | None) -> None:
        if failure is None:
            self._failure_label.clear()
            self._failure_label.setVisible(False)
            return
        self._failure_label.setText(
            self._service.translate("gui.queue.failure", {"code": failure.code})
        )
        self._failure_label.setVisible(True)

    def _update_empty_state(self) -> None:
        if self._controller.model.rowCount() == 0:
            self._content_stack.setCurrentWidget(self._empty_label)
        else:
            self._content_stack.setCurrentWidget(self._table)


__all__ = ["QueuePage"]
