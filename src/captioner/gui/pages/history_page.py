"""Recent History page filtering terminal Queue rows."""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.queue_projection import JobQueueItem, QueueSnapshot
from captioner.gui.batch_controller import BatchController
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.queue_table_model import QueueColumn
from captioner.gui.widgets.job_detail_panel import JobDetailPanel
from captioner.i18n.service import I18nService


class _TerminalFilterProxy(QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # type: ignore[override]
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        item = model.data(index, int(Qt.ItemDataRole.UserRole))
        return isinstance(item, JobQueueItem) and item.terminal


class HistoryPage(QWidget):
    def __init__(
        self,
        service: I18nService,
        controller: BatchController,
        operations: JobOperationsController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("historyPage")
        self._service = service
        self._controller = controller
        self._operations = operations

        root = QVBoxLayout(self)
        self._title = QLabel(service.translate("gui.history.title"))
        self._title.setObjectName("historyTitle")
        root.addWidget(self._title)

        self._summary = QLabel("")
        self._summary.setObjectName("historySummaryLabel")
        root.addWidget(self._summary)

        self._proxy = _TerminalFilterProxy(self)
        self._proxy.setSourceModel(controller.model)

        self._table = QTableView()
        self._table.setObjectName("historyTable")
        self._table.setModel(self._proxy)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(int(QueueColumn.INPUT), QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(int(QueueColumn.OUTPUT), QHeaderView.ResizeMode.Stretch)

        self._empty = QLabel(service.translate("gui.history.empty"))
        self._empty.setObjectName("historyEmptyLabel")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._detail = JobDetailPanel(service, operations)
        splitter = QSplitter()
        splitter.setObjectName("historyDetailSplitter")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._table)
        left_layout.addWidget(self._empty)
        splitter.addWidget(left)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        self._table.selectionModel().selectionChanged.connect(self._on_selection)
        controller.snapshot_changed.connect(self._on_snapshot)
        self._render_snapshot(controller.current_snapshot)
        self._update_empty()

    def _on_snapshot(self, snapshot: object) -> None:
        if isinstance(snapshot, QueueSnapshot):
            self._render_snapshot(snapshot)
        self._update_empty()

    def _render_snapshot(self, snapshot: QueueSnapshot | None) -> None:
        if snapshot is None:
            self._summary.clear()
            return
        self._summary.setText(
            self._service.translate(
                "gui.history.summary",
                {
                    "terminal": str(snapshot.terminal_count),
                    "hidden": str(snapshot.omitted_terminal_jobs),
                },
            )
        )

    def _update_empty(self) -> None:
        empty = self._proxy.rowCount() == 0
        self._empty.setVisible(empty)
        self._table.setVisible(not empty)

    def _on_selection(self, *_args: object) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            self._operations.select_job(None)
            return
        source = self._proxy.mapToSource(indexes[0])
        item = self._controller.model.item_at(source.row())
        self._operations.select_job(item)


__all__ = ["HistoryPage"]
