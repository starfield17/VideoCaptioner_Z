"""Read-only Qt table model projecting immutable Queue snapshots."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
)

from captioner.core.application.queue_projection import JobQueueItem, QueueSnapshot
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName
from captioner.i18n.service import I18nService

_ModelIndex = QModelIndex | QPersistentModelIndex
_INVALID_PARENT = QModelIndex()

_JOB_STATE_KEYS: dict[JobState, str] = {
    JobState.PENDING: "gui.queue.state.pending",
    JobState.RUNNING: "gui.queue.state.running",
    JobState.INTERRUPTED: "gui.queue.state.interrupted",
    JobState.FAILED: "gui.queue.state.failed",
    JobState.CANCELLED: "gui.queue.state.cancelled",
    JobState.SUCCEEDED: "gui.queue.state.succeeded",
}
_PROFILE_KEYS: dict[PipelineProfile, str] = {
    PipelineProfile.DETERMINISTIC: "gui.queue.profile.deterministic",
    PipelineProfile.FAST: "gui.queue.profile.fast",
    PipelineProfile.QUALITY: "gui.queue.profile.quality",
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
_COLUMN_HEADER_KEYS: dict[int, str] = {
    0: "gui.queue.column.input",
    1: "gui.queue.column.profile",
    2: "gui.queue.column.status",
    3: "gui.queue.column.stage",
    4: "gui.queue.column.attempt",
    5: "gui.queue.column.output",
    6: "gui.queue.column.batch",
}
_EMPTY_CELL = "—"
_DATA_CHANGE_ROLES = [
    Qt.ItemDataRole.DisplayRole,
    Qt.ItemDataRole.ToolTipRole,
    Qt.ItemDataRole.TextAlignmentRole,
    Qt.ItemDataRole.UserRole,
]


class QueueColumn(IntEnum):
    INPUT = 0
    PROFILE = 1
    STATUS = 2
    STAGE = 3
    ATTEMPT = 4
    OUTPUT = 5
    BATCH = 6


class QueueTableModel(QAbstractTableModel):
    """GUI-only projection of ``QueueSnapshot`` rows."""

    def __init__(self, service: I18nService, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._i18n = service
        self._items: tuple[JobQueueItem, ...] = ()
        self._snapshot: QueueSnapshot | None = None
        self._revision = 0

    @property
    def snapshot(self) -> QueueSnapshot | None:
        return self._snapshot

    @property
    def revision(self) -> int:
        return self._revision

    def rowCount(self, parent: _ModelIndex = _INVALID_PARENT) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def columnCount(self, parent: _ModelIndex = _INVALID_PARENT) -> int:
        if parent.isValid():
            return 0
        return len(QueueColumn)

    def flags(self, index: _ModelIndex) -> Qt.ItemFlag:
        if not index.isValid() or not self._is_in_range(index):
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object | None:
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role != int(Qt.ItemDataRole.DisplayRole):
            return None
        key = _COLUMN_HEADER_KEYS.get(section)
        if key is None:
            return None
        return self._i18n.translate(key)

    def data(
        self,
        index: _ModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object | None:
        if not index.isValid() or not self._is_in_range(index):
            return None
        item = self._items[index.row()]
        column = index.column()
        if role == int(Qt.ItemDataRole.UserRole):
            return item
        if role == int(Qt.ItemDataRole.TextAlignmentRole):
            if column == int(QueueColumn.ATTEMPT):
                return int(Qt.AlignmentFlag.AlignCenter)
            return None
        if role == int(Qt.ItemDataRole.DisplayRole):
            return self._display_value(item, column)
        if role == int(Qt.ItemDataRole.ToolTipRole):
            return self._tooltip_value(item, column)
        return None

    def item_at(self, row: int) -> JobQueueItem | None:
        if row < 0 or row >= len(self._items):
            return None
        return self._items[row]

    def key_at(self, row: int) -> tuple[str, str] | None:
        item = self.item_at(row)
        if item is None:
            return None
        return (item.batch_id, item.job_id)

    def row_for_key(self, key: tuple[str, str]) -> int | None:
        for index, item in enumerate(self._items):
            if (item.batch_id, item.job_id) == key:
                return index
        return None

    def apply_snapshot(self, snapshot: QueueSnapshot) -> bool:
        if snapshot.revision <= self._revision:
            return False

        old_items = self._items
        new_items = snapshot.items
        old_keys = tuple((item.batch_id, item.job_id) for item in old_items)
        new_keys = tuple((item.batch_id, item.job_id) for item in new_items)

        if old_keys != new_keys:
            self.beginResetModel()
            self._items = new_items
            self._snapshot = snapshot
            self._revision = snapshot.revision
            self.endResetModel()
            return True

        self._items = new_items
        self._snapshot = snapshot
        self._revision = snapshot.revision

        if old_items == new_items:
            return True

        changed_rows = [
            row
            for row, (old, new) in enumerate(zip(old_items, new_items, strict=True))
            if old != new
        ]
        for start, end in _contiguous_ranges(changed_rows):
            top_left = self.index(start, 0)
            bottom_right = self.index(end, len(QueueColumn) - 1)
            self.dataChanged.emit(top_left, bottom_right, _DATA_CHANGE_ROLES)
        return True

    def _is_in_range(self, index: _ModelIndex) -> bool:
        return 0 <= index.row() < len(self._items) and 0 <= index.column() < len(QueueColumn)

    def _display_value(self, item: JobQueueItem, column: int) -> str:
        if column == int(QueueColumn.INPUT):
            return Path(item.input_path).name
        if column == int(QueueColumn.PROFILE):
            return self._i18n.translate(_PROFILE_KEYS[item.pipeline_profile])
        if column == int(QueueColumn.STATUS):
            if item.cancel_requested and not item.terminal:
                return self._i18n.translate("gui.queue.state.cancelling")
            return self._i18n.translate(_JOB_STATE_KEYS[item.state])
        if column == int(QueueColumn.STAGE):
            if item.active_stage is None:
                return _EMPTY_CELL
            return self._i18n.translate(_STAGE_KEYS[item.active_stage])
        if column == int(QueueColumn.ATTEMPT):
            if item.active_stage_attempt > 0:
                return str(item.active_stage_attempt)
            return _EMPTY_CELL
        if column == int(QueueColumn.OUTPUT):
            return item.output_dir
        if column == int(QueueColumn.BATCH):
            return item.batch_id
        return _EMPTY_CELL

    def _tooltip_value(self, item: JobQueueItem, column: int) -> str | None:
        if column == int(QueueColumn.INPUT):
            return item.input_path
        if column == int(QueueColumn.OUTPUT):
            return item.output_dir
        if column == int(QueueColumn.BATCH):
            return self._i18n.translate(
                "gui.queue.batch_tooltip",
                {"batch_id": item.batch_id, "job_id": item.job_id},
            )
        return self._display_value(item, column)


def _contiguous_ranges(rows: list[int]) -> list[tuple[int, int]]:
    if not rows:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row == previous + 1:
            previous = row
            continue
        ranges.append((start, previous))
        start = previous = row
    ranges.append((start, previous))
    return ranges


__all__ = ["QueueColumn", "QueueTableModel"]
