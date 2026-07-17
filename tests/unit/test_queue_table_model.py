"""Unit tests for the read-only Queue table model."""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from captioner.core.application.queue_projection import (
    JobQueueItem,
    QueueLoadIssue,
    QueueSnapshot,
)
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName, StageState
from captioner.gui.queue_table_model import QueueColumn, QueueTableModel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-queue-table-model"])


def _item(
    *,
    batch_id: str = "batch-a",
    job_id: str = "job-000001",
    input_path: str = "/media/clips/sample.wav",
    output_dir: str = "/tmp/out",
    profile: PipelineProfile = PipelineProfile.DETERMINISTIC,
    state: JobState = JobState.PENDING,
    active_stage: StageName | None = None,
    active_stage_state: StageState | None = None,
    active_stage_attempt: int = 0,
    cancel_requested: bool = False,
    job_order: int = 0,
) -> JobQueueItem:
    return JobQueueItem(
        batch_id=batch_id,
        job_id=job_id,
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=job_order,
        input_path=input_path,
        output_dir=output_dir,
        pipeline_profile=profile,
        state=state,
        active_stage=active_stage,
        active_stage_state=active_stage_state,
        active_stage_attempt=active_stage_attempt,
        cancel_requested=cancel_requested,
        last_event_seq=1,
        journal_tail_status="clean",
        manifest_status="missing",
    )


def _snapshot(
    items: tuple[JobQueueItem, ...] = (),
    *,
    revision: int = 1,
    issues: tuple[QueueLoadIssue, ...] = (),
    omitted: int = 0,
) -> QueueSnapshot:
    return QueueSnapshot(
        schema_version=1,
        revision=revision,
        items=items,
        issues=issues,
        omitted_terminal_jobs=omitted,
    )


def test_structure_and_flags() -> None:
    service = I18nService("en")
    model = QueueTableModel(service)
    item = _item()
    assert model.apply_snapshot(_snapshot((item,), revision=1))
    assert model.columnCount() == 7
    assert model.rowCount() == 1
    assert model.headerData(int(QueueColumn.INPUT), Qt.Orientation.Horizontal) == "Input"
    index = model.index(0, 0)
    flags = model.flags(index)
    assert flags & Qt.ItemFlag.ItemIsEnabled
    assert flags & Qt.ItemFlag.ItemIsSelectable
    assert not (flags & Qt.ItemFlag.ItemIsEditable)
    assert model.data(model.index(-1, 0)) is None
    assert model.rowCount(model.index(0, 0)) == 0
    assert model.columnCount(model.index(0, 0)) == 0


def test_display_mappings_for_all_states_profiles_and_stages() -> None:
    service = I18nService("en")
    model = QueueTableModel(service)
    items: list[JobQueueItem] = []
    for index, state in enumerate(JobState):
        items.append(
            _item(
                job_id=f"job-state-{index}",
                job_order=index,
                state=state,
                active_stage=StageName.TRANSCRIBE if state is JobState.RUNNING else None,
                active_stage_state=StageState.RUNNING if state is JobState.RUNNING else None,
                active_stage_attempt=2 if state is JobState.RUNNING else 0,
            )
        )
    for index, profile in enumerate(PipelineProfile):
        items.append(
            _item(
                job_id=f"job-profile-{index}",
                job_order=100 + index,
                profile=profile,
            )
        )
    for index, stage in enumerate(StageName):
        items.append(
            _item(
                job_id=f"job-stage-{index}",
                job_order=200 + index,
                state=JobState.RUNNING,
                active_stage=stage,
                active_stage_state=StageState.RUNNING,
                active_stage_attempt=1,
            )
        )
    cancelling = _item(
        job_id="job-cancelling",
        job_order=300,
        state=JobState.RUNNING,
        cancel_requested=True,
        active_stage=StageName.TRANSCRIBE,
        active_stage_state=StageState.RUNNING,
        active_stage_attempt=1,
    )
    items.append(cancelling)
    assert model.apply_snapshot(_snapshot(tuple(items), revision=1))

    assert model.data(model.index(0, int(QueueColumn.INPUT))) == "sample.wav"
    assert model.data(model.index(0, int(QueueColumn.INPUT)), int(Qt.ItemDataRole.ToolTipRole)) == (
        "/media/clips/sample.wav"
    )
    assert model.data(model.index(0, int(QueueColumn.OUTPUT))) == "/tmp/out"
    assert model.data(model.index(0, int(QueueColumn.BATCH))) == "batch-a"
    assert model.data(model.index(0, int(QueueColumn.ATTEMPT))) == "—"

    def _require(row: int) -> JobQueueItem:
        item = model.item_at(row)
        assert item is not None
        return item

    running_row = next(
        row
        for row in range(model.rowCount())
        if (
            (item := model.item_at(row)) is not None
            and item.state is JobState.RUNNING
            and not item.cancel_requested
            and item.job_id.startswith("job-state")
        )
    )
    assert model.data(model.index(running_row, int(QueueColumn.ATTEMPT))) == "2"
    assert model.data(
        model.index(running_row, int(QueueColumn.ATTEMPT)),
        int(Qt.ItemDataRole.TextAlignmentRole),
    ) == int(Qt.AlignmentFlag.AlignCenter)

    expected_states = {
        JobState.PENDING: "Pending",
        JobState.RUNNING: "Running",
        JobState.INTERRUPTED: "Interrupted",
        JobState.FAILED: "Failed",
        JobState.CANCELLED: "Cancelled",
        JobState.SUCCEEDED: "Succeeded",
    }
    for state, label in expected_states.items():
        row = next(
            r
            for r in range(model.rowCount())
            if _require(r).job_id.startswith("job-state") and _require(r).state is state
        )
        assert model.data(model.index(row, int(QueueColumn.STATUS))) == label

    cancel_row = model.row_for_key(("batch-a", "job-cancelling"))
    assert cancel_row is not None
    assert model.data(model.index(cancel_row, int(QueueColumn.STATUS))) == "Cancelling…"

    for profile in PipelineProfile:
        row = next(
            r
            for r in range(model.rowCount())
            if _require(r).pipeline_profile is profile
            and _require(r).job_id.startswith("job-profile")
        )
        assert model.data(model.index(row, int(QueueColumn.PROFILE)))

    for stage in StageName:
        row = next(
            r
            for r in range(model.rowCount())
            if _require(r).active_stage is stage and _require(r).job_id.startswith("job-stage")
        )
        assert model.data(model.index(row, int(QueueColumn.STAGE)))

    pending_row = next(
        r
        for r in range(model.rowCount())
        if _require(r).state is JobState.PENDING and _require(r).job_id.startswith("job-state")
    )
    assert model.data(model.index(pending_row, int(QueueColumn.STAGE))) == "—"


def test_english_and_chinese_labels() -> None:
    en = QueueTableModel(I18nService("en"))
    zh = QueueTableModel(I18nService("zh-CN"))
    item = _item(
        state=JobState.RUNNING,
        profile=PipelineProfile.QUALITY,
        active_stage=StageName.TRANSCRIBE,
        active_stage_state=StageState.RUNNING,
        active_stage_attempt=1,
    )
    assert en.apply_snapshot(_snapshot((item,), revision=1))
    assert zh.apply_snapshot(_snapshot((item,), revision=1))
    assert en.headerData(int(QueueColumn.STATUS), Qt.Orientation.Horizontal) == "Status"
    assert zh.headerData(int(QueueColumn.STATUS), Qt.Orientation.Horizontal) == "状态"
    assert en.data(en.index(0, int(QueueColumn.STATUS))) == "Running"
    assert zh.data(zh.index(0, int(QueueColumn.STATUS))) == "运行中"
    assert en.data(en.index(0, int(QueueColumn.STAGE))) == "Transcribing"
    assert zh.data(zh.index(0, int(QueueColumn.STAGE))) == "语音识别"
    assert en.data(en.index(0, int(QueueColumn.PROFILE))) == "Quality"
    assert zh.data(zh.index(0, int(QueueColumn.PROFILE))) == "质量"
    assert I18nService("en").translate("gui.queue.title") == "Queue"
    assert I18nService("zh-CN").translate("gui.queue.title") == "队列"


def test_user_role_returns_immutable_item() -> None:
    model = QueueTableModel(I18nService("en"))
    item = _item()
    model.apply_snapshot(_snapshot((item,), revision=1))
    returned = model.data(model.index(0, 0), int(Qt.ItemDataRole.UserRole))
    assert returned is item


def test_stale_revision_rejected() -> None:
    model = QueueTableModel(I18nService("en"))
    item = _item()
    assert model.apply_snapshot(_snapshot((item,), revision=1)) is True
    assert model.apply_snapshot(_snapshot((item,), revision=1)) is False
    assert model.apply_snapshot(_snapshot((item,), revision=2)) is True
    assert model.apply_snapshot(_snapshot((item,), revision=2)) is False
    assert model.revision == 2


def test_state_only_update_emits_data_changed() -> None:
    model = QueueTableModel(I18nService("en"))
    first = _item(job_id="job-1", state=JobState.PENDING)
    second = _item(job_id="job-2", job_order=1, state=JobState.PENDING)
    model.apply_snapshot(_snapshot((first, second), revision=1))

    reset_spy = QSignalSpy(model.modelReset)
    changed_spy = QSignalSpy(model.dataChanged)
    updated_first = replace(
        first,
        state=JobState.RUNNING,
        active_stage=StageName.TRANSCRIBE,
        active_stage_state=StageState.RUNNING,
        active_stage_attempt=1,
    )
    assert model.apply_snapshot(_snapshot((updated_first, second), revision=2))
    assert reset_spy.count() == 0
    assert changed_spy.count() >= 1
    top_left = changed_spy.at(0)[0]
    bottom_right = changed_spy.at(0)[1]
    assert top_left.row() == 0
    assert bottom_right.row() == 0
    assert model.data(model.index(0, int(QueueColumn.STATUS))) == "Running"
    assert model.data(model.index(1, int(QueueColumn.STATUS))) == "Pending"


def test_structural_update_resets_model() -> None:
    model = QueueTableModel(I18nService("en"))
    first = _item(job_id="job-1")
    model.apply_snapshot(_snapshot((first,), revision=1))
    about_spy = QSignalSpy(model.modelAboutToBeReset)
    reset_spy = QSignalSpy(model.modelReset)
    second = _item(job_id="job-2")
    assert model.apply_snapshot(_snapshot((first, second), revision=2))
    assert about_spy.count() == 1
    assert reset_spy.count() == 1
    assert model.rowCount() == 2


def test_metadata_only_revision() -> None:
    model = QueueTableModel(I18nService("en"))
    item = _item()
    model.apply_snapshot(_snapshot((item,), revision=1))
    reset_spy = QSignalSpy(model.modelReset)
    changed_spy = QSignalSpy(model.dataChanged)
    next_snapshot = _snapshot(
        (item,),
        revision=2,
        issues=(QueueLoadIssue("broken-batch", "catalog.batch_corrupt"),),
        omitted=3,
    )
    assert model.apply_snapshot(next_snapshot)
    assert model.revision == 2
    assert model.snapshot is next_snapshot
    assert reset_spy.count() == 0
    assert changed_spy.count() == 0


def test_row_helpers() -> None:
    model = QueueTableModel(I18nService("en"))
    item = _item(batch_id="b1", job_id="j1")
    model.apply_snapshot(_snapshot((item,), revision=1))
    assert model.item_at(0) is item
    assert model.item_at(5) is None
    assert model.key_at(0) == ("b1", "j1")
    assert model.key_at(-1) is None
    assert model.row_for_key(("b1", "j1")) == 0
    assert model.row_for_key(("missing", "j1")) is None
