"""Unit tests for BatchController Queue presentation coordination."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from captioner.core.application.queue_projection import JobQueueItem, QueueSnapshot
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.batch_controller import BatchController
from captioner.gui.queue_table_model import QueueTableModel
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-batch-controller"])


def _item(
    job_id: str = "job-1",
    *,
    state: JobState = JobState.PENDING,
    batch_id: str = "batch-a",
) -> JobQueueItem:
    return JobQueueItem(
        batch_id=batch_id,
        job_id=job_id,
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=0,
        input_path="/media/a.wav",
        output_dir="/tmp/out",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        state=state,
        active_stage=None,
        active_stage_state=None,
        active_stage_attempt=0,
        cancel_requested=False,
        last_event_seq=1,
        journal_tail_status="clean",
        manifest_status="missing",
    )


def _snapshot(
    items: tuple[JobQueueItem, ...] = (),
    *,
    revision: int = 1,
) -> QueueSnapshot:
    return QueueSnapshot(1, revision, items, (), 0)


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.refresh_calls = 0
        self.stop_calls = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.start_calls += 1
        self._running = True
        self.started.emit()

    def request_refresh(self) -> None:
        self.refresh_calls += 1

    def stop(self, timeout_ms: int = 5000) -> bool:
        del timeout_ms
        self.stop_calls += 1
        self._running = False
        self.stopped.emit()
        return True


def _controller() -> tuple[BatchController, QueueTableModel, FakeRunner]:
    model = QueueTableModel(I18nService("en"))
    runner = FakeRunner()
    controller = BatchController(model, runner, refresh_interval_ms=1000)  # type: ignore[arg-type]
    return controller, model, runner


def test_start_is_idempotent_and_marks_busy() -> None:
    controller, _model, runner = _controller()
    running_spy = QSignalSpy(controller.running_changed)
    busy_spy = QSignalSpy(controller.busy_changed)
    controller.start()
    controller.start()
    assert runner.start_calls == 1
    assert controller.running is True
    assert controller.busy is True
    assert running_spy.count() == 1
    assert busy_spy.count() == 1
    controller.stop()


def test_initial_snapshot_updates_model_and_clears_busy() -> None:
    controller, model, runner = _controller()
    snapshot_spy = QSignalSpy(controller.snapshot_changed)
    busy_spy = QSignalSpy(controller.busy_changed)
    controller.start()
    snapshot = _snapshot((_item(),), revision=1)
    runner.snapshot_ready.emit(snapshot)
    assert model.rowCount() == 1
    assert controller.current_snapshot is snapshot
    assert controller.busy is False
    assert snapshot_spy.count() == 1
    assert controller.last_failure is None
    assert busy_spy.count() >= 2
    controller.stop()


def test_manual_refresh_when_idle() -> None:
    controller, _model, runner = _controller()
    controller.start()
    runner.snapshot_ready.emit(_snapshot(revision=1))
    before = runner.refresh_calls
    controller.refresh()
    assert runner.refresh_calls == before + 1
    assert controller.busy is True
    controller.stop()


def test_refresh_coalescing() -> None:
    controller, _model, runner = _controller()
    controller.start()
    # Initial start already marks in-flight without a request_refresh call.
    assert runner.refresh_calls == 0
    controller.refresh()
    controller.refresh()
    controller.refresh()
    assert runner.refresh_calls == 0
    assert controller.busy is True
    runner.snapshot_ready.emit(_snapshot(revision=1))
    assert runner.refresh_calls == 1
    assert controller.busy is True
    runner.snapshot_ready.emit(_snapshot(revision=2))
    assert runner.refresh_calls == 1
    assert controller.busy is False
    controller.stop()


def test_failure_retains_previous_snapshot() -> None:
    controller, model, runner = _controller()
    controller.start()
    snapshot = _snapshot((_item(state=JobState.RUNNING),), revision=1)
    runner.snapshot_ready.emit(snapshot)
    failure_spy = QSignalSpy(controller.failure_changed)
    runner.failure.emit(RunnerFailure(code="queue.test_failure", retryable=True))
    assert model.rowCount() == 1
    assert model.item_at(0) is not None
    assert controller.last_failure is not None
    assert controller.last_failure.code == "queue.test_failure"
    assert controller.busy is False
    assert failure_spy.count() == 1
    controller.stop()


def test_recovery_after_failure() -> None:
    controller, model, runner = _controller()
    controller.start()
    runner.snapshot_ready.emit(_snapshot((_item(),), revision=1))
    runner.failure.emit(RunnerFailure(code="queue.test_failure"))
    assert controller.last_failure is not None
    failure_spy = QSignalSpy(controller.failure_changed)
    newer = _snapshot((_item(state=JobState.SUCCEEDED),), revision=2)
    runner.snapshot_ready.emit(newer)
    assert controller.last_failure is None
    assert failure_spy.count() == 1
    assert failure_spy.at(0)[0] is None
    assert model.data(model.index(0, 2)) == "Succeeded"
    controller.stop()


def test_stale_snapshot_does_not_replace_current() -> None:
    controller, _model, runner = _controller()
    controller.start()
    current = _snapshot((_item(job_id="job-new"),), revision=2)
    runner.snapshot_ready.emit(current)
    stale = _snapshot((_item(job_id="job-old"),), revision=1)
    runner.snapshot_ready.emit(stale)
    assert controller.current_snapshot is current
    item = controller.model.item_at(0)
    assert item is not None
    assert item.job_id == "job-new"
    controller.stop()


def test_stop_is_idempotent() -> None:
    controller, _model, runner = _controller()
    running_spy = QSignalSpy(controller.running_changed)
    busy_spy = QSignalSpy(controller.busy_changed)
    controller.start()
    assert controller.stop()
    assert controller.stop()
    assert runner.stop_calls == 1
    assert controller.running is False
    assert controller.busy is False
    assert running_spy.count() == 2
    assert any(busy_spy.at(i)[0] is False for i in range(busy_spy.count()))
