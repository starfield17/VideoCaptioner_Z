"""Unit tests for RecoveryController."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from captioner.core.application.recovery import RecoveryItem, RecoveryRequest, RecoverySnapshot
from captioner.core.domain.batch import BatchState
from captioner.gui.recovery_controller import RecoveryController

_app = QApplication.instance() or QApplication(["test-recovery-controller"])


class FakeRunner(QObject):
    recovery_ready = Signal(object)
    recovery_failure = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.scans: list[RecoveryRequest] = []

    def request_recovery_scan(self, request: object) -> None:
        assert isinstance(request, RecoveryRequest)
        self.scans.append(request)


def test_prompt_once_per_batch() -> None:
    runner = FakeRunner()
    controller = RecoveryController(runner)  # type: ignore[arg-type]
    prompts: list[object] = []
    controller.prompt_requested.connect(prompts.append)
    controller.scan()
    assert len(runner.scans) == 1
    item = RecoveryItem(
        batch_id="batch-a",
        created_at_utc="t0",
        state=BatchState.PENDING,
        job_count=1,
        pause_requested=False,
        missing_input_paths=(),
        last_event_seq=1,
        blocked_code=None,
    )
    runner.recovery_ready.emit(RecoverySnapshot(1, runner.scans[0].request_id, (item,), ()))
    assert len(prompts) == 1
    controller.scan()
    runner.recovery_ready.emit(RecoverySnapshot(1, runner.scans[1].request_id, (item,), ()))
    assert len(prompts) == 1


def test_failure_and_coalesced_scan() -> None:
    from captioner.gui.application_runner import RunnerFailure

    runner = FakeRunner()
    controller = RecoveryController(runner)  # type: ignore[arg-type]
    controller.scan()
    controller.scan()  # coalesced while in flight
    assert len(runner.scans) == 1
    # completing first also dispatches queued second scan
    runner.recovery_ready.emit(RecoverySnapshot(1, runner.scans[0].request_id, (), ()))
    assert len(runner.scans) == 2
    runner.recovery_ready.emit(RecoverySnapshot(1, runner.scans[1].request_id, (), ()))
    assert controller.busy is False
    controller.scan()
    runner.recovery_failure.emit(RunnerFailure(code="recovery.failed", retryable=False))
    assert controller.busy is False
