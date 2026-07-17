"""Unit tests for CreateController."""

from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from captioner.core.application.configuration import default_configuration_snapshot
from captioner.core.application.input_selection import InputPreview, InputSelectionRequest
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.create_controller import CreateController

_app = QApplication.instance() or QApplication(["test-create-controller"])


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    input_preview_ready = Signal(object)
    configuration_ready = Signal(object)
    provider_test_ready = Signal(object)
    input_failure = Signal(object)
    configuration_failure = Signal(object)
    provider_test_failure = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.preview_requests: list[InputSelectionRequest] = []
        self._running = True
        self.auto_respond = True
        self.preview_result = InputPreview(accepted_paths=("/a.wav",), rejected=())

    @property
    def running(self) -> bool:
        return self._running

    def request_input_preview(self, request: InputSelectionRequest) -> None:
        self.preview_requests.append(request)
        if self.auto_respond:
            QTimer.singleShot(
                0,
                lambda: self.input_preview_ready.emit(self.preview_result),
            )

    def request_preset_save(self, preset: object) -> None:
        return None

    def request_preset_delete(self, name: str) -> None:
        return None


def _wait_until(predicate: Callable[[], bool], timeout_ms: int = 2000) -> bool:
    if predicate():
        return True
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(timeout_ms)
    loop.exec()
    timer.stop()
    return predicate()


def test_entry_order_duplicates_remove_clear() -> None:
    runner = FakeRunner()
    controller = CreateController(runner)  # type: ignore[arg-type]
    controller.append_entries(("/a.wav", "/b.mp4"))
    controller.append_entries(("/a.wav",))
    assert controller.entries == ("/a.wav", "/b.mp4", "/a.wav")
    controller.remove_entry(0)
    assert controller.entries == ("/b.mp4", "/a.wav")
    controller.clear_entries()
    assert controller.entries == ()
    assert _wait_until(lambda: controller.preview is not None)


def test_recursive_and_preview_coalescing() -> None:
    runner = FakeRunner()
    runner.auto_respond = False
    controller = CreateController(runner)  # type: ignore[arg-type]
    controller.set_entries(("/a.wav",))
    assert len(runner.preview_requests) == 1
    controller.append_entries(("/b.mp4",))
    controller.set_recursive(False)
    assert len(runner.preview_requests) == 1
    # Complete first request (stale) then follow-up.
    runner.input_preview_ready.emit(InputPreview(("/old.wav",), ()))
    assert _wait_until(lambda: len(runner.preview_requests) == 2)
    runner.preview_result = InputPreview(("/a.wav", "/b.mp4"), ())
    runner.input_preview_ready.emit(runner.preview_result)
    assert _wait_until(
        lambda: controller.preview is not None and controller.preview.accepted_count == 2
    )
    assert controller.recursive is False


def test_configuration_and_draft_lifecycle() -> None:
    runner = FakeRunner()
    controller = CreateController(runner)  # type: ignore[arg-type]
    controller.set_entries(("/a.wav",))
    assert _wait_until(lambda: controller.preview is not None)
    snapshot = default_configuration_snapshot()
    controller.set_configuration(snapshot)
    assert controller.configuration is snapshot
    draft = controller.validate_draft(
        output_root="/out",
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="auto",
        compute_type="default",
        source_language=None,
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    assert draft is not None
    assert draft.input_paths == ("/a.wav",)
    controller.append_entries(("/b.mp4",))
    assert controller.draft is None
    failed = controller.validate_draft(
        output_root="",
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="auto",
        compute_type="default",
        source_language=None,
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    assert failed is None
    assert controller.validation_error == "batch.draft_invalid"


def test_failure_does_not_clear_entries() -> None:
    runner = FakeRunner()
    runner.auto_respond = False
    controller = CreateController(runner)  # type: ignore[arg-type]
    controller.set_entries(("/a.wav",))
    runner.input_failure.emit(RunnerFailure(code="input.unreadable"))
    assert _wait_until(lambda: controller.last_failure is not None)
    assert controller.entries == ("/a.wav",)
