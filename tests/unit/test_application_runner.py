"""Unit tests for the dedicated Application runner bridge."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from captioner.core.application.queue_projection import QueueSnapshot
from captioner.core.domain.errors import AppError
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure

_app = QApplication.instance() or QApplication(["test-application-runner"])


def _empty_snapshot(revision: int = 1) -> QueueSnapshot:
    return QueueSnapshot(2, revision, (), (), 0)


@dataclass
class FakeBoundary:
    main_thread_id: int
    factory_thread_ids: list[int]
    refresh_thread_ids: list[int]
    get_calls: list[int]
    refresh_calls: list[int]
    snapshots: list[QueueSnapshot]
    get_error: BaseException | None = None
    refresh_error: BaseException | None = None
    block_get: threading.Event | None = None
    release_get: threading.Event | None = None
    operation_thread_ids: list[int] | None = None
    preview_error: BaseException | None = None
    config_error: BaseException | None = None
    provider_test_error: BaseException | None = None

    def get_queue_snapshot(self) -> QueueSnapshot:
        self.get_calls.append(threading.get_ident())
        if self.block_get is not None:
            self.block_get.set()
        if self.release_get is not None and not self.release_get.wait(timeout=5):
            raise RuntimeError("release_timeout")
        if self.get_error is not None:
            raise self.get_error
        snapshot = self.snapshots[len(self.get_calls) - 1] if self.snapshots else _empty_snapshot()
        return snapshot

    def refresh_queue(self) -> QueueSnapshot:
        self.refresh_calls.append(threading.get_ident())
        self.refresh_thread_ids.append(threading.get_ident())
        if self.refresh_error is not None:
            raise self.refresh_error
        index = len(self.get_calls) + len(self.refresh_calls) - 1
        if index < len(self.snapshots):
            return self.snapshots[index]
        return _empty_snapshot(revision=max(1, index + 1))

    def subscribe_queue(self, callback: Callable[..., Any]) -> Callable[[], None]:
        def unsubscribe() -> None:
            return None

        return unsubscribe

    def _record_op(self) -> None:
        if self.operation_thread_ids is not None:
            self.operation_thread_ids.append(threading.get_ident())

    def preview_inputs(self, request: object) -> Any:
        self._record_op()
        if self.preview_error is not None:
            raise self.preview_error
        from captioner.core.application.input_selection import InputPreview

        return InputPreview(accepted_paths=(), rejected=())

    def load_configuration(self) -> Any:
        self._record_op()
        if self.config_error is not None:
            raise self.config_error
        from captioner.core.application.configuration import default_configuration_snapshot

        return default_configuration_snapshot()

    def save_global_settings(self, settings: object) -> Any:
        return self.load_configuration()

    def save_provider_settings(self, update: object) -> Any:
        self._record_op()
        if self.config_error is not None:
            raise self.config_error
        from captioner.core.application.configuration import default_configuration_snapshot

        return default_configuration_snapshot()

    def save_user_preset(self, preset: object) -> Any:
        return self.load_configuration()

    def delete_user_preset(self, name: str) -> Any:
        return self.load_configuration()

    def test_provider_connection(self, update: object) -> Any:
        self._record_op()
        if self.provider_test_error is not None:
            raise self.provider_test_error
        from captioner.core.application.configuration import ProviderConnectionResult

        return ProviderConnectionResult(True, "llm.connection_ok")

    def submit_batch(self, request: object) -> Any:
        self._record_op()
        from captioner.core.application.batch_commands import (
            BatchCommandAck,
            BatchCommandKind,
        )

        return BatchCommandAck(
            request_id=getattr(request, "request_id", "req"),
            kind=BatchCommandKind.SUBMIT,
            batch_id="batch-a",
            job_id=None,
            accepted_at_utc="t0",
            scheduled=True,
            created_batch_id="batch-a",
        )

    def perform_batch_action(self, request: object) -> Any:
        self._record_op()
        from captioner.core.application.batch_commands import (
            BatchCommandAck,
            BatchCommandKind,
        )

        return BatchCommandAck(
            request_id=getattr(request, "request_id", "req"),
            kind=getattr(request, "kind", BatchCommandKind.PAUSE_BATCH),
            batch_id=getattr(request, "batch_id", "batch-a"),
            job_id=None,
            accepted_at_utc="t0",
            scheduled=False,
        )

    def perform_job_action(self, request: object) -> Any:
        self._record_op()
        from captioner.core.application.batch_commands import (
            BatchCommandAck,
            BatchCommandKind,
        )

        return BatchCommandAck(
            request_id=getattr(request, "request_id", "req"),
            kind=getattr(request, "kind", BatchCommandKind.CANCEL_JOB),
            batch_id=getattr(request, "batch_id", "batch-a"),
            job_id=getattr(request, "job_id", "job-000001"),
            accepted_at_utc="t0",
            scheduled=False,
        )

    def cancel_local_work(self, request: object) -> Any:
        self._record_op()
        from captioner.core.application.batch_commands import (
            BatchCommandAck,
            BatchCommandKind,
        )

        return BatchCommandAck(
            request_id=getattr(request, "request_id", "req"),
            kind=BatchCommandKind.CANCEL_LOCAL_WORK,
            batch_id=None,
            job_id=None,
            accepted_at_utc="t0",
            scheduled=False,
        )

    def load_job_detail(self, request: object) -> Any:
        self._record_op()
        raise AppError("batch.not_found")

    def scan_recovery(self, request: object) -> Any:
        self._record_op()
        from captioner.core.application.recovery import RecoverySnapshot

        return RecoverySnapshot(
            schema_version=1,
            request_id=getattr(request, "request_id", "req"),
            items=(),
            issues=(),
        )

    def poll_execution(self) -> Any:
        from captioner.core.application.batch_commands import LocalExecutionSnapshot
        from captioner.gui.application_boundary import ExecutionPoll

        return ExecutionPoll(
            state=LocalExecutionSnapshot(None, ()),
            completions=(),
        )

    def shutdown(self) -> None:
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


def test_factory_and_refresh_run_on_worker_thread() -> None:
    main_id = threading.get_ident()
    factory_ids: list[int] = []
    boundary_holder: list[FakeBoundary] = []

    def factory() -> FakeBoundary:
        factory_ids.append(threading.get_ident())
        boundary = FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=factory_ids,
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1), _empty_snapshot(2)],
        )
        boundary_holder.append(boundary)
        return boundary

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    snapshots: list[object] = []
    bridge.snapshot_ready.connect(snapshots.append)
    try:
        bridge.start()
        assert _wait_until(lambda: len(snapshots) >= 1)
        bridge.request_refresh()
        assert _wait_until(lambda: len(snapshots) >= 2)
        assert factory_ids
        assert factory_ids[0] != main_id
        boundary = boundary_holder[0]
        assert boundary.get_calls
        assert boundary.get_calls[0] != main_id
        assert boundary.refresh_calls
        assert boundary.refresh_calls[0] != main_id
    finally:
        assert bridge.stop()


def test_initial_snapshot_and_started() -> None:
    constructed: list[int] = []

    def factory() -> FakeBoundary:
        constructed.append(1)
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    started_spy = QSignalSpy(bridge.started)
    snapshots: list[QueueSnapshot] = []
    bridge.snapshot_ready.connect(snapshots.append)
    try:
        bridge.start()
        bridge.start()
        assert started_spy.count() == 1
        assert _wait_until(lambda: len(snapshots) >= 1)
        assert constructed == [1]
        assert snapshots[0].revision == 1
    finally:
        assert bridge.stop()


def test_manual_refresh_emits_new_snapshot() -> None:
    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1), _empty_snapshot(2)],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    snapshots: list[QueueSnapshot] = []
    bridge.snapshot_ready.connect(snapshots.append)
    try:
        bridge.start()
        assert _wait_until(lambda: len(snapshots) >= 1)
        bridge.request_refresh()
        assert _wait_until(lambda: len(snapshots) >= 2)
        assert snapshots[1].revision == 2
    finally:
        assert bridge.stop()


def test_structured_app_error() -> None:
    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[],
            get_error=AppError("queue.test_failure", retryable=True),
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    failures: list[RunnerFailure] = []
    bridge.failure.connect(failures.append)
    try:
        bridge.start()
        assert _wait_until(lambda: len(failures) >= 1)
        failure = failures[0]
        assert isinstance(failure, RunnerFailure)
        assert failure.code == "queue.test_failure"
        assert failure.retryable is True
        assert "secret" not in repr(failure)
    finally:
        assert bridge.stop()


def test_unexpected_exception_is_sanitized() -> None:
    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[],
            get_error=RuntimeError("secret raw text"),
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    failures: list[RunnerFailure] = []
    bridge.failure.connect(failures.append)
    try:
        bridge.start()
        assert _wait_until(lambda: len(failures) >= 1)
        failure = failures[0]
        assert isinstance(failure, RunnerFailure)
        assert failure.code == "gui.application_bridge_failed"
        assert "secret raw text" not in failure.code
        assert "secret raw text" not in repr(failure)
    finally:
        assert bridge.stop()


def test_ui_thread_remains_responsive_while_worker_blocks() -> None:
    entered = threading.Event()
    release = threading.Event()

    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            block_get=entered,
            release_get=release,
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    loop = QEventLoop()
    main_timer_fired: list[bool] = []
    snapshots: list[object] = []
    bridge.snapshot_ready.connect(snapshots.append)
    bridge.snapshot_ready.connect(loop.quit)
    try:
        bridge.start()
        assert entered.wait(timeout=2)
        QTimer.singleShot(0, lambda: main_timer_fired.append(True))
        QTimer.singleShot(25, release.set)
        QTimer.singleShot(2000, loop.quit)
        loop.exec()
        assert main_timer_fired == [True]
        assert len(snapshots) == 1
    finally:
        release.set()
        assert bridge.stop()


def test_stop_is_idempotent_and_clears_running_thread() -> None:
    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    stopped_spy = QSignalSpy(bridge.stopped)
    snapshots: list[object] = []
    bridge.snapshot_ready.connect(snapshots.append)
    try:
        bridge.start()
        assert _wait_until(lambda: len(snapshots) >= 1)
        assert bridge.stop()
        assert bridge.stop()
        assert not bridge.running
        assert stopped_spy.count() == 1
    finally:
        bridge.stop()


def test_input_config_and_provider_ops_run_on_worker_thread() -> None:
    from captioner.core.application.configuration import (
        ExecutionPreset,
        GlobalSettings,
        ProviderSettingsUpdate,
    )
    from captioner.core.application.input_selection import InputSelectionRequest
    from captioner.core.domain.stage import PipelineProfile

    main_id = threading.get_ident()
    op_ids: list[int] = []

    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            operation_thread_ids=op_ids,
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    previews: list[object] = []
    configs: list[object] = []
    tests: list[object] = []
    queue_failures: list[object] = []
    config_failures: list[object] = []
    bridge.input_preview_ready.connect(previews.append)
    bridge.configuration_loaded.connect(configs.append)
    bridge.global_settings_saved.connect(configs.append)
    bridge.provider_settings_saved.connect(configs.append)
    bridge.preset_saved.connect(configs.append)
    bridge.preset_deleted.connect(configs.append)
    bridge.provider_test_ready.connect(tests.append)
    bridge.failure.connect(queue_failures.append)
    bridge.configuration_load_failure.connect(config_failures.append)
    try:
        bridge.start()
        assert _wait_until(lambda: True)
        bridge.request_input_preview(InputSelectionRequest(entries=("/a.wav",)))
        bridge.request_configuration_load()
        bridge.request_global_save(GlobalSettings())
        bridge.request_provider_save(
            ProviderSettingsUpdate(
                profile_name="default",
                base_url="https://example.com/v1",
                model="m",
                api_key="secret-key",
            )
        )
        bridge.request_preset_save(
            ExecutionPreset(
                name="custom",
                display_name="Custom",
                built_in=False,
                pipeline_profile=PipelineProfile.FAST,
                model_ref="tiny",
                device="auto",
                compute_type="default",
                source_language=None,
                target_language="zh-CN",
                provider_profile="default",
            )
        )
        bridge.request_preset_delete("custom")
        bridge.request_provider_test(
            ProviderSettingsUpdate(
                profile_name="default",
                base_url="https://example.com/v1",
                model="m",
                api_key="secret-key",
            )
        )
        assert _wait_until(lambda: len(previews) >= 1 and len(configs) >= 1 and len(tests) >= 1)
        assert op_ids
        assert all(thread_id != main_id for thread_id in op_ids)
        assert queue_failures == []
        assert "secret-key" not in repr(previews)
        assert "secret-key" not in repr(configs)
        assert "secret-key" not in repr(tests)
        # Exactly one QThread remains (the bridge worker).
        assert bridge.running is True
    finally:
        assert bridge.stop()


def test_configuration_failure_is_operation_specific() -> None:
    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=threading.get_ident(),
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            config_error=AppError("config.write_failed"),
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    queue_failures: list[object] = []
    config_failures: list[object] = []
    bridge.failure.connect(queue_failures.append)
    bridge.configuration_load_failure.connect(config_failures.append)
    try:
        bridge.start()
        bridge.request_configuration_load()
        assert _wait_until(lambda: len(config_failures) >= 1)
        assert queue_failures == []
        failure = config_failures[0]
        assert isinstance(failure, RunnerFailure)
        assert failure.code == "config.write_failed"
    finally:
        assert bridge.stop()


def test_submit_and_execution_poll_and_shutdown() -> None:
    main_id = threading.get_ident()
    factory_ids: list[int] = []

    def factory() -> FakeBoundary:
        factory_ids.append(threading.get_ident())
        return FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=factory_ids,
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            operation_thread_ids=[],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    ready = QSignalSpy(bridge.snapshot_ready)
    command = QSignalSpy(bridge.batch_command_ready)
    bridge.start()
    assert _wait_until(lambda: ready.count() >= 1)
    from captioner.core.application.batch_commands import (
        SubmitBatchRequest,
    )
    from captioner.core.application.input_selection import BatchDraft
    from captioner.core.domain.stage import PipelineProfile

    draft = BatchDraft(
        input_paths=("/tmp/a.wav",),
        output_root="/tmp/out",
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language="en",
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    bridge.request_submit_batch(SubmitBatchRequest(request_id="req-submit", draft=draft))
    assert _wait_until(lambda: command.count() >= 1)
    assert bridge.stop(timeout_ms=3000)


def test_all_request_paths_and_failures() -> None:
    main_id = threading.get_ident()
    op_ids: list[int] = []

    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            operation_thread_ids=op_ids,
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    ready = QSignalSpy(bridge.snapshot_ready)
    command = QSignalSpy(bridge.batch_command_ready)
    recovery = QSignalSpy(bridge.recovery_ready)
    detail_fail = QSignalSpy(bridge.job_detail_failure)
    bridge.start()
    assert _wait_until(lambda: ready.count() >= 1)

    from captioner.core.application.batch_commands import (
        BatchActionRequest,
        BatchCommandKind,
        CancelLocalWorkRequest,
        JobActionRequest,
        SubmitBatchRequest,
    )
    from captioner.core.application.configuration import (
        ExecutionPreset,
        GlobalSettings,
        ProviderSettingsUpdate,
    )
    from captioner.core.application.input_selection import BatchDraft, InputSelectionRequest
    from captioner.core.application.job_detail import JobDetailRequest
    from captioner.core.application.recovery import RecoveryRequest
    from captioner.core.domain.stage import PipelineProfile

    bridge.request_refresh()
    bridge.request_input_preview(InputSelectionRequest(entries=("/a.wav",)))
    bridge.request_configuration_load()
    bridge.request_global_save(
        GlobalSettings(
            locale="en",
            default_preset_name="deterministic",
            default_output_root="/tmp",
            recursive_input=True,
            collision_policy="unique_subdir",
        )
    )
    bridge.request_provider_save(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com",
            model="m",
            api_key=None,
            max_concurrency=1,
            request_timeout_sec=30.0,
            max_retries=1,
            temperature=0.0,
            tokenizer="cl100k_base",
        )
    )
    bridge.request_preset_save(
        ExecutionPreset(
            name="user-a",
            display_name="User A",
            built_in=False,
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            source_language=None,
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    bridge.request_preset_delete("user-a")
    bridge.request_provider_test(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com",
            model="m",
            api_key=None,
            max_concurrency=1,
            request_timeout_sec=30.0,
            max_retries=1,
            temperature=0.0,
            tokenizer="cl100k_base",
        )
    )
    draft = BatchDraft(
        input_paths=("/tmp/a.wav",),
        output_root="/tmp/out",
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language="en",
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    bridge.request_submit_batch(SubmitBatchRequest(request_id="req-s", draft=draft))
    bridge.request_batch_action(
        BatchActionRequest(
            request_id="req-b",
            kind=BatchCommandKind.PAUSE_BATCH,
            batch_id="batch-a",
        )
    )
    bridge.request_job_action(
        JobActionRequest(
            request_id="req-j",
            kind=BatchCommandKind.CANCEL_JOB,
            batch_id="batch-a",
            job_id="job-000001",
        )
    )
    bridge.request_cancel_local_work(CancelLocalWorkRequest(request_id="req-c"))
    bridge.request_job_detail(
        JobDetailRequest(request_id="req-d", batch_id="batch-a", job_id="job-000001")
    )
    bridge.request_recovery_scan(RecoveryRequest(request_id="req-r"))
    assert _wait_until(lambda: command.count() >= 1)
    assert _wait_until(lambda: recovery.count() >= 1)
    assert _wait_until(lambda: detail_fail.count() >= 1)
    assert bridge.stop(timeout_ms=5000)


def test_failed_shutdown_preserves_boundary_and_allows_retry() -> None:
    """Active serial work: first stop fails, boundary stays usable, second stop succeeds."""
    from captioner.core.application.batch_commands import (
        CancelLocalWorkRequest,
        LocalExecutionSnapshot,
    )
    from captioner.gui.application_boundary import ExecutionPoll

    main_id = threading.get_ident()
    cancel_calls: list[int] = []
    poll_states: list[LocalExecutionSnapshot] = []
    failures: list[RunnerFailure] = []
    stopped_count = {"n": 0}
    active = {"has_work": True}
    shutdown_attempts = {"n": 0}

    class BusyBoundary(FakeBoundary):
        def poll_execution(self) -> Any:
            state = LocalExecutionSnapshot(
                "batch-a" if active["has_work"] else None,
                (),
            )
            return ExecutionPoll(state=state, completions=())

        def cancel_local_work(self, request: object) -> Any:
            cancel_calls.append(threading.get_ident())
            self._record_op()
            active["has_work"] = False
            from captioner.core.application.batch_commands import (
                BatchCommandAck,
                BatchCommandKind,
            )

            return BatchCommandAck(
                request_id=getattr(request, "request_id", "req"),
                kind=BatchCommandKind.CANCEL_LOCAL_WORK,
                batch_id=None,
                job_id=None,
                accepted_at_utc="t0",
                scheduled=False,
            )

        def shutdown(self) -> None:
            shutdown_attempts["n"] += 1
            if active["has_work"]:
                # Simulate coordinator rejecting shutdown while work is active.
                raise AppError("batch.execution_active")
            return None

    def factory() -> BusyBoundary:
        return BusyBoundary(
            main_thread_id=main_id,
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1), _empty_snapshot(2)],
            operation_thread_ids=[],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    bridge.failure.connect(failures.append)
    bridge.local_execution_state_changed.connect(poll_states.append)
    bridge.stopped.connect(lambda: stopped_count.__setitem__("n", stopped_count["n"] + 1))
    ready = QSignalSpy(bridge.snapshot_ready)
    cancel_ready = QSignalSpy(bridge.batch_command_ready)
    try:
        bridge.start()
        assert _wait_until(lambda: ready.count() >= 1)
        # First stop while "work" active → fails, boundary retained.
        assert bridge.stop(timeout_ms=800) is False
        assert stopped_count["n"] == 0
        assert _wait_until(lambda: shutdown_attempts["n"] >= 1)
        assert _wait_until(lambda: any(f.code == "batch.execution_active" for f in failures))
        for failure in failures:
            assert "Traceback" not in failure.code
            assert "secret" not in repr(failure)
        # Refresh still works after failed stop (same Worker thread).
        bridge.request_refresh()
        assert _wait_until(lambda: ready.count() >= 2)
        # Cancel still processed on worker thread.
        bridge.request_cancel_local_work(CancelLocalWorkRequest(request_id="req-cancel-all"))
        assert _wait_until(lambda: cancel_ready.count() >= 1)
        assert cancel_calls
        assert cancel_calls[0] != main_id
        assert active["has_work"] is False
        # Polling should eventually emit idle after cancel.
        assert _wait_until(
            lambda: any(not state.has_work for state in poll_states),
            timeout_ms=3000,
        )
        # Second stop succeeds once idle; no second Worker thread.
        thread_before = bridge._thread  # pyright: ignore[reportPrivateUsage]
        assert bridge.stop(timeout_ms=5000) is True
        assert stopped_count["n"] == 1
        assert bridge._thread is None  # pyright: ignore[reportPrivateUsage]
        assert thread_before is not None
        assert shutdown_attempts["n"] >= 2
    finally:
        if bridge.running:
            active["has_work"] = False
            bridge.stop(timeout_ms=5000)


def test_malformed_request_emits_safe_failure_on_worker() -> None:
    """Malformed request objects emit stable codes without raw exception text."""
    main_id = threading.get_ident()

    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            operation_thread_ids=[],
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    failures: list[object] = []
    bridge.batch_command_failure.connect(failures.append)
    ready = QSignalSpy(bridge.snapshot_ready)
    try:
        bridge.start()
        assert _wait_until(lambda: ready.count() >= 1)
        # Emit malformed payload through the same queued path as real callers.
        bridge._submit_batch_requested.emit("not-a-request")  # pyright: ignore[reportPrivateUsage]
        assert _wait_until(lambda: len(failures) >= 1)
        failure = failures[0]
        code = getattr(failure, "code", "")
        assert code == "gui.application_bridge_failed"
        assert "Traceback" not in repr(failure)
        assert "Exception" not in repr(failure)
    finally:
        assert bridge.stop(timeout_ms=5000)


def test_worker_command_thread_differs_from_main() -> None:
    """Worker slots run on the dedicated QThread, not the main GUI thread."""
    main_id = threading.get_ident()
    op_ids: list[int] = []

    def factory() -> FakeBoundary:
        return FakeBoundary(
            main_thread_id=main_id,
            factory_thread_ids=[],
            refresh_thread_ids=[],
            get_calls=[],
            refresh_calls=[],
            snapshots=[_empty_snapshot(1)],
            operation_thread_ids=op_ids,
        )

    bridge = ApplicationRunnerBridge(factory)  # type: ignore[arg-type]
    ready = QSignalSpy(bridge.batch_command_ready)
    snap = QSignalSpy(bridge.snapshot_ready)
    try:
        bridge.start()
        assert _wait_until(lambda: snap.count() >= 1)
        from captioner.core.application.batch_commands import CancelLocalWorkRequest

        bridge.request_cancel_local_work(CancelLocalWorkRequest(request_id="req-thread"))
        assert _wait_until(lambda: ready.count() >= 1)
        assert op_ids
        assert all(tid != main_id for tid in op_ids)
    finally:
        assert bridge.stop(timeout_ms=5000)
