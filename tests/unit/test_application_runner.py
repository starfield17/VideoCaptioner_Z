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
    return QueueSnapshot(1, revision, (), (), 0)


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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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

    bridge = ApplicationRunnerBridge(factory)
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
