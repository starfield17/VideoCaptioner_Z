"""Unit tests for SerialExecutionCoordinator."""

from __future__ import annotations

import threading
import time

import pytest

from captioner.core.application.batch_commands import BatchCommandKind
from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
from captioner.core.domain.errors import AppError


def test_executor_is_lazy() -> None:
    coordinator = SerialExecutionCoordinator()
    assert coordinator.snapshot().active_batch_id is None
    assert not coordinator.snapshot().has_work


def test_serial_execution_and_order() -> None:
    coordinator = SerialExecutionCoordinator()
    order: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def first() -> None:
        order.append("a-start")
        started.set()
        release.wait(timeout=2)
        order.append("a-end")

    def second() -> None:
        order.append("b")

    coordinator.schedule(
        batch_id="batch-a",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=first,
    )
    coordinator.schedule(
        batch_id="batch-b",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=second,
    )
    assert started.wait(timeout=2)
    snap = coordinator.snapshot()
    assert snap.active_batch_id == "batch-a"
    assert snap.queued_batch_ids == ("batch-b",)
    release.set()
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    assert order == ["a-start", "a-end", "b"]
    completions = coordinator.drain_completions()
    assert len(completions) == 2
    assert all(item.ok and item.code == "execution.completed" for item in completions)
    assert coordinator.drain_completions() == ()
    coordinator.shutdown()


def test_duplicate_batch_rejected() -> None:
    coordinator = SerialExecutionCoordinator()
    gate = threading.Event()

    def block() -> None:
        gate.wait(timeout=2)

    coordinator.schedule(
        batch_id="batch-a",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=block,
    )
    with pytest.raises(AppError, match=r"batch\.operation_conflict"):
        coordinator.schedule(
            batch_id="batch-a",
            kind=BatchCommandKind.RESUME_BATCH,
            job_id=None,
            operation=lambda: None,
        )
    gate.set()
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    coordinator.drain_completions()
    coordinator.shutdown()


def test_cancel_queued_not_active() -> None:
    coordinator = SerialExecutionCoordinator()
    active_started = threading.Event()
    release = threading.Event()

    def active() -> None:
        active_started.set()
        release.wait(timeout=2)

    coordinator.schedule(
        batch_id="batch-a",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=active,
    )
    assert active_started.wait(timeout=2)
    coordinator.schedule(
        batch_id="batch-b",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=lambda: None,
    )
    assert coordinator.cancel_queued("batch-b") is True
    assert coordinator.cancel_queued("batch-a") is False
    release.set()
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    coordinator.drain_completions()
    coordinator.shutdown()


def test_app_error_and_unexpected_sanitized() -> None:
    coordinator = SerialExecutionCoordinator()

    def boom() -> None:
        raise AppError("stage.test_failed")

    def unexpected() -> None:
        raise RuntimeError("secret-stack")

    coordinator.schedule(
        batch_id="batch-a", kind=BatchCommandKind.SUBMIT, job_id=None, operation=boom
    )
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    first = coordinator.drain_completions()
    assert len(first) == 1
    assert first[0].ok is False
    assert first[0].code == "stage.test_failed"

    coordinator.schedule(
        batch_id="batch-b",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=unexpected,
    )
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    second = coordinator.drain_completions()
    assert second[0].code == "gui.application_bridge_failed"
    assert "secret" not in second[0].code
    coordinator.shutdown()


def test_shutdown_rejects_active_and_finalizer_on_executor() -> None:
    coordinator = SerialExecutionCoordinator()
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        started.set()
        release.wait(timeout=2)

    coordinator.schedule(
        batch_id="batch-a",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=block,
    )
    assert started.wait(timeout=2)
    with pytest.raises(AppError, match=r"batch\.execution_active"):
        coordinator.shutdown()
    release.set()
    deadline = time.monotonic() + 2
    while coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    coordinator.drain_completions()

    names: list[str] = []

    def finalizer() -> None:
        names.append(threading.current_thread().name)

    coordinator.shutdown(finalizer=finalizer)
    assert names and names[0].startswith("captioner-pipeline")
    coordinator.shutdown()
