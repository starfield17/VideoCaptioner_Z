"""Serial single-worker coordinator for long Pipeline execution."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from captioner.core.application.batch_commands import (
    BatchCommandKind,
    ExecutionCompletion,
    LocalExecutionSnapshot,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier

_UNEXPECTED_FAILURE_CODE = "gui.application_bridge_failed"
_SUCCESS_CODE = "execution.completed"


@dataclass(slots=True)
class _ScheduledTask:
    batch_id: str
    kind: BatchCommandKind
    job_id: str | None
    operation: Callable[[], None]
    future: Future[None] | None = None


def _empty_queue() -> list[_ScheduledTask]:
    return []


def _empty_completions() -> queue.SimpleQueue[ExecutionCompletion]:
    return queue.SimpleQueue()


@dataclass(slots=True)
class SerialExecutionCoordinator:
    """One lazy ThreadPoolExecutor(max_workers=1) for Pipeline work."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _active: _ScheduledTask | None = field(default=None, init=False, repr=False)
    _queued: list[_ScheduledTask] = field(default_factory=_empty_queue, init=False, repr=False)
    _completions: queue.SimpleQueue[ExecutionCompletion] = field(
        default_factory=_empty_completions,
        init=False,
        repr=False,
    )
    _shut_down: bool = field(default=False, init=False, repr=False)

    def schedule(
        self,
        *,
        batch_id: str,
        kind: BatchCommandKind,
        job_id: str | None,
        operation: Callable[[], None],
    ) -> None:
        validated_batch = validate_identifier(batch_id, field="batch_id")
        validated_job = None if job_id is None else validate_identifier(job_id, field="job_id")
        with self._lock:
            if self._shut_down:
                raise AppError("batch.execution_active", {"reason": "shutdown"})
            if validated_batch in self._scheduled_ids_unlocked():
                raise AppError("batch.operation_conflict", {"batch_id": validated_batch})
            task = _ScheduledTask(
                batch_id=validated_batch,
                kind=kind,
                job_id=validated_job,
                operation=operation,
            )
            self._queued.append(task)
            self._ensure_executor_unlocked()
            self._pump_unlocked()

    def cancel_queued(self, batch_id: str) -> bool:
        validated = validate_identifier(batch_id, field="batch_id")
        with self._lock:
            remaining: list[_ScheduledTask] = []
            cancelled = False
            for task in self._queued:
                if task.batch_id != validated:
                    remaining.append(task)
                    continue
                if task.future is not None and task.future.cancel():
                    cancelled = True
                    continue
                if task.future is None:
                    cancelled = True
                    continue
                remaining.append(task)
            self._queued = remaining
            return cancelled

    def snapshot(self) -> LocalExecutionSnapshot:
        with self._lock:
            active = None if self._active is None else self._active.batch_id
            queued = tuple(task.batch_id for task in self._queued)
            return LocalExecutionSnapshot(active_batch_id=active, queued_batch_ids=queued)

    def scheduled_batch_ids(self) -> frozenset[str]:
        with self._lock:
            return self._scheduled_ids_unlocked()

    def drain_completions(self) -> tuple[ExecutionCompletion, ...]:
        drained: list[ExecutionCompletion] = []
        while True:
            try:
                drained.append(self._completions.get_nowait())
            except queue.Empty:
                break
        return tuple(drained)

    def shutdown(
        self,
        *,
        finalizer: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            if self._shut_down:
                return
            if self._active is not None or self._queued:
                raise AppError("batch.execution_active")
            executor = self._executor
            self._shut_down = True
            self._executor = None
        if executor is None:
            if finalizer is not None:
                finalizer()
            return
        if finalizer is not None:
            done = threading.Event()
            error: list[BaseException] = []

            def _run_finalizer() -> None:
                try:
                    finalizer()
                except BaseException as exc:
                    error.append(exc)
                finally:
                    done.set()

            future = executor.submit(_run_finalizer)
            try:
                future.result()
            except Exception as exc:
                executor.shutdown(wait=True, cancel_futures=False)
                raise AppError(_UNEXPECTED_FAILURE_CODE) from exc
            done.wait()
            if error:
                executor.shutdown(wait=True, cancel_futures=False)
                raise AppError(_UNEXPECTED_FAILURE_CODE) from error[0]
        executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_executor_unlocked(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="captioner-pipeline",
            )

    def _scheduled_ids_unlocked(self) -> frozenset[str]:
        ids = {task.batch_id for task in self._queued}
        if self._active is not None:
            ids.add(self._active.batch_id)
        return frozenset(ids)

    def _pump_unlocked(self) -> None:
        if self._active is not None:
            return
        if not self._queued:
            return
        assert self._executor is not None
        task = self._queued.pop(0)
        self._active = task
        future = self._executor.submit(self._run_task, task)
        task.future = future

    def _run_task(self, task: _ScheduledTask) -> None:
        code = _SUCCESS_CODE
        ok = True
        try:
            task.operation()
        except AppError as exc:
            ok = False
            code = exc.code
        except Exception:
            ok = False
            code = _UNEXPECTED_FAILURE_CODE
        finally:
            self._completions.put(
                ExecutionCompletion(
                    batch_id=task.batch_id,
                    kind=task.kind,
                    job_id=task.job_id,
                    ok=ok,
                    code=code,
                )
            )
            with self._lock:
                if self._active is task:
                    self._active = None
                self._pump_unlocked()


__all__ = ["SerialExecutionCoordinator"]
