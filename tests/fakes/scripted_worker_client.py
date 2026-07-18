"""Deterministic JSONL Worker Client fake."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.domain.worker_protocol import (
    CancelResult,
    ResultDescriptor,
    ShutdownResult,
    TranscribeRequest,
    WorkerError,
    WorkerErrorEvent,
    WorkerEvent,
    WorkerHandshake,
    WorkerProgressEvent,
    WorkerResultEvent,
    decode_stdout_line,
    validate_sequence,
)


def _empty_start_calls() -> list[tuple[RuntimeInstallation, Path]]:
    return []


def _empty_requests() -> list[TranscribeRequest]:
    return []


def _empty_cancel_calls() -> list[str]:
    return []


def _empty_shutdown_calls() -> list[bool]:
    return []


@dataclass(slots=True)
class ScriptedWorkerClient:
    handshake: WorkerHandshake
    progress_events: tuple[WorkerProgressEvent, ...] = ()
    result: ResultDescriptor | None = None
    error: WorkerError | None = None
    cancel_timed_out: bool = False
    contaminated_stdout_lines: tuple[str | bytes, ...] = ()
    start_calls: list[tuple[RuntimeInstallation, Path]] = field(default_factory=_empty_start_calls)
    transcribe_requests: list[TranscribeRequest] = field(default_factory=_empty_requests)
    cancel_calls: list[str] = field(default_factory=_empty_cancel_calls)
    shutdown_calls: list[bool] = field(default_factory=_empty_shutdown_calls)
    _active_request_id: str | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _shutdown: bool = field(default=False, init=False)
    _last_sequence: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.result is not None and self.error is not None:
            raise ValueError("result_and_error_are_mutually_exclusive")

    async def start(self, runtime: RuntimeInstallation, workspace: Path) -> WorkerHandshake:
        if self._shutdown:
            raise AppError("worker.shutdown")
        if not workspace.is_absolute():
            raise AppError("worker.workspace_invalid")
        self.start_calls.append((runtime, workspace))
        self._started = True
        self._last_sequence = None
        return self.handshake

    def transcribe(self, request: TranscribeRequest) -> AsyncIterator[WorkerEvent]:
        if not self._started or self._shutdown:
            raise AppError("worker.not_started")
        if self._active_request_id is not None:
            raise AppError("worker.busy")
        self._active_request_id = request.request_id
        self.transcribe_requests.append(request)

        async def scripted_events() -> AsyncIterator[WorkerEvent]:
            try:
                for line in self.contaminated_stdout_lines:
                    decode_stdout_line(line)
                for event in self.progress_events:
                    self._last_sequence = validate_sequence(self._last_sequence, event.sequence)
                    yield event
                if self.error is not None:
                    sequence = 0 if self._last_sequence is None else self._last_sequence + 1
                    self._last_sequence = validate_sequence(self._last_sequence, sequence)
                    yield WorkerErrorEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        sequence,
                        self.error,
                    )
                elif self.result is not None:
                    sequence = 0 if self._last_sequence is None else self._last_sequence + 1
                    self._last_sequence = validate_sequence(self._last_sequence, sequence)
                    yield WorkerResultEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        sequence,
                        self.result,
                    )
            finally:
                self._active_request_id = None

        return scripted_events()

    async def cancel(self, request_id: str) -> CancelResult:
        self.cancel_calls.append(request_id)
        if self._active_request_id != request_id:
            raise AppError("worker.request_not_found")
        if self.cancel_timed_out:
            return CancelResult(request_id, acknowledged=False, timed_out=True)
        return CancelResult(request_id, acknowledged=True, cancelled=True)

    async def shutdown(self) -> ShutdownResult:
        self.shutdown_calls.append(self._shutdown)
        self._shutdown = True
        self._started = False
        self._active_request_id = None
        return ShutdownResult(acknowledged=True)


__all__ = ["ScriptedWorkerClient"]
