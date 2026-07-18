"""Deterministic JSONL Worker Client fake."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.domain.worker_protocol import (
    CancelResult,
    DoctorRequest,
    DoctorResponse,
    HandshakeRequest,
    ResultDescriptor,
    ShutdownResult,
    TranscribeRequest,
    WorkerCancelledEvent,
    WorkerError,
    WorkerErrorEvent,
    WorkerEvent,
    WorkerHandshake,
    WorkerProgressEvent,
    WorkerResultEvent,
    decode_stdout_line,
    validate_sequence,
)


def _empty_handshake_requests() -> list[HandshakeRequest]:
    return []


def _empty_start_calls() -> list[tuple[RuntimeInstallation, Path, HandshakeRequest]]:
    return []


def _empty_requests() -> list[TranscribeRequest]:
    return []


def _empty_cancel_calls() -> list[str]:
    return []


def _empty_shutdown_calls() -> list[bool]:
    return []


def _empty_doctor_calls() -> list[DoctorRequest]:
    return []


@dataclass(slots=True)
class ScriptedWorkerClient:
    handshake: WorkerHandshake
    doctor_response: DoctorResponse | None = None
    progress_events: tuple[WorkerProgressEvent, ...] = ()
    result: ResultDescriptor | None = None
    error: WorkerError | None = None
    cancel_timed_out: bool = False
    contaminated_stdout_lines: tuple[str | bytes, ...] = ()
    start_calls: list[tuple[RuntimeInstallation, Path, HandshakeRequest]] = field(
        default_factory=_empty_start_calls
    )
    handshake_requests: list[HandshakeRequest] = field(default_factory=_empty_handshake_requests)
    transcribe_requests: list[TranscribeRequest] = field(default_factory=_empty_requests)
    cancel_calls: list[str] = field(default_factory=_empty_cancel_calls)
    shutdown_calls: list[bool] = field(default_factory=_empty_shutdown_calls)
    doctor_calls: list[DoctorRequest] = field(default_factory=_empty_doctor_calls)
    _active_request_id: str | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _shutdown: bool = field(default=False, init=False)
    _last_sequence: int | None = field(default=None, init=False)
    _cancel_requested: str | None = field(default=None, init=False)
    _termination_event: asyncio.Event | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.result is not None and self.error is not None:
            raise ValueError("result_and_error_are_mutually_exclusive")

    async def start(
        self,
        runtime: RuntimeInstallation,
        workspace: Path,
        request: HandshakeRequest,
    ) -> WorkerHandshake:
        if self._shutdown:
            raise AppError("worker.shutdown")
        if not workspace.is_absolute():
            raise AppError("worker.workspace_invalid")
        self.start_calls.append((runtime, workspace, request))
        self.handshake_requests.append(request)
        self._started = True
        self._last_sequence = None
        self._cancel_requested = None
        return self.handshake

    def transcribe(self, request: TranscribeRequest) -> AsyncIterator[WorkerEvent]:
        if not self._started or self._shutdown:
            raise AppError("worker.not_started")
        if self._active_request_id is not None:
            raise AppError("worker.busy")
        self._active_request_id = request.request_id
        self._termination_event = asyncio.Event()
        self.transcribe_requests.append(request)

        async def scripted_events() -> AsyncIterator[WorkerEvent]:
            try:
                for line in self.contaminated_stdout_lines:
                    decode_stdout_line(line)
                for event in self.progress_events:
                    _validate_event_correlation(event, request)
                    if self._cancel_requested == request.request_id:
                        sequence = 0 if self._last_sequence is None else self._last_sequence + 1
                        self._last_sequence = validate_sequence(self._last_sequence, sequence)
                        cancelled = WorkerCancelledEvent(
                            request.request_id,
                            request.job_id,
                            request.stage_attempt_id,
                            sequence,
                        )
                        self._active_request_id = None
                        self._cancel_requested = None
                        yield cancelled
                        return
                    self._last_sequence = validate_sequence(self._last_sequence, event.sequence)
                    yield event
                if self._cancel_requested == request.request_id:
                    sequence = 0 if self._last_sequence is None else self._last_sequence + 1
                    self._last_sequence = validate_sequence(self._last_sequence, sequence)
                    cancelled = WorkerCancelledEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        sequence,
                    )
                    self._active_request_id = None
                    self._cancel_requested = None
                    yield cancelled
                    return
                if self.cancel_timed_out:
                    termination_event = self._termination_event
                    if termination_event is None:
                        raise AppError("worker.termination_state_invalid")
                    await termination_event.wait()
                    return
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
                self._cancel_requested = None
                self._termination_event = None

        return scripted_events()

    async def cancel(self, request_id: str) -> CancelResult:
        self.cancel_calls.append(request_id)
        if self._active_request_id != request_id:
            raise AppError("worker.request_not_found")
        if self.cancel_timed_out:
            return CancelResult(request_id, acknowledged=False, timed_out=True)
        self._cancel_requested = request_id
        return CancelResult(request_id, acknowledged=True, cancelled=True)

    async def doctor(self, request: DoctorRequest) -> DoctorResponse:
        if not self._started or self._shutdown:
            raise AppError("worker.not_started")
        if self.doctor_response is None:
            raise AppError("worker.doctor_invalid")
        self.doctor_calls.append(request)
        return self.doctor_response

    async def shutdown(self) -> ShutdownResult:
        self.shutdown_calls.append(self._shutdown)
        self._shutdown = True
        self._started = False
        self._active_request_id = None
        self._cancel_requested = None
        if self._termination_event is not None:
            self._termination_event.set()
        return ShutdownResult(acknowledged=True)


def _validate_event_correlation(event: WorkerEvent, request: TranscribeRequest) -> None:
    if (
        event.request_id != request.request_id
        or event.job_id != request.job_id
        or event.stage_attempt_id != request.stage_attempt_id
    ):
        raise AppError("worker.event_correlation_invalid")


__all__ = ["ScriptedWorkerClient"]
