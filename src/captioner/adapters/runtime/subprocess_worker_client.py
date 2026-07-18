"""Real Core-side JSONL client for an isolated Runtime Worker process."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn, Protocol, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeState
from captioner.core.domain.worker_protocol import (
    WORKER_PROTOCOL_NAME,
    WORKER_PROTOCOL_VERSION,
    CancelAcknowledged,
    CancelRequest,
    CancelResult,
    DoctorRequest,
    DoctorResponse,
    HandshakeRequest,
    JsonlProtocolCodec,
    OperationCancelled,
    OperationProgress,
    ResultDescriptor,
    ShutdownAcknowledged,
    ShutdownRequest,
    ShutdownResult,
    TranscribeRequest,
    WorkerCancelledEvent,
    WorkerEnvelope,
    WorkerError,
    WorkerErrorEvent,
    WorkerEvent,
    WorkerHandshake,
    WorkerMessageType,
    WorkerProgressEvent,
    WorkerResultEvent,
    decode_typed_message,
    validate_sequence,
)
from captioner.core.ports.worker_client import WorkerClient

MAX_WORKER_LINE_BYTES = 1024 * 1024
DEFAULT_MESSAGE_TIMEOUT_SEC = 30.0


class _WorkerProcess(Protocol):
    pid: int
    returncode: int | None
    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None

    async def wait(self) -> int: ...


ProcessFactory = Callable[..., Awaitable[_WorkerProcess]]


@dataclass(frozen=True, slots=True)
class _WorkerEOF:
    pass


class SubprocessWorkerClient(WorkerClient):
    """One lazily started Worker session with isolated stdout/stderr pipes."""

    def __init__(
        self,
        *,
        log_dir: Path,
        process_factory: ProcessFactory | None = None,
        message_timeout_sec: float = DEFAULT_MESSAGE_TIMEOUT_SEC,
        cancellation_timeout_sec: float = 2.0,
        termination_grace_sec: float = 2.0,
    ) -> None:
        if message_timeout_sec <= 0 or cancellation_timeout_sec <= 0 or termination_grace_sec <= 0:
            raise ValueError
        self._log_dir = log_dir
        self._process_factory: ProcessFactory = process_factory or cast(
            ProcessFactory, _create_process
        )
        self._message_timeout = message_timeout_sec
        self._cancellation_timeout = cancellation_timeout_sec
        self._termination_grace = termination_grace_sec
        self._process: _WorkerProcess | None = None
        self._runtime: RuntimeInstallation | None = None
        self._workspace: Path | None = None
        self._queue: asyncio.Queue[WorkerEnvelope | AppError | _WorkerEOF] = asyncio.Queue()
        self._pending: list[WorkerEnvelope | AppError | _WorkerEOF] = []
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._log_stream: BinaryIO | None = None
        self._last_incoming_sequence: int | None = None
        self._outgoing_sequence = 0
        self._active_request_id: str | None = None
        self._active_request: TranscribeRequest | None = None
        self._cancel_ack_event: asyncio.Event | None = None
        self._shutdown_started = False
        self._codec = JsonlProtocolCodec()

    async def start(
        self,
        runtime: RuntimeInstallation,
        workspace: Path,
        request: HandshakeRequest,
    ) -> WorkerHandshake:
        if self._process is not None:
            raise AppError("worker.already_started")
        if runtime.state not in {
            RuntimeState.INSTALLED,
            RuntimeState.AVAILABLE,
            RuntimeState.EXTERNAL_UNMANAGED,
        }:
            raise AppError("runtime.not_available")
        if not workspace.is_absolute():
            raise AppError("worker.workspace_invalid")
        interpreter = runtime_interpreter(runtime)
        if not interpreter.is_file():
            raise AppError("runtime.interpreter_missing")
        session_id = uuid.uuid4().hex
        log_path = self._log_dir / "runtimes" / runtime.identity.runtime_id / f"{session_id}.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = log_path.open("ab")
            spawn_kwargs: dict[str, object] = {
                "cwd": str(runtime.install_path),
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": _worker_environment(),
                "limit": MAX_WORKER_LINE_BYTES + 1,
            }
            if os.name == "nt":
                spawn_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                spawn_kwargs["start_new_session"] = True
            self._process = await self._process_factory(
                str(interpreter),
                "-I",
                "-B",
                "-u",
                "-m",
                "captioner_runtime_worker",
                **spawn_kwargs,
            )
            if self._process.stdin is None or self._process.stdout is None:
                _fail("worker.pipe_missing")
            self._runtime = runtime
            self._workspace = workspace
            self._stderr_task = asyncio.create_task(self._pump_stderr())
            request_id = f"handshake-{uuid.uuid4().hex}"
            await self._send(
                WorkerEnvelope(
                    protocol=WORKER_PROTOCOL_NAME,
                    version=WORKER_PROTOCOL_VERSION,
                    message_type=WorkerMessageType.HANDSHAKE_REQUEST.value,
                    request_id=request_id,
                    sequence=self._next_outgoing_sequence(),
                    payload=request.to_payload(),
                )
            )
            envelope = await self._read_one(timeout=self._message_timeout)
            if envelope.request_id != request_id:
                _fail("worker.correlation_mismatch")
            if envelope.message_type != WorkerMessageType.HANDSHAKE_RESPONSE.value:
                _fail("worker.handshake_invalid")
            typed = decode_typed_message(envelope)
            if not isinstance(typed, WorkerHandshake):
                _fail("worker.handshake_invalid")
        except AppError:
            await self._cleanup(force=True)
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            await self._cleanup(force=True)
            raise AppError("worker.start_failed") from exc
        else:
            self._last_incoming_sequence = validate_sequence(None, envelope.sequence)
            self._stdout_task = asyncio.create_task(self._pump_stdout())
            return typed

    def transcribe(self, request: TranscribeRequest) -> AsyncIterator[WorkerEvent]:
        if self._process is None:
            raise AppError("worker.not_started")
        if self._active_request_id is not None:
            raise AppError("worker.busy")
        self._active_request_id = request.request_id
        self._active_request = request
        self._cancel_ack_event = asyncio.Event()
        return self._transcribe_events(request)

    async def doctor(self, request: DoctorRequest) -> DoctorResponse:
        if self._process is None:
            raise AppError("worker.not_started")
        request = DoctorRequest(
            request.nonce,
            request.probe_filename,
            request.workspace or self._workspace,
        )
        request_id = f"doctor-{uuid.uuid4().hex}"
        await self._send(
            WorkerEnvelope(
                protocol=WORKER_PROTOCOL_NAME,
                version=WORKER_PROTOCOL_VERSION,
                message_type=WorkerMessageType.DOCTOR_REQUEST.value,
                request_id=request_id,
                sequence=self._next_outgoing_sequence(),
                payload=request.to_payload(),
            )
        )
        envelope = await self._receive(timeout=self._message_timeout)
        if (
            envelope.request_id != request_id
            or envelope.message_type != WorkerMessageType.DOCTOR_RESPONSE.value
        ):
            raise AppError("worker.correlation_mismatch")
        typed = decode_typed_message(envelope)
        if not isinstance(typed, DoctorResponse):
            raise AppError("worker.doctor_invalid")
        return typed

    async def cancel(self, request_id: str) -> CancelResult:
        if self._process is None:
            return CancelResult(request_id=request_id, acknowledged=False, timed_out=True)
        if self._active_request_id != request_id:
            raise AppError("worker.cancel_wrong_request")
        await self._send(
            WorkerEnvelope(
                protocol=WORKER_PROTOCOL_NAME,
                version=WORKER_PROTOCOL_VERSION,
                message_type=WorkerMessageType.CANCEL_REQUEST.value,
                request_id=request_id,
                sequence=self._next_outgoing_sequence(),
                payload=CancelRequest(request_id).to_payload(),
                job_id=self._active_request.job_id if self._active_request else None,
                stage_attempt_id=(
                    self._active_request.stage_attempt_id if self._active_request else None
                ),
            )
        )
        try:
            acknowledgement = self._cancel_ack_event
            if acknowledgement is None:
                raise AppError("worker.cancel_invalid")
            await asyncio.wait_for(acknowledgement.wait(), timeout=self._cancellation_timeout)
            return CancelResult(request_id=request_id, acknowledged=True)
        except TimeoutError:
            return CancelResult(request_id=request_id, acknowledged=False, timed_out=True)

    async def shutdown(self) -> ShutdownResult:
        if self._process is None:
            return ShutdownResult(acknowledged=True)
        if self._active_request_id is not None:
            await self._terminate_process_tree()
            await self._cleanup(force=False)
            return ShutdownResult(acknowledged=True)
        if self._shutdown_started:
            await self._cleanup(force=True)
            return ShutdownResult(acknowledged=True)
        self._shutdown_started = True
        acknowledged = False
        request_id = f"shutdown-{uuid.uuid4().hex}"
        try:
            await self._send(
                WorkerEnvelope(
                    protocol=WORKER_PROTOCOL_NAME,
                    version=WORKER_PROTOCOL_VERSION,
                    message_type=WorkerMessageType.SHUTDOWN_REQUEST.value,
                    request_id=request_id,
                    sequence=self._next_outgoing_sequence(),
                    payload=ShutdownRequest().to_payload(),
                )
            )
            envelope = await self._receive(timeout=self._cancellation_timeout)
            if envelope.request_id == request_id:
                typed = decode_typed_message(envelope)
                acknowledged = isinstance(typed, ShutdownAcknowledged) and typed.acknowledged
        except (TimeoutError, AppError):
            acknowledged = False
        if not acknowledged or self._active_request_id is not None:
            await self._terminate_process_tree()
        else:
            await self._wait_process(self._termination_grace)
        await self._cleanup(force=False)
        return ShutdownResult(acknowledged=True)

    async def _transcribe_events(self, request: TranscribeRequest) -> AsyncIterator[WorkerEvent]:
        try:
            await self._send(
                WorkerEnvelope(
                    protocol=WORKER_PROTOCOL_NAME,
                    version=WORKER_PROTOCOL_VERSION,
                    message_type=WorkerMessageType.TRANSCRIBE_REQUEST.value,
                    request_id=request.request_id,
                    sequence=self._next_outgoing_sequence(),
                    payload=request.to_payload(),
                    job_id=request.job_id,
                    stage_attempt_id=request.stage_attempt_id,
                )
            )
            while True:
                envelope = await self._receive(timeout=None)
                self._validate_correlation(envelope, request)
                typed = decode_typed_message(envelope)
                if isinstance(typed, CancelAcknowledged):
                    if typed.target_request_id != request.request_id:
                        _fail("worker.correlation_mismatch")
                    if self._cancel_ack_event is not None:
                        self._cancel_ack_event.set()
                elif isinstance(typed, OperationProgress):
                    yield WorkerProgressEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        envelope.sequence,
                        typed,
                    )
                elif isinstance(typed, ResultDescriptor):
                    yield WorkerResultEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        envelope.sequence,
                        typed,
                    )
                    return
                elif isinstance(typed, WorkerError):
                    yield WorkerErrorEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        envelope.sequence,
                        typed,
                    )
                    return
                elif isinstance(typed, OperationCancelled):
                    yield WorkerCancelledEvent(
                        request.request_id,
                        request.job_id,
                        request.stage_attempt_id,
                        envelope.sequence,
                    )
                    return
                else:
                    _fail("worker.message_type_unexpected")
        except AppError:
            await self._cleanup(force=True)
            raise
        finally:
            self._active_request_id = None
            self._active_request = None
            self._cancel_ack_event = None

    def _validate_correlation(self, envelope: WorkerEnvelope, request: TranscribeRequest) -> None:
        if (
            envelope.request_id != request.request_id
            or envelope.job_id != request.job_id
            or envelope.stage_attempt_id != request.stage_attempt_id
        ):
            raise AppError("worker.correlation_mismatch")

    async def _read_one(self, *, timeout: float | None) -> WorkerEnvelope:
        stdout = self._process.stdout if self._process is not None else None
        if stdout is None:
            raise AppError("worker.pipe_missing")
        try:
            raw = await asyncio.wait_for(stdout.readline(), timeout=timeout)
        except asyncio.LimitOverrunError as exc:
            raise AppError("worker.protocol_line_too_large") from exc
        if not raw:
            raise AppError("worker.process_exit")
        if not raw.endswith(b"\n"):
            raise AppError("worker.protocol_partial_line")
        if len(raw) > MAX_WORKER_LINE_BYTES:
            raise AppError("worker.protocol_line_too_large")
        envelope = self._codec.decode_stdout(raw)
        self._last_incoming_sequence = validate_sequence(
            self._last_incoming_sequence, envelope.sequence
        )
        return envelope

    async def _receive(self, *, timeout: float | None) -> WorkerEnvelope:
        item: WorkerEnvelope | AppError | _WorkerEOF
        if self._pending:
            item = self._pending.pop(0)
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if isinstance(item, AppError):
            raise item
        if isinstance(item, _WorkerEOF):
            raise AppError("worker.process_exit")
        return item

    async def _pump_stdout(self) -> None:
        stdout = self._process.stdout if self._process is not None else None
        if stdout is None:
            return
        try:
            while True:
                try:
                    raw = await stdout.readline()
                except asyncio.LimitOverrunError:
                    await self._queue.put(AppError("worker.protocol_line_too_large"))
                    return
                if not raw:
                    await self._queue.put(_WorkerEOF())
                    return
                if len(raw) > MAX_WORKER_LINE_BYTES:
                    await self._queue.put(AppError("worker.protocol_line_too_large"))
                    return
                if not raw.endswith(b"\n"):
                    await self._queue.put(AppError("worker.protocol_partial_line"))
                    return
                try:
                    envelope = self._codec.decode_stdout(raw)
                    self._last_incoming_sequence = validate_sequence(
                        self._last_incoming_sequence, envelope.sequence
                    )
                except AppError as exc:
                    await self._queue.put(exc)
                    return
                await self._queue.put(envelope)
        except Exception:
            await self._queue.put(AppError("worker.protocol_read_failed"))
            raise

    async def _pump_stderr(self) -> None:
        stderr = self._process.stderr if self._process is not None else None
        stream = self._log_stream
        if stderr is None or stream is None:
            return
        while True:
            block = await stderr.read(64 * 1024)
            if not block:
                return
            stream.write(block)
            stream.flush()

    async def _send(self, envelope: WorkerEnvelope) -> None:
        stdin = self._process.stdin if self._process is not None else None
        if stdin is None:
            raise AppError("worker.pipe_missing")
        try:
            stdin.write(self._codec.encode(envelope))
            await stdin.drain()
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            raise AppError("worker.write_failed") from exc

    def _next_outgoing_sequence(self) -> int:
        value = self._outgoing_sequence
        self._outgoing_sequence += 1
        return value

    async def _wait_process(self, timeout: float) -> bool:
        if self._process is None:
            return True
        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def _terminate_process_tree(self) -> None:
        process = self._process
        if process is None:
            return
        from captioner.adapters.runtime.process_tree import terminate_process_tree

        await terminate_process_tree(process, grace_timeout=self._termination_grace)

    async def _cleanup(self, *, force: bool) -> None:
        if force and self._process is not None and self._process.returncode is None:
            await self._terminate_process_tree()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    continue
        if self._process is not None and self._process.stdin is not None:
            self._process.stdin.close()
        stream = self._log_stream
        if stream is not None:
            stream.close()
        self._stdout_task = None
        self._stderr_task = None
        self._log_stream = None
        self._process = None
        self._runtime = None
        self._workspace = None
        self._active_request_id = None
        self._active_request = None
        self._cancel_ack_event = None
        self._pending.clear()
        self._shutdown_started = False


async def _create_process(
    executable: str,
    *args: str,
    cwd: str,
    stdin: int,
    stdout: int,
    stderr: int,
    env: Mapping[str, str],
    limit: int,
    start_new_session: bool = False,
    creationflags: int = 0,
) -> _WorkerProcess:
    if os.name == "nt":
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            limit=limit,
            creationflags=creationflags,
        )
    else:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            cwd=cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            limit=limit,
            start_new_session=start_new_session,
        )
    return cast(_WorkerProcess, process)


def runtime_interpreter(runtime: RuntimeInstallation) -> Path:
    python_root = runtime.install_path / "payload" / "python"
    if runtime.manifest.target.platform == "windows":
        return python_root / "python.exe"
    return python_root / "bin" / "python3"


def _worker_environment() -> dict[str, str]:
    allowed_exact = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "TMP",
        "TEMP",
        "TMPDIR",
        "SYSTEMROOT",
        "WINDIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if (key in allowed_exact or key.startswith("LANG") or key.startswith("LC_"))
        and not _is_sensitive_environment_name(key)
    }
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CAPTIONER_RUNTIME_OFFLINE": "1",
        }
    )
    return environment


def _is_sensitive_environment_name(value: str) -> bool:
    upper = value.upper()
    return any(
        marker in upper
        for marker in (
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "API_KEY",
            "AUTHORIZATION",
            "CREDENTIAL",
        )
    )


def _fail(code: str) -> NoReturn:
    raise AppError(code)


__all__ = [
    "MAX_WORKER_LINE_BYTES",
    "SubprocessWorkerClient",
    "runtime_interpreter",
]
