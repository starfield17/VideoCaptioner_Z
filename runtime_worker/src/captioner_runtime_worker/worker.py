"""Async Runtime Worker main loop."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

from .backends.base import Backend
from .backends.faster_whisper import CancelledError, FasterWhisperCPUBackend
from .backends.mlx_whisper import MLXWhisperMetalBackend
from .build_info import load_build_info
from .protocol import decode, encode, write
from .transcript import sha256_file, write_result

_MESSAGE_TYPES = {
    "handshake.request",
    "doctor.request",
    "transcribe.request",
    "cancel.request",
    "shutdown.request",
    "model.load.request",
}


class ProtocolWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._sequence = 0

    def send(
        self,
        message_type: str,
        request_id: str,
        payload: Mapping[str, object],
        *,
        job_id: str | None = None,
        stage_attempt_id: str | None = None,
    ) -> None:
        with self._lock:
            message = encode(
                message_type,
                request_id,
                self._sequence,
                payload,
                job_id=job_id,
                stage_attempt_id=stage_attempt_id,
            )
            self._sequence += 1
            write(self._stream, message)


class RuntimeWorker:
    def __init__(self, *, runtime_root: Path | None = None) -> None:
        self._info = load_build_info(runtime_root)
        self._backend: Backend | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._active_request_id: str | None = None
        self._active_job_id: str | None = None
        self._active_attempt_id: str | None = None
        self._cancel_event: threading.Event | None = None
        self._active_operation_kind: str | None = None
        self._model_key: tuple[str, str] | None = None
        self._shutting_down = False
        self._last_incoming_sequence: int | None = None

    async def run(self, input_stream: BinaryIO, protocol_stream: BinaryIO) -> None:
        writer = ProtocolWriter(protocol_stream)
        while not self._shutting_down:
            raw = await asyncio.to_thread(input_stream.readline)
            if not raw:
                break
            message: dict[str, object] | None = None
            try:
                message = decode(raw)
                sequence = _int(message, "sequence")
                if (
                    self._last_incoming_sequence is not None
                    and sequence <= self._last_incoming_sequence
                ):
                    _invalid("sequence_not_monotonic")
                self._last_incoming_sequence = sequence
                message_type = _string(message, "message_type")
                if message_type not in _MESSAGE_TYPES:
                    _invalid("unknown_message_type")
                await self._dispatch(message, writer)
            except Exception as exc:
                if _can_report_operation_error(message):
                    self._error(
                        writer,
                        _string(message, "request_id"),
                        _string(message, "job_id"),
                        _string(message, "stage_attempt_id"),
                        "worker.protocol_invalid",
                        False,
                    )
                else:
                    print(
                        f"worker protocol failure: {_safe_reason(exc)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
        if self._active_task is not None:
            if self._cancel_event is not None:
                self._cancel_event.set()
            await self._active_task

    async def _dispatch(self, message: dict[str, object], writer: ProtocolWriter) -> None:
        message_type = _string(message, "message_type")
        request_id = _string(message, "request_id")
        payload = _object(message, "payload")
        if message_type == "handshake.request":
            self._handshake(request_id, payload, writer)
        elif message_type == "doctor.request":
            await self._doctor(request_id, payload, writer)
        elif message_type == "transcribe.request":
            self._start_transcribe(message, payload, writer)
        elif message_type == "cancel.request":
            self._cancel(request_id, message, payload, writer)
        elif message_type == "shutdown.request":
            await self._shutdown(request_id, writer)
        elif message_type == "model.load.request":
            self._start_model_load(request_id, payload, writer)

    def _handshake(
        self, request_id: str, payload: Mapping[str, object], writer: ProtocolWriter
    ) -> None:
        required = payload.get("required_capabilities", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ValueError("required_capabilities_invalid")
        capabilities = _string_list(self._info, "capabilities")
        writer.send(
            "handshake.response",
            request_id,
            {
                "protocol_version": _string(self._info, "protocol_version"),
                "runtime_id": _string(self._info, "runtime_id"),
                "runtime_version": _string(self._info, "runtime_version"),
                "backend_id": _string(self._info, "backend_id"),
                "backend_version": _string(self._info, "backend_version"),
                "worker_version": _string(self._info, "worker_version"),
                "platform": _string(self._info, "platform"),
                "architecture": _string(self._info, "architecture"),
                "capabilities": capabilities,
                "supported_devices": [_string(self._info, "device_kind")],
                "supported_model_formats": _string_list(self._info, "supported_model_formats"),
                "supported_result_schema_versions": [1],
            },
        )

    async def _doctor(
        self, request_id: str, payload: Mapping[str, object], writer: ProtocolWriter
    ) -> None:
        nonce = _string(payload, "nonce")
        probe_filename = _string(payload, "probe_filename")
        workspace_value = payload.get("workspace")
        workspace = Path(workspace_value) if isinstance(workspace_value, str) else Path.cwd()
        if (
            not workspace.is_absolute()
            or "/" in probe_filename
            or "\\" in probe_filename
            or ".." in Path(probe_filename).parts
        ):
            raise ValueError("doctor_path_invalid")
        backend_import_ok = False
        try:
            backend_import_ok = self._get_backend().doctor_import()
        except Exception:
            backend_import_ok = False
        probe = workspace / probe_filename
        workspace.mkdir(parents=True, exist_ok=True)
        temporary = probe.with_suffix(probe.suffix + ".tmp")
        data = (nonce + "\n").encode("utf-8")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, probe)
        writer.send(
            "doctor.response",
            request_id,
            {
                "nonce": nonce,
                "backend_import_ok": backend_import_ok,
                "device_kind": _string(self._info, "device_kind"),
                "probe_result": {
                    "relative_path": probe_filename,
                    "size_bytes": probe.stat().st_size,
                    "sha256": sha256_file(probe),
                    "schema_id": "captioner.runtime-doctor",
                    "schema_version": 1,
                },
                "details": {},
            },
        )

    def _start_model_load(
        self, request_id: str, payload: Mapping[str, object], writer: ProtocolWriter
    ) -> None:
        if self._active_task is not None and not self._active_task.done():
            writer.send(
                "model.load.response",
                request_id,
                {
                    "model_identity": dict(_object(payload, "model_identity")),
                    "backend_id": _string(self._info, "backend_id"),
                    "device_kind": _string(self._info, "device_kind"),
                    "loaded": False,
                    "details": {"code": "worker.busy"},
                },
            )
            return
        cancel_event = threading.Event()
        self._active_request_id = request_id
        self._active_job_id = None
        self._active_attempt_id = None
        self._active_operation_kind = "model_load"
        self._cancel_event = cancel_event
        self._active_task = asyncio.create_task(
            self._run_model_load(request_id, payload, writer, cancel_event)
        )

    async def _run_model_load(
        self,
        request_id: str,
        payload: Mapping[str, object],
        writer: ProtocolWriter,
        cancel_event: threading.Event,
    ) -> None:
        model_identity: Mapping[str, object] = {}
        loaded = False
        details: dict[str, object] = {}
        try:
            model_directory = Path(_string(payload, "model_directory"))
            model_identity = _object(payload, "model_identity")
            options = _object(payload, "backend_options")
            if not model_directory.is_absolute() or not model_directory.is_dir():
                _invalid("worker.remote_model_rejected")
            digest = _string(model_identity, "manifest_sha256")
            key = (str(model_directory.resolve()), digest)
            if self._model_key is not None and self._model_key != key:
                _invalid("worker.model_switch_requires_restart")
            if self._model_key == key:
                loaded = True
            else:
                loaded = await asyncio.to_thread(
                    self._get_backend().load_model,
                    model_directory=model_directory,
                    options=options,
                    model_identity=model_identity,
                )
                if loaded:
                    self._model_key = key
            if cancel_event.is_set():
                loaded = False
                details["code"] = "operation.cancelled"
        except ValueError as exc:
            details["code"] = str(exc)
        except Exception:
            details["code"] = "worker.model_load_failed"
        finally:
            writer.send(
                "model.load.response",
                request_id,
                {
                    "model_identity": dict(model_identity),
                    "backend_id": _string(self._info, "backend_id"),
                    "device_kind": _string(self._info, "device_kind"),
                    "loaded": loaded,
                    "details": details,
                },
            )
            self._active_task = None
            self._active_request_id = None
            self._active_job_id = None
            self._active_attempt_id = None
            self._active_operation_kind = None
            self._cancel_event = None

    def _start_transcribe(
        self,
        message: Mapping[str, object],
        payload: Mapping[str, object],
        writer: ProtocolWriter,
    ) -> None:
        request_id = _string(message, "request_id")
        job_id = _string(message, "job_id")
        attempt_id = _string(message, "stage_attempt_id")
        if self._active_task is not None and not self._active_task.done():
            self._error(writer, request_id, job_id, attempt_id, "worker.busy", False)
            return
        model_directory = Path(_string(payload, "model_directory"))
        if not model_directory.is_absolute() or not model_directory.is_dir():
            self._error(
                writer, request_id, job_id, attempt_id, "worker.remote_model_rejected", False
            )
            return
        model_identity = _object(payload, "model_identity")
        digest = _string(model_identity, "manifest_sha256")
        key = (str(model_directory.resolve()), digest)
        if self._model_key is not None and self._model_key != key:
            self._error(
                writer,
                request_id,
                job_id,
                attempt_id,
                "worker.model_switch_requires_restart",
                False,
            )
            return
        self._model_key = key
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self._active_request_id = request_id
        self._active_job_id = job_id
        self._active_attempt_id = attempt_id
        self._active_operation_kind = "transcribe"
        self._active_task = asyncio.create_task(
            self._run_transcribe(
                request_id,
                job_id,
                attempt_id,
                payload,
                model_directory,
                model_identity,
                cancel_event,
                writer,
            )
        )

    async def _run_transcribe(
        self,
        request_id: str,
        job_id: str,
        attempt_id: str,
        payload: Mapping[str, object],
        model_directory: Path,
        model_identity: Mapping[str, object],
        cancel_event: threading.Event,
        writer: ProtocolWriter,
    ) -> None:
        try:
            options = payload.get("backend_options", {})
            if not isinstance(options, dict):
                _invalid_type("backend_options_invalid")
            result = await asyncio.to_thread(
                self._get_backend().transcribe,
                audio_path=Path(_string(payload, "normalized_audio_path")),
                model_directory=model_directory,
                language=_optional_string(payload, "language"),
                task=_string(payload, "task"),
                initial_prompt=_optional_string(payload, "initial_prompt"),
                options=options,
                cancelled=cancel_event,
                progress=lambda phase: writer.send(
                    "operation.progress",
                    request_id,
                    {
                        "operation": "asr",
                        "phase": phase,
                        "message_code": f"asr.{phase}",
                        "detail_parameters": {},
                    },
                    job_id=job_id,
                    stage_attempt_id=attempt_id,
                ),
                model_identity=model_identity,
                runtime_info=self._info,
            )
            if cancel_event.is_set():
                _raise_cancelled()
            workspace = Path(_string(payload, "attempt_workspace"))
            descriptor = write_result(workspace, result)
            writer.send(
                "operation.result",
                request_id,
                descriptor,
                job_id=job_id,
                stage_attempt_id=attempt_id,
            )
        except CancelledError:
            writer.send(
                "operation.cancelled",
                request_id,
                {"target_request_id": request_id},
                job_id=job_id,
                stage_attempt_id=attempt_id,
            )
        except Exception as exc:
            print(f"worker backend failure: {_safe_reason(exc)}", file=sys.stderr)
            self._error(
                writer, request_id, job_id, attempt_id, "worker.transcription_failed", False
            )
        finally:
            self._active_task = None
            self._active_request_id = None
            self._active_job_id = None
            self._active_attempt_id = None
            self._active_operation_kind = None
            self._cancel_event = None

    def _cancel(
        self,
        request_id: str,
        message: Mapping[str, object],
        payload: Mapping[str, object],
        writer: ProtocolWriter,
    ) -> None:
        target = _string(payload, "target_request_id")
        if target != self._active_request_id:
            self._error(
                writer,
                request_id,
                _string(message, "job_id"),
                _string(message, "stage_attempt_id"),
                "worker.cancel_wrong_request",
                False,
            )
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        writer.send(
            "cancel.acknowledged",
            request_id,
            {"target_request_id": target},
            job_id=self._active_job_id,
            stage_attempt_id=self._active_attempt_id,
        )

    async def _shutdown(self, request_id: str, writer: ProtocolWriter) -> None:
        self._shutting_down = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        writer.send("shutdown.acknowledged", request_id, {"acknowledged": True})
        if self._active_task is not None:
            await self._active_task

    def _get_backend(self) -> Backend:
        if self._backend is None:
            backend_id = _string(self._info, "backend_id")
            backend_version = _string(self._info, "backend_version")
            if backend_id == "faster-whisper":
                self._backend = FasterWhisperCPUBackend(backend_version=backend_version)
            elif backend_id == "mlx-whisper":
                self._backend = MLXWhisperMetalBackend(backend_version=backend_version)
            else:
                raise ValueError("worker.backend_unknown")
        return self._backend

    @staticmethod
    def _error(
        writer: ProtocolWriter,
        request_id: str,
        job_id: str | None,
        attempt_id: str | None,
        code: str,
        retryable: bool,
    ) -> None:
        writer.send(
            "operation.error",
            request_id,
            {
                "code": code,
                "message_code": code,
                "retryable": retryable,
                "details": {},
            },
            job_id=job_id,
            stage_attempt_id=attempt_id,
        )


def run_worker() -> None:
    protocol_stream = sys.stdout.buffer
    sys.stdout = sys.stderr
    worker = RuntimeWorker()
    asyncio.run(worker.run(sys.stdin.buffer, protocol_stream))


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key}_invalid")
    return item


def _optional_string(value: Mapping[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise ValueError(f"{key}_invalid")
    return item


def _string_list(value: Mapping[str, object], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
        _invalid_type(f"{key}_invalid")
    return cast(list[str], item)


def _object(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        _invalid_type(f"{key}_invalid")
    return cast(Mapping[str, object], item)


def _int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise TypeError(f"{key}_invalid")
    return cast(int, item)


def _can_report_operation_error(message: Mapping[str, object] | None) -> bool:
    if message is None or message.get("message_type") not in {
        "transcribe.request",
        "cancel.request",
    }:
        return False
    return all(
        isinstance(message.get(key), str) and bool(cast(str, message[key]).strip())
        for key in ("request_id", "job_id", "stage_attempt_id")
    )


def _invalid(reason: str) -> None:
    raise ValueError(reason)


def _invalid_type(reason: str) -> None:
    raise TypeError(reason)


def _raise_cancelled() -> None:
    raise CancelledError


def _safe_reason(exc: Exception) -> str:
    value = type(exc).__name__
    return value if value else "worker_error"


__all__ = ["ProtocolWriter", "RuntimeWorker", "run_worker"]
