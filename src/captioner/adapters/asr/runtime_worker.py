"""ASR Engine adapter backed by an isolated Runtime Worker session."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from captioner.adapters.persistence.domain_codecs import decode_transcript
from captioner.core.application.model_compatibility import ensure_model_compatibility
from captioner.core.application.worker_handshake_validation import validate_worker_handshake
from captioner.core.application.worker_result_validation import validate_worker_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.model import ModelIdentity, ModelInstallation, ModelState
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeState
from captioner.core.domain.transcript import Transcript
from captioner.core.domain.worker_protocol import (
    HandshakeRequest,
    ModelLoadRequest,
    ModelLoadResponse,
    WorkerCancelledEvent,
    WorkerError,
    WorkerErrorEvent,
    WorkerEvent,
    WorkerHandshake,
    WorkerProgressEvent,
    WorkerResultEvent,
)
from captioner.core.domain.worker_protocol import (
    TranscribeRequest as WorkerTranscribeRequest,
)
from captioner.core.ports.asr import ASRCapabilities, ASREngine, TranscriptionRequest
from captioner.core.ports.worker_client import WorkerClient


@dataclass(slots=True)
class WorkerBackedASREngine(ASREngine):
    runtime: RuntimeInstallation
    model: ModelInstallation
    worker_client: WorkerClient
    session_workspace_root: Path
    model_use_lock: Callable[[ModelIdentity], AbstractContextManager[None]] | None = None
    load_on_start: bool = False
    backend_options: Mapping[str, JsonValue] | None = None
    _handshake: WorkerHandshake | None = None
    _started: bool = False
    _model_use_lock_context: AbstractContextManager[None] | None = None

    def __post_init__(self) -> None:
        if not self.session_workspace_root.is_absolute():
            raise AppError("worker.workspace_invalid")
        _require_runtime_and_model(self.runtime, self.model)
        ensure_model_compatibility(self.runtime, self.model)

    @property
    def engine_id(self) -> str:
        return self.runtime.manifest.backend_id

    @property
    def capabilities(self) -> ASRCapabilities:
        capability = self.runtime.manifest.capabilities
        return ASRCapabilities(
            word_timestamps=capability.word_timestamps,
            segment_timestamps=True,
            language_detection=capability.language_detection,
            native_long_audio=True,
            internal_batching=False,
            supported_languages=None,
            supported_devices=frozenset({self.runtime.manifest.target.device_kind}),
        )

    async def transcribe(
        self, request: TranscriptionRequest, context: ExecutionContext
    ) -> Transcript:
        context.raise_if_cancelled()
        _require_runtime_and_model(self.runtime, self.model)
        ensure_model_compatibility(self.runtime, self.model)
        if request.job_id is None or request.stage_attempt_id is None:
            raise AppError("worker.correlation_invalid")
        if request.attempt_workspace is None or not request.attempt_workspace.is_absolute():
            raise AppError("worker.workspace_invalid")
        await self._ensure_started()
        request_id = f"transcribe-{uuid.uuid4().hex}"
        worker_request = WorkerTranscribeRequest(
            normalized_audio_path=request.audio.path.expanduser().resolve(),
            attempt_workspace=request.attempt_workspace.expanduser().resolve(),
            model_directory=self.model.model_directory.expanduser().resolve(),
            backend_id=self.runtime.manifest.backend_id,
            runtime_identity=self.runtime.identity,
            model_identity=self.model.identity,
            result_schema_version=1,
            language=request.language,
            task=request.task,
            word_timestamps=request.word_timestamps,
            initial_prompt=request.initial_prompt,
            backend_options=_merged_backend_options(self.backend_options, request.backend_options),
            request_id=request_id,
            job_id=request.job_id,
            stage_attempt_id=request.stage_attempt_id,
        )
        try:
            async for event in self._events_with_cancellation(worker_request, context):
                if isinstance(event, WorkerProgressEvent):
                    continue
                if isinstance(event, WorkerErrorEvent):
                    _raise_worker_error(event.error)
                if isinstance(event, WorkerCancelledEvent):
                    _raise_cancelled()
                return self._decode_worker_result(event, request)
        except BaseException:
            # Preserve the original failure while ensuring a cancelled request,
            # worker crash, or injected client exception cannot retain the model
            # use-lock or a stale Worker session.
            try:
                await self._shutdown_after_failure()
            except BaseException:
                self._reset_session()
            raise
        raise AppError("worker.result_missing")

    async def close(self) -> None:
        if not self._started and self._model_use_lock_context is None:
            return
        try:
            await self.worker_client.shutdown()
        finally:
            self._reset_session()

    async def _shutdown_after_failure(self) -> None:
        try:
            await self.worker_client.shutdown()
        finally:
            self._reset_session()

    async def _ensure_started(self) -> None:
        if self._started:
            return
        self.session_workspace_root.mkdir(parents=True, exist_ok=True)
        lock = None if self.model_use_lock is None else self.model_use_lock(self.model.identity)
        if lock is not None:
            lock.__enter__()
            self._model_use_lock_context = lock
        required = tuple(sorted(self.runtime.manifest.capabilities.advertised_capabilities))
        request = HandshakeRequest(
            required_capabilities=required,
            required_backend_id=self.runtime.manifest.backend_id,
            required_result_schema_versions=(1,),
        )
        worker_started = False
        try:
            handshake = await self.worker_client.start(
                self.runtime, self.session_workspace_root, request
            )
            worker_started = True
            result = validate_worker_handshake(self.runtime.manifest, request, handshake)
            if not result.ok:
                _raise_handshake_failure(result.error_code, result.reasons)
            if self.load_on_start:
                response = await self.worker_client.load_model(
                    ModelLoadRequest(
                        model_directory=self.model.model_directory.expanduser().resolve(),
                        model_identity=self.model.identity,
                        backend_options=_merged_backend_options(self.backend_options, None),
                    )
                )
                _validate_model_load_response(response, self.runtime, self.model)
        except BaseException as exc:
            if worker_started:
                try:
                    await self.worker_client.shutdown()
                except BaseException as shutdown_error:
                    self._reset_session()
                    raise shutdown_error from exc
            self._reset_session()
            raise
        self._handshake = handshake
        self._started = True

    async def _events_with_cancellation(
        self,
        request: WorkerTranscribeRequest,
        context: ExecutionContext,
    ) -> AsyncIterator[WorkerEvent]:
        iterator = self.worker_client.transcribe(request)
        next_event: asyncio.Task[WorkerEvent] = asyncio.create_task(_next_event(iterator))
        cancellation = asyncio.create_task(context.wait_cancelled())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {next_event, cancellation}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancellation in done:
                    await self._cancel_worker(request.request_id, next_event)
                    raise AppError("operation.cancelled")
                try:
                    event = next_event.result()
                except StopAsyncIteration as exc:
                    raise AppError("worker.result_missing") from exc
                yield event
                next_event = asyncio.create_task(_next_event(iterator))
        finally:
            if not cancellation.done():
                cancellation.cancel()
            if not next_event.done():
                next_event.cancel()
            with suppress(AppError, StopAsyncIteration, asyncio.CancelledError):
                await next_event

    async def _cancel_worker(self, request_id: str, next_event: asyncio.Task[WorkerEvent]) -> None:
        try:
            result = await self.worker_client.cancel(request_id)
        except AppError as exc:
            if exc.code != "worker.request_not_found":
                raise
            await self.worker_client.shutdown()
            self._reset_session()
            return
        if result.timed_out:
            await self.worker_client.shutdown()
            self._reset_session()
            return
        try:
            await asyncio.wait_for(asyncio.shield(next_event), timeout=2.0)
        except (TimeoutError, StopAsyncIteration):
            await self.worker_client.shutdown()
            self._reset_session()

    def _decode_worker_result(
        self, event: WorkerResultEvent, request: TranscriptionRequest
    ) -> Transcript:
        workspace = request.attempt_workspace
        if workspace is None:
            raise AppError("worker.workspace_invalid")
        path = validate_worker_result(
            event.result,
            workspace,
            supported_schema_versions={"captioner.transcript": {1}},
        )
        transcript = decode_transcript(path.read_bytes())
        expected_model_id = (
            f"{self.model.identity.backend_id}:{self.model.identity.manifest_sha256}"
        )
        if transcript.engine_id != self.runtime.manifest.backend_id:
            raise AppError("worker.result_identity_mismatch", {"field": "engine_id"})
        if transcript.model_id != expected_model_id:
            raise AppError("worker.result_identity_mismatch", {"field": "model_id"})
        metadata = cast(Mapping[str, object], transcript.metadata)
        if metadata.get("runtime_identity") != self.runtime.identity.runtime_id:
            raise AppError("worker.result_identity_mismatch", {"field": "runtime_identity"})
        if metadata.get("runtime_version") != self.runtime.identity.version:
            raise AppError("worker.result_identity_mismatch", {"field": "runtime_version"})
        if metadata.get("backend_version") != self.runtime.manifest.backend_version:
            raise AppError("worker.result_identity_mismatch", {"field": "backend_version"})
        if (
            self._handshake is None
            or metadata.get("worker_version") != self._handshake.worker_version
        ):
            raise AppError("worker.result_identity_mismatch", {"field": "worker_version"})
        if metadata.get("device_kind") != self.runtime.manifest.target.device_kind:
            raise AppError("worker.result_identity_mismatch", {"field": "device_kind"})
        if metadata.get("word_timestamps") is not True:
            raise AppError("worker.result_identity_mismatch", {"field": "word_timestamps"})
        raw_model_identity = metadata.get("model_identity")
        if raw_model_identity != self.model.identity.to_dict():
            raise AppError("worker.result_identity_mismatch", {"field": "model_identity"})
        return transcript

    def _reset_session(self) -> None:
        self._started = False
        self._handshake = None
        lock = self._model_use_lock_context
        self._model_use_lock_context = None
        if lock is not None:
            lock.__exit__(None, None, None)


def _validate_model_load_response(
    response: ModelLoadResponse,
    runtime: RuntimeInstallation,
    model: ModelInstallation,
) -> None:
    if not response.loaded:
        raise AppError("model.load_failed")
    if response.model_identity != model.identity:
        raise AppError("model.load_identity_mismatch")
    if response.backend_id != runtime.manifest.backend_id:
        raise AppError("model.load_identity_mismatch", {"field": "backend_id"})
    if response.device_kind != runtime.manifest.target.device_kind:
        raise AppError("model.load_identity_mismatch", {"field": "device_kind"})


def _raise_handshake_failure(code: str | None, reasons: tuple[str, ...]) -> NoReturn:
    raise AppError(code or "worker.handshake_invalid", {"reasons": list(reasons)})


def _require_runtime_and_model(runtime: RuntimeInstallation, model: ModelInstallation) -> None:
    if not runtime.is_available:
        raise AppError("runtime.not_available")
    if (
        model.state
        not in {
            ModelState.INSTALLED,
            ModelState.LOAD_VERIFIED,
            ModelState.EXTERNAL_UNMANAGED,
        }
        or not model.is_validated
    ):
        raise AppError("model.not_installed")
    if runtime.state not in {RuntimeState.AVAILABLE, RuntimeState.EXTERNAL_UNMANAGED}:
        raise AppError("runtime.not_available")


async def _next_event(iterator: AsyncIterator[WorkerEvent]) -> WorkerEvent:
    return await iterator.__anext__()


def _merged_backend_options(
    configured: Mapping[str, JsonValue] | None,
    request: Mapping[str, JsonValue] | None,
) -> Mapping[str, JsonValue]:
    merged: dict[str, JsonValue] = {} if configured is None else dict(configured)
    if request is not None:
        merged.update(request)
    return merged


def _raise_worker_error(error: WorkerError) -> NoReturn:
    raise AppError(error.code, dict(error.details), error.retryable)


def _raise_cancelled() -> NoReturn:
    raise AppError("operation.cancelled")


__all__ = ["WorkerBackedASREngine"]
