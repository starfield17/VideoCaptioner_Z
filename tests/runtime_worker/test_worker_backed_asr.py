from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from tests.fakes.phase6_values import (
    model_installation,
    result_descriptor,
    runtime_installation,
    worker_handshake,
)
from tests.fakes.scripted_worker_client import ScriptedWorkerClient
from tests.support import make_audio, make_transcript

from captioner.adapters.asr.runtime_worker import WorkerBackedASREngine
from captioner.adapters.persistence.domain_codecs import encode_transcript
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import CancellationToken, ExecutionContext
from captioner.core.domain.model import ModelInstallation
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeState
from captioner.core.domain.worker_protocol import ResultDescriptor
from captioner.core.ports.asr import TranscriptionRequest


def _transcription_request(tmp_path: Path, *, job_id: str = "job-1") -> TranscriptionRequest:
    attempt = tmp_path / "attempt-1"
    return TranscriptionRequest(
        audio=make_audio(tmp_path / "audio.wav"),
        language="en",
        job_id=job_id,
        stage_attempt_id=f"{job_id}-transcribe-1",
        attempt_workspace=attempt,
    )


def _result_for(
    runtime: RuntimeInstallation, model: ModelInstallation, workspace: Path
) -> ResultDescriptor:
    transcript = replace(
        make_transcript(),
        engine_id=runtime.manifest.backend_id,
        model_id=f"{model.identity.backend_id}:{model.identity.manifest_sha256}",
        metadata={
            "runtime_identity": runtime.identity.runtime_id,
            "runtime_version": runtime.identity.version,
            "backend_version": runtime.manifest.backend_version,
            "worker_version": "1.0.0",
            "device_kind": runtime.manifest.target.device_kind,
            "model_identity": model.identity.to_dict(),
            "word_timestamps": True,
        },
    )
    data = encode_transcript(transcript)
    workspace.mkdir(parents=True)
    (workspace / "result.json").write_bytes(data)
    descriptor = result_descriptor(
        relative_path="result.json",
    )
    return replace(
        descriptor,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        schema_id="captioner.transcript",
    )


def test_worker_backed_engine_lazy_starts_reuses_session_and_validates_result(
    tmp_path: Path,
) -> None:
    runtime = runtime_installation()
    model = model_installation()
    attempt = tmp_path / "attempt-1"
    descriptor = _result_for(runtime, model, attempt)
    client = ScriptedWorkerClient(worker_handshake(), result=descriptor)
    engine = WorkerBackedASREngine(runtime, model, client, tmp_path / "session")

    result = asyncio.run(
        engine.transcribe(
            _transcription_request(tmp_path),
            ExecutionContext(),
        )
    )

    assert result.engine_id == runtime.manifest.backend_id
    assert len(client.start_calls) == 1
    assert client.transcribe_requests[0].job_id == "job-1"
    assert client.transcribe_requests[0].stage_attempt_id == "job-1-transcribe-1"
    asyncio.run(engine.close())
    assert len(client.shutdown_calls) == 1


def test_worker_backed_engine_requires_available_runtime_and_validated_model(
    tmp_path: Path,
) -> None:
    model = model_installation()
    runtime = runtime_installation(state=RuntimeState.INSTALLED)
    with pytest.raises(AppError, match=r"runtime\.not_available"):
        WorkerBackedASREngine(runtime, model, ScriptedWorkerClient(worker_handshake()), tmp_path)


def test_worker_backed_engine_bridges_cancel_timeout_to_shutdown(tmp_path: Path) -> None:
    runtime = runtime_installation()
    model = model_installation()
    client = ScriptedWorkerClient(worker_handshake(), cancel_timed_out=True)
    engine = WorkerBackedASREngine(runtime, model, client, tmp_path / "session")
    execution = ExecutionContext(CancellationToken())
    request = _transcription_request(tmp_path)

    async def run() -> None:
        task = asyncio.create_task(engine.transcribe(request, execution))
        await asyncio.sleep(0)
        execution.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await task

    asyncio.run(run())
    assert client.shutdown_calls
