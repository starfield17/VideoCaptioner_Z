from __future__ import annotations

import asyncio
import importlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
from tests.fakes.fake_model_source import FakeModelSource
from tests.fakes.fake_runtime_doctor import FakeRuntimeDoctor
from tests.fakes.in_memory_model_repository import InMemoryModelRepository
from tests.fakes.in_memory_runtime_repository import InMemoryRuntimeRepository
from tests.fakes.phase6_values import (
    model_installation,
    runtime_installation,
    transcribe_request,
    worker_handshake,
)
from tests.fakes.scripted_worker_client import ScriptedWorkerClient

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelSourceCandidate, ModelState
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.domain.runtime import DoctorCheck, DoctorPhase, DoctorReport, RuntimeState
from captioner.core.domain.worker_protocol import (
    HandshakeRequest,
    WorkerCancelledEvent,
    WorkerProgressEvent,
)


def test_phase6_core_layers_do_not_import_adapters() -> None:
    modules = (
        "captioner.core.domain.runtime",
        "captioner.core.domain.model",
        "captioner.core.domain.worker_protocol",
        "captioner.core.ports.runtime_repository",
        "captioner.core.ports.worker_client",
        "captioner.core.application.runtime_selection",
        "captioner.core.application.model_compatibility",
        "captioner.core.application.worker_handshake_validation",
    )
    for module_name in modules:
        source = inspect.getsource(importlib.import_module(module_name))
        assert "captioner.adapters" not in source


def test_runtime_repository_only_activates_available_and_registered_runtime() -> None:
    available = runtime_installation()
    installed = runtime_installation(
        state=RuntimeState.INSTALLED,
        runtime_id="faster-whisper-cpu-macos-arm64-installed",
    )
    repository = InMemoryRuntimeRepository((available, installed))
    repository.set_active_runtime(
        available.identity,
        available.manifest.backend_id,
        available.manifest.target,
    )
    assert (
        repository.get_active_runtime(available.manifest.backend_id, available.manifest.target)
        == available
    )
    with pytest.raises(AppError, match=r"runtime\.not_available"):
        repository.set_active_runtime(
            installed.identity,
            installed.manifest.backend_id,
            installed.manifest.target,
        )


def test_external_runtime_can_be_active_only_after_activation_doctor() -> None:
    external = runtime_installation(
        state=RuntimeState.EXTERNAL_UNMANAGED,
        doctor_passed=True,
    )
    repository = InMemoryRuntimeRepository((external,))
    repository.set_active_runtime(
        external.identity,
        external.manifest.backend_id,
        external.manifest.target,
    )
    assert (
        repository.get_active_runtime(external.manifest.backend_id, external.manifest.target)
        == external
    )


def test_runtime_repository_reports_typed_busy_error_for_in_use_record() -> None:
    runtime = runtime_installation()
    repository = InMemoryRuntimeRepository((runtime,))
    repository.set_in_use(runtime.identity)
    with pytest.raises(AppError, match=r"runtime\.busy"):
        repository.remove_installation_record(runtime.identity)


def test_model_repository_preserves_external_delete_boundary() -> None:
    managed = model_installation()
    external = model_installation(
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        source_id="external-path",
        repository_id="external-model",
    )
    repository = InMemoryModelRepository()
    repository.register_managed_model(managed)
    repository.register_external_model(external)
    repository.mark_load_verified(managed.identity)
    with pytest.raises(AppError, match=r"model\.external_unmanaged"):
        repository.remove_managed_model_record(external.identity)
    repository.remove_managed_model_record(managed.identity)
    assert repository.get(managed.identity) is None


def test_modelscope_source_supports_exact_lookup_but_not_search() -> None:
    candidate = model_installation().manifest
    source = FakeModelSource(
        "modelscope",
        search_supported=False,
        exact_results=(ModelSourceCandidate(candidate.identity, candidate.display_name),),
    )
    assert not source.capabilities().search
    with pytest.raises(AppError, match=r"model\.source_search_unsupported"):
        source.search("model", "faster-whisper", 10)
    assert source.resolve_exact(
        candidate.identity.repository_id, candidate.identity.revision, "faster-whisper"
    )


def test_fake_runtime_doctor_exposes_static_and_activation_reports() -> None:
    report = DoctorReport(True, DoctorPhase.STATIC.value, (DoctorCheck("manifest", True),))
    activation = DoctorReport(True, DoctorPhase.ACTIVATION.value, (DoctorCheck("handshake", True),))
    doctor = FakeRuntimeDoctor(report, activation)
    runtime = runtime_installation()
    assert doctor.static_doctor(runtime) == report
    assert doctor.activation_doctor(runtime, Path("/workspace")) == activation


def test_scripted_worker_is_ordered_single_request_and_shutdown_is_idempotent() -> None:
    async def scenario() -> None:
        request = transcribe_request()
        progress = WorkerProgressEvent(
            request.request_id,
            request.job_id,
            request.stage_attempt_id,
            0,
            OperationProgress("asr", "transcribing", "worker.transcribing", {}),
        )
        client = ScriptedWorkerClient(
            handshake=worker_handshake(),
            progress_events=(progress,),
        )
        await client.start(
            runtime_installation(),
            Path("/workspace"),
            HandshakeRequest(
                required_capabilities=("word_timestamps",),
                required_backend_id="faster-whisper",
                required_result_schema_versions=(1,),
            ),
        )
        first = client.transcribe(request)
        with pytest.raises(AppError, match=r"worker\.busy"):
            client.transcribe(request)
        events = [event async for event in first]
        assert events == [progress]
        assert client.handshake_requests == [
            HandshakeRequest(
                required_capabilities=("word_timestamps",),
                required_backend_id="faster-whisper",
                required_result_schema_versions=(1,),
            )
        ]
        assert client.transcribe_requests == [request]
        await client.shutdown()
        await client.shutdown()
        assert client.shutdown_calls == [False, True]

    asyncio.run(scenario())


def test_scripted_worker_cancellation_is_terminal_and_releases_request() -> None:
    async def scenario() -> None:
        request = transcribe_request()
        progress = WorkerProgressEvent(
            request.request_id,
            request.job_id,
            request.stage_attempt_id,
            0,
            OperationProgress("asr", "transcribing", "worker.transcribing", {}),
        )
        client = ScriptedWorkerClient(
            handshake=worker_handshake(),
            progress_events=(progress,),
        )
        handshake_request = HandshakeRequest(
            required_capabilities=("word_timestamps",),
            required_backend_id="faster-whisper",
            required_result_schema_versions=(1,),
        )
        await client.start(runtime_installation(), Path("/workspace"), handshake_request)
        events = client.transcribe(request)
        assert await anext(events) == progress
        cancellation = await client.cancel(request.request_id)
        assert cancellation.acknowledged
        assert cancellation.cancelled
        cancelled = await anext(events)
        assert isinstance(cancelled, WorkerCancelledEvent)
        with pytest.raises(StopAsyncIteration):
            await anext(events)
        # A terminal cancellation releases the session for a new request.
        second_progress = replace(progress, sequence=2)
        client.progress_events = (second_progress,)
        second_events = client.transcribe(request)
        assert [event async for event in second_events] == [second_progress]

    asyncio.run(scenario())


def test_scripted_worker_cancel_timeout_keeps_request_busy_until_shutdown() -> None:
    async def scenario() -> None:
        request = transcribe_request()
        client = ScriptedWorkerClient(
            handshake=worker_handshake(),
            cancel_timed_out=True,
        )
        await client.start(runtime_installation(), Path("/workspace"), HandshakeRequest())
        client.transcribe(request)
        result = await client.cancel(request.request_id)
        assert not result.acknowledged
        assert result.timed_out
        with pytest.raises(AppError, match=r"worker\.busy"):
            client.transcribe(request)
        await client.shutdown()

    asyncio.run(scenario())


def test_scripted_worker_cancel_requires_matching_request_id() -> None:
    async def scenario() -> None:
        request = transcribe_request()
        client = ScriptedWorkerClient(handshake=worker_handshake())
        await client.start(runtime_installation(), Path("/workspace"), HandshakeRequest())
        events = client.transcribe(request)
        with pytest.raises(AppError, match=r"worker\.request_not_found"):
            await client.cancel("other-request")
        with pytest.raises(StopAsyncIteration):
            await anext(events)

    asyncio.run(scenario())


@pytest.mark.parametrize("non_monotonic", [True, False])
def test_scripted_worker_validates_event_correlation_and_sequence(
    non_monotonic: bool,
) -> None:
    async def scenario() -> None:
        request = transcribe_request()
        first = WorkerProgressEvent(
            request.request_id,
            request.job_id,
            request.stage_attempt_id,
            0,
            OperationProgress("asr", "preparing_audio", "worker.preparing", {}),
        )
        second = WorkerProgressEvent(
            request.request_id if non_monotonic else "wrong-request",
            request.job_id,
            request.stage_attempt_id,
            0,
            OperationProgress("asr", "transcribing", "worker.transcribing", {}),
        )
        client = ScriptedWorkerClient(
            handshake=worker_handshake(),
            progress_events=(first, second),
        )
        await client.start(runtime_installation(), Path("/workspace"), HandshakeRequest())
        events = client.transcribe(request)
        assert await anext(events) == first
        expected_error = (
            "worker.sequence_invalid" if non_monotonic else "worker.event_correlation_invalid"
        )
        with pytest.raises(AppError, match=expected_error):
            await anext(events)

    asyncio.run(scenario())
