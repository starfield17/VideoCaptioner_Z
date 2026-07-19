from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from tests.fakes.phase6_values import runtime_installation, transcribe_request

from captioner.adapters.runtime.filesystem_runtime_repository import FilesystemRuntimeRepository
from captioner.adapters.runtime.subprocess_worker_client import (
    ProcessFactory,
    SubprocessWorkerClient,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeState
from captioner.core.domain.worker_protocol import (
    HandshakeRequest,
    TranscribeRequest,
    WorkerCancelledEvent,
    WorkerEvent,
    WorkerResultEvent,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "runtime" / "fake_worker.py"


def _runtime(tmp_path: Path):
    runtime = runtime_installation()
    install_path = tmp_path / "runtime"
    interpreter = install_path / "payload" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"fake interpreter")
    interpreter.chmod(0o755)
    return replace(runtime, install_path=install_path)


def _request(tmp_path: Path) -> TranscribeRequest:
    request = transcribe_request()
    return replace(
        request,
        normalized_audio_path=tmp_path / "audio.wav",
        attempt_workspace=tmp_path / "attempt",
        model_directory=tmp_path / "model",
    )


def _factory(mode: str):
    async def create(
        executable: str,
        *args: str,
        **kwargs: object,
    ) -> asyncio.subprocess.Process:
        del executable, args
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            str(FIXTURE),
            mode,
            cwd=cast(str, kwargs["cwd"]),
            stdin=cast(int, kwargs["stdin"]),
            stdout=cast(int, kwargs["stdout"]),
            stderr=cast(int, kwargs["stderr"]),
            env=cast(Mapping[str, str], kwargs["env"]),
            limit=cast(int, kwargs["limit"]),
            start_new_session=cast(bool, kwargs.get("start_new_session", False)),
        )

    return create


def _sequence_factory(modes: tuple[str, ...]):
    remaining = list(modes)

    async def create(
        executable: str,
        *args: str,
        **kwargs: object,
    ) -> asyncio.subprocess.Process:
        if not remaining:
            raise AssertionError
        mode = remaining.pop(0)
        del executable, args
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            str(FIXTURE),
            mode,
            cwd=cast(str, kwargs["cwd"]),
            stdin=cast(int, kwargs["stdin"]),
            stdout=cast(int, kwargs["stdout"]),
            stderr=cast(int, kwargs["stderr"]),
            env=cast(Mapping[str, str], kwargs["env"]),
            limit=cast(int, kwargs["limit"]),
            start_new_session=cast(bool, kwargs.get("start_new_session", False)),
        )

    return create


async def _start(tmp_path: Path, mode: str) -> tuple[SubprocessWorkerClient, TranscribeRequest]:
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, _factory(mode)),
        message_timeout_sec=2.0,
        cancellation_timeout_sec=0.5,
        termination_grace_sec=0.5,
    )
    runtime = _runtime(tmp_path)
    await client.start(runtime, tmp_path / "session", HandshakeRequest())
    return client, _request(tmp_path)


def test_subprocess_client_handshake_result_and_stderr_log(tmp_path: Path) -> None:
    asyncio.run(_test_subprocess_client_handshake_result_and_stderr_log(tmp_path))


async def _test_subprocess_client_handshake_result_and_stderr_log(tmp_path: Path) -> None:
    client, request = await _start(tmp_path, "stderr")
    events = [event async for event in client.transcribe(request)]
    assert isinstance(events[0], WorkerResultEvent)
    await client.shutdown()
    await client.shutdown()
    logs = tuple((tmp_path / "logs").rglob("*.log"))
    assert logs and b"worker diagnostic" in logs[0].read_bytes()


@pytest.mark.parametrize(
    "mode", ("contamination", "partial", "wrong-correlation", "wrong-sequence")
)
def test_subprocess_client_rejects_protocol_failures(tmp_path: Path, mode: str) -> None:
    asyncio.run(_test_subprocess_client_rejects_protocol_failures(tmp_path, mode))


async def _test_subprocess_client_rejects_protocol_failures(tmp_path: Path, mode: str) -> None:
    client, request = await _start(tmp_path, mode)
    with pytest.raises(AppError):
        _ = [event async for event in client.transcribe(request)]
    await client.shutdown()


def test_subprocess_client_cooperative_cancel_is_correlated(tmp_path: Path) -> None:
    asyncio.run(_test_subprocess_client_cooperative_cancel_is_correlated(tmp_path))


async def _test_subprocess_client_cooperative_cancel_is_correlated(tmp_path: Path) -> None:
    client, request = await _start(tmp_path, "wait-cancel")
    events = client.transcribe(request)
    send_task = asyncio.create_task(_consume_one(events))
    await asyncio.sleep(0.05)
    result = await client.cancel(request.request_id)
    assert result.acknowledged and not result.timed_out
    event = await send_task
    assert isinstance(event, WorkerCancelledEvent)
    await client.shutdown()


def test_subprocess_client_wrong_cancel_request_is_typed_failure(tmp_path: Path) -> None:
    asyncio.run(_test_subprocess_client_wrong_cancel_request_is_typed_failure(tmp_path))


async def _test_subprocess_client_wrong_cancel_request_is_typed_failure(tmp_path: Path) -> None:
    client, request = await _start(tmp_path, "wait-cancel")
    _ = client.transcribe(request)
    with pytest.raises(AppError, match=r"worker\.cancel_wrong_request"):
        await client.cancel("other-request")
    await client.shutdown()


def test_same_client_can_restart_after_graceful_shutdown(tmp_path: Path) -> None:
    asyncio.run(_test_same_client_can_restart_after_graceful_shutdown(tmp_path))


async def _test_same_client_can_restart_after_graceful_shutdown(tmp_path: Path) -> None:
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, _factory("stderr")),
        message_timeout_sec=2.0,
        termination_grace_sec=0.5,
    )
    runtime = _runtime(tmp_path)
    request = _request(tmp_path)
    await client.start(runtime, tmp_path / "session-a", HandshakeRequest())
    assert [event async for event in client.transcribe(request)]
    await client.shutdown()

    await client.start(runtime, tmp_path / "session-b", HandshakeRequest())
    assert [event async for event in client.transcribe(request)]
    await client.shutdown()


def test_same_client_restarts_after_worker_crash_during_transcribe(tmp_path: Path) -> None:
    asyncio.run(_test_same_client_restarts_after_worker_crash_during_transcribe(tmp_path))


async def _test_same_client_restarts_after_worker_crash_during_transcribe(
    tmp_path: Path,
) -> None:
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(
            ProcessFactory,
            _sequence_factory(("crash-during-transcribe", "stderr")),
        ),
        message_timeout_sec=2.0,
        termination_grace_sec=0.5,
    )
    runtime = _runtime(tmp_path)
    request = _request(tmp_path)
    await client.start(runtime, tmp_path / "session-a", HandshakeRequest())
    with pytest.raises(AppError, match=r"worker\.process_exit"):
        _ = [event async for event in client.transcribe(request)]

    await client.start(runtime, tmp_path / "session-b", HandshakeRequest())
    assert [event async for event in client.transcribe(request)]
    await client.shutdown()


def test_start_failure_releases_runtime_use_lock(tmp_path: Path) -> None:
    asyncio.run(_test_start_failure_releases_runtime_use_lock(tmp_path))


async def _test_start_failure_releases_runtime_use_lock(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    base = runtime_installation()
    runtime = replace(
        base,
        install_path=repository.root / base.identity.runtime_id / base.identity.version,
    )
    repository.register_installation(runtime)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        runtime_use_lock=repository.use_lock,
    )

    with pytest.raises(AppError, match=r"runtime\.interpreter_missing"):
        await client.start(runtime, tmp_path / "session", HandshakeRequest())

    repository.remove_managed_files(runtime.identity)


def test_handshake_failure_releases_runtime_use_lock(tmp_path: Path) -> None:
    asyncio.run(_test_handshake_failure_releases_runtime_use_lock(tmp_path))


async def _test_handshake_failure_releases_runtime_use_lock(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    base = _runtime(tmp_path)
    runtime = replace(
        base,
        install_path=repository.root / base.identity.runtime_id / base.identity.version,
    )
    interpreter = runtime.install_path / "payload" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fake interpreter")
    interpreter.chmod(0o755)
    repository.register_installation(runtime)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, _factory("contamination")),
        message_timeout_sec=2.0,
        termination_grace_sec=0.2,
        runtime_use_lock=repository.use_lock,
    )

    await client.start(runtime, tmp_path / "session", HandshakeRequest())
    with pytest.raises(AppError):
        _ = [event async for event in client.transcribe(_request(tmp_path))]

    repository.remove_managed_files(runtime.identity)


def test_active_worker_blocks_runtime_removal_until_shutdown(tmp_path: Path) -> None:
    asyncio.run(_test_active_worker_blocks_runtime_removal_until_shutdown(tmp_path))


async def _test_active_worker_blocks_runtime_removal_until_shutdown(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    base = _runtime(tmp_path)
    runtime = replace(
        base,
        install_path=repository.root / base.identity.runtime_id / base.identity.version,
    )
    interpreter = runtime.install_path / "payload" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fake interpreter")
    interpreter.chmod(0o755)
    repository.register_installation(runtime)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, _factory("stderr")),
        message_timeout_sec=2.0,
        termination_grace_sec=0.2,
        runtime_use_lock=repository.use_lock,
    )
    await client.start(runtime, tmp_path / "session", HandshakeRequest())

    with pytest.raises(AppError, match=r"runtime\.in_use"):
        repository.remove_managed_files(runtime.identity)

    await client.shutdown()
    repository.remove_managed_files(runtime.identity)


def test_active_external_worker_blocks_record_removal_until_shutdown(tmp_path: Path) -> None:
    asyncio.run(_test_active_external_worker_blocks_record_removal_until_shutdown(tmp_path))


async def _test_active_external_worker_blocks_record_removal_until_shutdown(
    tmp_path: Path,
) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    runtime = replace(
        _runtime(tmp_path),
        state=RuntimeState.EXTERNAL_UNMANAGED,
        managed=False,
        doctor_passed=True,
    )
    marker = runtime.install_path / "external-model.bin"
    marker.write_bytes(b"external")
    repository.register_installation(runtime)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, _factory("stderr")),
        message_timeout_sec=2.0,
        termination_grace_sec=0.2,
        runtime_use_lock=repository.use_lock,
    )
    await client.start(runtime, tmp_path / "session", HandshakeRequest())

    with pytest.raises(AppError, match=r"runtime\.in_use"):
        repository.remove_installation_record(runtime.identity)

    await client.shutdown()
    repository.remove_installation_record(runtime.identity)
    assert marker.read_bytes() == b"external"
    assert runtime.install_path.is_dir()


def test_shutdown_acknowledgement_still_escalates_for_hanging_process(tmp_path: Path) -> None:
    asyncio.run(_test_shutdown_acknowledgement_still_escalates_for_hanging_process(tmp_path))


async def _test_shutdown_acknowledgement_still_escalates_for_hanging_process(
    tmp_path: Path,
) -> None:
    client, _request_value = await _start(tmp_path, "ack-shutdown-but-hang")
    await client.shutdown()
    runtime = _runtime(tmp_path)
    await client.start(runtime, tmp_path / "second-session", HandshakeRequest())
    await client.shutdown()


async def _consume_one(events: AsyncIterator[WorkerEvent]) -> WorkerEvent:
    return await events.__anext__()
