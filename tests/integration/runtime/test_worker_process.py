from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from tests.fakes.phase6_values import runtime_installation, transcribe_request

from captioner.adapters.runtime.subprocess_worker_client import (
    ProcessFactory,
    SubprocessWorkerClient,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.worker_protocol import HandshakeRequest, WorkerResultEvent

FIXTURE = Path(__file__).parents[2] / "fixtures" / "runtime" / "fake_worker.py"


def test_real_subprocess_worker_round_trip_stays_on_jsonl_boundary(tmp_path: Path) -> None:
    asyncio.run(_run_round_trip(tmp_path))


async def _run_round_trip(tmp_path: Path) -> None:
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
            "stderr",
            cwd=cast(str, kwargs["cwd"]),
            stdin=cast(int, kwargs["stdin"]),
            stdout=cast(int, kwargs["stdout"]),
            stderr=cast(int, kwargs["stderr"]),
            env=cast(Mapping[str, str], kwargs["env"]),
            limit=cast(int, kwargs["limit"]),
            start_new_session=cast(bool, kwargs.get("start_new_session", False)),
        )

    runtime = runtime_installation()
    install_path = tmp_path / "runtime"
    interpreter = install_path / "payload" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fixture interpreter")
    interpreter.chmod(0o755)
    runtime = replace(runtime, install_path=install_path)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, create),
        message_timeout_sec=2.0,
        termination_grace_sec=0.5,
    )
    request = replace(
        transcribe_request(),
        normalized_audio_path=tmp_path / "audio.wav",
        attempt_workspace=tmp_path / "attempt",
        model_directory=tmp_path / "model",
    )
    handshake = await client.start(runtime, tmp_path / "session", HandshakeRequest())
    assert handshake.runtime_id == runtime.identity.runtime_id
    events = [event async for event in client.transcribe(request)]
    assert isinstance(events[0], WorkerResultEvent)
    await client.shutdown()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
def test_real_subprocess_worker_kills_spawned_child_on_cancel_escalation(
    tmp_path: Path,
) -> None:
    asyncio.run(_run_child_termination(tmp_path))


async def _run_child_termination(tmp_path: Path) -> None:
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
            "spawn-child",
            cwd=cast(str, kwargs["cwd"]),
            stdin=cast(int, kwargs["stdin"]),
            stdout=cast(int, kwargs["stdout"]),
            stderr=cast(int, kwargs["stderr"]),
            env=cast(Mapping[str, str], kwargs["env"]),
            limit=cast(int, kwargs["limit"]),
            start_new_session=cast(bool, kwargs.get("start_new_session", False)),
        )

    runtime = runtime_installation()
    install_path = tmp_path / "runtime"
    interpreter = install_path / "payload" / "python" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fixture interpreter")
    interpreter.chmod(0o755)
    runtime = replace(runtime, install_path=install_path)
    client = SubprocessWorkerClient(
        log_dir=tmp_path / "logs",
        process_factory=cast(ProcessFactory, create),
        message_timeout_sec=2.0,
        cancellation_timeout_sec=0.05,
        termination_grace_sec=0.05,
    )
    request = replace(
        transcribe_request(),
        normalized_audio_path=tmp_path / "audio.wav",
        attempt_workspace=tmp_path / "attempt",
        model_directory=tmp_path / "model",
    )
    await client.start(runtime, tmp_path / "session", HandshakeRequest())
    events = client.transcribe(request)
    consumer = asyncio.create_task(_consume_first(events))
    try:
        child_pid: int | None = None
        for _ in range(40):
            logs = tuple((tmp_path / "logs").rglob("*.log"))
            for log in logs:
                text = log.read_text(encoding="utf-8", errors="replace")
                marker = "child_pid="
                if marker in text:
                    child_pid = int(text.split(marker, 1)[1].splitlines()[0])
                    break
            if child_pid is not None:
                break
            await asyncio.sleep(0.01)
        assert child_pid is not None
        cancelled = await client.cancel(request.request_id)
        assert cancelled.timed_out
        await client.shutdown()

        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, AppError):
            consumer.cancel()
            await consumer
        for _ in range(40):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError
    finally:
        if not consumer.done():
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
        await client.shutdown()


async def _consume_first(events: AsyncIterator[object]) -> object:
    return await events.__anext__()
