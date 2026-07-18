from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from tests.fakes.phase6_values import runtime_installation, transcribe_request

from captioner.adapters.runtime.subprocess_worker_client import (
    ProcessFactory,
    SubprocessWorkerClient,
)
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
