from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import cast

import pytest

import captioner.adapters.process.asyncio_subprocess as process_module
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext


def test_process_runner_captures_output_and_numeric_exit_code() -> None:
    async def scenario() -> None:
        result = await AsyncioSubprocessRunner().run(
            (sys.executable, "-c", "print('路径 with spaces')"), ExecutionContext()
        )
        assert result.returncode == 0
        assert "路径 with spaces" in result.stdout.decode("utf-8")

        failed = await AsyncioSubprocessRunner().run(
            (sys.executable, "-c", "raise SystemExit(7)"), ExecutionContext()
        )
        assert failed.returncode == 7

    asyncio.run(scenario())


def test_process_runner_distinguishes_missing_executable() -> None:
    async def scenario() -> None:
        with pytest.raises(AppError, match="executable_not_found"):
            await AsyncioSubprocessRunner().run(
                ("missing-captioner-executable",), ExecutionContext()
            )

    asyncio.run(scenario())


def test_process_runner_terminates_and_reaps_on_cancellation() -> None:
    async def scenario() -> None:
        context = ExecutionContext()
        task = asyncio.create_task(
            AsyncioSubprocessRunner().run(
                (sys.executable, "-c", "import time; time.sleep(10)"), context
            )
        )
        await asyncio.sleep(0.1)
        context.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await task

    asyncio.run(scenario())


@dataclass
class RaceProcess:
    wait_delay: float = 0.0
    terminate_lookup_error: bool = False
    kill_lookup_error: bool = False
    returncode: int | None = None

    def __post_init__(self) -> None:
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.finished = asyncio.Event()

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_lookup_error:
            raise ProcessLookupError

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_lookup_error:
            raise ProcessLookupError

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.wait_delay:
            await asyncio.sleep(self.wait_delay)
        self.returncode = 0
        self.finished.set()
        return 0

    async def communicate(self) -> tuple[bytes, bytes]:
        await self.finished.wait()
        return b"", b""


def test_termination_ignores_process_lookup_races_and_reaps() -> None:
    async def scenario() -> None:
        process = RaceProcess(terminate_lookup_error=True)
        communication = asyncio.create_task(process.communicate())
        await process_module.terminate_and_reap(
            cast(asyncio.subprocess.Process, process), communication
        )
        assert process.wait_calls == 1
        assert process.terminate_calls == 1
        assert communication.done()

    asyncio.run(scenario())


def test_kill_lookup_race_is_ignored_after_grace_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = RaceProcess(wait_delay=0.02, kill_lookup_error=True)
        communication = asyncio.create_task(process.communicate())
        await process_module.terminate_and_reap(
            cast(asyncio.subprocess.Process, process), communication
        )
        assert process.kill_calls == 1
        assert process.wait_calls == 1

    monkeypatch.setattr(process_module, "_TERMINATION_GRACE_SECONDS", 0.001)
    asyncio.run(scenario())


def test_outer_task_cancellation_shields_cleanup_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        process = RaceProcess()

        async def start_process(*args: object, **kwargs: object) -> RaceProcess:
            del args, kwargs
            return process

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", start_process)
        task = asyncio.create_task(
            AsyncioSubprocessRunner().run(("fake-process",), ExecutionContext())
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.wait_calls == 1
        assert process.terminate_calls == 1

    asyncio.run(scenario())
