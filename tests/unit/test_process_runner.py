from __future__ import annotations

import asyncio
import sys

import pytest

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
