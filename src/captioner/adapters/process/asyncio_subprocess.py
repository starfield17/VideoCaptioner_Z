"""Cancellable subprocess execution using argument arrays."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.process import ProcessResult


class AsyncioSubprocessRunner:
    """Run external programs without invoking a shell."""

    async def run(self, arguments: Sequence[str], context: ExecutionContext) -> ProcessResult:
        context.raise_if_cancelled()
        command = tuple(str(argument) for argument in arguments)
        if not command or not command[0].strip():
            raise AppError("process.invalid_arguments")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AppError("process.executable_not_found", {"executable": command[0]}) from exc
        except OSError as exc:
            raise AppError("process.start_failed", {"executable": command[0]}) from exc

        communication = asyncio.create_task(process.communicate())
        try:
            while not communication.done():
                if context.is_cancelled:
                    await _terminate_and_reap(process)
                    await communication
                    raise AppError("operation.cancelled")
                await asyncio.sleep(0.02)
            stdout, stderr = await communication
        except asyncio.CancelledError:
            await _terminate_and_reap(process)
            await communication
            raise
        return ProcessResult(stdout=stdout, stderr=stderr, returncode=process.returncode or 0)


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    """Terminate a child, escalate after a short grace period, and reap it."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except TimeoutError:
        process.kill()
        await process.wait()
