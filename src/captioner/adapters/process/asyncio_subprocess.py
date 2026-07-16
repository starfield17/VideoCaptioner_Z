"""Cancellable subprocess execution using argument arrays."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.process import ProcessResult

_TERMINATION_GRACE_SECONDS = 2.0


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
                    await _cleanup_shielded(process, communication)
                    raise AppError("operation.cancelled")
                await asyncio.sleep(0.02)
            stdout, stderr = await communication
            context.raise_if_cancelled()
        except asyncio.CancelledError:
            await _cleanup_shielded(process, communication)
            raise
        return ProcessResult(stdout=stdout, stderr=stderr, returncode=process.returncode or 0)


async def _cleanup_shielded(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
) -> None:
    """Terminate, reap and collect a child even if the caller is cancelled."""
    cleanup = asyncio.create_task(terminate_and_reap(process, communication))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    await cleanup


async def terminate_and_reap(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]] | None = None,
) -> None:
    """Terminate a child, escalate after a short grace period, and reap it."""
    wait_task = asyncio.create_task(process.wait())
    try:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task), timeout=_TERMINATION_GRACE_SECONDS
                )
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await asyncio.shield(wait_task)
        else:
            await asyncio.shield(wait_task)
    finally:
        if communication is not None:
            await asyncio.shield(communication)
