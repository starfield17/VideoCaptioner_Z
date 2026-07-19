"""Process-group termination for Runtime Workers."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Protocol

from captioner.core.domain.errors import AppError


class ProcessLike(Protocol):
    pid: int
    returncode: int | None

    async def wait(self) -> int: ...


async def terminate_process_tree(
    process: ProcessLike,
    *,
    grace_timeout: float = 2.0,
    kill_timeout: float = 2.0,
) -> None:
    """Terminate a whole Worker process tree, escalating once if necessary."""
    if grace_timeout <= 0 or kill_timeout <= 0:
        raise ValueError
    if process.returncode is not None:
        return
    if os.name == "nt":
        _taskkill(process.pid, force=False)
    else:
        _signal_group_if_present(process.pid, signal.SIGTERM)
    if await _wait_until_exit(process, grace_timeout):
        return
    if os.name == "nt":
        _taskkill(process.pid, force=True)
    else:
        _signal_group_if_present(process.pid, signal.SIGKILL)
    if not await _wait_until_exit(process, kill_timeout):
        raise AppError("worker.process_termination_timeout")


async def _wait_until_exit(process: ProcessLike, timeout: float) -> bool:
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        return False
    return True


def _signal_group(pid: int, value: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(pid), value)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise AppError("worker.process_termination_failed") from exc


def _signal_group_if_present(pid: int, value: signal.Signals) -> None:
    try:
        _signal_group(pid, value)
    except ProcessLookupError:
        return


def _taskkill(pid: int, *, force: bool) -> None:
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError as exc:
        raise AppError("worker.process_termination_failed") from exc
    if result.returncode not in {0, 128, 255}:
        raise AppError("worker.process_termination_failed")


__all__ = ["ProcessLike", "terminate_process_tree"]
