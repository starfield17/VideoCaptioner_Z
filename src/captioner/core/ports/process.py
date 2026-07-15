"""Argument-array process execution boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from captioner.core.domain.execution import ExecutionContext


@dataclass(frozen=True, slots=True)
class ProcessResult:
    stdout: bytes
    stderr: bytes
    returncode: int


class ProcessPort(Protocol):
    async def run(self, arguments: Sequence[str], context: ExecutionContext) -> ProcessResult:
        """Run one executable without a shell."""
        ...
