"""Cancellation primitives shared by all Phase 1 boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event

from captioner.core.domain.errors import AppError


class CancellationToken:
    """A small thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise AppError("operation.cancelled")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Context passed through one cancellable operation."""

    cancellation: CancellationToken = field(default_factory=CancellationToken)
    checkpoint_hook: Callable[[str], None] | None = field(default=None, repr=False, compare=False)

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation.is_cancelled

    def cancel(self) -> None:
        self.cancellation.cancel()

    def raise_if_cancelled(self) -> None:
        self.cancellation.raise_if_cancelled()

    async def wait_cancelled(self, poll_interval_sec: float = 0.02) -> None:
        """Wait cooperatively until the thread-safe cancellation marker is set."""
        if poll_interval_sec <= 0:
            raise ValueError
        while not self.is_cancelled:
            await asyncio.sleep(poll_interval_sec)

    def checkpoint(self, point: str) -> None:
        self.raise_if_cancelled()
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(point)
