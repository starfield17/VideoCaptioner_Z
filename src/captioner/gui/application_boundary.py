"""GUI consumer Protocol for Application-owned Queue projections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from captioner.core.application.queue_projection import QueueSnapshot


class GuiApplicationBoundary(Protocol):
    def get_queue_snapshot(self) -> QueueSnapshot: ...

    def refresh_queue(self) -> QueueSnapshot: ...

    def subscribe_queue(
        self,
        callback: Callable[[QueueSnapshot], None],
    ) -> Callable[[], None]: ...


__all__ = ["GuiApplicationBoundary"]
