"""Append-only durable journal boundary."""

from collections.abc import Sequence
from typing import Protocol

from captioner.core.domain.journal import JournalEvent


class JournalPort(Protocol):
    def read(self) -> tuple[JournalEvent, ...]: ...

    def append(self, event: JournalEvent) -> None:
        """Durably append exactly one event, reconciling uncertain writes."""
        ...

    def append_many(self, events: Sequence[JournalEvent]) -> None: ...
