"""Append-only durable journal boundary."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from captioner.core.domain.journal import JournalEvent


@dataclass(frozen=True, slots=True)
class JournalSnapshot:
    events: tuple[JournalEvent, ...]
    tail_status: Literal["clean", "incomplete"]


class JournalPort(Protocol):
    def read_snapshot(self) -> JournalSnapshot: ...

    def repair_and_read(self) -> tuple[JournalEvent, ...]: ...

    def append(self, event: JournalEvent) -> None:
        """Durably append exactly one event, reconciling uncertain writes."""
        ...

    def append_many(self, events: Sequence[JournalEvent]) -> None: ...
