"""Repairable, append-only, fsynced JSONL Journal."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.journal import JournalEvent
from captioner.core.ports.journal import JournalSnapshot

MAX_EVENT_LINE_BYTES = 1024 * 1024


def canonical_event_bytes(event: JournalEvent) -> bytes:
    encoded = (
        json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > MAX_EVENT_LINE_BYTES:
        raise AppError("journal.event_too_large", {"size_bytes": len(encoded)})
    return encoded


@dataclass(frozen=True, slots=True)
class JsonlJournal:
    path: Path

    def read_snapshot(self) -> JournalSnapshot:
        if not self.path.exists():
            return JournalSnapshot((), "clean")
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise AppError("journal.read_failed", {"path": str(self.path)}) from exc
        tail_status = "clean" if not data or data.endswith(b"\n") else "incomplete"
        complete = data if tail_status == "clean" else data[: data.rfind(b"\n") + 1]
        events = self._parse_complete_lines(complete)
        return JournalSnapshot(events, tail_status)

    def repair_and_read(self) -> tuple[JournalEvent, ...]:
        if not self.path.exists():
            return ()
        self._repair_tail()
        return self.read_snapshot().events

    def _parse_complete_lines(self, data: bytes) -> tuple[JournalEvent, ...]:
        raw_lines = data.splitlines(keepends=True)
        events: list[JournalEvent] = []
        for line_number, line in enumerate(raw_lines, start=1):
            if len(line) > MAX_EVENT_LINE_BYTES:
                raise AppError("journal.corrupt", {"reason": "line_too_large", "line": line_number})
            try:
                decoded = line[:-1].decode("utf-8")
                value = cast(object, json.loads(decoded))
                event = JournalEvent.from_dict(value)
            except (UnicodeDecodeError, json.JSONDecodeError, AppError) as exc:
                raise AppError(
                    "journal.corrupt", {"reason": "complete_line", "line": line_number}
                ) from exc
            if event.seq != line_number:
                raise AppError("journal.corrupt", {"reason": "sequence", "line": line_number})
            if events and event.batch_id != events[0].batch_id:
                raise AppError("journal.corrupt", {"reason": "batch_identity", "line": line_number})
            if any(previous.event_id == event.event_id for previous in events):
                raise AppError(
                    "journal.corrupt", {"reason": "duplicate_event_id", "line": line_number}
                )
            events.append(event)
        return tuple(events)

    def append(self, event: JournalEvent) -> None:
        events = self.read_snapshot().events
        if event.seq != len(events) + 1:
            raise AppError("journal.append_failed", {"reason": "sequence"})
        if events and event.batch_id != events[0].batch_id:
            raise AppError("journal.append_failed", {"reason": "batch_identity"})
        encoded = canonical_event_bytes(event)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            if self._event_is_durable(event):
                return
            raise AppError("journal.append_failed", {"seq": event.seq}) from exc

    def append_many(self, events: Sequence[JournalEvent]) -> None:
        for event in events:
            self.append(event)

    def _event_is_durable(self, expected: JournalEvent) -> bool:
        snapshot = self.read_snapshot()
        if snapshot.tail_status == "incomplete":
            return False
        events = snapshot.events
        matching = [event for event in events if event.event_id == expected.event_id]
        if not matching:
            return False
        if len(matching) != 1 or matching[0].seq != expected.seq or matching[0] != expected:
            raise AppError("journal.corrupt", {"reason": "event_identity_conflict"})
        return True

    def _repair_tail(self) -> None:
        try:
            data = self.path.read_bytes()
            if not data or data.endswith(b"\n"):
                return
            final_newline = data.rfind(b"\n")
            keep = 0 if final_newline < 0 else final_newline + 1
            with self.path.open("r+b") as handle:
                handle.truncate(keep)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise AppError("journal.repair_failed", {"path": str(self.path)}) from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
