from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.persistence.jsonl_journal import JsonlJournal, canonical_event_bytes
from captioner.core.domain.errors import AppError
from captioner.core.domain.journal import JournalEvent


def _event() -> JournalEvent:
    return JournalEvent(1, "event-1", "2026-01-01T00:00:00+00:00", "batch-a", "batch.created", {})


@pytest.mark.parametrize("tail", [b"{", canonical_event_bytes(_event()).rstrip(b"\n")])
def test_only_unterminated_tail_is_repaired(tmp_path: Path, tail: bytes) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(canonical_event_bytes(_event()) + tail)
    assert JsonlJournal(path).repair_and_read() == (_event(),)


@pytest.mark.parametrize("bad", [b"bad\n", b"\xff\n"])
def test_complete_bad_line_is_never_repaired(tmp_path: Path, bad: bytes) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(canonical_event_bytes(_event()) + bad)
    with pytest.raises(AppError, match=r"journal\.corrupt"):
        JsonlJournal(path).repair_and_read()
