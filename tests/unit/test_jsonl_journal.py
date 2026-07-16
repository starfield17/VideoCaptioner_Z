from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from captioner.adapters.persistence.jsonl_journal import JsonlJournal, canonical_event_bytes
from captioner.core.domain.errors import AppError
from captioner.core.domain.journal import JournalEvent


def _event(seq: int = 1) -> JournalEvent:
    return JournalEvent(
        seq,
        f"event-{seq}",
        "2026-01-01T00:00:00+00:00",
        "batch-a",
        "batch.created" if seq == 1 else "job.cancelled",
        {} if seq == 1 else {"job_id": "job-000001"},
    )


def test_empty_and_complete_journal(tmp_path: Path) -> None:
    journal = JsonlJournal(tmp_path / "journal.jsonl")
    assert journal.read_snapshot().events == ()
    journal.append(_event())
    assert journal.read_snapshot().events == (_event(),)
    assert journal.path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "fragment",
    [b'{"partial":', canonical_event_bytes(_event()).rstrip(b"\n")],
)
def test_unterminated_tail_is_truncated(tmp_path: Path, fragment: bytes) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(canonical_event_bytes(_event()) + fragment)
    assert JsonlJournal(path).repair_and_read() == (_event(),)
    assert path.read_bytes() == canonical_event_bytes(_event())


@pytest.mark.parametrize(
    "content",
    [
        b"not-json\n",
        canonical_event_bytes(_event()) + b"not-json\n",
        b"\xff\n",
    ],
)
def test_complete_corrupt_line_is_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(content)
    with pytest.raises(AppError, match=r"journal\.corrupt"):
        JsonlJournal(path).repair_and_read()


def test_missing_and_duplicate_sequence_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    second = _event(2).to_dict()
    path.write_bytes((json.dumps(second) + "\n").encode())
    with pytest.raises(AppError, match=r"journal\.corrupt"):
        JsonlJournal(path).repair_and_read()
    path.write_bytes(canonical_event_bytes(_event()) * 2)
    with pytest.raises(AppError, match=r"journal\.corrupt"):
        JsonlJournal(path).repair_and_read()


def test_event_codec_is_canonical_and_round_trips() -> None:
    encoded = canonical_event_bytes(_event())
    assert b" " not in encoded
    assert JournalEvent.from_dict(json.loads(encoded)) == _event()


def test_uncertain_append_is_reconciled_by_event_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = JsonlJournal(tmp_path / "journal.jsonl")
    real_fsync = os.fsync
    calls = 0

    def fsync_then_fail(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        real_fsync(descriptor)
        if calls == 1:
            raise OSError

    monkeypatch.setattr(os, "fsync", fsync_then_fail)
    journal.append(_event())
    assert journal.read_snapshot().events == (_event(),)


def test_snapshot_reports_incomplete_tail_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    original = canonical_event_bytes(_event()) + b'{"partial":'
    path.write_bytes(original)
    snapshot = JsonlJournal(path).read_snapshot()
    assert snapshot.events == (_event(),)
    assert snapshot.tail_status == "incomplete"
    assert path.read_bytes() == original
