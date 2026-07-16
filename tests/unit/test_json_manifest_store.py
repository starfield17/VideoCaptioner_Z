from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError


def test_manifest_atomic_write_and_current_reconcile(tmp_path: Path) -> None:
    store = JsonManifestStore(tmp_path / "manifest.json")
    projection = BatchProjection("batch-a", last_event_seq=1, event_ids=frozenset({"event-1"}))
    store.write(projection)
    assert store.reconcile(projection) == "current"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_missing_and_stale_manifest_are_rebuilt(tmp_path: Path) -> None:
    store = JsonManifestStore(tmp_path / "manifest.json")
    first = BatchProjection("batch-a", last_event_seq=1)
    second = BatchProjection("batch-a", last_event_seq=2)
    assert store.reconcile(first) == "rebuilt"
    assert store.reconcile(second) == "rebuilt"
    assert store.read()["last_event_seq"] == 2  # type: ignore[index]  # read is asserted non-None


def test_manifest_ahead_and_same_seq_mismatch_are_rejected(tmp_path: Path) -> None:
    store = JsonManifestStore(tmp_path / "manifest.json")
    projection = BatchProjection("batch-a", last_event_seq=1)
    store.write(BatchProjection("batch-a", last_event_seq=2))
    with pytest.raises(AppError, match="ahead_of_journal"):
        store.reconcile(projection)
    store.write(projection)
    value = store.read()
    assert value is not None
    value["state"] = "succeeded"
    store.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AppError, match="projection_mismatch"):
        store.reconcile(projection)
