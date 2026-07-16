from __future__ import annotations

import os
from pathlib import Path

import pytest

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError


def test_interruption_before_replace_preserves_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonManifestStore(tmp_path / "manifest.json")
    store.write(BatchProjection("batch-a", last_event_seq=1))
    previous = store.path.read_bytes()

    def fail_before_replace(
        _source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        _target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        raise OSError

    monkeypatch.setattr(os, "replace", fail_before_replace)
    with pytest.raises(AppError, match=r"manifest\.projection_failed"):
        store.write(BatchProjection("batch-a", last_event_seq=2))
    assert store.path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_interruption_after_replace_leaves_complete_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonManifestStore(tmp_path / "manifest.json")
    store.write(BatchProjection("batch-a", last_event_seq=1))
    real_replace = os.replace

    def replace_then_fail(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        real_replace(source, target)
        raise OSError

    monkeypatch.setattr(os, "replace", replace_then_fail)
    with pytest.raises(AppError, match=r"manifest\.projection_failed"):
        store.write(BatchProjection("batch-a", last_event_seq=2))
    assert store.read()["last_event_seq"] == 2  # type: ignore[index]  # manifest exists after replacement
