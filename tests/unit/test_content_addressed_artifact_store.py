from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.core.domain.errors import AppError


def test_put_deduplicates_and_verifies(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes(b"same", kind="test", media_type="text/plain", logical_name="a.txt")
    second = store.put_bytes(b"same", kind="test", media_type="text/plain", logical_name="b.txt")
    assert first.sha256 == second.sha256
    assert store.read_bytes(first) == b"same"
    assert len(list((store.root / "sha256").rglob(first.sha256))) == 1
    assert not list((store.root / ".incoming").iterdir())


def test_put_file_and_materialize_overwrite_policy(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    ref = store.put_file(
        source, kind="binary", media_type="application/octet-stream", logical_name="a.bin"
    )
    target = tmp_path / "output" / "a.bin"
    store.materialize(ref, target, overwrite=False)
    with pytest.raises(AppError, match=r"output\.exists"):
        store.materialize(ref, target, overwrite=False)
    target.write_bytes(b"old")
    store.materialize(ref, target, overwrite=True)
    assert target.read_bytes() == b"content"


def test_existing_content_path_with_wrong_bytes_is_corrupt(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"right", kind="test", media_type="text/plain", logical_name="a.txt")
    store.resolve(ref).write_bytes(b"wrong")
    with pytest.raises(AppError, match=r"artifact\.corrupt"):
        store.verify(ref)
