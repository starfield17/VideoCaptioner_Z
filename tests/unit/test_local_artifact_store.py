from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.core.domain.errors import AppError


def test_local_artifact_store_writes_atomically_and_requires_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    store = LocalArtifactStore(root)
    target = store.write_bytes("nested/字幕.srt", "你好\n".encode())
    assert target.read_text(encoding="utf-8") == "你好\n"
    with pytest.raises(AppError, match=r"output\.exists"):
        store.write_bytes("nested/字幕.srt", b"new")
    store.write_bytes("nested/字幕.srt", b"new", overwrite=True)
    assert store.read_bytes("nested/字幕.srt") == b"new"
    assert not list(target.parent.glob(".*.tmp"))
    store.delete("nested/字幕.srt")
    assert not store.exists("nested/字幕.srt")


@pytest.mark.parametrize("key", ["../escape", "/absolute", "C:\\absolute", "nested/../../escape"])
def test_local_artifact_store_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(AppError, match="path_invalid"):
        LocalArtifactStore(root).write_bytes(key, b"data")
