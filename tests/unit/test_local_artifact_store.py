from __future__ import annotations

from pathlib import Path

import pytest

import captioner.adapters.persistence.local_artifact_store as local_store_module
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


def test_staged_artifact_is_single_use_and_discard_removes_temp(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    store = LocalArtifactStore(root)
    staged = store.stage_bytes("result.srt", b"data")
    assert list(root.glob(".*.tmp"))
    assert staged.commit(overwrite=False) == root / "result.srt"
    with pytest.raises(AppError, match=r"output\.stage_invalid"):
        staged.commit(overwrite=True)
    staged.discard()

    discarded = store.stage_bytes("discarded.srt", b"data")
    discarded.discard()
    with pytest.raises(AppError, match=r"output\.stage_invalid"):
        discarded.commit(overwrite=False)
    assert not (root / "discarded.srt").exists()
    assert not list(root.glob(".*.tmp"))


def test_local_store_does_not_follow_symlink_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    (root / "escape.txt").symlink_to(outside)
    with pytest.raises(AppError, match=r"output\.path_invalid"):
        LocalArtifactStore(root).stage_bytes("escape.txt", b"new")


def test_keyboard_interrupt_during_staging_runs_finally_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "output"
    root.mkdir()

    def interrupt(_descriptor: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(local_store_module.os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt):
        LocalArtifactStore(root).stage_bytes("interrupted.srt", b"data")
    assert not list(root.glob(".*.tmp"))


@pytest.mark.parametrize("key", ["../escape", "/absolute", "C:\\absolute", "nested/../../escape"])
def test_local_artifact_store_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(AppError, match="path_invalid"):
        LocalArtifactStore(root).write_bytes(key, b"data")
