"""Unit tests for filesystem input discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.persistence.filesystem_input_discovery import (
    FilesystemInputDiscovery,
)
from captioner.core.application.input_selection import InputSelectionRequest


@pytest.fixture
def discovery() -> FilesystemInputDiscovery:
    return FilesystemInputDiscovery()


def test_explicit_file_and_case_insensitive_extension(
    tmp_path: Path, discovery: FilesystemInputDiscovery
) -> None:
    media = tmp_path / "clip.MP4"
    media.write_bytes(b"")
    preview = discovery.preview(InputSelectionRequest(entries=(str(media),), recursive=True))
    assert preview.accepted_paths == (str(media.resolve()),)
    assert preview.rejected == ()


def test_folder_direct_and_recursive_scan(
    tmp_path: Path, discovery: FilesystemInputDiscovery
) -> None:
    root = tmp_path / "media"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "b.wav").write_bytes(b"")
    (root / "a.mp3").write_bytes(b"")
    (nested / "c.flac").write_bytes(b"")
    (root / "notes.txt").write_text("x", encoding="utf-8")

    direct = discovery.preview(InputSelectionRequest(entries=(str(root),), recursive=False))
    assert [Path(p).name for p in direct.accepted_paths] == ["a.mp3", "b.wav"]
    assert any(item.code == "input.unsupported" for item in direct.rejected)

    recursive = discovery.preview(InputSelectionRequest(entries=(str(root),), recursive=True))
    names = [Path(p).name for p in recursive.accepted_paths]
    assert names == ["a.mp3", "b.wav", "c.flac"]


def test_missing_unsupported_and_duplicates(
    tmp_path: Path, discovery: FilesystemInputDiscovery
) -> None:
    media = tmp_path / "a.wav"
    media.write_bytes(b"")
    text = tmp_path / "notes.txt"
    text.write_text("x", encoding="utf-8")
    missing = tmp_path / "gone.mp4"
    preview = discovery.preview(
        InputSelectionRequest(
            entries=(str(media), str(media), str(text), str(missing), str(tmp_path)),
            recursive=False,
        )
    )
    resolved = str(media.resolve())
    assert preview.accepted_paths.count(resolved) == 3
    codes = {item.code for item in preview.rejected}
    assert "input.unsupported" in codes
    assert "input.not_found" in codes


def test_result_limit(tmp_path: Path, discovery: FilesystemInputDiscovery) -> None:
    for index in range(5):
        (tmp_path / f"{index}.wav").write_bytes(b"")
    preview = discovery.preview(
        InputSelectionRequest(entries=(str(tmp_path),), recursive=False, maximum_results=2)
    )
    assert len(preview.accepted_paths) == 2
    assert any(item.code == "input.result_limit" for item in preview.rejected)


def test_directory_symlink_not_followed(
    tmp_path: Path, discovery: FilesystemInputDiscovery
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "hidden.wav").write_bytes(b"")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    preview = discovery.preview(InputSelectionRequest(entries=(str(link),), recursive=True))
    assert preview.accepted_paths == ()
    assert any(item.code == "input.unsupported" for item in preview.rejected)


def test_no_filesystem_mutation_and_no_ffprobe(
    tmp_path: Path, discovery: FilesystemInputDiscovery, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "a.wav"
    media.write_bytes(b"")
    before = {path.name for path in tmp_path.iterdir()}

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr("subprocess.Popen", boom)
    discovery.preview(InputSelectionRequest(entries=(str(media),)))
    after = {path.name for path in tmp_path.iterdir()}
    assert before == after


def test_unreadable_directory(
    tmp_path: Path, discovery: FilesystemInputDiscovery, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()

    original_iterdir = Path.iterdir

    def broken_iterdir(self: Path):  # type: ignore[no-untyped-def]
        if self == folder:
            raise OSError
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", broken_iterdir)
    preview = discovery.preview(InputSelectionRequest(entries=(str(folder),), recursive=False))
    assert preview.accepted_paths == ()
    assert any(item.code == "input.directory_unreadable" for item in preview.rejected)
