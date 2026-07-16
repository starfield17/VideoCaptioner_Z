from __future__ import annotations

from pathlib import Path

import pytest

from captioner.cli.commands import batch
from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import resolve_app_paths


def test_same_stem_inputs_are_rejected_before_batch_creation(tmp_path: Path) -> None:
    first = tmp_path / "one" / "news.wav"
    second = tmp_path / "two" / "news.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    options = batch.BatchRunOptions(
        (first, second),
        tmp_path / "output",
        "tiny",
        "cpu",
        "int8",
        "en",
        "ffmpeg",
        "ffprobe",
        True,
    )
    with pytest.raises(AppError, match=r"batch\.output_collision"):
        batch.run(options, paths=paths)
    assert not paths.batches_dir.exists() or not list(paths.batches_dir.iterdir())
