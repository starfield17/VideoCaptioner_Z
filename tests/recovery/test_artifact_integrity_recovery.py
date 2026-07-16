from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import StageName


def test_missing_committed_artifact_is_reported_then_invalidated(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    ref = projection.job("job-000001").stage(StageName.TRANSCRIBE).artifacts[0]
    current.executor.artifact_store.resolve(ref).unlink()
    with pytest.raises(AppError, match=r"artifact\.missing"):
        current.status()
    result = asyncio.run(service(tmp_path, counts).resume())
    assert result.job("job-000001").stage(StageName.INSPECT).attempt == 1
    assert result.job("job-000001").stage(StageName.NORMALIZE).attempt == 1
    assert result.job("job-000001").stage(StageName.TRANSCRIBE).attempt == 2


def test_corrupt_committed_artifact_is_invalidated(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    ref = projection.job("job-000001").stage(StageName.SEGMENT).artifacts[0]
    current.executor.artifact_store.resolve(ref).write_bytes(b"corrupt")
    result = asyncio.run(service(tmp_path, counts).resume())
    assert result.job("job-000001").stage(StageName.TRANSCRIBE).attempt == 1
    assert result.job("job-000001").stage(StageName.SEGMENT).attempt == 2
