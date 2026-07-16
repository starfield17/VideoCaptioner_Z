from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import StageName


def _events(current: DurablePipelineService) -> tuple[str, ...]:
    return tuple(event.type for event in current.journal.read_snapshot().events)


def test_manifest_failure_after_commit_does_not_fail_or_rerun_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    manifest_type = type(current.manifest)
    real_write = manifest_type.write
    failed = False

    def fail_after_inspect(self: JsonManifestStore, projected: BatchProjection) -> None:
        nonlocal failed
        if not failed and projected.job("job-000001").stage(StageName.INSPECT).artifacts:
            failed = True
            raise AppError("manifest.projection_failed")
        real_write(self, projected)

    monkeypatch.setattr(manifest_type, "write", fail_after_inspect)
    with pytest.raises(AppError, match=r"stage\.post_commit_failed"):
        asyncio.run(current.run(projection))
    assert _events(current).count("stage.committed") == 1
    assert "stage.failed" not in _events(current)
    assert "job.failed" not in _events(current)
    monkeypatch.setattr(manifest_type, "write", real_write)
    recovered = asyncio.run(service(tmp_path, counts).resume())
    assert recovered.job("job-000001").state.value == "succeeded"
    assert counts[StageName.INSPECT] == 1


def test_workspace_cleanup_failure_after_commit_is_post_commit_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    real_rmtree = shutil.rmtree
    failed = False

    def fail_once(path: str | Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        del args, kwargs
        if not failed and "attempt-1" in str(path):
            failed = True
            raise OSError
        real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", fail_once)
    with pytest.raises(AppError, match=r"stage\.post_commit_failed"):
        asyncio.run(current.run(projection))
    assert "stage.failed" not in _events(current)
    assert "job.failed" not in _events(current)
