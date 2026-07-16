from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.core.application.durable_pipeline import write_cancel_marker
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import StageName


def _three_jobs(root: Path):
    return tuple(
        (f"job-{index:06d}", _input(root, index), config(root / f"input-{index}"))
        for index in range(1, 4)
    )


def _input(root: Path, index: int) -> Path:
    path = root / f"input-{index}.wav"
    path.write_bytes(b"source")
    return path


def test_batch_cancel_before_first_job_cancels_all_nonterminal_jobs(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    write_cancel_marker(current.control_dir, job_id=None)
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(current.run(projection))
    result = current.status()
    assert all(job.state is JobState.CANCELLED for job in result.jobs)
    assert all(event.type != "job.failed" for event in current.journal.read_snapshot().events)
    assert not (current.control_dir / "cancel-batch").exists()


def test_batch_cancel_between_jobs_preserves_success_and_cancels_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    manifest_type = type(current.manifest)
    real_write = manifest_type.write
    marker_written = False

    def write_and_cancel(self: JsonManifestStore, projected: BatchProjection) -> None:
        nonlocal marker_written
        real_write(self, projected)
        if not marker_written and projected.job("job-000001").state is JobState.SUCCEEDED:
            marker_written = True
            write_cancel_marker(current.control_dir, job_id=None)

    monkeypatch.setattr(manifest_type, "write", write_and_cancel)
    result = asyncio.run(current.run(projection))
    assert result.job("job-000001").state is JobState.SUCCEEDED
    assert result.job("job-000002").state is JobState.CANCELLED
    assert result.job("job-000003").state is JobState.CANCELLED
    assert all(event.type != "job.failed" for event in current.journal.read_snapshot().events)


def test_job_cancel_marker_does_not_cancel_unrelated_jobs(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    write_cancel_marker(current.control_dir, job_id="job-000002")
    result = asyncio.run(current.run(projection))
    assert result.job("job-000001").state is JobState.SUCCEEDED
    assert result.job("job-000002").state is JobState.CANCELLED
    assert result.job("job-000003").state is JobState.SUCCEEDED
