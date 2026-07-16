from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.core.application.durable_pipeline import write_cancel_marker
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import StageName
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


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


@dataclass(slots=True)
class CancelAtCheckpoint:
    name: StageName
    control_dir: Path
    job_id: str
    batch: bool
    version: str = "fake-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        if request.job_id == self.job_id:
            write_cancel_marker(
                self.control_dir,
                job_id=None if self.batch else request.job_id,
            )
        context.checkpoint("mid_execute")
        return (
            ProducedArtifact(
                self.name.value,
                "application/octet-stream",
                f"{self.name.value}.bin",
                data=self.name.value.encode(),
            ),
        )


@pytest.mark.parametrize("stage", [StageName.NORMALIZE, StageName.TRANSCRIBE])
def test_job_cancel_during_active_stage_clears_only_job_marker(
    tmp_path: Path, stage: StageName
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    current.runners = {
        **current.runners,
        stage: CancelAtCheckpoint(stage, current.control_dir, "job-000002", False),
    }

    result = asyncio.run(current.run(projection))

    assert result.job("job-000001").state is JobState.SUCCEEDED
    assert result.job("job-000002").state is JobState.CANCELLED
    assert result.job("job-000003").state is JobState.SUCCEEDED
    assert not (current.control_dir / "cancel-job-000002").exists()
    assert not (current.control_dir / "cancel-batch").exists()
    assert all(
        event.type not in {"stage.failed", "job.failed"}
        for event in current.journal.read_snapshot().events
    )


@pytest.mark.parametrize("stage", [StageName.NORMALIZE, StageName.TRANSCRIBE])
def test_batch_cancel_during_active_stage_cancels_all_remaining_jobs(
    tmp_path: Path, stage: StageName
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    current.runners = {
        **current.runners,
        stage: CancelAtCheckpoint(stage, current.control_dir, "job-000001", True),
    }

    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(current.run(projection))

    result = current.read_status().projection
    assert all(job.state is JobState.CANCELLED for job in result.jobs)
    assert not (current.control_dir / "cancel-batch").exists()
    assert all(
        event.type not in {"stage.failed", "job.failed"}
        for event in current.journal.read_snapshot().events
    )


def test_batch_cancel_accepts_interrupted_job(tmp_path: Path) -> None:
    from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector

    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector("normalize", "before_execute"))
    projection = current.create("batch-a", _three_jobs(tmp_path))
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    write_cancel_marker(current.control_dir, job_id=None)

    result = asyncio.run(service(tmp_path, counts).resume())

    assert all(job.state is JobState.CANCELLED for job in result.jobs)
    assert result.job("job-000001").stage(StageName.NORMALIZE).state.value == "interrupted"
    assert not (current.control_dir / "cancel-batch").exists()


def test_batch_cancel_projection_failure_keeps_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    write_cancel_marker(current.control_dir, job_id=None)
    manifest_type = type(current.manifest)
    real_write = manifest_type.write

    def fail_cancel_projection(self: JsonManifestStore, projected: BatchProjection) -> None:
        if projected.jobs and projected.jobs[0].state is JobState.CANCELLED:
            raise AppError("manifest.projection_failed")
        real_write(self, projected)

    monkeypatch.setattr(manifest_type, "write", fail_cancel_projection)
    with pytest.raises(AppError, match=r"manifest\.projection_failed"):
        asyncio.run(current.run(projection))
    assert (current.control_dir / "cancel-batch").exists()
    assert all(
        event.type not in {"stage.failed", "job.failed"}
        for event in current.journal.read_snapshot().events
    )

    monkeypatch.setattr(manifest_type, "write", real_write)
    result = asyncio.run(service(tmp_path, counts).resume())
    assert all(job.state is JobState.CANCELLED for job in result.jobs)
    assert not (current.control_dir / "cancel-batch").exists()
