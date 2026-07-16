from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName


def _three_jobs(root: Path) -> tuple[tuple[str, Path, JobConfig], ...]:
    jobs: list[tuple[str, Path, JobConfig]] = []
    for index in range(1, 4):
        source = root / f"input-{index}.wav"
        source.write_bytes(b"source")
        jobs.append((f"job-{index:06d}", source, config(root)))
    return tuple(jobs)


def test_batch_config_update_is_one_atomic_event(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    projection = asyncio.run(current.run(projection))
    updated_config = config(tmp_path, model="small")

    projection = current.update_config(
        projection, config=updated_config, earliest_stage=StageName.TRANSCRIBE
    )

    events = current.journal.read_snapshot().events
    assert [event.type for event in events].count("batch.config_updated") == 1
    assert "job.config_updated" not in [event.type for event in events]
    assert {job.config.runtime_signature for job in projection.jobs} == {
        updated_config.runtime_signature
    }
    assert all(
        job.stage(StageName.INSPECT).attempt == 1
        and job.stage(StageName.NORMALIZE).attempt == 1
        and all(job.stage(stage).attempt == 1 for stage in STAGE_PLAN[2:])
        for job in projection.jobs
    )

    result = asyncio.run(current.run(projection))
    assert all(job.state is JobState.SUCCEEDED for job in result.jobs)
    assert all(counts[stage] == (3 if STAGE_PLAN.index(stage) < 2 else 6) for stage in STAGE_PLAN)


def test_batch_config_crash_after_commit_recovers_without_divergence(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    projection = asyncio.run(current.run(projection))
    updated_config = config(tmp_path, model="small")
    current.executor.fault_injector = ScriptedFaultInjector(
        "batch-config", "after_batch_config_commit"
    )

    with pytest.raises(InjectedCrash):
        current.update_config(
            projection, config=updated_config, earliest_stage=StageName.TRANSCRIBE
        )

    recovered = service(tmp_path, counts)
    result = asyncio.run(recovered.resume())
    events = recovered.journal.read_snapshot().events
    assert [event.type for event in events].count("batch.config_updated") == 1
    assert {job.config.runtime_signature for job in result.jobs} == {
        updated_config.runtime_signature
    }
    assert all(job.state is JobState.SUCCEEDED for job in result.jobs)


def test_batch_config_manifest_failure_leaves_durable_common_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    projection = asyncio.run(current.run(projection))
    updated_config = config(tmp_path, model="small")
    real_write = JsonManifestStore.write

    def fail_config_projection(self: JsonManifestStore, projected: BatchProjection) -> None:
        if projected.jobs[0].config.model_ref == "small":
            raise AppError("manifest.projection_failed")
        real_write(self, projected)

    monkeypatch.setattr(JsonManifestStore, "write", fail_config_projection)
    with pytest.raises(AppError, match=r"manifest\.projection_failed"):
        current.update_config(
            projection, config=updated_config, earliest_stage=StageName.TRANSCRIBE
        )

    monkeypatch.setattr(JsonManifestStore, "write", real_write)
    result = asyncio.run(service(tmp_path, counts).resume())
    events = current.journal.read_snapshot().events
    assert [event.type for event in events].count("batch.config_updated") == 1
    assert {job.config.runtime_signature for job in result.jobs} == {
        updated_config.runtime_signature
    }


def test_output_config_update_invalidates_publish_only(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create("batch-a", _three_jobs(tmp_path))
    projection = asyncio.run(current.run(projection))
    updated_config = replace(
        projection.jobs[0].config,
        output_dir=str((tmp_path / "new-output").resolve()),
    )

    updated = current.update_config(
        projection, config=updated_config, earliest_stage=StageName.PUBLISH
    )

    assert all(
        job.stage(stage).state.value
        == ("invalidated" if stage is StageName.PUBLISH else "committed")
        for job in updated.jobs
        for stage in STAGE_PLAN
    )
    result = asyncio.run(current.run(updated))
    assert all(job.state is JobState.SUCCEEDED for job in result.jobs)
    assert all(
        counts[stage] == (3 if stage is not StageName.PUBLISH else 6) for stage in STAGE_PLAN
    )
