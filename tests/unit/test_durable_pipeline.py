from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.application.durable_pipeline import DurablePipelineService, write_cancel_marker
from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageState
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


@dataclass(slots=True)
class FakeStage:
    name: StageName
    counts: dict[StageName, int]
    version: str = "fake-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        del request
        context.execution.raise_if_cancelled()
        self.counts[self.name] = self.counts.get(self.name, 0) + 1
        return (
            ProducedArtifact(
                self.name.value,
                "application/octet-stream",
                f"{self.name.value}.bin",
                data=self.name.value.encode(),
            ),
        )


def _config(tmp_path: Path) -> JobConfig:
    return JobConfig(
        "tiny",
        "faster-whisper:tiny",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"rate": 16000},
        {"limit": 84},
        str((tmp_path / "output").resolve()),
        False,
        {stage.value: "fake-v1" for stage in STAGE_PLAN},
    )


def _service(
    tmp_path: Path, counts: dict[StageName, int], fault: ScriptedFaultInjector | None = None
) -> DurablePipelineService:
    batch_dir = tmp_path / "batch"
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    manifest = JsonManifestStore(batch_dir / "manifest.json")
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    sequence = len(journal.read())

    def next_id() -> str:
        nonlocal sequence
        sequence += 1
        return f"event-{sequence}"

    factory = EventFactory(next_id, lambda: "2026-01-01T00:00:00+00:00")
    executor = StageExecutor(journal, manifest, artifacts, factory, batch_dir / "work")
    if fault is not None:
        executor.fault_injector = fault
    runners: Mapping[StageName, FakeStage] = {
        stage: FakeStage(stage, counts) for stage in STAGE_PLAN
    }
    return DurablePipelineService(
        journal, manifest, executor, factory, runners, batch_dir / "control"
    )


def test_pipeline_runs_all_stages_and_retry_only_suffix(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(tmp_path, counts)
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    projection = asyncio.run(service.run(projection))
    assert projection.job("job-000001").state is JobState.SUCCEEDED
    assert all(counts[stage] == 1 for stage in STAGE_PLAN)
    projection = asyncio.run(service.retry("job-000001", StageName.SEGMENT))
    assert projection.job("job-000001").state is JobState.SUCCEEDED
    assert (
        counts[StageName.INSPECT]
        == counts[StageName.NORMALIZE]
        == counts[StageName.TRANSCRIBE]
        == 1
    )
    assert all(
        counts[stage] == 2 for stage in (StageName.SEGMENT, StageName.EXPORT, StageName.PUBLISH)
    )


def test_after_journal_commit_crash_resumes_without_rerun(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(
        tmp_path, counts, ScriptedFaultInjector("transcribe", "after_journal_commit")
    )
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(service.run(projection))
    recovered = _service(tmp_path, counts)
    projection = asyncio.run(recovered.resume())
    assert projection.job("job-000001").state is JobState.SUCCEEDED
    assert counts[StageName.TRANSCRIBE] == 1


def test_before_commit_crash_records_interrupted_and_reruns(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(
        tmp_path, counts, ScriptedFaultInjector("transcribe", "before_journal_commit")
    )
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(service.run(projection))
    projection = asyncio.run(_service(tmp_path, counts).resume())
    assert projection.job("job-000001").stage(StageName.TRANSCRIBE).attempt == 2
    assert counts[StageName.TRANSCRIBE] == 2


def test_cancel_marker_produces_cancelled_not_failed(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(tmp_path, counts)
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    write_cancel_marker(service.control_dir, job_id="job-000001")
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(service.run(projection))
    status = service.status()
    assert status.job("job-000001").state is JobState.CANCELLED
    assert all(stage.state is StageState.PENDING for stage in status.job("job-000001").stages)
    assert all(event.type not in {"stage.failed", "job.failed"} for event in service.journal.read())
