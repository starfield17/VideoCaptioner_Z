"""Pause marker behavior on DurablePipelineService."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.core.application.durable_pipeline import (
    DurablePipelineService,
    clear_pause_marker,
    write_pause_marker,
)
from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.domain.job import JobConfig, JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageState
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


@dataclass(slots=True)
class CountingStage:
    name: StageName
    counts: dict[StageName, int]
    pause_after: StageName | None = None
    control_dir: Path | None = None
    version: str = "fake-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        del request
        context.execution.raise_if_cancelled()
        self.counts[self.name] = self.counts.get(self.name, 0) + 1
        if self.pause_after is self.name and self.control_dir is not None:
            write_pause_marker(self.control_dir)
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
    tmp_path: Path,
    counts: dict[StageName, int],
    *,
    pause_after: StageName | None = None,
) -> DurablePipelineService:
    (tmp_path / "input.wav").write_bytes(b"source")
    batch_dir = tmp_path / "batch"
    control = batch_dir / "control"
    control.mkdir(parents=True, exist_ok=True)
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    manifest = JsonManifestStore(batch_dir / "manifest.json")
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    sequence = 0

    def next_id() -> str:
        nonlocal sequence
        sequence += 1
        return f"event-{sequence}"

    factory = EventFactory(next_id, lambda: "2026-01-01T00:00:00+00:00")
    executor = StageExecutor(journal, manifest, artifacts, factory, batch_dir / "work")
    runners: Mapping[StageName, CountingStage] = {
        stage: CountingStage(stage, counts, pause_after=pause_after, control_dir=control)
        for stage in STAGE_PLAN
    }
    return DurablePipelineService(journal, manifest, executor, factory, runners, control)


def test_pause_before_first_stage(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(tmp_path, counts)
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    write_pause_marker(service.control_dir)
    projection = asyncio.run(service.run(projection))
    assert projection.job("job-000001").state is JobState.PENDING
    assert counts == {}
    assert (service.control_dir / "pause-batch").exists()


def test_pause_during_stage_commits_then_stops(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(tmp_path, counts, pause_after=StageName.INSPECT)
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    projection = asyncio.run(service.run(projection))
    assert counts.get(StageName.INSPECT) == 1
    assert counts.get(StageName.NORMALIZE) is None
    assert projection.job("job-000001").stage(StageName.INSPECT).state is StageState.COMMITTED
    assert (service.control_dir / "pause-batch").exists()


def test_resume_clears_marker_and_continues(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    service = _service(tmp_path, counts, pause_after=StageName.INSPECT)
    projection = service.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", _config(tmp_path)),)
    )
    asyncio.run(service.run(projection))
    assert (service.control_dir / "pause-batch").exists()
    # Clear pause and continue with the same service so event IDs remain unique.
    clear_pause_marker(service.control_dir)
    # Re-write pause then resume to prove resume clears the marker.
    write_pause_marker(service.control_dir)
    # Manually clear via resume path by calling resume after removing the
    # pause-after side effect runners.
    service.runners = {
        stage: CountingStage(stage, counts, control_dir=service.control_dir) for stage in STAGE_PLAN
    }
    projection = asyncio.run(service.resume())
    assert not (service.control_dir / "pause-batch").exists()
    assert projection.job("job-000001").state is JobState.SUCCEEDED
    assert counts[StageName.INSPECT] == 1
    assert counts[StageName.NORMALIZE] == 1
