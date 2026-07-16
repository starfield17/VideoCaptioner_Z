from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.adapters.testing.fault_injector import ScriptedFaultInjector
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.domain.job import JobConfig
from captioner.core.domain.stage import STAGE_PLAN, StageName
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
        (context.workspace / "partial-output").write_bytes(self.name.value.encode())
        context.checkpoint("mid_execute")
        return (
            ProducedArtifact(
                self.name.value,
                "application/octet-stream",
                f"{self.name.value}.bin",
                data=self.name.value.encode(),
            ),
        )


def config(root: Path, *, model: str = "tiny", output: Path | None = None) -> JobConfig:
    return JobConfig(
        model,
        f"faster-whisper:{model}",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"rate": 16000},
        {"limit": 84},
        str((output or root / "output").resolve()),
        False,
        {stage.value: "fake-v1" for stage in STAGE_PLAN},
    )


def service(
    root: Path, counts: dict[StageName, int], fault: ScriptedFaultInjector | None = None
) -> DurablePipelineService:
    (root / "input.wav").write_bytes(b"source")
    batch_dir = root / "batch"
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    manifest = JsonManifestStore(batch_dir / "manifest.json")
    artifacts = ContentAddressedArtifactStore(root / "artifacts")
    sequence = len(journal.read_snapshot().events)

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
