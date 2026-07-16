from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.core.domain.stage import STAGE_PLAN, StageName
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


@dataclass(slots=True)
class MultiArtifactStage:
    name: StageName
    outputs: tuple[tuple[str, bytes], ...]
    version: str = "fake-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        del request
        context.execution.raise_if_cancelled()
        return tuple(
            ProducedArtifact(self.name.value, "application/octet-stream", name, data=data)
            for name, data in self.outputs
        )


@pytest.mark.parametrize("bad_index", [0, 1])
@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_normalize_second_artifact_corruption_removes_exact_blob(
    tmp_path: Path, bad_index: int, corruption: str
) -> None:
    _run_and_recover_multi_artifact(
        tmp_path,
        StageName.NORMALIZE,
        ("normalized.wav", "normalized-audio.json"),
        bad_index,
        corruption,
    )


@pytest.mark.parametrize("bad_index", [0, 1])
@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_export_second_artifact_corruption_removes_exact_blob(
    tmp_path: Path, bad_index: int, corruption: str
) -> None:
    _run_and_recover_multi_artifact(
        tmp_path,
        StageName.EXPORT,
        ("final-transcript.json", "final-subtitle.srt"),
        bad_index,
        corruption,
    )


def _run_and_recover_multi_artifact(
    tmp_path: Path,
    stage: StageName,
    logical_names: tuple[str, str],
    bad_index: int,
    corruption: str,
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    current.runners = {
        **current.runners,
        stage: MultiArtifactStage(
            stage,
            tuple((name, f"{name}-healthy".encode()) for name in logical_names),
        ),
    }
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    refs = projection.job("job-000001").stage(stage).artifacts
    bad_ref = refs[bad_index]
    healthy_ref = refs[1 - bad_index]
    bad_path = current.executor.artifact_store.resolve(bad_ref)
    healthy_path = current.executor.artifact_store.resolve(healthy_ref)
    if corruption == "missing":
        bad_path.unlink()
    else:
        bad_path.write_bytes(b"corrupt")

    result = asyncio.run(service(tmp_path, counts).resume())

    assert result.job("job-000001").stage(stage).attempt == 2
    assert all(
        result.job("job-000001").stage(name).attempt
        == (1 if STAGE_PLAN.index(name) < STAGE_PLAN.index(stage) else 2)
        for name in STAGE_PLAN
    )
    assert healthy_path.is_file()
    current.executor.artifact_store.verify(healthy_ref)
    assert not bad_path.exists()
