from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName

POINTS = (
    "before_execute",
    "mid_execute",
    "after_artifact_write",
    "before_journal_commit",
    "after_journal_commit",
    "before_manifest_projection",
)


@pytest.mark.parametrize("stage", STAGE_PLAN)
@pytest.mark.parametrize("point", POINTS)
def test_every_stage_fault_point_recovers(stage: StageName, point: str, tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector(stage.value, point))
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    result = asyncio.run(service(tmp_path, counts).resume())
    assert result.job("job-000001").state is JobState.SUCCEEDED
    expected = (
        1
        if point in {"after_journal_commit", "before_manifest_projection"}
        else (1 if point == "before_execute" else 2)
    )
    assert counts[stage] == expected
    for upstream in STAGE_PLAN[: STAGE_PLAN.index(stage)]:
        assert counts[upstream] == 1
