from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName, stage_plan_for

POINTS = (
    "before_execute",
    "mid_execute",
    "after_artifact_write",
    "before_journal_commit",
    "after_journal_commit",
    "before_manifest_projection",
)


@pytest.mark.parametrize("profile", tuple(PipelineProfile))
@pytest.mark.parametrize("stage", tuple(StageName))
@pytest.mark.parametrize("point", POINTS)
def test_every_stage_fault_point_recovers(
    profile: PipelineProfile, stage: StageName, point: str, tmp_path: Path
) -> None:
    plan = stage_plan_for(profile)
    if stage not in plan:
        pytest.skip("stage is not part of this profile")
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector(stage.value, point), profile)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path, profile=profile)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    result = asyncio.run(service(tmp_path, counts, profile=profile).resume())
    assert result.job("job-000001").state is JobState.SUCCEEDED
    expected = (
        1
        if point in {"after_journal_commit", "before_manifest_projection"}
        else (1 if point == "before_execute" else 2)
    )
    assert counts[stage] == expected
    for upstream in plan[: plan.index(stage)]:
        assert counts[upstream] == 1
