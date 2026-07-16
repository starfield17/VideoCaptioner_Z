from __future__ import annotations

import asyncio
from pathlib import Path

from tests.recovery.support import config, service

from captioner.core.domain.stage import STAGE_PLAN, StageName


def test_retry_publish_does_not_rerun_other_stages(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    asyncio.run(current.run(projection))
    result = asyncio.run(current.retry("job-000001", StageName.PUBLISH))
    assert all(result.job("job-000001").stage(stage).attempt == 1 for stage in STAGE_PLAN[:-1])
    assert result.job("job-000001").stage(StageName.PUBLISH).attempt == 2
