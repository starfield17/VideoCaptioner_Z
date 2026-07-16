from __future__ import annotations

import asyncio
from pathlib import Path

from tests.recovery.support import config, service

from captioner.core.domain.stage import STAGE_PLAN, StageName


def test_model_change_invalidates_transcribe_suffix(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    projection = current.update_config(
        projection,
        job_id="job-000001",
        config=config(tmp_path, model="small"),
        earliest_stage=StageName.TRANSCRIBE,
    )
    result = asyncio.run(current.run(projection))
    assert result.job("job-000001").stage(StageName.NORMALIZE).attempt == 1
    assert all(result.job("job-000001").stage(stage).attempt == 2 for stage in STAGE_PLAN[2:])


def test_output_change_invalidates_publish_only(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    projection = current.update_config(
        projection,
        job_id="job-000001",
        config=config(tmp_path, output=tmp_path / "other"),
        earliest_stage=StageName.PUBLISH,
    )
    result = asyncio.run(current.run(projection))
    assert all(result.job("job-000001").stage(stage).attempt == 1 for stage in STAGE_PLAN[:-1])
    assert result.job("job-000001").stage(StageName.PUBLISH).attempt == 2


def test_source_content_change_reruns_complete_suffix(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    (tmp_path / "input.wav").write_bytes(b"changed-source")
    result = asyncio.run(current.run(projection))
    assert all(result.job("job-000001").stage(stage).attempt == 2 for stage in STAGE_PLAN)
