from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import STAGE_PLAN, StageName


def test_divergent_runtime_configs_are_rejected_before_journal_write(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    with pytest.raises(AppError, match=r"batch\.config_inconsistent"):
        current.create(
            "batch-a",
            (
                ("job-000001", tmp_path / "input-1.wav", config(tmp_path, model="tiny")),
                ("job-000002", tmp_path / "input-2.wav", config(tmp_path, model="small")),
            ),
        )
    assert isinstance(current.journal, JsonlJournal)
    assert not current.journal.path.exists()


def test_model_change_updates_every_job_and_suffix_only(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    for index in range(1, 4):
        (tmp_path / f"input-{index}.wav").write_bytes(b"source")
    projection = current.create(
        "batch-a",
        (
            ("job-000001", tmp_path / "input-1.wav", config(tmp_path / "one")),
            ("job-000002", tmp_path / "input-2.wav", config(tmp_path / "two")),
            ("job-000003", tmp_path / "input-3.wav", config(tmp_path / "three")),
        ),
    )
    projection = asyncio.run(current.run(projection))
    for job in projection.jobs:
        projection = current.update_config(
            projection,
            job_id=job.job_id,
            config=config(tmp_path, model="small"),
            earliest_stage=StageName.TRANSCRIBE,
        )
    result = asyncio.run(current.run(projection))
    for job in result.jobs:
        assert job.stage(StageName.NORMALIZE).attempt == 1
        assert all(job.stage(stage).attempt == 2 for stage in STAGE_PLAN[2:])
