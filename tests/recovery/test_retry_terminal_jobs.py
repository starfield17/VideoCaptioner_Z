from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.core.application.durable_pipeline import write_cancel_marker
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName


def test_retry_is_a_durable_event_and_reopens_cancelled_job(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    write_cancel_marker(current.control_dir, job_id="job-000001")
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(current.run(projection))
    assert current.status().job("job-000001").state is JobState.CANCELLED
    result = asyncio.run(current.retry("job-000001", StageName.INSPECT))
    assert result.job("job-000001").state is JobState.SUCCEEDED
    events = current.journal.read_snapshot().events
    assert any(event.type == "job.retry_requested" for event in events)
    assert counts[StageName.INSPECT] == 1
    assert all(result.job("job-000001").stage(stage).attempt == 1 for stage in STAGE_PLAN)
