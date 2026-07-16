from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.core.application.durable_pipeline import write_cancel_marker
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import StageName


def test_cancelled_job_is_not_resumed(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    write_cancel_marker(current.control_dir, job_id="job-000001")
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(current.run(projection))
    result = asyncio.run(service(tmp_path, counts).resume())
    assert result.job("job-000001").state is JobState.CANCELLED
    assert counts == {}
