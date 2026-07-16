from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import JournalEvent, apply_event, replay
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value
from captioner.core.domain.stage import STAGE_PLAN


@given(attempt=st.integers(min_value=1, max_value=100))
def test_attempts_must_strictly_increase(attempt: int) -> None:
    root = (Path.cwd() / "captioner-property").resolve()
    config = JobConfig(
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
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    base = [
        JournalEvent(1, "event-1", "2026-01-01T00:00:00+00:00", "batch-a", "batch.created", {}),
        JournalEvent(
            2,
            "event-2",
            "2026-01-01T00:00:00+00:00",
            "batch-a",
            "job.created",
            cast(
                dict[str, FrozenJsonValue],
                freeze_json_value(
                    {
                        "job_id": "job-000001",
                        "input_path": str(root / "a.wav"),
                        "config": config.to_dict(),
                    }
                ),
            ),
        ),
        JournalEvent(
            3,
            "event-3",
            "2026-01-01T00:00:00+00:00",
            "batch-a",
            "stage.started",
            {
                "job_id": "job-000001",
                "stage_name": "inspect",
                "attempt": attempt,
            },
        ),
        JournalEvent(
            4,
            "event-4",
            "2026-01-01T00:00:00+00:00",
            "batch-a",
            "stage.interrupted",
            {
                "job_id": "job-000001",
                "stage_name": "inspect",
                "attempt": attempt,
            },
        ),
    ]
    projection = replay(base)
    cancelled = apply_event(
        projection,
        JournalEvent(
            5,
            "event-5",
            "2026-01-01T00:00:00+00:00",
            "batch-a",
            "job.cancelled",
            {"job_id": "job-000001"},
        ),
    )
    assert cancelled.jobs[0].state.value == "cancelled"
    with pytest.raises(AppError, match="attempt_reused"):
        apply_event(
            cancelled,
            JournalEvent(
                6,
                "event-6",
                "2026-01-01T00:00:00+00:00",
                "batch-a",
                "stage.started",
                {"job_id": "job-000001", "stage_name": "inspect", "attempt": attempt},
            ),
        )
