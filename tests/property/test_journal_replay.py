from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import JournalEvent, apply_event, replay
from captioner.core.domain.stage import STAGE_PLAN


def _event(seq: int, event_type: str, payload: dict[str, object]) -> JournalEvent:
    return JournalEvent(
        seq,
        f"event-{seq}",
        "2026-01-01T00:00:00+00:00",
        "batch-a",
        event_type,
        payload,  # type: ignore[arg-type]  # generated values are JSON-compatible
    )


@given(job_count=st.integers(min_value=1, max_value=20))
def test_sequential_application_matches_complete_replay(job_count: int) -> None:
    root = "/tmp/captioner-property"
    config = JobConfig(
        "tiny",
        "faster-whisper:tiny",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        root,
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    events = [_event(1, "batch.created", {})]
    for number in range(1, job_count + 1):
        events.append(
            _event(
                number + 1,
                "job.created",
                {
                    "job_id": f"job-{number:06d}",
                    "input_path": f"{root}/{number}.wav",
                    "config": config.to_dict(),
                },
            )
        )
    sequential = None
    for event in events:
        sequential = apply_event(sequential, event)
    assert sequential == replay(events)
    assert replay(events) == replay(events)
