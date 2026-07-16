from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
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
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        str(root),
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
                    "input_path": str(root / f"{number}.wav"),
                    "config": config.to_dict(),
                },
            )
        )
    sequential = None
    for event in events:
        sequential = apply_event(sequential, event)
    assert sequential == replay(events)
    assert replay(events) == replay(events)


@given(committed_count=st.integers(min_value=1, max_value=len(STAGE_PLAN)))
def test_generated_committed_prefix_obeys_dependency_transitions(committed_count: int) -> None:
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
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    events = [
        _event(1, "batch.created", {}),
        _event(
            2,
            "job.created",
            {
                "job_id": "job-000001",
                "input_path": str(root / "input.wav"),
                "config": config.to_dict(),
            },
        ),
    ]
    sequence = 2
    for stage in STAGE_PLAN[:committed_count]:
        sequence += 1
        events.append(
            _event(
                sequence,
                "stage.started",
                {"job_id": "job-000001", "stage_name": stage.value, "attempt": 1},
            )
        )
        sequence += 1
        events.append(
            _event(
                sequence,
                "stage.committed",
                {
                    "job_id": "job-000001",
                    "stage_name": stage.value,
                    "attempt": 1,
                    "cache_key": f"sha256:{'a' * 64}",
                    "artifacts": [
                        {
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                            "kind": "test",
                            "media_type": "application/octet-stream",
                            "logical_name": f"{stage.value}.bin",
                        }
                    ],
                },
            )
        )
    projection = replay(events)
    assert (
        sum(stage.state.value == "committed" for stage in projection.jobs[0].stages)
        == committed_count
    )
    assert replay(events) == projection


def test_reversing_stage_commit_order_is_rejected() -> None:
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
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    events = [
        _event(1, "batch.created", {}),
        _event(
            2,
            "job.created",
            {"job_id": "job-000001", "input_path": str(root / "a.wav"), "config": config.to_dict()},
        ),
        _event(
            3,
            "stage.committed",
            {
                "job_id": "job-000001",
                "stage_name": "inspect",
                "attempt": 1,
                "cache_key": f"sha256:{'a' * 64}",
                "artifacts": [],
            },
        ),
    ]

    with pytest.raises(AppError):
        replay(events)


def test_batch_config_updated_changes_every_job_at_one_replay_boundary() -> None:
    root = (Path.cwd() / "captioner-property").resolve()
    old_config = JobConfig(
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
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    new_config = JobConfig(
        "small",
        "faster-whisper:small",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    events = [
        _event(1, "batch.created", {}),
        _event(
            2,
            "job.created",
            {
                "job_id": "job-000001",
                "input_path": str(root / "one.wav"),
                "config": old_config.to_dict(),
            },
        ),
        _event(
            3,
            "job.created",
            {
                "job_id": "job-000002",
                "input_path": str(root / "two.wav"),
                "config": old_config.to_dict(),
            },
        ),
        _event(
            4,
            "batch.config_updated",
            {"config": new_config.to_dict(), "earliest_stage": "transcribe"},
        ),
    ]

    projection = replay(events)

    assert {job.config.runtime_signature for job in projection.jobs} == {
        new_config.runtime_signature
    }
    assert all(
        job.stage(STAGE_PLAN[0]).state.value == "pending"
        and job.stage(STAGE_PLAN[2]).state.value == "pending"
        for job in projection.jobs
    )
    assert apply_event(replay(events[:-1]), events[-1]) == projection


def test_interrupted_job_can_be_cancelled_without_rewriting_stage_history() -> None:
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
        {"sample_rate": 16000},
        {"max_duration_ms": 7000},
        str(root),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )
    events = [
        _event(1, "batch.created", {}),
        _event(
            2,
            "job.created",
            {
                "job_id": "job-000001",
                "input_path": str(root / "one.wav"),
                "config": config.to_dict(),
            },
        ),
        _event(
            3,
            "stage.started",
            {"job_id": "job-000001", "stage_name": "inspect", "attempt": 1},
        ),
        _event(
            4,
            "stage.interrupted",
            {"job_id": "job-000001", "stage_name": "inspect", "attempt": 1},
        ),
        _event(5, "job.cancelled", {"job_id": "job-000001"}),
    ]

    projection = replay(events)

    assert projection.jobs[0].state.value == "cancelled"
    assert projection.jobs[0].stage(STAGE_PLAN[0]).state.value == "interrupted"
