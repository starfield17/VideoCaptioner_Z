from __future__ import annotations

from pathlib import Path

import pytest

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.batch import BatchState
from captioner.core.domain.cache_key import derive_stage_cache_key
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import JournalEvent, apply_event, replay
from captioner.core.domain.result import freeze_json_value
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageState


def _config(tmp_path: Path) -> JobConfig:
    return JobConfig(
        model_ref="tiny",
        model_identity="faster-whisper:tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        vad_filter=True,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        normalization={"sample_rate": 16000, "channels": 1},
        segmentation={"max_duration_ms": 7000},
        output_dir=str(tmp_path.resolve()),
        overwrite=False,
        stage_versions={stage.value: "1" for stage in STAGE_PLAN},
    )


def _event(seq: int, event_type: str, payload: dict[str, object]) -> JournalEvent:
    return JournalEvent(
        seq=seq,
        event_id=f"event-{seq:06d}",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        batch_id="batch-0123456789abcdef",
        type=event_type,
        payload=payload,  # type: ignore[arg-type]  # test helper accepts JSON-compatible literals
    )


def _initial_events(tmp_path: Path) -> list[JournalEvent]:
    return [
        _event(1, "batch.created", {}),
        _event(
            2,
            "job.created",
            {
                "job_id": "job-000001",
                "input_path": str((tmp_path / "input.wav").resolve()),
                "config": _config(tmp_path).to_dict(),
            },
        ),
    ]


def _artifact(name: str = "media.json") -> ArtifactRef:
    return ArtifactRef("a" * 64, 10, "media", "application/json", name)


def test_artifact_ref_rejects_unsafe_identity() -> None:
    with pytest.raises(AppError, match=r"artifact\.invalid"):
        ArtifactRef("A" * 64, 1, "media", "application/json", "media.json")
    with pytest.raises(AppError, match=r"artifact\.invalid"):
        ArtifactRef("a" * 64, 1, "media", "application/json", "../media.json")


def test_replay_fixed_pipeline_to_success(tmp_path: Path) -> None:
    events = _initial_events(tmp_path)
    seq = len(events)
    for stage in STAGE_PLAN:
        seq += 1
        events.append(
            _event(
                seq,
                "stage.started",
                {"job_id": "job-000001", "stage_name": stage.value, "attempt": 1},
            )
        )
        seq += 1
        events.append(
            _event(
                seq,
                "stage.committed",
                {
                    "job_id": "job-000001",
                    "stage_name": stage.value,
                    "attempt": 1,
                    "cache_key": f"sha256:{'b' * 64}",
                    "artifacts": [_artifact(f"{stage.value}.json").to_dict()],
                },
            )
        )
    events.append(_event(seq + 1, "job.succeeded", {"job_id": "job-000001"}))
    first = replay(events)
    assert first == replay(events)
    assert first.state is BatchState.SUCCEEDED
    assert all(stage.state is StageState.COMMITTED for stage in first.jobs[0].stages)


def test_stage_cannot_start_before_dependency(tmp_path: Path) -> None:
    projection = replay(_initial_events(tmp_path))
    with pytest.raises(AppError, match="dependency_not_committed"):
        apply_event(
            projection,
            _event(
                3,
                "stage.started",
                {"job_id": "job-000001", "stage_name": "normalize", "attempt": 1},
            ),
        )


def test_attempt_number_cannot_be_reused(tmp_path: Path) -> None:
    projection = replay(
        [
            *_initial_events(tmp_path),
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
        ]
    )
    with pytest.raises(AppError, match="attempt_reused"):
        apply_event(
            projection,
            _event(
                5,
                "stage.started",
                {"job_id": "job-000001", "stage_name": "inspect", "attempt": 1},
            ),
        )


def test_job_cannot_succeed_with_uncommitted_stages(tmp_path: Path) -> None:
    projection = replay(_initial_events(tmp_path))
    with pytest.raises(AppError, match="stages_uncommitted"):
        apply_event(projection, _event(3, "job.succeeded", {"job_id": "job-000001"}))


def test_cache_key_is_canonical_and_input_sensitive() -> None:
    artifact = _artifact()
    left = derive_stage_cache_key(
        stage_name=StageName.INSPECT,
        stage_version="1",
        input_artifacts=(artifact,),
        config=freeze_json_value({"b": 2, "a": 1}),  # type: ignore[arg-type]  # mapping shape known
    )
    right = derive_stage_cache_key(
        stage_name=StageName.INSPECT,
        stage_version="1",
        input_artifacts=(artifact,),
        config=freeze_json_value({"a": 1, "b": 2}),  # type: ignore[arg-type]  # mapping shape known
    )
    changed = derive_stage_cache_key(
        stage_name=StageName.INSPECT,
        stage_version="1",
        input_artifacts=(ArtifactRef("c" * 64, 10, "media", "application/json", "media.json"),),
        config=freeze_json_value({"a": 1, "b": 2}),  # type: ignore[arg-type]  # mapping shape known
    )
    assert left == right
    assert changed != left
