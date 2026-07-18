"""Unit tests for the immutable Application Queue projection service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from captioner.core.application.queue_projection import (
    JobQueueItem,
    QueueLoadIssue,
    QueueProjectionService,
    QueueSnapshot,
)
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.job import JobConfig, JobProjection, JobState
from captioner.core.domain.stage import (
    PipelineProfile,
    StageName,
    StageProjection,
    StageState,
    stage_plan_for,
)
from captioner.core.ports.batch_catalog import (
    BatchCatalogEntry,
    BatchCatalogIssue,
    BatchCatalogSnapshot,
    LeaseExecutionState,
)


def _config(tmp_path: Path, *, output_name: str = "output") -> JobConfig:
    plan = stage_plan_for(PipelineProfile.DETERMINISTIC)
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
        output_dir=str((tmp_path / output_name).resolve()),
        overwrite=False,
        stage_versions={stage.value: "1" for stage in plan},
    )


def _job(
    tmp_path: Path,
    job_id: str,
    *,
    input_name: str = "input.wav",
    state: JobState = JobState.PENDING,
    stages: tuple[StageProjection, ...] | None = None,
    output_name: str = "output",
) -> JobProjection:
    return JobProjection(
        job_id,
        str((tmp_path / input_name).resolve()),
        _config(tmp_path, output_name=output_name),
        state=state,
        stages=() if stages is None else stages,
    )


def _stages(*pairs: tuple[StageName, StageState, int]) -> tuple[StageProjection, ...]:
    plan = stage_plan_for(PipelineProfile.DETERMINISTIC)
    by_name = {name: (state, attempt) for name, state, attempt in pairs}
    stages: list[StageProjection] = []
    for name in plan:
        state, attempt = by_name.get(name, (StageState.PENDING, 0))
        stages.append(StageProjection(name, state=state, attempt=attempt))
    return tuple(stages)


def _entry(
    batch_id: str,
    jobs: tuple[JobProjection, ...],
    *,
    created_at_utc: str = "2026-01-01T00:00:00+00:00",
    lease_state: LeaseExecutionState = "missing",
    batch_cancel_requested: bool = False,
    job_cancel_requests: frozenset[str] = frozenset(),
    batch_pause_requested: bool = False,
    journal_tail_status: str = "clean",
    manifest_status: str = "missing",
    last_event_seq: int = 1,
) -> BatchCatalogEntry:
    return BatchCatalogEntry(
        batch_id=batch_id,
        created_at_utc=created_at_utc,
        projection=BatchProjection(batch_id, jobs, last_event_seq=last_event_seq),
        journal_tail_status=journal_tail_status,  # type: ignore[arg-type]
        manifest_status=manifest_status,  # type: ignore[arg-type]
        lease_state=lease_state,
        batch_cancel_requested=batch_cancel_requested,
        job_cancel_requests=job_cancel_requests,
        batch_pause_requested=batch_pause_requested,
    )


class FakeCatalog:
    def __init__(self, snapshot: BatchCatalogSnapshot | None = None) -> None:
        self.snapshot = snapshot or BatchCatalogSnapshot((), ())

    def read_snapshot(self) -> BatchCatalogSnapshot:
        return self.snapshot


def test_one_row_per_job(tmp_path: Path) -> None:
    jobs = (
        _job(tmp_path, "job-000001", input_name="a.wav"),
        _job(tmp_path, "job-000002", input_name="b.wav"),
    )
    catalog = FakeCatalog(BatchCatalogSnapshot((_entry("batch-a", jobs),), ()))
    snapshot = QueueProjectionService(catalog).refresh_queue()
    assert len(snapshot.items) == 2
    assert {item.batch_id for item in snapshot.items} == {"batch-a"}
    assert [item.job_id for item in snapshot.items] == ["job-000001", "job-000002"]
    assert [item.job_order for item in snapshot.items] == [0, 1]


def test_duplicate_input_paths_remain_separate(tmp_path: Path) -> None:
    path = str((tmp_path / "same.wav").resolve())
    jobs = (
        replace(_job(tmp_path, "job-000001"), input_path=path),
        replace(_job(tmp_path, "job-000002"), input_path=path),
    )
    catalog = FakeCatalog(BatchCatalogSnapshot((_entry("batch-a", jobs),), ()))
    items = QueueProjectionService(catalog).refresh_queue().items
    assert len(items) == 2
    assert items[0].input_path == items[1].input_path == path
    assert items[0].job_id != items[1].job_id


def test_stable_submission_order_is_state_independent(tmp_path: Path) -> None:
    first = _entry(
        "batch-b",
        (_job(tmp_path, "job-000001", state=JobState.SUCCEEDED),),
        created_at_utc="2026-01-01T00:00:01+00:00",
    )
    second = _entry(
        "batch-a",
        (_job(tmp_path, "job-000001", state=JobState.PENDING),),
        created_at_utc="2026-01-01T00:00:02+00:00",
    )
    third = _entry(
        "batch-c",
        (
            _job(tmp_path, "job-000002", state=JobState.FAILED),
            _job(tmp_path, "job-000001", state=JobState.RUNNING),
        ),
        created_at_utc="2026-01-01T00:00:01+00:00",
        lease_state="active_local",
    )
    catalog = FakeCatalog(BatchCatalogSnapshot((third, second, first), ()))
    service = QueueProjectionService(catalog)
    first_order = [item.job_id + "@" + item.batch_id for item in service.refresh_queue().items]
    catalog.snapshot = BatchCatalogSnapshot(
        (
            replace(
                third,
                projection=BatchProjection(
                    "batch-c",
                    (
                        replace(third.projection.jobs[0], state=JobState.SUCCEEDED),
                        replace(third.projection.jobs[1], state=JobState.FAILED),
                    ),
                    last_event_seq=2,
                ),
            ),
            second,
            first,
        ),
        (),
    )
    second_order = [item.job_id + "@" + item.batch_id for item in service.refresh_queue().items]
    assert first_order == [
        "job-000001@batch-b",
        "job-000002@batch-c",
        "job-000001@batch-c",
        "job-000001@batch-a",
    ]
    assert second_order == first_order


def test_active_plus_recent_terminal_limit(tmp_path: Path) -> None:
    active_jobs = tuple(
        _job(tmp_path, f"job-active-{index:03d}", state=JobState.PENDING) for index in range(3)
    )
    terminal_jobs = tuple(
        _job(
            tmp_path,
            f"job-term-{index:03d}",
            state=JobState.SUCCEEDED if index % 2 == 0 else JobState.FAILED,
        )
        for index in range(105)
    )
    entries = (
        _entry(
            "batch-active",
            active_jobs,
            created_at_utc="2026-01-02T00:00:00+00:00",
        ),
        _entry(
            "batch-terminal",
            terminal_jobs,
            created_at_utc="2026-01-01T00:00:00+00:00",
        ),
    )
    snapshot = QueueProjectionService(
        FakeCatalog(BatchCatalogSnapshot(entries, ())),
        recent_terminal_limit=100,
    ).refresh_queue()
    assert snapshot.active_count == 3
    assert snapshot.terminal_count == 100
    assert snapshot.omitted_terminal_jobs == 5
    terminal_ids = [item.job_id for item in snapshot.items if item.terminal]
    assert terminal_ids == [f"job-term-{index:03d}" for index in range(5, 105)]
    active_ids = [item.job_id for item in snapshot.items if not item.terminal]
    assert active_ids == [f"job-active-{index:03d}" for index in range(3)]
    assert [item.job_id for item in snapshot.items] == [
        *terminal_ids,
        *active_ids,
    ]


def test_zero_terminal_limit_keeps_only_active(tmp_path: Path) -> None:
    entries = (
        _entry(
            "batch-a",
            (
                _job(tmp_path, "job-000001", state=JobState.PENDING),
                _job(tmp_path, "job-000002", state=JobState.SUCCEEDED),
            ),
        ),
    )
    snapshot = QueueProjectionService(
        FakeCatalog(BatchCatalogSnapshot(entries, ())),
        recent_terminal_limit=0,
    ).refresh_queue()
    assert [item.job_id for item in snapshot.items] == ["job-000001"]
    assert snapshot.omitted_terminal_jobs == 1


@pytest.mark.parametrize(
    ("lease_state", "expected_job", "expected_stage"),
    [
        ("stale", JobState.INTERRUPTED, StageState.INTERRUPTED),
        ("missing", JobState.INTERRUPTED, StageState.INTERRUPTED),
        ("invalid", JobState.INTERRUPTED, StageState.INTERRUPTED),
        ("active_local", JobState.RUNNING, StageState.RUNNING),
        ("active_remote", JobState.RUNNING, StageState.RUNNING),
    ],
)
def test_stale_running_projection(
    tmp_path: Path,
    lease_state: LeaseExecutionState,
    expected_job: JobState,
    expected_stage: StageState,
) -> None:
    stages = _stages((StageName.TRANSCRIBE, StageState.RUNNING, 1))
    entry = _entry(
        "batch-a",
        (_job(tmp_path, "job-000001", state=JobState.RUNNING, stages=stages),),
        lease_state=lease_state,
    )
    item = (
        QueueProjectionService(FakeCatalog(BatchCatalogSnapshot((entry,), ())))
        .refresh_queue()
        .items[0]
    )
    assert item.state is expected_job
    assert item.active_stage is StageName.TRANSCRIBE
    assert item.active_stage_state is expected_stage
    assert item.active_stage_attempt == 1


def test_interrupted_job_does_not_rewrite_non_running_stage(tmp_path: Path) -> None:
    """Only a RUNNING active Stage is projected as INTERRUPTED with the Job."""
    stages = _stages(
        (StageName.INSPECT, StageState.COMMITTED, 1),
        (StageName.NORMALIZE, StageState.FAILED, 1),
    )
    entry = _entry(
        "batch-a",
        (_job(tmp_path, "job-000001", state=JobState.RUNNING, stages=stages),),
        lease_state="stale",
    )
    item = (
        QueueProjectionService(FakeCatalog(BatchCatalogSnapshot((entry,), ())))
        .refresh_queue()
        .items[0]
    )
    assert item.state is JobState.INTERRUPTED
    assert item.active_stage is StageName.NORMALIZE
    assert item.active_stage_state is StageState.FAILED


def test_current_stage_selection(tmp_path: Path) -> None:
    cases = [
        (
            _stages((StageName.INSPECT, StageState.RUNNING, 1)),
            StageName.INSPECT,
            StageState.RUNNING,
            1,
        ),
        (
            _stages(
                (StageName.INSPECT, StageState.COMMITTED, 1),
                (StageName.NORMALIZE, StageState.INTERRUPTED, 2),
            ),
            StageName.NORMALIZE,
            StageState.INTERRUPTED,
            2,
        ),
        (
            _stages(
                (StageName.INSPECT, StageState.COMMITTED, 1),
                (StageName.NORMALIZE, StageState.FAILED, 1),
            ),
            StageName.NORMALIZE,
            StageState.FAILED,
            1,
        ),
        (
            _stages((StageName.INSPECT, StageState.COMMITTED, 1)),
            StageName.NORMALIZE,
            StageState.PENDING,
            0,
        ),
        (
            tuple(
                StageProjection(name, state=StageState.COMMITTED, attempt=1)
                for name in stage_plan_for(PipelineProfile.DETERMINISTIC)
            ),
            None,
            None,
            0,
        ),
    ]
    for stages, stage_name, stage_state, attempt in cases:
        state = JobState.SUCCEEDED if stage_name is None else JobState.RUNNING
        entry = _entry(
            "batch-a",
            (_job(tmp_path, "job-000001", state=state, stages=stages),),
            lease_state="active_local",
        )
        item = (
            QueueProjectionService(FakeCatalog(BatchCatalogSnapshot((entry,), ())))
            .refresh_queue()
            .items[0]
        )
        assert item.active_stage is stage_name
        assert item.active_stage_state is stage_state
        assert item.active_stage_attempt == attempt


def test_cancellation_projection(tmp_path: Path) -> None:
    batch_marker = _entry(
        "batch-a",
        (_job(tmp_path, "job-000001"),),
        batch_cancel_requested=True,
    )
    job_marker = _entry(
        "batch-b",
        (_job(tmp_path, "job-000001"), _job(tmp_path, "job-000002")),
        job_cancel_requests=frozenset({"job-000002"}),
        created_at_utc="2026-01-02T00:00:00+00:00",
    )
    none_marker = _entry(
        "batch-c",
        (_job(tmp_path, "job-000001"),),
        created_at_utc="2026-01-03T00:00:00+00:00",
    )
    items = (
        QueueProjectionService(
            FakeCatalog(BatchCatalogSnapshot((batch_marker, job_marker, none_marker), ()))
        )
        .refresh_queue()
        .items
    )
    by_key = {(item.batch_id, item.job_id): item.cancel_requested for item in items}
    assert by_key[("batch-a", "job-000001")] is True
    assert by_key[("batch-b", "job-000001")] is False
    assert by_key[("batch-b", "job-000002")] is True
    assert by_key[("batch-c", "job-000001")] is False


def test_revision_behavior(tmp_path: Path) -> None:
    entry = _entry("batch-a", (_job(tmp_path, "job-000001"),))
    catalog = FakeCatalog(BatchCatalogSnapshot((entry,), ()))
    service = QueueProjectionService(catalog)
    first = service.refresh_queue()
    second = service.refresh_queue()
    assert first.revision == 1
    assert second.revision == 1
    catalog.snapshot = BatchCatalogSnapshot(
        (
            replace(
                entry,
                projection=BatchProjection(
                    "batch-a",
                    (replace(entry.projection.jobs[0], state=JobState.RUNNING),),
                    last_event_seq=2,
                ),
                lease_state="active_local",
            ),
        ),
        (),
    )
    third = service.refresh_queue()
    assert third.revision == 2
    catalog.snapshot = BatchCatalogSnapshot(
        catalog.snapshot.batches,
        (BatchCatalogIssue("bad-batch", "journal.corrupt"),),
    )
    fourth = service.refresh_queue()
    assert fourth.revision == 3
    many_terminal = _entry(
        "batch-terminal",
        tuple(_job(tmp_path, f"job-{index:06d}", state=JobState.SUCCEEDED) for index in range(3)),
        created_at_utc="2026-01-01T00:00:00+00:00",
    )
    limited = QueueProjectionService(
        FakeCatalog(BatchCatalogSnapshot((many_terminal,), ())),
        recent_terminal_limit=1,
    )
    first_limited = limited.refresh_queue()
    assert first_limited.omitted_terminal_jobs == 2
    assert first_limited.revision == 1
    limited.catalog.snapshot = BatchCatalogSnapshot(  # type: ignore[attr-defined]
        (
            _entry(
                "batch-terminal",
                tuple(
                    _job(tmp_path, f"job-{index:06d}", state=JobState.SUCCEEDED)
                    for index in range(4)
                ),
                created_at_utc="2026-01-01T00:00:00+00:00",
            ),
        ),
        (),
    )
    second_limited = limited.refresh_queue()
    assert second_limited.omitted_terminal_jobs == 3
    assert second_limited.revision == 2


def test_subscription_behavior(tmp_path: Path) -> None:
    entry = _entry("batch-a", (_job(tmp_path, "job-000001"),))
    catalog = FakeCatalog(BatchCatalogSnapshot((entry,), ()))
    service = QueueProjectionService(catalog)
    received: list[QueueSnapshot] = []
    unsubscribe = service.subscribe_queue(received.append)
    unchanged = service.refresh_queue()
    assert unchanged.revision == 1
    assert len(received) == 1
    assert received[0].revision == 1
    same = service.refresh_queue()
    assert same.revision == 1
    assert len(received) == 1
    catalog.snapshot = BatchCatalogSnapshot(
        (
            replace(
                entry,
                projection=BatchProjection(
                    "batch-a",
                    (replace(entry.projection.jobs[0], state=JobState.FAILED),),
                    last_event_seq=2,
                ),
            ),
        ),
        (),
    )
    changed = service.refresh_queue()
    assert changed.revision == 2
    assert len(received) == 2
    assert received[1].revision == 2
    unsubscribe()
    unsubscribe()
    catalog.snapshot = BatchCatalogSnapshot(
        (
            replace(
                entry,
                projection=BatchProjection(
                    "batch-a",
                    (replace(entry.projection.jobs[0], state=JobState.CANCELLED),),
                    last_event_seq=3,
                ),
            ),
        ),
        (),
    )
    after = service.refresh_queue()
    assert after.revision == 3
    assert len(received) == 2


def test_immutability(tmp_path: Path) -> None:
    entry = _entry("batch-a", (_job(tmp_path, "job-000001"),))
    snapshot = QueueProjectionService(
        FakeCatalog(BatchCatalogSnapshot((entry,), (BatchCatalogIssue("x", "journal.corrupt"),)))
    ).refresh_queue()
    item = snapshot.items[0]
    issue = snapshot.issues[0]
    with pytest.raises((AttributeError, TypeError)):
        item.state = JobState.FAILED  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        snapshot.items = ()  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        issue.code = "other"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        entry.batch_id = "other"  # type: ignore[misc]
    assert isinstance(item, JobQueueItem)
    assert isinstance(issue, QueueLoadIssue)


def test_get_queue_snapshot_delegates(tmp_path: Path) -> None:
    catalog = FakeCatalog()
    service = QueueProjectionService(catalog)
    assert service.get_queue_snapshot() == service.refresh_queue()


def test_recent_terminal_limit_rejects_negative() -> None:
    with pytest.raises(ValueError, match="recent_terminal_limit"):
        QueueProjectionService(FakeCatalog(), recent_terminal_limit=-1)
