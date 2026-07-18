"""Unit tests for recovery discovery projections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
from captioner.core.application.recovery import RecoveryRequest, RecoveryService
from captioner.core.domain.batch import BatchProjection, BatchState
from captioner.core.domain.job import JobConfig, JobProjection, JobState
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageProjection, StageState
from captioner.core.ports.batch_gateway import RecoverySource


def _job(job_id: str, *, state: JobState, input_path: str) -> JobProjection:
    stages = tuple(
        StageProjection(
            name, StageState.PENDING if state is JobState.PENDING else StageState.COMMITTED
        )
        for name in STAGE_PLAN
    )
    if state is JobState.INTERRUPTED:
        stages = (
            StageProjection(StageName.INSPECT, StageState.COMMITTED),
            StageProjection(StageName.NORMALIZE, StageState.INTERRUPTED, attempt=1),
            *tuple(
                StageProjection(name, StageState.PENDING)
                for name in STAGE_PLAN
                if name not in {StageName.INSPECT, StageName.NORMALIZE}
            ),
        )
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
        "/tmp/out",
        False,
        {stage.value: "v1" for stage in STAGE_PLAN},
    )
    return JobProjection(job_id, input_path, config, state, stages)


class FakeGateway:
    def __init__(
        self,
        sources: tuple[RecoverySource, ...],
        issues: tuple[object, ...] = (),
    ) -> None:
        self.sources = sources
        self.issues = issues

    def read_recovery_sources(self) -> object:
        from captioner.core.ports.batch_gateway import RecoveryReadResult, RecoverySourceIssue

        typed_issues = tuple(
            issue
            if isinstance(issue, RecoverySourceIssue)
            else RecoverySourceIssue(
                batch_name=getattr(issue, "batch_name", "x"),
                code=getattr(issue, "code", "queue.batch_read_failed"),
            )
            for issue in self.issues
        )
        return RecoveryReadResult(sources=self.sources, issues=typed_issues)

    def create_batch(self, draft: object) -> object:
        raise NotImplementedError

    def execute_created_batch(self, batch_id: str) -> None:
        raise NotImplementedError

    def validate_resume(self, batch_id: str) -> None:
        raise NotImplementedError

    def resume_batch(self, batch_id: str) -> None:
        raise NotImplementedError

    def resolve_retry_stage(self, batch_id: str, job_id: str) -> StageName:
        raise NotImplementedError

    def retry_job(self, batch_id: str, job_id: str, stage: StageName) -> None:
        raise NotImplementedError

    def request_cancel(
        self, batch_id: str, *, job_id: str | None, execution_scheduled: bool
    ) -> None:
        raise NotImplementedError

    def request_pause(self, batch_id: str, *, execution_scheduled: bool) -> None:
        raise NotImplementedError

    def create_run_again(self, batch_id: str, job_id: str) -> object:
        raise NotImplementedError

    def read_job_detail_source(self, batch_id: str, job_id: str) -> object:
        raise NotImplementedError

    def close_shared_runtime(self) -> None:
        return None


def test_recovery_filters_active_and_blocks_missing_input(tmp_path: Path) -> None:
    present = tmp_path / "ok.wav"
    present.write_bytes(b"x")
    missing = str(tmp_path / "missing.wav")
    sources = (
        RecoverySource(
            batch_id="batch-pending",
            created_at_utc="2026-01-01T00:00:00+00:00",
            state=BatchState.PENDING,
            job_count=1,
            pause_requested=False,
            missing_input_paths=(),
            last_event_seq=1,
            lease_state="missing",
            projection=BatchProjection(
                "batch-pending",
                (_job("job-000001", state=JobState.PENDING, input_path=str(present)),),
                last_event_seq=1,
            ),
        ),
        RecoverySource(
            batch_id="batch-active",
            created_at_utc="2026-01-01T00:00:01+00:00",
            state=BatchState.RUNNING,
            job_count=1,
            pause_requested=False,
            missing_input_paths=(),
            last_event_seq=2,
            lease_state="active_local",
            projection=BatchProjection(
                "batch-active",
                (_job("job-000001", state=JobState.RUNNING, input_path=str(present)),),
                last_event_seq=2,
            ),
        ),
        RecoverySource(
            batch_id="batch-missing",
            created_at_utc="2026-01-01T00:00:02+00:00",
            state=BatchState.INTERRUPTED,
            job_count=1,
            pause_requested=False,
            missing_input_paths=(missing,),
            last_event_seq=3,
            lease_state="stale",
            projection=BatchProjection(
                "batch-missing",
                (_job("job-000001", state=JobState.INTERRUPTED, input_path=missing),),
                last_event_seq=3,
            ),
        ),
    )
    service = RecoveryService(cast(Any, FakeGateway(sources)), SerialExecutionCoordinator())
    snapshot = service.scan(RecoveryRequest(request_id="req-1"))
    ids = [item.batch_id for item in snapshot.items]
    assert "batch-pending" in ids
    assert "batch-active" not in ids
    blocked = next(item for item in snapshot.items if item.batch_id == "batch-missing")
    assert blocked.can_resume is False
    assert blocked.blocked_code == "recovery.input_missing"


def test_recovery_propagates_catalog_issues(tmp_path: Path) -> None:
    from captioner.core.ports.batch_gateway import RecoverySourceIssue

    present = tmp_path / "ok.wav"
    present.write_bytes(b"x")
    sources = (
        RecoverySource(
            batch_id="batch-valid",
            created_at_utc="2026-01-01T00:00:00+00:00",
            state=BatchState.PENDING,
            job_count=1,
            pause_requested=False,
            missing_input_paths=(),
            last_event_seq=1,
            lease_state="missing",
            projection=BatchProjection(
                "batch-valid",
                (_job("job-000001", state=JobState.PENDING, input_path=str(present)),),
                last_event_seq=1,
            ),
        ),
    )
    issues = (
        RecoverySourceIssue(batch_name="batch-corrupt", code="queue.batch_read_failed"),
        RecoverySourceIssue(batch_name="batch-a-corrupt", code="queue.journal_corrupt"),
    )
    service = RecoveryService(
        cast(Any, FakeGateway(sources, issues=issues)),
        SerialExecutionCoordinator(),
    )
    snapshot = service.scan(RecoveryRequest(request_id="req-2"))
    assert any(item.batch_id == "batch-valid" for item in snapshot.items)
    assert len(snapshot.issues) == 2
    # Stable ordering by batch_name then code.
    assert snapshot.issues[0].batch_name == "batch-a-corrupt"
    assert snapshot.issues[1].batch_name == "batch-corrupt"
    for issue in snapshot.issues:
        assert "/" not in issue.batch_name
        assert "Traceback" not in issue.code
        assert "Exception" not in issue.code
