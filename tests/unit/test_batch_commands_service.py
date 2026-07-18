"""Unit tests for BatchCommandService."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from captioner.core.application.batch_commands import (
    BatchActionRequest,
    BatchCommandKind,
    BatchCommandService,
    CancelLocalWorkRequest,
    JobActionRequest,
    SubmitBatchRequest,
)
from captioner.core.application.execution_coordinator import SerialExecutionCoordinator
from captioner.core.application.input_selection import BatchDraft
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile, StageName
from captioner.core.ports.batch_gateway import BatchGatewayPort, CreatedBatch


def _empty_drafts() -> list[BatchDraft]:
    return []


def _empty_strings() -> list[str]:
    return []


def _empty_pairs() -> list[tuple[str, str]]:
    return []


def _empty_cancels() -> list[tuple[str, str | None, bool]]:
    return []


def _empty_pauses() -> list[tuple[str, bool]]:
    return []


@dataclass(slots=True)
class FakeGateway:
    created: list[BatchDraft] = field(default_factory=_empty_drafts)
    executed: list[str] = field(default_factory=_empty_strings)
    resumed: list[str] = field(default_factory=_empty_strings)
    retried: list[tuple[str, str]] = field(default_factory=_empty_pairs)
    cancelled: list[tuple[str, str | None, bool]] = field(default_factory=_empty_cancels)
    paused: list[tuple[str, bool]] = field(default_factory=_empty_pauses)
    run_again: list[tuple[str, str]] = field(default_factory=_empty_pairs)

    def create_batch(self, draft: BatchDraft) -> CreatedBatch:
        self.created.append(draft)
        return CreatedBatch("batch-new", ("job-000001",))

    def execute_created_batch(self, batch_id: str) -> None:
        self.executed.append(batch_id)

    def resume_batch(self, batch_id: str) -> None:
        self.resumed.append(batch_id)

    def retry_job(self, batch_id: str, job_id: str) -> StageName:
        self.retried.append((batch_id, job_id))
        return StageName.SEGMENT

    def request_cancel(
        self,
        batch_id: str,
        *,
        job_id: str | None,
        execution_scheduled: bool,
    ) -> None:
        self.cancelled.append((batch_id, job_id, execution_scheduled))

    def request_pause(self, batch_id: str, *, execution_scheduled: bool) -> None:
        self.paused.append((batch_id, execution_scheduled))

    def create_run_again(self, batch_id: str, job_id: str) -> CreatedBatch:
        self.run_again.append((batch_id, job_id))
        return CreatedBatch("batch-again", ("job-000001",))

    def read_job_detail_source(self, batch_id: str, job_id: str) -> Any:
        raise NotImplementedError

    def read_recovery_sources(self) -> tuple[Any, ...]:
        return ()

    def close_shared_runtime(self) -> None:
        return None


def _draft(tmp_path: Path) -> BatchDraft:
    media = tmp_path / "a.wav"
    media.write_bytes(b"x")
    return BatchDraft(
        input_paths=(str(media),),
        output_root=str(tmp_path / "out"),
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language="en",
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )


def _service(gateway: FakeGateway) -> BatchCommandService:
    return BatchCommandService(
        cast(BatchGatewayPort, gateway),
        SerialExecutionCoordinator(),
        now_utc=lambda: "t0",
    )


def test_submit_creates_then_schedules(tmp_path: Path) -> None:
    gateway = FakeGateway()
    service = _service(gateway)
    draft = _draft(tmp_path)
    ack = service.submit(SubmitBatchRequest(request_id="req-1", draft=draft))
    assert ack.kind is BatchCommandKind.SUBMIT
    assert ack.batch_id == "batch-new"
    assert ack.created_batch_id == "batch-new"
    assert ack.scheduled is True
    assert gateway.created == [draft]
    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gateway.executed == ["batch-new"]
    service.coordinator.drain_completions()
    service.coordinator.shutdown()


def test_resume_retry_run_again_and_cancel(tmp_path: Path) -> None:
    del tmp_path
    gateway = FakeGateway()
    service = _service(gateway)

    ack = service.perform_batch_action(
        BatchActionRequest(
            request_id="req-r",
            kind=BatchCommandKind.RESUME_BATCH,
            batch_id="batch-a",
        )
    )
    assert ack.scheduled is True

    with pytest.raises(AppError, match=r"batch\.operation_conflict"):
        service.perform_batch_action(
            BatchActionRequest(
                request_id="req-r2",
                kind=BatchCommandKind.RESUME_BATCH,
                batch_id="batch-a",
            )
        )

    pause = service.perform_batch_action(
        BatchActionRequest(
            request_id="req-p",
            kind=BatchCommandKind.PAUSE_BATCH,
            batch_id="batch-a",
        )
    )
    assert pause.scheduled is False
    assert gateway.paused

    cancel = service.perform_job_action(
        JobActionRequest(
            request_id="req-c",
            kind=BatchCommandKind.CANCEL_JOB,
            batch_id="batch-a",
            job_id="job-000001",
        )
    )
    assert cancel.kind is BatchCommandKind.CANCEL_JOB

    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    service.coordinator.drain_completions()

    again = service.perform_job_action(
        JobActionRequest(
            request_id="req-a",
            kind=BatchCommandKind.RUN_AGAIN,
            batch_id="batch-old",
            job_id="job-000001",
        )
    )
    assert again.created_batch_id == "batch-again"
    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    service.coordinator.drain_completions()
    service.coordinator.shutdown()


def test_cancel_local_work() -> None:
    gateway = FakeGateway()
    service = _service(gateway)
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        started.set()
        release.wait(timeout=2)

    service.coordinator.schedule(
        batch_id="batch-a",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=block,
    )
    assert started.wait(timeout=2)
    service.coordinator.schedule(
        batch_id="batch-b",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=lambda: None,
    )
    ack = service.cancel_local_work(CancelLocalWorkRequest(request_id="req-x"))
    assert "batch-a" in ack.affected_batch_ids
    release.set()
    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    service.coordinator.drain_completions()
    service.coordinator.shutdown()


def test_retry_job_schedules() -> None:
    gateway = FakeGateway()
    service = _service(gateway)
    ack = service.perform_job_action(
        JobActionRequest(
            request_id="req-retry",
            kind=BatchCommandKind.RETRY_JOB,
            batch_id="batch-r",
            job_id="job-000001",
        )
    )
    assert ack.scheduled is True
    import time

    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gateway.retried == [("batch-r", "job-000001")]
    service.coordinator.drain_completions()
    service.coordinator.shutdown()


def test_cancel_batch_queued_and_retry_conflict() -> None:
    import threading
    import time

    gateway = FakeGateway()
    service = _service(gateway)
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        started.set()
        release.wait(timeout=2)

    service.coordinator.schedule(
        batch_id="batch-active",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=block,
    )
    assert started.wait(timeout=2)
    service.coordinator.schedule(
        batch_id="batch-queued",
        kind=BatchCommandKind.SUBMIT,
        job_id=None,
        operation=lambda: None,
    )
    ack = service.perform_batch_action(
        BatchActionRequest(
            request_id="req-cb",
            kind=BatchCommandKind.CANCEL_BATCH,
            batch_id="batch-queued",
        )
    )
    assert ack.kind is BatchCommandKind.CANCEL_BATCH
    with pytest.raises(AppError, match=r"batch\.operation_conflict"):
        service.perform_job_action(
            JobActionRequest(
                request_id="req-rt",
                kind=BatchCommandKind.RETRY_JOB,
                batch_id="batch-active",
                job_id="job-000001",
            )
        )
    release.set()
    deadline = time.monotonic() + 2
    while service.coordinator.snapshot().has_work and time.monotonic() < deadline:
        time.sleep(0.01)
    service.coordinator.drain_completions()
    service.coordinator.shutdown()
