"""Unit tests for DiagnosticsService aggregation and export."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from captioner.core.application.configuration import (
    ConfigurationIssue,
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderPublicSettings,
    built_in_presets,
)
from captioner.core.application.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticExportRequest,
    DiagnosticExportResult,
    DiagnosticsRequest,
    DiagnosticsService,
    DiagnosticsSnapshot,
    DiagnosticsStorageLocations,
    RuntimeAvailability,
)
from captioner.core.application.queue_projection import (
    JobQueueItem,
    QueueLoadIssue,
    QueueSnapshot,
)
from captioner.core.application.recovery import (
    RecoveryIssue,
    RecoveryItem,
    RecoveryRequest,
    RecoverySnapshot,
)
from captioner.core.domain.batch import BatchState
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName, StageState


class FakeQueue:
    def __init__(self, snapshot: QueueSnapshot) -> None:
        self.snapshot = snapshot

    def get_queue_snapshot(self) -> QueueSnapshot:
        return self.snapshot


class FakeConfiguration:
    def __init__(self, snapshot: ConfigurationSnapshot) -> None:
        self.snapshot = snapshot
        self.load_calls = 0

    def load(self) -> ConfigurationSnapshot:
        self.load_calls += 1
        return self.snapshot


class FakeRecovery:
    def __init__(self, snapshot: RecoverySnapshot) -> None:
        self.snapshot = snapshot
        self.requests: list[RecoveryRequest] = []

    def scan(self, request: RecoveryRequest) -> RecoverySnapshot:
        self.requests.append(request)
        return self.snapshot


class FakeEnvironment:
    def collect_runtime_availability(
        self,
        *,
        provider_configured: bool,
        credential_source: Literal["config", "environment", "missing"],
    ) -> RuntimeAvailability:
        return RuntimeAvailability(
            packaged=False,
            operating_system="Linux",
            architecture="x86_64",
            python_version="3.13.0",
            app_version="0.0.0",
            ffmpeg_available=True,
            ffprobe_available=True,
            asr_runtime_available=False,
            provider_configured=provider_configured,
            credential_source=credential_source,
        )

    def collect_storage_locations(self) -> DiagnosticsStorageLocations:
        return DiagnosticsStorageLocations.empty()


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[DiagnosticExportRequest, DiagnosticsSnapshot]] = []

    def write_bundle(
        self,
        request: DiagnosticExportRequest,
        *,
        snapshot: DiagnosticsSnapshot,
    ) -> DiagnosticExportResult:
        self.calls.append((request, snapshot))
        return DiagnosticExportResult(
            request_id=request.request_id,
            destination=request.destination,
            size_bytes=12,
            sha256="a" * 64,
        )


def _job(
    *,
    state: JobState = JobState.RUNNING,
    profile: PipelineProfile = PipelineProfile.FAST,
    stage: StageName | None = StageName.TRANSCRIBE,
    input_path: str = "/private/home/alice/media/input-secret.wav",
    batch_id: str = "batch-secret",
    job_id: str = "job-secret",
) -> JobQueueItem:
    return JobQueueItem(
        batch_id=batch_id,
        job_id=job_id,
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=1,
        input_path=input_path,
        output_dir="/private/home/alice/out",
        pipeline_profile=profile,
        state=state,
        active_stage=stage,
        active_stage_state=StageState.RUNNING if stage is not None else None,
        active_stage_attempt=1,
        cancel_requested=False,
        pause_requested=False,
        paused=False,
        last_event_seq=1,
        journal_tail_status="clean",
        manifest_status="current",
    )


def _service() -> tuple[DiagnosticsService, FakeWriter, FakeQueue, FakeConfiguration, FakeRecovery]:
    queue_snapshot = QueueSnapshot(
        schema_version=2,
        revision=3,
        items=(
            _job(state=JobState.RUNNING, profile=PipelineProfile.FAST),
            _job(
                state=JobState.SUCCEEDED,
                profile=PipelineProfile.QUALITY,
                stage=None,
                batch_id="batch-b",
                job_id="job-b",
            ),
        ),
        issues=(QueueLoadIssue(batch_name="bad", code="queue.batch_read_failed"),),
        omitted_terminal_jobs=2,
    )
    config = ConfigurationSnapshot(
        global_settings=GlobalSettings(locale="zh-CN"),
        presets=(
            *built_in_presets(),
            ExecutionPreset(
                name="user-secret-preset",
                display_name="PRIVATE_PRESET",
                built_in=False,
                pipeline_profile=PipelineProfile.FAST,
                model_ref="tiny",
                device="cpu",
                compute_type="default",
                source_language=None,
                target_language="zh-CN",
                provider_profile="default",
            ),
        ),
        provider=ProviderPublicSettings(
            profile_name="secret-profile",
            base_url="https://user:password@example.internal/v1",
            model="gpt-secret",
            max_concurrency=4,
            request_timeout_sec=120.0,
            max_retries=5,
            temperature=0.1,
            tokenizer="cl100k_base",
            credential_source="config",
        ),
        issues=(ConfigurationIssue(code="config.settings_invalid"),),
    )
    recovery = RecoverySnapshot(
        schema_version=1,
        request_id="req-1",
        items=(
            RecoveryItem(
                batch_id="batch-recover",
                created_at_utc="t0",
                state=BatchState.INTERRUPTED,
                job_count=1,
                pause_requested=True,
                missing_input_paths=("/private/home/alice/media/input-secret.wav",),
                last_event_seq=2,
                blocked_code=None,
            ),
            RecoveryItem(
                batch_id="batch-blocked",
                created_at_utc="t1",
                state=BatchState.FAILED,
                job_count=1,
                pause_requested=False,
                missing_input_paths=("C:\\Users\\Alice\\Videos\\input-secret.mp4",),
                last_event_seq=1,
                blocked_code="recovery.input_missing",
            ),
        ),
        issues=(RecoveryIssue(batch_name="corrupt", code="recovery.read_failed"),),
    )
    writer = FakeWriter()
    service = DiagnosticsService(
        queue=FakeQueue(queue_snapshot),  # type: ignore[arg-type]
        configuration=FakeConfiguration(config),  # type: ignore[arg-type]
        recovery=FakeRecovery(recovery),  # type: ignore[arg-type]
        environment=FakeEnvironment(),
        writer=writer,
        now_utc=lambda: "2026-07-18T00:00:00+00:00",
    )
    return (
        service,
        writer,
        FakeQueue(queue_snapshot),
        FakeConfiguration(config),
        FakeRecovery(recovery),
    )


def test_load_aggregates_without_paths_or_ids() -> None:
    service, _writer, *_rest = _service()
    snapshot = service.load(DiagnosticsRequest(request_id="req-diag-1"))
    assert snapshot.schema_version == DIAGNOSTICS_SCHEMA_VERSION
    assert snapshot.queue.active_jobs == 1
    assert snapshot.queue.terminal_jobs == 1
    assert snapshot.queue.omitted_terminal_jobs == 2
    assert snapshot.queue.state_counts == (("running", 1), ("succeeded", 1))
    assert snapshot.queue.profile_counts == (("fast", 1), ("quality", 1))
    assert snapshot.queue.stage_counts == (("transcribe", 1),)
    assert snapshot.queue.issue_codes == (("queue.batch_read_failed", 1),)
    assert snapshot.configuration.locale == "zh-CN"
    assert snapshot.configuration.built_in_preset_count == 3
    assert snapshot.configuration.user_preset_count == 1
    assert snapshot.configuration.provider_configured is True
    assert snapshot.configuration.credential_source == "config"
    assert snapshot.configuration.issue_codes == ("config.settings_invalid",)
    assert snapshot.recovery.recoverable_batches == 1
    assert snapshot.recovery.blocked_batches == 1
    assert snapshot.recovery.paused_batches == 1
    assert snapshot.recovery.issue_codes == (("recovery.read_failed", 1),)
    rendered = repr(snapshot)
    for sentinel in (
        "batch-secret",
        "job-secret",
        "/private/home/alice",
        "input-secret",
        "PRIVATE_PRESET",
        "user-secret-preset",
        "secret-profile",
        "example.internal",
        "gpt-secret",
        "sk-diagnostic",
    ):
        assert sentinel not in rendered


def test_export_uses_supplied_or_generated_snapshot() -> None:
    service, writer, *_rest = _service()
    snapshot = service.load(DiagnosticsRequest(request_id="req-export-1"))
    result = service.export(
        DiagnosticExportRequest(
            request_id="req-export-1",
            destination="/tmp/out.zip",
            overwrite=True,
        ),
        snapshot=snapshot,
    )
    assert result.request_id == "req-export-1"
    assert len(writer.calls) == 1
    assert writer.calls[0][1] is snapshot

    result2 = service.export(
        DiagnosticExportRequest(
            request_id="req-export-2",
            destination="/tmp/out2.zip",
        )
    )
    assert result2.request_id == "req-export-2"
    assert len(writer.calls) == 2
    assert writer.calls[1][1].request_id == "req-export-2"


def test_source_snapshots_are_not_mutated() -> None:
    service, _writer, queue, config, recovery = _service()
    before_queue = queue.snapshot
    before_config = config.snapshot
    before_recovery = recovery.snapshot
    service.load(DiagnosticsRequest(request_id="req-immut"))
    assert queue.snapshot is before_queue
    assert config.snapshot is before_config
    assert recovery.snapshot is before_recovery
    # replace would raise if frozen fields were mutated incorrectly
    assert replace(before_queue, revision=before_queue.revision).revision == before_queue.revision
