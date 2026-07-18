"""Aggregate diagnostics DTOs and Application service (no paths/IDs/secrets)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState, validate_identifier

if TYPE_CHECKING:
    from captioner.core.application.configuration import ConfigurationService
    from captioner.core.application.queue_projection import QueueProjectionService
    from captioner.core.application.recovery import RecoveryService
    from captioner.core.ports.diagnostics import (
        DiagnosticBundleWriterPort,
        DiagnosticsEnvironmentPort,
    )

DIAGNOSTICS_SCHEMA_VERSION = 1
DIAGNOSTIC_BUNDLE_SCHEMA_VERSION = 1

CredentialSourceLabel = Literal["config", "environment", "missing"]
_KNOWN_CREDENTIAL_SOURCES = frozenset({"config", "environment", "missing"})
_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)


def _require_nonnegative(value: int, *, field: str) -> None:
    if type(value) is not int or value < 0:
        raise AppError("diagnostics.snapshot_invalid", {"field": field})


def _sorted_count_tuples(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class RuntimeAvailability:
    packaged: bool
    operating_system: str
    architecture: str
    python_version: str
    app_version: str
    ffmpeg_available: bool
    ffprobe_available: bool
    asr_runtime_available: bool
    provider_configured: bool
    credential_source: CredentialSourceLabel

    def __post_init__(self) -> None:
        if self.credential_source not in _KNOWN_CREDENTIAL_SOURCES:
            raise AppError(
                "diagnostics.snapshot_invalid",
                {"field": "credential_source"},
            )
        for field_name, value in (
            ("operating_system", self.operating_system),
            ("architecture", self.architecture),
            ("python_version", self.python_version),
            ("app_version", self.app_version),
        ):
            if not value.strip():
                raise AppError("diagnostics.snapshot_invalid", {"field": field_name})


@dataclass(frozen=True, slots=True)
class DiagnosticsQueueSummary:
    schema_version: int
    revision: int
    active_jobs: int
    terminal_jobs: int
    omitted_terminal_jobs: int
    state_counts: tuple[tuple[str, int], ...]
    profile_counts: tuple[tuple[str, int], ...]
    stage_counts: tuple[tuple[str, int], ...]
    issue_codes: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise AppError("diagnostics.snapshot_invalid", {"field": "queue.schema_version"})
        _require_nonnegative(self.revision, field="queue.revision")
        _require_nonnegative(self.active_jobs, field="queue.active_jobs")
        _require_nonnegative(self.terminal_jobs, field="queue.terminal_jobs")
        _require_nonnegative(self.omitted_terminal_jobs, field="queue.omitted_terminal_jobs")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfigurationSummary:
    schema_version: int
    locale: str
    built_in_preset_count: int
    user_preset_count: int
    provider_configured: bool
    credential_source: CredentialSourceLabel
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise AppError(
                "diagnostics.snapshot_invalid",
                {"field": "configuration.schema_version"},
            )
        if not self.locale.strip():
            raise AppError("diagnostics.snapshot_invalid", {"field": "configuration.locale"})
        _require_nonnegative(
            self.built_in_preset_count, field="configuration.built_in_preset_count"
        )
        _require_nonnegative(self.user_preset_count, field="configuration.user_preset_count")
        if self.credential_source not in _KNOWN_CREDENTIAL_SOURCES:
            raise AppError(
                "diagnostics.snapshot_invalid",
                {"field": "configuration.credential_source"},
            )


@dataclass(frozen=True, slots=True)
class DiagnosticsRecoverySummary:
    schema_version: int
    recoverable_batches: int
    blocked_batches: int
    paused_batches: int
    issue_codes: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise AppError(
                "diagnostics.snapshot_invalid",
                {"field": "recovery.schema_version"},
            )
        _require_nonnegative(self.recoverable_batches, field="recovery.recoverable_batches")
        _require_nonnegative(self.blocked_batches, field="recovery.blocked_batches")
        _require_nonnegative(self.paused_batches, field="recovery.paused_batches")


@dataclass(frozen=True, slots=True)
class DiagnosticsSnapshot:
    schema_version: int
    request_id: str
    generated_at_utc: str
    runtime: RuntimeAvailability
    queue: DiagnosticsQueueSummary
    configuration: DiagnosticsConfigurationSummary
    recovery: DiagnosticsRecoverySummary

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION:
            raise AppError("diagnostics.snapshot_invalid", {"field": "schema_version"})
        validate_identifier(self.request_id, field="request_id")
        if not self.generated_at_utc.strip():
            raise AppError("diagnostics.snapshot_invalid", {"field": "generated_at_utc"})


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    request_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")


@dataclass(frozen=True, slots=True)
class DiagnosticExportRequest:
    request_id: str
    destination: str
    overwrite: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")
        if not self.destination.strip():
            raise AppError("diagnostics.destination_invalid", {"field": "destination"})


@dataclass(frozen=True, slots=True)
class DiagnosticExportResult:
    request_id: str
    destination: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, field="request_id")
        _require_nonnegative(self.size_bytes, field="size_bytes")
        if len(self.sha256) != 64:
            raise AppError("diagnostics.bundle_invalid", {"field": "sha256"})


def _provider_configured(credential_source: CredentialSourceLabel) -> bool:
    return credential_source in {"config", "environment"}


def _summarize_queue(snapshot: object) -> DiagnosticsQueueSummary:
    from captioner.core.application.queue_projection import QueueSnapshot

    if not isinstance(snapshot, QueueSnapshot):
        raise AppError("diagnostics.snapshot_invalid", {"field": "queue"})

    state_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    issue_codes: Counter[str] = Counter()
    active = 0
    terminal = 0
    for item in snapshot.items:
        state_counts[item.state.value] += 1
        profile_counts[item.pipeline_profile.value] += 1
        if item.active_stage is not None:
            stage_counts[item.active_stage.value] += 1
        if item.state in _TERMINAL_JOB_STATES:
            terminal += 1
        else:
            active += 1
    for issue in snapshot.issues:
        issue_codes[issue.code] += 1
    return DiagnosticsQueueSummary(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        revision=snapshot.revision,
        active_jobs=active,
        terminal_jobs=terminal,
        omitted_terminal_jobs=snapshot.omitted_terminal_jobs,
        state_counts=_sorted_count_tuples(state_counts),
        profile_counts=_sorted_count_tuples(profile_counts),
        stage_counts=_sorted_count_tuples(stage_counts),
        issue_codes=_sorted_count_tuples(issue_codes),
    )


def _summarize_configuration(snapshot: object) -> DiagnosticsConfigurationSummary:
    from captioner.core.application.configuration import ConfigurationSnapshot

    if not isinstance(snapshot, ConfigurationSnapshot):
        raise AppError("diagnostics.snapshot_invalid", {"field": "configuration"})

    built_in = sum(1 for preset in snapshot.presets if preset.built_in)
    user = sum(1 for preset in snapshot.presets if not preset.built_in)
    credential_source = snapshot.provider.credential_source
    issue_codes = tuple(sorted({issue.code for issue in snapshot.issues}))
    return DiagnosticsConfigurationSummary(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        locale=snapshot.global_settings.locale,
        built_in_preset_count=built_in,
        user_preset_count=user,
        provider_configured=_provider_configured(credential_source),
        credential_source=credential_source,
        issue_codes=issue_codes,
    )


def _summarize_recovery(snapshot: object) -> DiagnosticsRecoverySummary:
    from captioner.core.application.recovery import RecoverySnapshot

    if not isinstance(snapshot, RecoverySnapshot):
        raise AppError("diagnostics.snapshot_invalid", {"field": "recovery"})

    recoverable = 0
    blocked = 0
    paused = 0
    issue_codes: Counter[str] = Counter()
    for item in snapshot.items:
        if item.blocked_code is not None:
            blocked += 1
        else:
            recoverable += 1
        if item.pause_requested:
            paused += 1
    for issue in snapshot.issues:
        issue_codes[issue.code] += 1
    return DiagnosticsRecoverySummary(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        recoverable_batches=recoverable,
        blocked_batches=blocked,
        paused_batches=paused,
        issue_codes=_sorted_count_tuples(issue_codes),
    )


@dataclass(slots=True)
class DiagnosticsService:
    queue: QueueProjectionService
    configuration: ConfigurationService
    recovery: RecoveryService
    environment: DiagnosticsEnvironmentPort
    writer: DiagnosticBundleWriterPort
    now_utc: Callable[[], str]

    def load(self, request: DiagnosticsRequest) -> DiagnosticsSnapshot:
        queue_snapshot = self.queue.get_queue_snapshot()
        configuration_snapshot = self.configuration.load()
        from captioner.core.application.recovery import RecoveryRequest

        recovery_snapshot = self.recovery.scan(RecoveryRequest(request_id=request.request_id))
        queue_summary = _summarize_queue(queue_snapshot)
        configuration_summary = _summarize_configuration(configuration_snapshot)
        recovery_summary = _summarize_recovery(recovery_snapshot)
        runtime = self.environment.collect_runtime_availability(
            provider_configured=configuration_summary.provider_configured,
            credential_source=configuration_summary.credential_source,
        )
        return DiagnosticsSnapshot(
            schema_version=DIAGNOSTICS_SCHEMA_VERSION,
            request_id=request.request_id,
            generated_at_utc=self.now_utc(),
            runtime=runtime,
            queue=queue_summary,
            configuration=configuration_summary,
            recovery=recovery_summary,
        )

    def export(
        self,
        request: DiagnosticExportRequest,
        *,
        snapshot: DiagnosticsSnapshot | None = None,
    ) -> DiagnosticExportResult:
        resolved = snapshot
        if resolved is None:
            resolved = self.load(DiagnosticsRequest(request_id=request.request_id))
        elif resolved.request_id != request.request_id:
            # Keep correlation explicit; regenerate when IDs diverge.
            resolved = self.load(DiagnosticsRequest(request_id=request.request_id))
        return self.writer.write_bundle(request, snapshot=resolved)


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "DIAGNOSTIC_BUNDLE_SCHEMA_VERSION",
    "DiagnosticExportRequest",
    "DiagnosticExportResult",
    "DiagnosticsConfigurationSummary",
    "DiagnosticsQueueSummary",
    "DiagnosticsRecoverySummary",
    "DiagnosticsRequest",
    "DiagnosticsService",
    "DiagnosticsSnapshot",
    "RuntimeAvailability",
]
