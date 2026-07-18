"""Unit tests for DiagnosticsController refresh/export correlation."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from captioner.core.application.diagnostics import (
    DiagnosticExportRequest,
    DiagnosticExportResult,
    DiagnosticsConfigurationSummary,
    DiagnosticsQueueSummary,
    DiagnosticsRecoverySummary,
    DiagnosticsRequest,
    DiagnosticsSnapshot,
    RuntimeAvailability,
)
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.diagnostics_controller import DiagnosticsController

_app = QApplication.instance() or QApplication(["test-diagnostics-controller"])


def _snapshot(request_id: str = "req-1") -> DiagnosticsSnapshot:
    return DiagnosticsSnapshot(
        schema_version=1,
        request_id=request_id,
        generated_at_utc="2026-07-18T00:00:00+00:00",
        runtime=RuntimeAvailability(
            packaged=False,
            operating_system="Linux",
            architecture="x86_64",
            python_version="3.13.0",
            app_version="0.0.0",
            ffmpeg_available=True,
            ffprobe_available=True,
            asr_runtime_available=False,
            provider_configured=False,
            credential_source="missing",
        ),
        queue=DiagnosticsQueueSummary(
            schema_version=1,
            revision=1,
            active_jobs=0,
            terminal_jobs=0,
            omitted_terminal_jobs=0,
            state_counts=(),
            profile_counts=(),
            stage_counts=(),
            issue_codes=(),
        ),
        configuration=DiagnosticsConfigurationSummary(
            schema_version=1,
            locale="en",
            built_in_preset_count=3,
            user_preset_count=0,
            provider_configured=False,
            credential_source="missing",
            issue_codes=(),
        ),
        recovery=DiagnosticsRecoverySummary(
            schema_version=1,
            recoverable_batches=0,
            blocked_batches=0,
            paused_batches=0,
            issue_codes=(),
        ),
    )


class FakeRunner(QObject):
    diagnostics_ready = Signal(object)
    diagnostics_failure = Signal(object)
    diagnostic_export_ready = Signal(object)
    diagnostic_export_failure = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.loads: list[DiagnosticsRequest] = []
        self.exports: list[DiagnosticExportRequest] = []

    def request_diagnostics_load(self, request: object) -> None:
        assert isinstance(request, DiagnosticsRequest)
        self.loads.append(request)

    def request_diagnostics_export(self, request: object) -> None:
        assert isinstance(request, DiagnosticExportRequest)
        self.exports.append(request)


def test_refresh_coalesce_and_stale_rejection() -> None:
    runner = FakeRunner()
    controller = DiagnosticsController(runner)  # type: ignore[arg-type]
    snapshots: list[object] = []
    controller.snapshot_changed.connect(snapshots.append)
    controller.refresh()
    controller.refresh()  # queued while in flight
    assert len(runner.loads) == 1
    first_id = runner.loads[0].request_id
    # Stale generation after extra refresh while in flight: completing first dispatches second.
    runner.diagnostics_ready.emit(_snapshot(first_id))
    assert len(runner.loads) == 2
    runner.diagnostics_ready.emit(_snapshot(runner.loads[1].request_id))
    assert len(snapshots) == 1
    assert controller.refresh_busy is False


def test_refresh_failure_keeps_last_snapshot() -> None:
    runner = FakeRunner()
    controller = DiagnosticsController(runner)  # type: ignore[arg-type]
    controller.refresh()
    snap = _snapshot(runner.loads[0].request_id)
    runner.diagnostics_ready.emit(snap)
    assert controller.snapshot is snap
    controller.refresh()
    runner.diagnostics_failure.emit(RunnerFailure(code="diagnostics.failed", retryable=False))
    assert controller.snapshot is snap
    assert controller.refresh_busy is False


def test_export_correlation_and_duplicate_block() -> None:
    runner = FakeRunner()
    controller = DiagnosticsController(runner)  # type: ignore[arg-type]
    successes: list[object] = []
    failures: list[object] = []
    controller.export_succeeded.connect(successes.append)
    controller.export_failure_changed.connect(failures.append)
    controller.export("/tmp/a.zip", overwrite=False)
    controller.export("/tmp/b.zip", overwrite=True)  # blocked while in flight
    assert len(runner.exports) == 1
    request = runner.exports[0]
    # Unrelated signal ignored.
    runner.diagnostic_export_ready.emit(
        DiagnosticExportResult(
            request_id="other-req",
            destination="/tmp/other.zip",
            size_bytes=1,
            sha256="b" * 64,
        )
    )
    assert successes == []
    runner.diagnostic_export_ready.emit(
        DiagnosticExportResult(
            request_id=request.request_id,
            destination=request.destination,
            size_bytes=10,
            sha256="c" * 64,
        )
    )
    assert len(successes) == 1
    assert controller.export_busy is False

    controller.export("/tmp/c.zip", overwrite=False)
    runner.diagnostic_export_failure.emit(RunnerFailure(code="diagnostics.write_failed"))
    assert failures[-1] is not None
    assert controller.export_busy is False
