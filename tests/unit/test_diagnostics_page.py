"""Unit tests for DiagnosticsPage presentation."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from captioner.core.application.diagnostics import (
    DiagnosticsConfigurationSummary,
    DiagnosticsQueueSummary,
    DiagnosticsRecoverySummary,
    DiagnosticsSnapshot,
    RuntimeAvailability,
)
from captioner.gui.diagnostics_controller import DiagnosticsController
from captioner.gui.pages.diagnostics_page import DiagnosticsPage
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-diagnostics-page"])


class FakeRunner(QObject):
    diagnostics_ready = Signal(object)
    diagnostics_failure = Signal(object)
    diagnostic_export_ready = Signal(object)
    diagnostic_export_failure = Signal(object)

    def request_diagnostics_load(self, request: object) -> None:
        return None

    def request_diagnostics_export(self, request: object) -> None:
        return None


def _snapshot() -> DiagnosticsSnapshot:
    return DiagnosticsSnapshot(
        schema_version=1,
        request_id="req-page",
        generated_at_utc="2026-07-18T00:00:00+00:00",
        runtime=RuntimeAvailability(
            packaged=False,
            operating_system="Linux",
            architecture="x86_64",
            python_version="3.13.0",
            app_version="1.2.3",
            ffmpeg_available=True,
            ffprobe_available=False,
            asr_runtime_available=False,
            provider_configured=True,
            credential_source="environment",
        ),
        queue=DiagnosticsQueueSummary(
            schema_version=1,
            revision=1,
            active_jobs=2,
            terminal_jobs=1,
            omitted_terminal_jobs=3,
            state_counts=(("running", 2),),
            profile_counts=(),
            stage_counts=(),
            issue_codes=(("queue.batch_read_failed", 1),),
        ),
        configuration=DiagnosticsConfigurationSummary(
            schema_version=1,
            locale="en",
            built_in_preset_count=3,
            user_preset_count=0,
            provider_configured=True,
            credential_source="environment",
            issue_codes=(),
        ),
        recovery=DiagnosticsRecoverySummary(
            schema_version=1,
            recoverable_batches=1,
            blocked_batches=0,
            paused_batches=1,
            issue_codes=(),
        ),
    )


def _page(locale: str = "en") -> DiagnosticsPage:
    service = I18nService(locale)
    runner = FakeRunner()
    controller = DiagnosticsController(runner)  # type: ignore[arg-type]
    return DiagnosticsPage(service, controller)


def test_required_object_names_and_future_controls() -> None:
    page = _page("en")
    assert page.objectName() == "diagnosticsPage"
    for name in (
        "diagnosticsTitle",
        "diagnosticsRefreshButton",
        "diagnosticsRefreshingLabel",
        "diagnosticsFailureLabel",
        "diagnosticsApplicationGroup",
        "diagnosticsAppVersionLabel",
        "diagnosticsOsLabel",
        "diagnosticsArchitectureLabel",
        "diagnosticsPythonLabel",
        "diagnosticsPackagedLabel",
        "diagnosticsCapabilitiesGroup",
        "diagnosticsFfmpegLabel",
        "diagnosticsFfprobeLabel",
        "diagnosticsAsrRuntimeLabel",
        "diagnosticsProviderLabel",
        "diagnosticsCredentialSourceLabel",
        "diagnosticsQueueGroup",
        "diagnosticsQueueSummaryLabel",
        "diagnosticsQueueIssuesLabel",
        "diagnosticsRecoveryGroup",
        "diagnosticsRecoverySummaryLabel",
        "diagnosticsRecoveryIssuesLabel",
        "diagnosticsRuntimeGroup",
        "diagnosticsInstallRuntimeButton",
        "diagnosticsManageModelsButton",
        "diagnosticsPrivacyLabel",
        "diagnosticsExportButton",
        "diagnosticsExportingLabel",
        "diagnosticsExportFailureLabel",
        "diagnosticsExportSuccessLabel",
    ):
        assert page.findChild(QObject, name) is not None, name
    install = page.findChild(QPushButton, "diagnosticsInstallRuntimeButton")
    manage = page.findChild(QPushButton, "diagnosticsManageModelsButton")
    assert install is not None and manage is not None
    assert not install.isEnabled()
    assert not manage.isEnabled()
    privacy = page.findChild(QLabel, "diagnosticsPrivacyLabel")
    assert privacy is not None
    assert privacy.text()


def test_snapshot_render_english_and_chinese() -> None:
    for locale in ("en", "zh-CN"):
        service = I18nService(locale)
        runner = FakeRunner()
        controller = DiagnosticsController(runner)  # type: ignore[arg-type]
        page = DiagnosticsPage(service, controller)
        controller.snapshot_changed.emit(_snapshot())
        version = page.findChild(QLabel, "diagnosticsAppVersionLabel")
        assert version is not None
        assert version.text() == "1.2.3"
        ffmpeg = page.findChild(QLabel, "diagnosticsFfmpegLabel")
        assert ffmpeg is not None
        assert ffmpeg.text()
        # No raw credential_source enum value as complete label.
        cred = page.findChild(QLabel, "diagnosticsCredentialSourceLabel")
        assert cred is not None
        assert cred.text() != "environment"
        assert cred.text() != "config"
        assert cred.text() != "missing"
        queue = page.findChild(QLabel, "diagnosticsQueueSummaryLabel")
        assert queue is not None
        assert "2" in queue.text()
