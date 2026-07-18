"""Unit tests for DiagnosticsPage presentation."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from captioner.core.application.diagnostics import (
    DiagnosticsConfigurationSummary,
    DiagnosticsQueueSummary,
    DiagnosticsRecoverySummary,
    DiagnosticsSnapshot,
    DiagnosticsStorageLocations,
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
        storage=DiagnosticsStorageLocations(
            config_dir="/tmp/captioner/config",
            data_dir="/tmp/captioner/data",
            models_dir="/tmp/captioner/data/models",
            runtimes_dir="/tmp/captioner/data/runtimes",
            workspaces_dir="/tmp/captioner/data/workspaces",
            cache_dir="/tmp/captioner/cache",
            log_dir="/tmp/captioner/log",
            downloads_dir="/tmp/captioner/data/downloads",
            artifacts_dir="/tmp/captioner/data/artifacts",
            staging_dir="/tmp/captioner/data/staging",
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
        "diagnosticsStorageGroup",
        "diagnosticsConfigPathLabel",
        "diagnosticsDataPathLabel",
        "diagnosticsModelsPathLabel",
        "diagnosticsRuntimesPathLabel",
        "diagnosticsWorkspacesPathLabel",
        "diagnosticsCachePathLabel",
        "diagnosticsLogsPathLabel",
        "diagnosticsOpenConfigFolderButton",
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
        config_path = page.findChild(QLabel, "diagnosticsConfigPathLabel")
        assert config_path is not None
        assert config_path.text() == "/tmp/captioner/config"
        assert config_path.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_open_folder_uses_snapshot_path_as_local_file_url(tmp_path: Path) -> None:
    page = _page("en")
    target = tmp_path / "config"
    target.mkdir()
    storage = replace(_snapshot().storage, config_dir=str(target))
    page._on_snapshot(  # pyright: ignore[reportPrivateUsage]  # render injected snapshot
        replace(_snapshot(), storage=storage)
    )
    with patch(
        "captioner.gui.pages.diagnostics_page.QDesktopServices.openUrl",
        return_value=True,
    ) as open_url:
        page._on_open_storage(  # pyright: ignore[reportPrivateUsage]  # invoke button slot
            "config_dir"
        )
    url = open_url.call_args.args[0]
    assert url.isLocalFile()
    assert Path(url.toLocalFile()) == target
    page.close()
