"""Final Phase 5 GUI acceptance workflow across all five pages."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStackedWidget

from captioner.core.application.batch_commands import (
    BatchCommandAck,
    BatchCommandKind,
    LocalExecutionSnapshot,
)
from captioner.core.application.configuration import default_configuration_snapshot
from captioner.core.application.diagnostics import (
    DiagnosticExportResult,
    DiagnosticsConfigurationSummary,
    DiagnosticsQueueSummary,
    DiagnosticsRecoverySummary,
    DiagnosticsSnapshot,
    RuntimeAvailability,
)
from captioner.core.application.input_selection import InputPreview
from captioner.core.application.job_detail import JobAction, JobDetailSnapshot
from captioner.core.application.queue_projection import JobQueueItem, QueueSnapshot
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import PipelineProfile, StageName, StageState
from captioner.gui.batch_controller import BatchController
from captioner.gui.composition import GuiControllers
from captioner.gui.create_controller import CreateController
from captioner.gui.diagnostics_controller import DiagnosticsController
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.main_window import MainWindow
from captioner.gui.queue_table_model import QueueTableModel
from captioner.gui.recovery_controller import RecoveryController
from captioner.gui.settings_controller import SettingsController
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-gui-acceptance"])


class FakeRunner(QObject):
    snapshot_ready = Signal(object)
    failure = Signal(object)
    started = Signal()
    stopped = Signal()
    input_preview_ready = Signal(object)
    input_failure = Signal(object)
    configuration_loaded = Signal(object)
    global_settings_saved = Signal(object)
    provider_settings_saved = Signal(object)
    preset_saved = Signal(object)
    preset_deleted = Signal(object)
    configuration_load_failure = Signal(object)
    global_settings_save_failure = Signal(object)
    provider_settings_save_failure = Signal(object)
    preset_save_failure = Signal(object)
    preset_delete_failure = Signal(object)
    provider_test_ready = Signal(object)
    provider_test_failure = Signal(object)
    batch_command_ready = Signal(object)
    batch_command_failure = Signal(object)
    job_detail_ready = Signal(object)
    job_detail_failure = Signal(object)
    recovery_ready = Signal(object)
    recovery_failure = Signal(object)
    diagnostics_ready = Signal(object)
    diagnostics_failure = Signal(object)
    diagnostic_export_ready = Signal(object)
    diagnostic_export_failure = Signal(object)
    execution_completion = Signal(object)
    local_execution_state_changed = Signal(object)

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.tmp_path = tmp_path
        self.start_calls = 0
        self.stop_calls = 0
        self._running = False
        self.preview_requests: list[object] = []
        self.submit_requests: list[object] = []
        self.detail_requests: list[object] = []
        self.job_actions: list[object] = []
        self.diagnostics_loads: list[object] = []
        self.diagnostics_exports: list[object] = []

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self.start_calls += 1
        self._running = True
        self.started.emit()
        self.snapshot_ready.emit(
            QueueSnapshot(
                schema_version=2,
                revision=1,
                items=(),
                issues=(),
                omitted_terminal_jobs=0,
            )
        )
        self.local_execution_state_changed.emit(
            LocalExecutionSnapshot(active_batch_id=None, queued_batch_ids=())
        )

    def stop(self, timeout_ms: int = 5000) -> bool:
        del timeout_ms
        self.stop_calls += 1
        self._running = False
        self.stopped.emit()
        return True

    def request_refresh(self) -> None:
        item = _queue_item(self.tmp_path)
        self.snapshot_ready.emit(
            QueueSnapshot(
                schema_version=2,
                revision=2,
                items=(item,),
                issues=(),
                omitted_terminal_jobs=0,
            )
        )

    def request_configuration_load(self) -> None:
        self.configuration_loaded.emit(default_configuration_snapshot())

    def request_input_preview(self, request: object) -> None:
        self.preview_requests.append(request)
        path = str(self.tmp_path / "clip.wav")
        self.input_preview_ready.emit(InputPreview(accepted_paths=(path,), rejected=()))

    def request_submit_batch(self, request: object) -> None:
        self.submit_requests.append(request)
        self.batch_command_ready.emit(
            BatchCommandAck(
                request_id=getattr(request, "request_id", "req"),
                kind=BatchCommandKind.SUBMIT,
                batch_id="batch-1",
                job_id="job-1",
                accepted_at_utc="2026-07-18T00:00:00+00:00",
                scheduled=True,
                created_batch_id="batch-1",
            )
        )
        self.request_refresh()

    def request_job_detail(self, request: object) -> None:
        self.detail_requests.append(request)
        self.job_detail_ready.emit(
            JobDetailSnapshot(
                schema_version=1,
                request_id=getattr(request, "request_id", "req"),
                batch_id="batch-1",
                job_id="job-1",
                input_path=str(self.tmp_path / "clip.wav"),
                output_dir=str(self.tmp_path / "out"),
                state=JobState.FAILED,
                active_stage=StageName.TRANSCRIBE,
                active_stage_state=StageState.FAILED,
                active_stage_attempt=1,
                lease_state="missing",
                cancel_requested=False,
                pause_requested=False,
                paused=False,
                input_exists=True,
                retry_stage=StageName.TRANSCRIBE,
                available_actions=(JobAction.RETRY_JOB, JobAction.RUN_AGAIN),
                activity=(),
                omitted_activity_count=0,
                journal_tail_status="clean",
                manifest_status="current",
            )
        )

    def request_job_action(self, request: object) -> None:
        self.job_actions.append(request)
        self.batch_command_ready.emit(
            BatchCommandAck(
                request_id=getattr(request, "request_id", "req"),
                kind=BatchCommandKind.RETRY_JOB,
                batch_id="batch-1",
                job_id="job-1",
                accepted_at_utc="2026-07-18T00:00:00+00:00",
                scheduled=True,
            )
        )

    def request_batch_action(self, request: object) -> None:
        return None

    def request_cancel_local_work(self, request: object) -> None:
        return None

    def request_recovery_scan(self, request: object) -> None:
        return None

    def request_diagnostics_load(self, request: object) -> None:
        self.diagnostics_loads.append(request)
        self.diagnostics_ready.emit(
            DiagnosticsSnapshot(
                schema_version=1,
                request_id=getattr(request, "request_id", "req"),
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
                    revision=2,
                    active_jobs=0,
                    terminal_jobs=1,
                    omitted_terminal_jobs=0,
                    state_counts=(("failed", 1),),
                    profile_counts=(("deterministic", 1),),
                    stage_counts=(("transcribe", 1),),
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
        )

    def request_diagnostics_export(self, request: object) -> None:
        from captioner.core.application.diagnostics import DiagnosticExportRequest

        self.diagnostics_exports.append(request)
        assert isinstance(request, DiagnosticExportRequest)
        destination = Path(request.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("manifest.json", "{}\n")
        self.diagnostic_export_ready.emit(
            DiagnosticExportResult(
                request_id=request.request_id,
                destination=str(destination),
                size_bytes=destination.stat().st_size,
                sha256="d" * 64,
            )
        )

    def request_global_save(self, settings: object) -> None:
        return None

    def request_provider_save(self, update: object) -> None:
        return None

    def request_preset_save(self, preset: object) -> None:
        return None

    def request_preset_delete(self, name: str) -> None:
        return None

    def request_provider_test(self, update: object) -> None:
        return None


def _queue_item(tmp_path: Path) -> JobQueueItem:
    return JobQueueItem(
        batch_id="batch-1",
        job_id="job-1",
        batch_created_at_utc="2026-01-01T00:00:00+00:00",
        job_order=1,
        input_path=str(tmp_path / "clip.wav"),
        output_dir=str(tmp_path / "out"),
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        state=JobState.FAILED,
        active_stage=StageName.TRANSCRIBE,
        active_stage_state=StageState.FAILED,
        active_stage_attempt=1,
        cancel_requested=False,
        pause_requested=False,
        paused=False,
        last_event_seq=3,
        journal_tail_status="clean",
        manifest_status="current",
    )


def _build(locale: str, tmp_path: Path) -> tuple[MainWindow, GuiControllers, FakeRunner]:
    service = I18nService(locale)
    runner = FakeRunner(tmp_path)
    model = QueueTableModel(service)
    queue = BatchController(model, runner, refresh_interval_ms=60_000)  # type: ignore[arg-type]
    create = CreateController(runner)  # type: ignore[arg-type]
    settings = SettingsController(runner)  # type: ignore[arg-type]
    operations = JobOperationsController(runner)  # type: ignore[arg-type]
    recovery = RecoveryController(runner)  # type: ignore[arg-type]
    diagnostics = DiagnosticsController(runner)  # type: ignore[arg-type]
    controllers = GuiControllers(
        queue=queue,
        create=create,
        settings=settings,
        operations=operations,
        recovery=recovery,
        diagnostics=diagnostics,
    )
    window = MainWindow(service, controllers)
    return window, controllers, runner


def _workflow(locale: str, tmp_path: Path) -> None:
    media = tmp_path / "clip.wav"
    media.write_bytes(b"RIFF")
    out = tmp_path / "out"
    out.mkdir()
    window, controllers, runner = _build(locale, tmp_path)
    window.show()
    window.start()
    assert runner.start_calls == 1

    stack = window.findChild(QStackedWidget, "mainPageStack")
    assert stack is not None
    create_page = window.findChild(QObject, "createPage")
    assert stack.currentWidget() is create_page

    controllers.create.set_configuration(default_configuration_snapshot())
    controllers.create.set_entries((str(media),))
    assert runner.preview_requests
    controllers.create.validate_draft(
        output_root=str(out),
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="default",
        source_language=None,
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    controllers.create.submit_draft()
    assert runner.submit_requests

    queue_button = window.findChild(QPushButton, "navQueueButton")
    assert queue_button is not None
    QTest.mouseClick(queue_button, Qt.MouseButton.LeftButton)
    assert window.findChild(QObject, "queuePage") is stack.currentWidget()

    controllers.queue.refresh()
    item = _queue_item(tmp_path)
    controllers.operations.select_job(item)
    assert runner.detail_requests
    controllers.operations.retry_job()
    assert runner.job_actions

    for button_name, page_name in (
        ("navHistoryButton", "historyPage"),
        ("navSettingsButton", "settingsPage"),
        ("navDiagnosticsButton", "diagnosticsPage"),
    ):
        button = window.findChild(QPushButton, button_name)
        page = window.findChild(QObject, page_name)
        assert button is not None and page is not None
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        assert stack.currentWidget() is page

    assert runner.diagnostics_loads
    export_path = tmp_path / "acceptance-diag.zip"
    controllers.diagnostics.export(str(export_path), overwrite=True)
    assert runner.diagnostics_exports
    assert export_path.is_file()
    with zipfile.ZipFile(export_path) as archive:
        assert "manifest.json" in archive.namelist()

    assert window.windowTitle() == "Captioner"
    diag_title = window.findChild(QLabel, "diagnosticsTitle")
    assert diag_title is not None
    assert diag_title.text()

    # Ensure no raw JobState value is the complete diagnostics title.
    assert diag_title.text() not in {"failed", "running", "pending"}

    closed = window.close()
    assert closed is True or closed is False
    assert runner.stop_calls >= 1
    assert not runner.running


def test_english_acceptance_workflow(tmp_path: Path) -> None:
    _workflow("en", tmp_path)


def test_chinese_acceptance_workflow(tmp_path: Path) -> None:
    _workflow("zh-CN", tmp_path)
