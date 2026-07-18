"""Diagnostics page: aggregate runtime summary and redacted bundle export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.diagnostics import (
    DiagnosticExportResult,
    DiagnosticsSnapshot,
    DiagnosticsStorageLocations,
)
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.diagnostics_controller import DiagnosticsController
from captioner.i18n.service import I18nService


class DiagnosticsPage(QWidget):
    def __init__(
        self,
        service: I18nService,
        controller: DiagnosticsController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("diagnosticsPage")
        self._service = service
        self._controller = controller
        self._visited = False
        self._storage_paths: dict[str, str] = {}

        root = QVBoxLayout(self)
        title = QLabel(service.translate("gui.diagnostics.title"))
        title.setObjectName("diagnosticsTitle")
        root.addWidget(title)

        toolbar = QHBoxLayout()
        self._refresh_button = QPushButton(service.translate("gui.diagnostics.refresh"))
        self._refresh_button.setObjectName("diagnosticsRefreshButton")
        self._refresh_button.clicked.connect(self._on_refresh)
        toolbar.addWidget(self._refresh_button)
        self._refreshing = QLabel(service.translate("gui.diagnostics.refreshing"))
        self._refreshing.setObjectName("diagnosticsRefreshingLabel")
        self._refreshing.setVisible(False)
        toolbar.addWidget(self._refreshing)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        self._failure = QLabel("")
        self._failure.setObjectName("diagnosticsFailureLabel")
        self._failure.setVisible(False)
        root.addWidget(self._failure)

        app_group = QGroupBox(service.translate("gui.diagnostics.application"))
        app_group.setObjectName("diagnosticsApplicationGroup")
        app_form = QFormLayout(app_group)
        self._app_version = QLabel("")
        self._app_version.setObjectName("diagnosticsAppVersionLabel")
        self._os = QLabel("")
        self._os.setObjectName("diagnosticsOsLabel")
        self._architecture = QLabel("")
        self._architecture.setObjectName("diagnosticsArchitectureLabel")
        self._python = QLabel("")
        self._python.setObjectName("diagnosticsPythonLabel")
        self._packaged = QLabel("")
        self._packaged.setObjectName("diagnosticsPackagedLabel")
        app_form.addRow(service.translate("gui.diagnostics.app_version"), self._app_version)
        app_form.addRow(service.translate("gui.diagnostics.operating_system"), self._os)
        app_form.addRow(service.translate("gui.diagnostics.architecture"), self._architecture)
        app_form.addRow(service.translate("gui.diagnostics.python"), self._python)
        app_form.addRow(service.translate("gui.diagnostics.packaged"), self._packaged)
        root.addWidget(app_group)

        cap_group = QGroupBox(service.translate("gui.diagnostics.capabilities"))
        cap_group.setObjectName("diagnosticsCapabilitiesGroup")
        cap_form = QFormLayout(cap_group)
        self._ffmpeg = QLabel("")
        self._ffmpeg.setObjectName("diagnosticsFfmpegLabel")
        self._ffprobe = QLabel("")
        self._ffprobe.setObjectName("diagnosticsFfprobeLabel")
        self._asr = QLabel("")
        self._asr.setObjectName("diagnosticsAsrRuntimeLabel")
        self._provider = QLabel("")
        self._provider.setObjectName("diagnosticsProviderLabel")
        self._credential_source = QLabel("")
        self._credential_source.setObjectName("diagnosticsCredentialSourceLabel")
        cap_form.addRow(service.translate("gui.diagnostics.ffmpeg"), self._ffmpeg)
        cap_form.addRow(service.translate("gui.diagnostics.ffprobe"), self._ffprobe)
        cap_form.addRow(service.translate("gui.diagnostics.asr_runtime"), self._asr)
        cap_form.addRow(service.translate("gui.diagnostics.provider"), self._provider)
        cap_form.addRow(
            service.translate("gui.diagnostics.credential_source"),
            self._credential_source,
        )
        root.addWidget(cap_group)

        storage_group = QGroupBox(service.translate("gui.diagnostics.storage"))
        storage_group.setObjectName("diagnosticsStorageGroup")
        storage_form = QFormLayout(storage_group)
        self._storage_labels: dict[str, QLabel] = {}
        self._storage_buttons: dict[str, QPushButton] = {}
        for key, label_key, label_name, button_name in (
            (
                "config_dir",
                "gui.diagnostics.storage.configuration",
                "diagnosticsConfigPathLabel",
                "diagnosticsOpenConfigFolderButton",
            ),
            (
                "data_dir",
                "gui.diagnostics.storage.application_data",
                "diagnosticsDataPathLabel",
                "diagnosticsOpenDataFolderButton",
            ),
            (
                "models_dir",
                "gui.diagnostics.storage.models",
                "diagnosticsModelsPathLabel",
                "diagnosticsOpenModelsFolderButton",
            ),
            (
                "runtimes_dir",
                "gui.diagnostics.storage.runtimes",
                "diagnosticsRuntimesPathLabel",
                "diagnosticsOpenRuntimesFolderButton",
            ),
            (
                "workspaces_dir",
                "gui.diagnostics.storage.workspaces",
                "diagnosticsWorkspacesPathLabel",
                "diagnosticsOpenWorkspacesFolderButton",
            ),
            (
                "cache_dir",
                "gui.diagnostics.storage.cache",
                "diagnosticsCachePathLabel",
                "diagnosticsOpenCacheFolderButton",
            ),
            (
                "log_dir",
                "gui.diagnostics.storage.logs",
                "diagnosticsLogsPathLabel",
                "diagnosticsOpenLogsFolderButton",
            ),
        ):
            path_label = QLabel("")
            path_label.setObjectName(label_name)
            path_label.setWordWrap(True)
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            path_label.setToolTip("")
            open_button = QPushButton(service.translate("gui.diagnostics.open_folder"))
            open_button.setObjectName(button_name)
            open_button.setEnabled(False)
            open_button.clicked.connect(
                lambda _checked=False, storage_key=key: self._on_open_storage(storage_key)
            )
            row = QHBoxLayout()
            row.addWidget(path_label, stretch=1)
            row.addWidget(open_button)
            storage_form.addRow(service.translate(label_key), row)
            self._storage_labels[key] = path_label
            self._storage_buttons[key] = open_button
        root.addWidget(storage_group)

        queue_group = QGroupBox(service.translate("gui.diagnostics.queue"))
        queue_group.setObjectName("diagnosticsQueueGroup")
        queue_layout = QVBoxLayout(queue_group)
        self._queue_summary = QLabel("")
        self._queue_summary.setObjectName("diagnosticsQueueSummaryLabel")
        self._queue_summary.setWordWrap(True)
        self._queue_issues = QLabel("")
        self._queue_issues.setObjectName("diagnosticsQueueIssuesLabel")
        self._queue_issues.setWordWrap(True)
        queue_layout.addWidget(self._queue_summary)
        queue_layout.addWidget(self._queue_issues)
        root.addWidget(queue_group)

        recovery_group = QGroupBox(service.translate("gui.diagnostics.recovery"))
        recovery_group.setObjectName("diagnosticsRecoveryGroup")
        recovery_layout = QVBoxLayout(recovery_group)
        self._recovery_summary = QLabel("")
        self._recovery_summary.setObjectName("diagnosticsRecoverySummaryLabel")
        self._recovery_summary.setWordWrap(True)
        self._recovery_issues = QLabel("")
        self._recovery_issues.setObjectName("diagnosticsRecoveryIssuesLabel")
        self._recovery_issues.setWordWrap(True)
        recovery_layout.addWidget(self._recovery_summary)
        recovery_layout.addWidget(self._recovery_issues)
        root.addWidget(recovery_group)

        runtime_group = QGroupBox(service.translate("gui.diagnostics.runtime"))
        runtime_group.setObjectName("diagnosticsRuntimeGroup")
        runtime_layout = QHBoxLayout(runtime_group)
        future_tip = service.translate("gui.diagnostics.future_control_tooltip")
        self._install_runtime = QPushButton(service.translate("gui.diagnostics.install_runtime"))
        self._install_runtime.setObjectName("diagnosticsInstallRuntimeButton")
        self._install_runtime.setEnabled(False)
        self._install_runtime.setToolTip(future_tip)
        self._manage_models = QPushButton(service.translate("gui.diagnostics.manage_models"))
        self._manage_models.setObjectName("diagnosticsManageModelsButton")
        self._manage_models.setEnabled(False)
        self._manage_models.setToolTip(future_tip)
        runtime_layout.addWidget(self._install_runtime)
        runtime_layout.addWidget(self._manage_models)
        runtime_layout.addStretch(1)
        root.addWidget(runtime_group)

        privacy = QLabel(service.translate("gui.diagnostics.privacy"))
        privacy.setObjectName("diagnosticsPrivacyLabel")
        privacy.setWordWrap(True)
        root.addWidget(privacy)

        export_row = QHBoxLayout()
        self._export_button = QPushButton(service.translate("gui.diagnostics.export"))
        self._export_button.setObjectName("diagnosticsExportButton")
        self._export_button.clicked.connect(self._on_export)
        export_row.addWidget(self._export_button)
        self._exporting = QLabel(service.translate("gui.diagnostics.exporting"))
        self._exporting.setObjectName("diagnosticsExportingLabel")
        self._exporting.setVisible(False)
        export_row.addWidget(self._exporting)
        export_row.addStretch(1)
        root.addLayout(export_row)

        self._export_failure = QLabel("")
        self._export_failure.setObjectName("diagnosticsExportFailureLabel")
        self._export_failure.setVisible(False)
        root.addWidget(self._export_failure)

        self._export_success = QLabel("")
        self._export_success.setObjectName("diagnosticsExportSuccessLabel")
        self._export_success.setVisible(False)
        self._export_success.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._export_success.setWordWrap(True)
        root.addWidget(self._export_success)
        root.addStretch(1)

        controller.snapshot_changed.connect(self._on_snapshot)
        controller.refresh_busy_changed.connect(self._on_refresh_busy)
        controller.refresh_failure_changed.connect(self._on_refresh_failure)
        controller.export_busy_changed.connect(self._on_export_busy)
        controller.export_succeeded.connect(self._on_export_succeeded)
        controller.export_failure_changed.connect(self._on_export_failure)

    def on_shown(self) -> None:
        """Called when the page becomes the active sidebar page."""
        first = not self._visited
        self._visited = True
        self._controller.refresh()
        if first:
            return

    def _on_refresh(self) -> None:
        self._controller.refresh()

    def _on_export(self) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        suggested = f"captioner-diagnostics-{stamp}.zip"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            self._service.translate("gui.diagnostics.export"),
            suggested,
            "ZIP (*.zip)",
        )
        if not path:
            return
        destination = path if path.lower().endswith(".zip") else f"{path}.zip"
        overwrite = False
        if Path(destination).exists():
            box = QMessageBox(self)
            box.setObjectName("diagnosticsExportExistsDialog")
            box.setWindowTitle(self._service.translate("gui.diagnostics.export_exists.title"))
            box.setText(self._service.translate("gui.diagnostics.export_exists.message"))
            overwrite_btn = box.addButton(
                self._service.translate("gui.diagnostics.export_overwrite"),
                QMessageBox.ButtonRole.AcceptRole,
            )
            cancel_btn = box.addButton(
                self._service.translate("gui.diagnostics.export_cancel"),
                QMessageBox.ButtonRole.RejectRole,
            )
            box.setDefaultButton(cancel_btn)
            box.exec()
            if box.clickedButton() is not overwrite_btn:
                return
            overwrite = True
        self._export_success.clear()
        self._export_success.setVisible(False)
        self._export_failure.clear()
        self._export_failure.setVisible(False)
        self._controller.export(destination, overwrite=overwrite)

    def _on_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, DiagnosticsSnapshot):
            return
        runtime = snapshot.runtime
        self._app_version.setText(runtime.app_version)
        self._os.setText(runtime.operating_system)
        self._architecture.setText(runtime.architecture)
        self._python.setText(runtime.python_version)
        self._packaged.setText(
            self._service.translate(
                "gui.diagnostics.packaged.yes"
                if runtime.packaged
                else "gui.diagnostics.packaged.no"
            )
        )
        self._ffmpeg.setText(self._availability(runtime.ffmpeg_available))
        self._ffprobe.setText(self._availability(runtime.ffprobe_available))
        self._asr.setText(self._availability(runtime.asr_runtime_available))
        self._provider.setText(
            self._service.translate(
                "gui.diagnostics.provider_configured"
                if runtime.provider_configured
                else "gui.diagnostics.provider_missing"
            )
        )
        self._credential_source.setText(self._credential_label(runtime.credential_source))

        storage = snapshot.storage
        self._storage_paths = self._storage_values(storage)
        unavailable = self._service.translate("gui.value.unavailable")
        for key, label in self._storage_labels.items():
            path = self._storage_paths[key]
            label.setText(path or unavailable)
            label.setToolTip(path)
            self._storage_buttons[key].setEnabled(bool(path))

        queue = snapshot.queue
        self._queue_summary.setText(
            self._service.translate(
                "gui.diagnostics.queue_summary",
                {
                    "active": str(queue.active_jobs),
                    "terminal": str(queue.terminal_jobs),
                    "hidden": str(queue.omitted_terminal_jobs),
                },
            )
        )
        if queue.issue_codes:
            codes = ", ".join(f"{code}x{count}" for code, count in queue.issue_codes)
            self._queue_issues.setText(
                self._service.translate(
                    "gui.diagnostics.queue_issues",
                    {"codes": codes},
                )
            )
            self._queue_issues.setVisible(True)
        else:
            self._queue_issues.setText(self._service.translate("gui.value.none"))
            self._queue_issues.setVisible(True)

        recovery = snapshot.recovery
        self._recovery_summary.setText(
            self._service.translate(
                "gui.diagnostics.recovery_summary",
                {
                    "recoverable": str(recovery.recoverable_batches),
                    "blocked": str(recovery.blocked_batches),
                    "paused": str(recovery.paused_batches),
                },
            )
        )
        if recovery.issue_codes:
            codes = ", ".join(f"{code}x{count}" for code, count in recovery.issue_codes)
            self._recovery_issues.setText(
                self._service.translate(
                    "gui.diagnostics.recovery_issues",
                    {"codes": codes},
                )
            )
        else:
            self._recovery_issues.setText(self._service.translate("gui.value.none"))
        self._failure.clear()
        self._failure.setVisible(False)

    def _on_refresh_busy(self, busy: object) -> None:
        active = bool(busy)
        self._refreshing.setVisible(active)
        self._refresh_button.setEnabled(not active)

    def _on_refresh_failure(self, failure: object) -> None:
        if failure is None:
            self._failure.clear()
            self._failure.setVisible(False)
            return
        code = (
            failure.code if isinstance(failure, RunnerFailure) else "gui.application_bridge_failed"
        )
        self._failure.setText(self._service.translate("gui.diagnostics.failure", {"code": code}))
        self._failure.setVisible(True)

    def _on_export_busy(self, busy: object) -> None:
        active = bool(busy)
        self._exporting.setVisible(active)
        self._export_button.setEnabled(not active)

    def _on_export_succeeded(self, result: object) -> None:
        if not isinstance(result, DiagnosticExportResult):
            return
        self._export_success.setText(
            self._service.translate(
                "gui.diagnostics.export_success",
                {"path": result.destination},
            )
        )
        self._export_success.setVisible(True)
        self._export_failure.clear()
        self._export_failure.setVisible(False)

    def _on_export_failure(self, failure: object) -> None:
        if failure is None:
            self._export_failure.clear()
            self._export_failure.setVisible(False)
            return
        code = (
            failure.code if isinstance(failure, RunnerFailure) else "gui.application_bridge_failed"
        )
        self._export_failure.setText(
            self._service.translate("gui.diagnostics.export_failure", {"code": code})
        )
        self._export_failure.setVisible(True)

    def _availability(self, available: bool) -> str:
        key = "gui.value.available" if available else "gui.value.unavailable"
        return self._service.translate(key)

    def _credential_label(self, source: str) -> str:
        key = {
            "config": "gui.value.config",
            "environment": "gui.value.environment",
            "missing": "gui.value.missing",
        }.get(source, "gui.value.missing")
        return self._service.translate(key)

    def _on_open_storage(self, key: str) -> None:
        path_text = self._storage_paths.get(key, "")
        if not path_text:
            return
        path = Path(path_text)
        if not path.is_dir():
            self._show_storage_error("gui.diagnostics.storage_missing", path_text)
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._show_storage_error("gui.diagnostics.storage_open_failed", path_text)

    def _show_storage_error(self, message_key: str, path: str) -> None:
        box = QMessageBox(self)
        box.setObjectName("diagnosticsStorageErrorDialog")
        box.setWindowTitle(self._service.translate("gui.diagnostics.storage_error_title"))
        box.setText(self._service.translate(message_key, {"path": path}))
        box.exec()

    @staticmethod
    def _storage_values(storage: DiagnosticsStorageLocations) -> dict[str, str]:
        return {
            "config_dir": storage.config_dir,
            "data_dir": storage.data_dir,
            "models_dir": storage.models_dir,
            "runtimes_dir": storage.runtimes_dir,
            "workspaces_dir": storage.workspaces_dir,
            "cache_dir": storage.cache_dir,
            "log_dir": storage.log_dir,
        }


__all__ = ["DiagnosticsPage"]
