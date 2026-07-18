"""Create page for input selection, presets, and draft validation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.configuration import ConfigurationSnapshot, ExecutionPreset
from captioner.core.application.input_selection import BatchDraft, InputPreview
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.create_controller import CreateController
from captioner.gui.widgets.collapsible_section import CollapsibleSection
from captioner.i18n.service import I18nService

_REJECTION_KEYS = {
    "input.not_found": "gui.create.input.rejection.not_found",
    "input.unsupported": "gui.create.input.rejection.unsupported",
    "input.unreadable": "gui.create.input.rejection.unreadable",
    "input.directory_unreadable": "gui.create.input.rejection.directory_unreadable",
    "input.result_limit": "gui.create.input.rejection.result_limit",
}


class CreatePage(QWidget):
    """Functional Create surface for PR5.3."""

    def __init__(
        self,
        service: I18nService,
        controller: CreateController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("createPage")
        self.setAcceptDrops(True)
        self._service = service
        self._controller = controller
        self._applying_preset = False

        root = QVBoxLayout(self)
        title = QLabel(service.translate("gui.create.title"))
        title.setObjectName("createTitle")
        root.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.create.section.input"),
                self._build_input_section(),
                object_name="createInputSection",
            )
        )
        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.create.section.recognition"),
                self._build_recognition_section(),
                object_name="createRecognitionSection",
            )
        )
        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.create.section.language"),
                self._build_language_section(),
                object_name="createLanguageSection",
            )
        )
        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.create.section.subtitle"),
                self._build_subtitle_section(),
                object_name="createSubtitleSection",
            )
        )
        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.create.section.output"),
                self._build_output_section(),
                object_name="createOutputSection",
            )
        )

        validate_row = QHBoxLayout()
        self._validate_button = QPushButton(service.translate("gui.create.validate"))
        self._validate_button.setObjectName("createValidateButton")
        self._validate_button.clicked.connect(self._on_validate)
        self._submit_button = QPushButton(service.translate("gui.create.submit"))
        self._submit_button.setObjectName("createSubmitButton")
        self._submit_button.clicked.connect(self._on_submit)
        self._submit_button.setEnabled(False)
        validate_row.addWidget(self._validate_button)
        validate_row.addWidget(self._submit_button)
        validate_row.addStretch(1)
        body_layout.addLayout(validate_row)

        self._submit_busy = QLabel(service.translate("gui.create.submitting"))
        self._submit_busy.setObjectName("createSubmitBusyLabel")
        self._submit_busy.setVisible(False)
        body_layout.addWidget(self._submit_busy)

        self._submit_failure = QLabel("")
        self._submit_failure.setObjectName("createSubmitFailureLabel")
        self._submit_failure.setVisible(False)
        body_layout.addWidget(self._submit_failure)

        self._draft_status = QLabel("")
        self._draft_status.setObjectName("createDraftStatusLabel")
        body_layout.addWidget(self._draft_status)

        self._draft_failure = QLabel("")
        self._draft_failure.setObjectName("createDraftFailureLabel")
        self._draft_failure.setVisible(False)
        body_layout.addWidget(self._draft_failure)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

        controller.entries_changed.connect(self._on_entries)
        controller.preview_changed.connect(self._on_preview)
        controller.configuration_changed.connect(self._on_configuration)
        controller.draft_changed.connect(self._on_draft)
        controller.busy_changed.connect(self._on_busy)
        controller.failure_changed.connect(self._on_failure)
        controller.preset_busy_changed.connect(self._on_preset_busy)
        controller.submission_busy_changed.connect(self._on_submission_busy)
        controller.submission_failure_changed.connect(self._on_submission_failure)
        controller.batch_submitted.connect(self._on_batch_submitted)
        self._wire_draft_invalidation()
        self._update_llm_enabled()
        self._update_submit_enabled()

    def _build_input_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        buttons = QHBoxLayout()
        add_files = QPushButton(self._service.translate("gui.create.input.add_files"))
        add_files.setObjectName("createAddFilesButton")
        add_files.clicked.connect(self._on_add_files)
        add_folder = QPushButton(self._service.translate("gui.create.input.add_folder"))
        add_folder.setObjectName("createAddFolderButton")
        add_folder.clicked.connect(self._on_add_folder)
        remove = QPushButton(self._service.translate("gui.create.input.remove"))
        remove.setObjectName("createRemoveInputButton")
        remove.clicked.connect(self._on_remove)
        clear = QPushButton(self._service.translate("gui.create.input.clear"))
        clear.setObjectName("createClearInputsButton")
        clear.clicked.connect(self._controller.clear_entries)
        buttons.addWidget(add_files)
        buttons.addWidget(add_folder)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._recursive = QCheckBox(self._service.translate("gui.create.input.recursive"))
        self._recursive.setObjectName("createRecursiveCheck")
        self._recursive.setChecked(True)
        self._recursive.toggled.connect(self._controller.set_recursive)
        layout.addWidget(self._recursive)

        self._input_list = QListWidget()
        self._input_list.setObjectName("createInputList")
        layout.addWidget(self._input_list)

        self._accepted_label = QLabel(
            self._service.translate("gui.create.input.accepted", {"count": "0"})
        )
        self._accepted_label.setObjectName("createAcceptedLabel")
        layout.addWidget(self._accepted_label)

        self._rejected_label = QLabel(
            self._service.translate("gui.create.input.rejected", {"count": "0"})
        )
        self._rejected_label.setObjectName("createRejectedLabel")
        layout.addWidget(self._rejected_label)

        self._input_failure = QLabel("")
        self._input_failure.setObjectName("createInputFailureLabel")
        self._input_failure.setVisible(False)
        layout.addWidget(self._input_failure)
        return widget

    def _build_recognition_section(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self._preset_combo = QComboBox()
        self._preset_combo.setObjectName("createPresetCombo")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        form.addRow(self._service.translate("gui.create.preset"), self._preset_combo)

        self._preset_name = QLineEdit()
        self._preset_name.setObjectName("createPresetNameEdit")
        form.addRow(self._service.translate("gui.create.preset_name"), self._preset_name)

        preset_buttons = QHBoxLayout()
        save_preset = QPushButton(self._service.translate("gui.create.preset.save"))
        save_preset.setObjectName("createSavePresetButton")
        save_preset.clicked.connect(self._on_save_preset)
        delete_preset = QPushButton(self._service.translate("gui.create.preset.delete"))
        delete_preset.setObjectName("createDeletePresetButton")
        delete_preset.clicked.connect(self._on_delete_preset)
        preset_buttons.addWidget(save_preset)
        preset_buttons.addWidget(delete_preset)
        preset_buttons.addStretch(1)
        form.addRow(preset_buttons)

        self._profile_combo = QComboBox()
        self._profile_combo.setObjectName("createProfileCombo")
        for profile, key in (
            (PipelineProfile.DETERMINISTIC, "gui.profile.deterministic"),
            (PipelineProfile.FAST, "gui.profile.fast"),
            (PipelineProfile.QUALITY, "gui.profile.quality"),
        ):
            self._profile_combo.addItem(self._service.translate(key), profile.value)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        form.addRow(self._service.translate("gui.create.profile"), self._profile_combo)

        self._model_edit = QLineEdit("tiny")
        self._model_edit.setObjectName("createModelEdit")
        form.addRow(self._service.translate("gui.create.model"), self._model_edit)

        self._device_combo = QComboBox()
        self._device_combo.setObjectName("createDeviceCombo")
        for value, key in (
            ("auto", "gui.device.auto"),
            ("cpu", "gui.device.cpu"),
            ("cuda", "gui.device.cuda"),
        ):
            self._device_combo.addItem(self._service.translate(key), value)
        form.addRow(self._service.translate("gui.create.device"), self._device_combo)

        self._compute_edit = QLineEdit("default")
        self._compute_edit.setObjectName("createComputeTypeEdit")
        form.addRow(self._service.translate("gui.create.compute_type"), self._compute_edit)
        return widget

    def _build_language_section(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self._source_edit = QLineEdit()
        self._source_edit.setObjectName("createSourceLanguageEdit")
        form.addRow(self._service.translate("gui.create.source_language"), self._source_edit)

        self._source_auto = QCheckBox(self._service.translate("gui.create.source_auto"))
        self._source_auto.setObjectName("createSourceLanguageAutoCheck")
        self._source_auto.setChecked(True)
        self._source_auto.toggled.connect(self._on_source_auto_toggled)
        form.addRow(self._source_auto)
        self._source_edit.setEnabled(False)

        self._target_edit = QLineEdit("zh-CN")
        self._target_edit.setObjectName("createTargetLanguageEdit")
        form.addRow(self._service.translate("gui.create.target_language"), self._target_edit)

        self._provider_profile = QLineEdit("default")
        self._provider_profile.setObjectName("createProviderProfileEdit")
        form.addRow(
            self._service.translate("gui.create.provider_profile"),
            self._provider_profile,
        )
        return widget

    def _build_subtitle_section(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        description = QLabel(self._service.translate("gui.create.subtitle.description"))
        description.setWordWrap(True)
        description.setObjectName("createSubtitleDescription")
        layout.addWidget(description)
        return widget

    def _build_output_section(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setObjectName("createOutputRootEdit")
        browse = QPushButton(self._service.translate("gui.create.output_browse"))
        browse.setObjectName("createBrowseOutputButton")
        browse.clicked.connect(self._on_browse_output)
        output_row.addWidget(self._output_edit)
        output_row.addWidget(browse)
        form.addRow(self._service.translate("gui.create.output_root"), output_row)

        self._collision_combo = QComboBox()
        self._collision_combo.setObjectName("createCollisionPolicyCombo")
        for value, key in (
            ("unique_subdir", "gui.create.collision.unique_subdir"),
            ("fail", "gui.create.collision.fail"),
            ("overwrite", "gui.create.collision.overwrite"),
        ):
            self._collision_combo.addItem(self._service.translate(key), value)
        form.addRow(
            self._service.translate("gui.create.collision_policy"),
            self._collision_combo,
        )
        return widget

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        paths = tuple(url.toLocalFile() for url in urls if url.isLocalFile() and url.toLocalFile())
        if paths:
            self._controller.append_entries(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self)
        if paths:
            self._controller.append_entries(tuple(paths))

    def _on_add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self)
        if directory:
            self._controller.append_entries((directory,))

    def _on_remove(self) -> None:
        row = self._input_list.currentRow()
        if row >= 0:
            self._controller.remove_entry(row)

    def _wire_draft_invalidation(self) -> None:
        """Any form field that participates in BatchDraft must clear a prior draft."""
        self._model_edit.textEdited.connect(self._controller.invalidate_draft)
        self._compute_edit.textEdited.connect(self._controller.invalidate_draft)
        self._source_edit.textEdited.connect(self._controller.invalidate_draft)
        self._target_edit.textEdited.connect(self._controller.invalidate_draft)
        self._provider_profile.textEdited.connect(self._controller.invalidate_draft)
        self._output_edit.textEdited.connect(self._controller.invalidate_draft)
        self._device_combo.currentIndexChanged.connect(self._controller.invalidate_draft)
        self._collision_combo.currentIndexChanged.connect(self._controller.invalidate_draft)

    def _on_browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self)
        if directory:
            self._output_edit.setText(directory)
            self._controller.invalidate_draft()

    def _on_source_auto_toggled(self, checked: bool) -> None:
        self._source_edit.setEnabled(not checked)
        if not self._applying_preset:
            self._controller.invalidate_draft()

    def _on_profile_changed(self) -> None:
        self._update_llm_enabled()
        if not self._applying_preset:
            self._controller.invalidate_draft()

    def _on_entries(self, entries: object) -> None:
        if not isinstance(entries, tuple):
            return
        self._input_list.clear()
        for entry in cast(tuple[object, ...], entries):
            self._input_list.addItem(QListWidgetItem(str(entry)))

    def _on_preview(self, preview: object) -> None:
        if not isinstance(preview, InputPreview):
            return
        self._accepted_label.setText(
            self._service.translate(
                "gui.create.input.accepted",
                {"count": str(preview.accepted_count)},
            )
        )
        self._rejected_label.setText(
            self._service.translate(
                "gui.create.input.rejected",
                {"count": str(preview.rejected_count)},
            )
        )
        if preview.rejected:
            lines: list[str] = []
            for item in preview.rejected[:20]:
                key = _REJECTION_KEYS.get(item.code, "gui.create.input.failure")
                reason = self._service.translate(key)
                name = Path(item.path).name if item.path else item.code
                lines.append(f"{name}: {reason}")
            self._input_failure.setText("\n".join(lines))
            self._input_failure.setVisible(True)
        else:
            self._input_failure.clear()
            self._input_failure.setVisible(False)

    def _on_configuration(self, snapshot: object) -> None:
        if not isinstance(snapshot, ConfigurationSnapshot):
            return
        self._applying_preset = True
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for preset in snapshot.presets:
            self._preset_combo.addItem(preset.display_name, preset.name)
        default = snapshot.global_settings.default_preset_name
        index = self._preset_combo.findData(default)
        if index < 0:
            index = 0
        self._preset_combo.setCurrentIndex(index)
        self._preset_combo.blockSignals(False)
        self._applying_preset = False
        if snapshot.global_settings.default_output_root:
            self._output_edit.setText(snapshot.global_settings.default_output_root)
        self._recursive.setChecked(snapshot.global_settings.recursive_input)
        collision = snapshot.global_settings.collision_policy
        cindex = self._collision_combo.findData(collision)
        if cindex >= 0:
            self._collision_combo.setCurrentIndex(cindex)
        self._apply_preset_fields(self._current_preset())

    def _current_preset(self) -> ExecutionPreset | None:
        name = self._preset_combo.currentData()
        if name is None or self._controller.configuration is None:
            return None
        for preset in self._controller.configuration.presets:
            if preset.name == name:
                return preset
        return None

    def _on_preset_selected(self) -> None:
        if self._applying_preset:
            return
        preset = self._current_preset()
        if preset is not None:
            self._controller.select_preset(preset.name)
            self._apply_preset_fields(preset)

    def _apply_preset_fields(self, preset: ExecutionPreset | None) -> None:
        if preset is None:
            return
        self._applying_preset = True
        profile_index = self._profile_combo.findData(preset.pipeline_profile.value)
        if profile_index >= 0:
            self._profile_combo.setCurrentIndex(profile_index)
        self._model_edit.setText(preset.model_ref)
        device_index = self._device_combo.findData(preset.device)
        if device_index >= 0:
            self._device_combo.setCurrentIndex(device_index)
        self._compute_edit.setText(preset.compute_type)
        if preset.source_language is None:
            self._source_auto.setChecked(True)
            self._source_edit.clear()
        else:
            self._source_auto.setChecked(False)
            self._source_edit.setText(preset.source_language)
        self._target_edit.setText(preset.target_language or "")
        self._provider_profile.setText(preset.provider_profile)
        if not preset.built_in:
            self._preset_name.setText(preset.name)
        self._applying_preset = False
        self._update_llm_enabled()
        delete = self.findChild(QPushButton, "createDeletePresetButton")
        if delete is not None:
            delete.setEnabled(not preset.built_in)

    def _update_llm_enabled(self) -> None:
        profile = self._profile_combo.currentData()
        llm_enabled = profile in {
            PipelineProfile.FAST.value,
            PipelineProfile.QUALITY.value,
        }
        self._target_edit.setEnabled(llm_enabled)
        self._provider_profile.setEnabled(True)

    def _on_save_preset(self) -> None:
        name = self._preset_name.text().strip()
        if not name:
            return
        profile = PipelineProfile(str(self._profile_combo.currentData()))
        target = (
            None
            if profile is PipelineProfile.DETERMINISTIC
            else self._target_edit.text().strip() or None
        )
        source = None if self._source_auto.isChecked() else self._source_edit.text().strip() or None
        preset = ExecutionPreset(
            name=name,
            display_name=name,
            built_in=False,
            pipeline_profile=profile,
            model_ref=self._model_edit.text().strip() or "tiny",
            device=str(self._device_combo.currentData()),  # type: ignore[arg-type]
            compute_type=self._compute_edit.text().strip() or "default",
            source_language=source,
            target_language=target,
            provider_profile=self._provider_profile.text().strip() or "default",
        )
        self._controller.save_user_preset(preset)

    def _on_delete_preset(self) -> None:
        preset = self._current_preset()
        if preset is None or preset.built_in:
            return
        self._controller.delete_user_preset(preset.name)

    def _on_validate(self) -> None:
        profile = PipelineProfile(str(self._profile_combo.currentData()))
        source = None if self._source_auto.isChecked() else self._source_edit.text().strip() or None
        target = (
            None
            if profile is PipelineProfile.DETERMINISTIC
            else self._target_edit.text().strip() or None
        )
        self._controller.validate_draft(
            output_root=self._output_edit.text().strip(),
            preset_name=str(self._preset_combo.currentData() or "deterministic"),
            pipeline_profile=profile,
            model_ref=self._model_edit.text().strip() or "tiny",
            device=str(self._device_combo.currentData() or "auto"),
            compute_type=self._compute_edit.text().strip() or "default",
            source_language=source,
            target_language=target,
            provider_profile=self._provider_profile.text().strip() or "default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy=str(self._collision_combo.currentData() or "unique_subdir"),
        )

    def _on_submit(self) -> None:
        self._controller.submit_draft()

    def _on_draft(self, draft: object) -> None:
        if isinstance(draft, BatchDraft):
            profile_key = {
                PipelineProfile.DETERMINISTIC: "gui.queue.profile.deterministic",
                PipelineProfile.FAST: "gui.queue.profile.fast",
                PipelineProfile.QUALITY: "gui.queue.profile.quality",
            }[draft.pipeline_profile]
            self._draft_status.setText(
                self._service.translate(
                    "gui.create.draft.ready",
                    {
                        "count": str(len(draft.input_paths)),
                        "profile": self._service.translate(profile_key),
                        "output": draft.output_root,
                    },
                )
            )
            self._draft_failure.clear()
            self._draft_failure.setVisible(False)
            self._update_submit_enabled()
            return
        self._draft_status.clear()
        self._update_submit_enabled()

    def _on_busy(self, busy: bool) -> None:
        self._validate_button.setEnabled(
            not busy and not self._controller.preset_busy and not self._controller.submission_busy
        )
        self._update_submit_enabled()

    def _on_preset_busy(self, busy: bool) -> None:
        for name in ("createSavePresetButton", "createDeletePresetButton"):
            button = self.findChild(QPushButton, name)
            if button is not None:
                button.setEnabled(not busy)
        self._validate_button.setEnabled(
            not busy and not self._controller.busy and not self._controller.submission_busy
        )
        self._update_submit_enabled()

    def _on_submission_busy(self, busy: bool) -> None:
        self._submit_busy.setVisible(busy)
        self._validate_button.setEnabled(
            not busy and not self._controller.busy and not self._controller.preset_busy
        )
        self._update_submit_enabled()

    def _on_submission_failure(self, failure: object) -> None:
        if failure is None:
            self._submit_failure.clear()
            self._submit_failure.setVisible(False)
            return
        code = (
            failure.code if isinstance(failure, RunnerFailure) else "gui.application_bridge_failed"
        )
        self._submit_failure.setText(
            self._service.translate("gui.create.submit_failure", {"code": code})
        )
        self._submit_failure.setVisible(True)

    def _on_batch_submitted(self, _ack: object) -> None:
        self._submit_failure.clear()
        self._submit_failure.setVisible(False)
        self._draft_status.setText(self._service.translate("gui.create.submitted"))
        self._update_submit_enabled()

    def _update_submit_enabled(self) -> None:
        enabled = (
            self._controller.draft is not None
            and not self._controller.busy
            and not self._controller.preset_busy
            and not self._controller.submission_busy
        )
        self._submit_button.setEnabled(enabled)

    def _on_failure(self, failure: object) -> None:
        if failure is None:
            return
        if isinstance(failure, RunnerFailure):
            code = failure.code
        else:
            code = "gui.application_bridge_failed"
        if code.startswith("batch.") or code.startswith("input."):
            self._draft_failure.setText(
                self._service.translate("gui.create.draft.failure", {"code": code})
            )
            self._draft_failure.setVisible(True)
        else:
            self._input_failure.setText(
                self._service.translate("gui.create.input.failure", {"code": code})
            )
            self._input_failure.setVisible(True)
        self._update_submit_enabled()


__all__ = ["CreatePage"]
