"""Settings page for global defaults and provider configuration."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
)
from captioner.gui.application_runner import RunnerFailure
from captioner.gui.settings_controller import SettingsController
from captioner.gui.widgets.collapsible_section import CollapsibleSection
from captioner.i18n.service import I18nService


class SettingsPage(QWidget):
    """Functional Settings surface for PR5.3."""

    def __init__(
        self,
        service: I18nService,
        controller: SettingsController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self._service = service
        self._controller = controller

        root = QVBoxLayout(self)
        title = QLabel(service.translate("gui.settings.title"))
        title.setObjectName("settingsTitle")
        root.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.settings.application"),
                self._build_global_section(),
                object_name="settingsApplicationSection",
            )
        )
        body_layout.addWidget(
            CollapsibleSection(
                service.translate("gui.settings.provider"),
                self._build_provider_section(),
                object_name="settingsProviderSection",
            )
        )
        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

        controller.configuration_changed.connect(self._on_configuration)
        controller.busy_changed.connect(self._on_busy)
        controller.failure_changed.connect(self._on_failure)
        controller.provider_test_changed.connect(self._on_provider_test)
        controller.restart_required_changed.connect(self._on_restart_required)

        if controller.startup_issue is not None:
            self._global_failure.setText(
                service.translate(
                    "gui.settings.invalid_loaded",
                    {"code": controller.startup_issue},
                )
            )
            self._global_failure.setVisible(True)

    def _build_global_section(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self._locale_combo = QComboBox()
        self._locale_combo.setObjectName("settingsLocaleCombo")
        self._locale_combo.addItem("English", "en")
        self._locale_combo.addItem("简体中文", "zh-CN")
        form.addRow(self._service.translate("gui.settings.locale"), self._locale_combo)

        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setObjectName("settingsOutputRootEdit")
        browse = QPushButton(self._service.translate("gui.create.output_browse"))
        browse.setObjectName("settingsBrowseOutputButton")
        browse.clicked.connect(self._on_browse_output)
        output_row.addWidget(self._output_edit)
        output_row.addWidget(browse)
        form.addRow(self._service.translate("gui.settings.default_output"), output_row)

        self._recursive = QCheckBox(self._service.translate("gui.settings.recursive"))
        self._recursive.setObjectName("settingsRecursiveCheck")
        form.addRow(self._recursive)

        self._default_preset = QComboBox()
        self._default_preset.setObjectName("settingsDefaultPresetCombo")
        form.addRow(
            self._service.translate("gui.settings.default_preset"),
            self._default_preset,
        )

        self._collision = QComboBox()
        self._collision.setObjectName("settingsCollisionPolicyCombo")
        for value, key in (
            ("unique_subdir", "gui.create.collision.unique_subdir"),
            ("fail", "gui.create.collision.fail"),
            ("overwrite", "gui.create.collision.overwrite"),
        ):
            self._collision.addItem(self._service.translate(key), value)
        form.addRow(
            self._service.translate("gui.settings.collision_policy"),
            self._collision,
        )

        save = QPushButton(self._service.translate("gui.settings.save"))
        save.setObjectName("settingsSaveGlobalButton")
        save.clicked.connect(self._on_save_global)
        form.addRow(save)

        self._restart_label = QLabel(self._service.translate("gui.settings.restart_required"))
        self._restart_label.setObjectName("settingsRestartLabel")
        self._restart_label.setVisible(False)
        form.addRow(self._restart_label)

        self._global_failure = QLabel("")
        self._global_failure.setObjectName("settingsGlobalFailureLabel")
        self._global_failure.setVisible(False)
        form.addRow(self._global_failure)
        return widget

    def _build_provider_section(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self._profile_edit = QLineEdit("default")
        self._profile_edit.setObjectName("settingsProviderProfileEdit")
        form.addRow(
            self._service.translate("gui.settings.provider_profile"),
            self._profile_edit,
        )

        self._base_url = QLineEdit()
        self._base_url.setObjectName("settingsBaseUrlEdit")
        form.addRow(self._service.translate("gui.settings.base_url"), self._base_url)

        self._model = QLineEdit()
        self._model.setObjectName("settingsModelEdit")
        form.addRow(self._service.translate("gui.settings.model"), self._model)

        self._api_key = QLineEdit()
        self._api_key.setObjectName("settingsApiKeyEdit")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText(
            self._service.translate("gui.settings.api_key_placeholder")
        )
        form.addRow(self._service.translate("gui.settings.api_key"), self._api_key)

        self._credential_source = QLabel("")
        self._credential_source.setObjectName("settingsCredentialSourceLabel")
        form.addRow(
            self._service.translate("gui.settings.credential_source"),
            self._credential_source,
        )

        self._max_concurrency = QSpinBox()
        self._max_concurrency.setObjectName("settingsMaxConcurrencySpin")
        self._max_concurrency.setRange(1, 64)
        self._max_concurrency.setValue(4)
        form.addRow(
            self._service.translate("gui.settings.max_concurrency"),
            self._max_concurrency,
        )

        self._timeout = QDoubleSpinBox()
        self._timeout.setObjectName("settingsTimeoutSpin")
        self._timeout.setRange(1.0, 3600.0)
        self._timeout.setValue(120.0)
        form.addRow(self._service.translate("gui.settings.timeout"), self._timeout)

        self._max_retries = QSpinBox()
        self._max_retries.setObjectName("settingsMaxRetriesSpin")
        self._max_retries.setRange(0, 20)
        self._max_retries.setValue(5)
        form.addRow(
            self._service.translate("gui.settings.max_retries"),
            self._max_retries,
        )

        self._temperature = QDoubleSpinBox()
        self._temperature.setObjectName("settingsTemperatureSpin")
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.1)
        self._temperature.setValue(0.1)
        form.addRow(
            self._service.translate("gui.settings.temperature"),
            self._temperature,
        )

        self._tokenizer = QComboBox()
        self._tokenizer.setObjectName("settingsTokenizerCombo")
        for value in ("cl100k_base", "o200k_base", "auto"):
            self._tokenizer.addItem(value, value)
        form.addRow(self._service.translate("gui.settings.tokenizer"), self._tokenizer)

        buttons = QHBoxLayout()
        save_provider = QPushButton(self._service.translate("gui.settings.provider_save"))
        save_provider.setObjectName("settingsSaveProviderButton")
        save_provider.clicked.connect(self._on_save_provider)
        test_provider = QPushButton(self._service.translate("gui.settings.provider_test"))
        test_provider.setObjectName("settingsTestProviderButton")
        test_provider.clicked.connect(self._on_test_provider)
        buttons.addWidget(save_provider)
        buttons.addWidget(test_provider)
        buttons.addStretch(1)
        form.addRow(buttons)

        self._provider_result = QLabel("")
        self._provider_result.setObjectName("settingsProviderResultLabel")
        form.addRow(self._provider_result)

        self._provider_failure = QLabel("")
        self._provider_failure.setObjectName("settingsProviderFailureLabel")
        self._provider_failure.setVisible(False)
        form.addRow(self._provider_failure)
        return widget

    def _on_browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self)
        if directory:
            self._output_edit.setText(directory)

    def _on_save_global(self) -> None:
        settings = GlobalSettings(
            locale=str(self._locale_combo.currentData()),  # type: ignore[arg-type]
            default_output_root=self._output_edit.text().strip() or None,
            recursive_input=self._recursive.isChecked(),
            default_preset_name=str(self._default_preset.currentData() or "deterministic"),
            collision_policy=str(self._collision.currentData() or "unique_subdir"),  # type: ignore[arg-type]
        )
        self._controller.save_global(settings)

    def _provider_update(self) -> ProviderSettingsUpdate:
        key_text = self._api_key.text()
        api_key = None if not key_text.strip() else key_text.strip()
        return ProviderSettingsUpdate(
            profile_name=self._profile_edit.text().strip() or "default",
            base_url=self._base_url.text().strip(),
            model=self._model.text().strip(),
            api_key=api_key,
            max_concurrency=int(self._max_concurrency.value()),
            request_timeout_sec=float(self._timeout.value()),
            max_retries=int(self._max_retries.value()),
            temperature=float(self._temperature.value()),
            tokenizer=str(self._tokenizer.currentData() or "cl100k_base"),
        )

    def _on_save_provider(self) -> None:
        update = self._provider_update()
        self._api_key.clear()
        self._controller.save_provider(update)

    def _on_test_provider(self) -> None:
        update = self._provider_update()
        self._api_key.clear()
        self._controller.test_provider(update)

    def _on_configuration(self, snapshot: object) -> None:
        if not isinstance(snapshot, ConfigurationSnapshot):
            return
        settings = snapshot.global_settings
        locale_index = self._locale_combo.findData(settings.locale)
        if locale_index >= 0:
            self._locale_combo.setCurrentIndex(locale_index)
        self._output_edit.setText(settings.default_output_root or "")
        self._recursive.setChecked(settings.recursive_input)
        self._default_preset.blockSignals(True)
        self._default_preset.clear()
        for preset in snapshot.presets:
            self._default_preset.addItem(preset.display_name, preset.name)
        preset_index = self._default_preset.findData(settings.default_preset_name)
        if preset_index >= 0:
            self._default_preset.setCurrentIndex(preset_index)
        self._default_preset.blockSignals(False)
        collision_index = self._collision.findData(settings.collision_policy)
        if collision_index >= 0:
            self._collision.setCurrentIndex(collision_index)

        provider = snapshot.provider
        self._profile_edit.setText(provider.profile_name)
        self._base_url.setText(provider.base_url)
        self._model.setText(provider.model)
        self._api_key.clear()
        source_key = {
            "config": "gui.settings.credential.config",
            "environment": "gui.settings.credential.environment",
            "missing": "gui.settings.credential.missing",
        }[provider.credential_source]
        self._credential_source.setText(self._service.translate(source_key))
        self._max_concurrency.setValue(provider.max_concurrency)
        self._timeout.setValue(provider.request_timeout_sec)
        self._max_retries.setValue(provider.max_retries)
        self._temperature.setValue(provider.temperature)
        tokenizer_index = self._tokenizer.findData(provider.tokenizer)
        if tokenizer_index >= 0:
            self._tokenizer.setCurrentIndex(tokenizer_index)

        if snapshot.issues:
            codes = ", ".join(issue.code for issue in snapshot.issues)
            self._global_failure.setText(
                self._service.translate(
                    "gui.settings.invalid_loaded",
                    {"code": codes},
                )
            )
            self._global_failure.setVisible(True)
        else:
            self._global_failure.clear()
            self._global_failure.setVisible(False)

    def _on_busy(self, busy: bool) -> None:
        for name in (
            "settingsSaveGlobalButton",
            "settingsSaveProviderButton",
            "settingsTestProviderButton",
        ):
            button = self.findChild(QPushButton, name)
            if button is not None:
                button.setEnabled(not busy)

    def _on_failure(self, failure: object) -> None:
        if failure is None:
            self._provider_failure.clear()
            self._provider_failure.setVisible(False)
            return
        code = (
            failure.code if isinstance(failure, RunnerFailure) else "gui.application_bridge_failed"
        )
        self._provider_failure.setText(
            self._service.translate("gui.settings.failure", {"code": code})
        )
        self._provider_failure.setVisible(True)

    def _on_provider_test(self, result: object) -> None:
        if not isinstance(result, ProviderConnectionResult):
            return
        if result.ok:
            self._provider_result.setText(self._service.translate("gui.settings.provider_test_ok"))
            self._provider_failure.clear()
            self._provider_failure.setVisible(False)
            return
        self._provider_result.setText(
            self._service.translate(
                "gui.settings.provider_test_failed",
                {"code": result.code},
            )
        )

    def _on_restart_required(self, required: bool) -> None:
        self._restart_label.setVisible(required)


__all__ = ["SettingsPage"]
