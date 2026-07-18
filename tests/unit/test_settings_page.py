"""Unit tests for SettingsPage widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton

from captioner.core.application.configuration import (
    ConfigurationIssue,
    ConfigurationSnapshot,
    ProviderConnectionResult,
    ProviderPublicSettings,
    built_in_presets,
    default_configuration_snapshot,
    default_global_settings,
)
from captioner.gui.pages.settings_page import SettingsPage
from captioner.gui.settings_controller import SettingsController
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-settings-page"])


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

    def __init__(self) -> None:
        super().__init__()
        self._running = True
        self.provider_updates: list[object] = []

    @property
    def running(self) -> bool:
        return self._running

    def request_configuration_load(self) -> None:
        self.configuration_loaded.emit(default_configuration_snapshot())

    def request_global_save(self, settings: object) -> None:
        self.global_settings_saved.emit(
            ConfigurationSnapshot(
                global_settings=settings,  # type: ignore[arg-type]
                presets=built_in_presets(),
                provider=default_configuration_snapshot().provider,
                issues=(),
            )
        )

    def request_provider_save(self, update: object) -> None:
        self.provider_updates.append(update)
        self.provider_settings_saved.emit(default_configuration_snapshot())

    def request_provider_test(self, update: object) -> None:
        self.provider_updates.append(update)
        self.provider_test_ready.emit(ProviderConnectionResult(True, "llm.connection_ok"))


def _page(
    locale: str = "en", startup_issue: str | None = None
) -> tuple[SettingsPage, SettingsController, FakeRunner]:
    service = I18nService(locale)
    runner = FakeRunner()
    controller = SettingsController(runner, startup_issue=startup_issue)  # type: ignore[arg-type]
    page = SettingsPage(service, controller)
    page.show()
    return page, controller, runner


def test_required_object_names_and_password_echo() -> None:
    page, controller, _runner = _page()
    assert page.objectName() == "settingsPage"
    for name in (
        "settingsLocaleCombo",
        "settingsOutputRootEdit",
        "settingsBrowseOutputButton",
        "settingsRecursiveCheck",
        "settingsDefaultPresetCombo",
        "settingsCollisionPolicyCombo",
        "settingsSaveGlobalButton",
        "settingsRestartLabel",
        "settingsGlobalFailureLabel",
        "settingsProviderProfileEdit",
        "settingsBaseUrlEdit",
        "settingsModelEdit",
        "settingsApiKeyEdit",
        "settingsCredentialSourceLabel",
        "settingsMaxConcurrencySpin",
        "settingsTimeoutSpin",
        "settingsMaxRetriesSpin",
        "settingsTemperatureSpin",
        "settingsTokenizerCombo",
        "settingsSaveProviderButton",
        "settingsTestProviderButton",
        "settingsProviderResultLabel",
        "settingsProviderFailureLabel",
    ):
        assert page.findChild(QObject, name) is not None, name
    api_key = page.findChild(QLineEdit, "settingsApiKeyEdit")
    assert api_key is not None
    assert api_key.echoMode() == QLineEdit.EchoMode.Password
    controller.load()
    page.close()


def test_configuration_rendering_and_api_key_cleared() -> None:
    page, controller, runner = _page()
    snapshot = ConfigurationSnapshot(
        global_settings=default_global_settings(),
        presets=built_in_presets(),
        provider=ProviderPublicSettings(
            profile_name="default",
            base_url="https://example.com/v1",
            model="unit-model",
            max_concurrency=3,
            request_timeout_sec=30.0,
            max_retries=1,
            temperature=0.2,
            tokenizer="cl100k_base",
            credential_source="config",
        ),
        issues=(),
    )
    controller.configuration_changed.emit(snapshot)
    base = page.findChild(QLineEdit, "settingsBaseUrlEdit")
    model = page.findChild(QLineEdit, "settingsModelEdit")
    api_key = page.findChild(QLineEdit, "settingsApiKeyEdit")
    source = page.findChild(QLabel, "settingsCredentialSourceLabel")
    assert base is not None and model is not None and api_key is not None and source is not None
    assert base.text() == "https://example.com/v1"
    assert model.text() == "unit-model"
    assert api_key.text() == ""
    assert "Config" in source.text() or "配置" in source.text()
    api_key.setText("should-clear")
    save = page.findChild(QPushButton, "settingsSaveProviderButton")
    assert save is not None
    save.click()
    assert api_key.text() == ""
    assert len(runner.provider_updates) == 1
    page.close()


def test_invalid_config_and_restart_label() -> None:
    page, controller, _runner = _page(startup_issue="config.settings_invalid")
    failure = page.findChild(QLabel, "settingsGlobalFailureLabel")
    assert failure is not None
    assert failure.isVisible()
    snapshot = default_configuration_snapshot(
        issues=(ConfigurationIssue(code="config.settings_invalid"),)
    )
    controller.configuration_changed.emit(snapshot)
    assert failure.isVisible()
    controller.restart_required_changed.emit(True)
    restart = page.findChild(QLabel, "settingsRestartLabel")
    assert restart is not None
    assert restart.isVisible()
    page.close()


def test_chinese_labels() -> None:
    page, _controller, _runner = _page("zh-CN")
    title = page.findChild(QLabel, "settingsTitle")
    assert title is not None
    assert title.text() == "设置"
    save = page.findChild(QPushButton, "settingsSaveGlobalButton")
    assert save is not None
    assert "保存" in save.text()
    page.close()
