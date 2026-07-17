"""Unit tests for CreatePage widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
)

from captioner.core.application.configuration import default_configuration_snapshot
from captioner.core.application.input_selection import InputPreview
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.create_controller import CreateController
from captioner.gui.pages.create_page import CreatePage
from captioner.i18n.service import I18nService

_app = QApplication.instance() or QApplication(["test-create-page"])


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

    @property
    def running(self) -> bool:
        return self._running

    def request_input_preview(self, request: object) -> None:
        self.input_preview_ready.emit(
            InputPreview(accepted_paths=("/a.wav", "/a.wav"), rejected=())
        )

    def request_preset_save(self, preset: object) -> None:
        return None

    def request_preset_delete(self, name: str) -> None:
        return None


def _page(locale: str = "en") -> tuple[CreatePage, CreateController]:
    service = I18nService(locale)
    runner = FakeRunner()
    controller = CreateController(runner)  # type: ignore[arg-type]
    page = CreatePage(service, controller)
    page.show()
    controller.set_configuration(default_configuration_snapshot())
    return page, controller


def test_required_object_names_and_no_start_button() -> None:
    page, controller = _page()
    assert page.objectName() == "createPage"
    for name in (
        "createAddFilesButton",
        "createAddFolderButton",
        "createRemoveInputButton",
        "createClearInputsButton",
        "createRecursiveCheck",
        "createInputList",
        "createAcceptedLabel",
        "createRejectedLabel",
        "createInputFailureLabel",
        "createPresetCombo",
        "createPresetNameEdit",
        "createSavePresetButton",
        "createDeletePresetButton",
        "createProfileCombo",
        "createModelEdit",
        "createDeviceCombo",
        "createComputeTypeEdit",
        "createSourceLanguageEdit",
        "createSourceLanguageAutoCheck",
        "createTargetLanguageEdit",
        "createProviderProfileEdit",
        "createOutputRootEdit",
        "createBrowseOutputButton",
        "createCollisionPolicyCombo",
        "createValidateButton",
        "createDraftStatusLabel",
        "createDraftFailureLabel",
        "createInputSection",
        "createRecognitionSection",
        "createLanguageSection",
        "createSubtitleSection",
        "createOutputSection",
    ):
        assert page.findChild(QObject, name) is not None, name
    for forbidden in ("Start", "Run", "Add to Queue", "Submit", "开始", "运行"):
        for button in page.findChildren(QPushButton):
            assert forbidden not in button.text()
    controller.clear_entries()
    page.close()


def test_duplicate_display_and_validation() -> None:
    page, controller = _page()
    controller.set_entries(("/a.wav", "/a.wav"))
    listing = page.findChild(QListWidget, "createInputList")
    assert listing is not None
    assert listing.count() == 2
    output = page.findChild(QLineEdit, "createOutputRootEdit")
    assert output is not None
    output.setText("/tmp/out")
    validate = page.findChild(QPushButton, "createValidateButton")
    assert validate is not None
    validate.click()
    status = page.findChild(QLabel, "createDraftStatusLabel")
    assert status is not None
    assert "2" in status.text()
    page.close()


def test_profile_enables_target_language() -> None:
    page, _controller = _page()
    profile = page.findChild(QComboBox, "createProfileCombo")
    target = page.findChild(QLineEdit, "createTargetLanguageEdit")
    assert profile is not None and target is not None
    index = profile.findData(PipelineProfile.DETERMINISTIC.value)
    profile.setCurrentIndex(index)
    assert target.isEnabled() is False
    index = profile.findData(PipelineProfile.FAST.value)
    profile.setCurrentIndex(index)
    assert target.isEnabled() is True
    page.close()


def test_chinese_labels() -> None:
    page, _controller = _page("zh-CN")
    validate = page.findChild(QPushButton, "createValidateButton")
    assert validate is not None
    assert validate.text() == "验证配置"
    recursive = page.findChild(QCheckBox, "createRecursiveCheck")
    assert recursive is not None
    assert "递归" in recursive.text()
    page.close()
