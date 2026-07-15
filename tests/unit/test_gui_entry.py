from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from captioner.gui.gui_entry import main
from captioner.gui.main_window import MainWindow
from captioner.i18n.service import I18nService


def test_main_window_uses_i18n() -> None:
    app = QApplication.instance() or QApplication(["test"])
    window = MainWindow(I18nService("zh-CN"))
    assert window.windowTitle() == "Captioner"
    label = window.findChild(QLabel, "phase0Message")
    assert label is not None
    assert label.text() == "Phase 0 项目骨架"
    assert app is not None
    window.close()


def test_gui_smoke_test_returns_zero() -> None:
    assert main(["--lang", "zh-CN", "--smoke-test"]) == 0
