"""Minimal Phase 0 Qt window."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from captioner.i18n.service import I18nService


class MainWindow(QMainWindow):
    """A deliberately small, business-free application window."""

    def __init__(self, service: I18nService | None = None) -> None:
        super().__init__()
        message_service = I18nService() if service is None else service
        self.setWindowTitle(message_service.translate("gui.window.title"))
        label = QLabel(message_service.translate("gui.phase0.message"))
        label.setObjectName("phase0Message")
        label.setMargin(24)
        layout = QVBoxLayout()
        layout.addWidget(label)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.resize(480, 180)
