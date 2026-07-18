"""Reusable placeholder page for unimplemented Phase 5 surfaces."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    """Localized placeholder without Application logic."""

    def __init__(
        self,
        title: str,
        message: str,
        object_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        title_label = QLabel(title)
        title_label.setObjectName(f"{object_name}Title")
        message_label = QLabel(message)
        message_label.setObjectName(f"{object_name}Message")
        message_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(title_label)
        layout.addWidget(message_label)
        layout.addStretch(1)


__all__ = ["PlaceholderPage"]
