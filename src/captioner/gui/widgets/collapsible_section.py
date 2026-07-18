"""Simple collapsible section used by Create and Settings pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Header toggle controlling visibility of a content widget."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        object_name: str | None = None,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if object_name is not None:
            self.setObjectName(object_name)
        self._content = content
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._toggle.toggled.connect(self._on_toggled)
        content.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle)
        layout.addWidget(content)

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)


__all__ = ["CollapsibleSection"]
