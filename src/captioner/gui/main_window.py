"""Phase 5 desktop shell with sidebar navigation and Queue page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from captioner.gui.batch_controller import BatchController
from captioner.gui.pages.placeholder_page import PlaceholderPage
from captioner.gui.pages.queue_page import QueuePage
from captioner.i18n.service import I18nService


class MainWindow(QMainWindow):
    """Sidebar shell owning navigation and controller lifecycle binding."""

    def __init__(
        self,
        service: I18nService,
        controller: BatchController,
    ) -> None:
        super().__init__()
        self._controller = controller
        self.setWindowTitle(service.translate("gui.window.title"))
        self.resize(1100, 700)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        create_button = self._nav_button(
            "navCreateButton",
            service.translate("gui.nav.create"),
        )
        queue_button = self._nav_button(
            "navQueueButton",
            service.translate("gui.nav.queue"),
        )
        history_button = self._nav_button(
            "navHistoryButton",
            service.translate("gui.nav.history"),
        )
        settings_button = self._nav_button(
            "navSettingsButton",
            service.translate("gui.nav.settings"),
        )
        diagnostics_button = self._nav_button(
            "navDiagnosticsButton",
            service.translate("gui.nav.diagnostics"),
        )

        for button in (
            create_button,
            queue_button,
            history_button,
            settings_button,
            diagnostics_button,
        ):
            self._nav_group.addButton(button)

        nav_layout = QVBoxLayout()
        nav_layout.addWidget(create_button)
        nav_layout.addWidget(queue_button)
        nav_layout.addWidget(history_button)
        nav_layout.addWidget(settings_button)
        nav_layout.addWidget(diagnostics_button)
        nav_layout.addStretch(1)
        nav_panel = QWidget()
        nav_panel.setObjectName("navPanel")
        nav_panel.setLayout(nav_layout)
        nav_panel.setFixedWidth(160)

        create_page = PlaceholderPage(
            service.translate("gui.nav.create"),
            service.translate(
                "gui.placeholder.message", {"page": service.translate("gui.nav.create")}
            ),
            "createPage",
        )
        queue_page = QueuePage(service, controller)
        history_page = PlaceholderPage(
            service.translate("gui.nav.history"),
            service.translate(
                "gui.placeholder.message",
                {"page": service.translate("gui.nav.history")},
            ),
            "historyPage",
        )
        settings_page = PlaceholderPage(
            service.translate("gui.nav.settings"),
            service.translate(
                "gui.placeholder.message",
                {"page": service.translate("gui.nav.settings")},
            ),
            "settingsPage",
        )
        diagnostics_page = PlaceholderPage(
            service.translate("gui.nav.diagnostics"),
            service.translate(
                "gui.placeholder.message",
                {"page": service.translate("gui.nav.diagnostics")},
            ),
            "diagnosticsPage",
        )

        self._page_stack = QStackedWidget()
        self._page_stack.setObjectName("mainPageStack")
        self._page_stack.addWidget(create_page)
        self._page_stack.addWidget(queue_page)
        self._page_stack.addWidget(history_page)
        self._page_stack.addWidget(settings_page)
        self._page_stack.addWidget(diagnostics_page)

        create_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(create_page))
        queue_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(queue_page))
        history_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(history_page))
        settings_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(settings_page))
        diagnostics_button.clicked.connect(
            lambda: self._page_stack.setCurrentWidget(diagnostics_page)
        )

        body = QHBoxLayout()
        body.addWidget(nav_panel)
        body.addWidget(self._page_stack, stretch=1)
        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        queue_button.setChecked(True)
        self._page_stack.setCurrentWidget(queue_page)

    def start(self) -> None:
        self._controller.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._controller.stop():
            event.accept()
            return
        event.ignore()

    def _nav_button(self, object_name: str, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        return button


__all__ = ["MainWindow"]
