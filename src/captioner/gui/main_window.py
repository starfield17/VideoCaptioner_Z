"""Phase 5 desktop shell with sidebar navigation and functional pages."""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from captioner.core.application.batch_commands import LocalExecutionSnapshot
from captioner.core.application.recovery import RecoveryItem
from captioner.gui.composition import GuiControllers
from captioner.gui.pages.create_page import CreatePage
from captioner.gui.pages.history_page import HistoryPage
from captioner.gui.pages.placeholder_page import PlaceholderPage
from captioner.gui.pages.queue_page import QueuePage
from captioner.gui.pages.settings_page import SettingsPage
from captioner.gui.widgets.recovery_dialog import RecoveryDialog
from captioner.i18n.service import I18nService


class MainWindow(QMainWindow):
    """Sidebar shell owning navigation and controller lifecycle binding."""

    def __init__(
        self,
        service: I18nService,
        controllers: GuiControllers,
    ) -> None:
        super().__init__()
        self._service = service
        self._controllers = controllers
        self._started = False
        self._close_when_idle = False
        self.setWindowTitle(service.translate("gui.window.title"))
        self.resize(1100, 700)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        self._create_button = self._nav_button(
            "navCreateButton",
            service.translate("gui.nav.create"),
        )
        self._queue_button = self._nav_button(
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
            self._create_button,
            self._queue_button,
            history_button,
            settings_button,
            diagnostics_button,
        ):
            self._nav_group.addButton(button)

        nav_layout = QVBoxLayout()
        nav_layout.addWidget(self._create_button)
        nav_layout.addWidget(self._queue_button)
        nav_layout.addWidget(history_button)
        nav_layout.addWidget(settings_button)
        nav_layout.addWidget(diagnostics_button)
        nav_layout.addStretch(1)
        nav_panel = QWidget()
        nav_panel.setObjectName("navPanel")
        nav_panel.setLayout(nav_layout)
        nav_panel.setFixedWidth(160)

        create_page = CreatePage(service, controllers.create)
        queue_page = QueuePage(service, controllers.queue, controllers.operations)
        history_page = HistoryPage(service, controllers.queue, controllers.operations)
        settings_page = SettingsPage(service, controllers.settings)
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

        self._create_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(create_page))
        self._queue_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(queue_page))
        history_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(history_page))
        settings_button.clicked.connect(lambda: self._page_stack.setCurrentWidget(settings_page))
        diagnostics_button.clicked.connect(
            lambda: self._page_stack.setCurrentWidget(diagnostics_page)
        )

        self._notification = QLabel("")
        self._notification.setObjectName("globalNotificationLabel")
        self._notification.setVisible(False)
        self._notification_timer = QTimer(self)
        self._notification_timer.setSingleShot(True)
        self._notification_timer.timeout.connect(self._hide_success_notification)

        body = QVBoxLayout()
        body.addWidget(self._notification)
        content = QHBoxLayout()
        content.addWidget(nav_panel)
        content.addWidget(self._page_stack, stretch=1)
        body.addLayout(content, stretch=1)
        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        self._create_button.setChecked(True)
        self._page_stack.setCurrentWidget(create_page)

        controllers.create.batch_submitted.connect(self._on_batch_submitted)
        controllers.operations.notification_changed.connect(self._on_notification)
        controllers.operations.local_execution_state_changed.connect(self._on_execution_state)
        controllers.operations.close_cancellation_failed.connect(self._on_close_cancellation_failed)
        controllers.recovery.prompt_requested.connect(self._on_recovery_prompt)

        self._create_page = create_page
        self._queue_page = queue_page

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._controllers.queue.start()
        self._controllers.settings.load()
        self._controllers.recovery.scan()

    def closeEvent(self, event: QCloseEvent) -> None:
        operations = self._controllers.operations
        if operations.has_local_work:
            choice = QMessageBox.question(
                self,
                self._service.translate("gui.close.active.title"),
                self._service.translate("gui.close.active.message"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_when_idle = True
            self._show_notification(
                self._service.translate("gui.close.cancelling"),
                auto_hide=False,
            )
            operations.cancel_all_local_work()
            event.ignore()
            return

        if self._controllers.queue.stop():
            event.accept()
            return
        self._show_notification(
            self._service.translate("gui.close.shutdown_failed"),
            auto_hide=False,
        )
        event.ignore()

    def _on_batch_submitted(self, _ack: object) -> None:
        self._page_stack.setCurrentWidget(self._queue_page)
        self._queue_button.setChecked(True)
        self._controllers.queue.refresh()
        self._show_notification(
            self._service.translate("gui.notification.batch_submitted"),
            auto_hide=True,
        )

    def _on_notification(self, message: object) -> None:
        if not isinstance(message, str):
            return
        if message.startswith("command.failed:"):
            code = message.split(":", 1)[1]
            self._show_notification(
                self._service.translate(
                    "gui.notification.command_failed",
                    {"code": code},
                ),
                auto_hide=False,
            )
            return
        if message.startswith("execution.failed:"):
            code = message.split(":", 1)[1]
            self._show_notification(
                self._service.translate(
                    "gui.notification.execution_failed",
                    {"code": code},
                ),
                auto_hide=False,
            )
            return
        if message.startswith("execution.completed:"):
            self._show_notification(
                self._service.translate("gui.notification.execution_completed"),
                auto_hide=True,
            )
            return
        if message.startswith("command.accepted:"):
            self._show_notification(
                self._service.translate("gui.notification.command_accepted"),
                auto_hide=True,
            )

    def _on_execution_state(self, state: object) -> None:
        if not isinstance(state, LocalExecutionSnapshot):
            return
        if self._close_when_idle and not state.has_work:
            self._close_when_idle = False
            self.close()

    def _on_close_cancellation_failed(self, failure: object) -> None:
        # CancelLocalWork failed: abort close-when-idle and keep the window usable.
        self._close_when_idle = False
        code = getattr(failure, "code", "gui.application_bridge_failed")
        if not isinstance(code, str):
            code = "gui.application_bridge_failed"
        self._show_notification(
            self._service.translate(
                "gui.notification.command_failed",
                {"code": code},
            ),
            auto_hide=False,
        )

    def _on_recovery_prompt(self, items: object) -> None:
        if not isinstance(items, tuple) or not items:
            return
        candidates = cast(tuple[object, ...], items)
        typed = tuple(item for item in candidates if isinstance(item, RecoveryItem))
        if not typed:
            return
        dialog = RecoveryDialog(self._service, typed, parent=self)
        result = dialog.exec()
        if result != RecoveryDialog.DialogCode.Accepted:
            return
        batch_id = dialog.selected_batch_id
        if batch_id is None:
            return
        if dialog.action == "resume":
            self._controllers.operations.resume_batch_id(batch_id)
            self._controllers.queue.refresh()
        elif dialog.action == "cancel":
            self._controllers.operations.cancel_batch_id(batch_id)
            self._controllers.queue.refresh()

    def _show_notification(self, text: str, *, auto_hide: bool) -> None:
        self._notification.setText(text)
        self._notification.setVisible(True)
        self._notification_timer.stop()
        if auto_hide:
            self._notification_timer.start(5000)

    def _hide_success_notification(self) -> None:
        self._notification.clear()
        self._notification.setVisible(False)

    def _nav_button(self, object_name: str, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        return button


__all__ = ["MainWindow"]
