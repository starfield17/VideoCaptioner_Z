"""Thin PySide6 GUI entry point with an offscreen smoke mode."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from captioner.core.domain.errors import AppError
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import CompiledRuntime, resolve_app_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal GUI parser without importing Qt."""
    parser = argparse.ArgumentParser(prog="captioner-gui")
    parser.add_argument(
        "--lang",
        default=None,
        help="Locale override, for example zh-CN (default: settings.toml or en)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Open and close automatically")
    return parser


def _smoke_assert(condition: bool, message: str) -> None:
    if not condition:
        raise AppError("gui.smoke_failed", {"reason": message})


def _run_smoke_invariants(window: object) -> None:
    from PySide6.QtWidgets import QWidget

    from captioner.gui.main_window import MainWindow
    from captioner.gui.pages.diagnostics_page import DiagnosticsPage
    from captioner.gui.pages.placeholder_page import PlaceholderPage

    if not isinstance(window, MainWindow):
        raise AppError("gui.smoke_failed", {"reason": "main_window_type"})
    main = window
    stack = main.findChild(QWidget, "mainPageStack")
    _smoke_assert(stack is not None, "page_stack_missing")
    # Required navigation buttons.
    for name in (
        "navCreateButton",
        "navQueueButton",
        "navHistoryButton",
        "navSettingsButton",
        "navDiagnosticsButton",
    ):
        button = main.findChild(QWidget, name)
        _smoke_assert(button is not None, f"missing_{name}")

    diagnostics = main.findChild(DiagnosticsPage, "diagnosticsPage")
    if diagnostics is None:
        raise AppError("gui.smoke_failed", {"reason": "diagnostics_page_missing"})
    placeholder = main.findChild(PlaceholderPage, "diagnosticsPage")
    _smoke_assert(placeholder is None, "diagnostics_still_placeholder")

    for name in (
        "diagnosticsTitle",
        "diagnosticsRefreshButton",
        "diagnosticsExportButton",
        "diagnosticsInstallRuntimeButton",
        "diagnosticsManageModelsButton",
        "diagnosticsPrivacyLabel",
    ):
        control = diagnostics.findChild(QWidget, name)
        _smoke_assert(control is not None, f"missing_{name}")

    for name in ("createPage", "queuePage", "historyPage", "settingsPage", "diagnosticsPage"):
        page = main.findChild(QWidget, name)
        _smoke_assert(page is not None, f"missing_{name}")


def main(
    argv: Sequence[str] | None = None,
    *,
    compiled_runtime: CompiledRuntime | None = None,
) -> int:
    """Create a window and optionally close it after one event-loop turn."""
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        namespace = build_parser().parse_args(arguments)
        paths = resolve_app_paths(compiled_runtime=compiled_runtime)

        from captioner.gui_bootstrap import load_startup_locale

        locale, startup_issue = load_startup_locale(
            paths=paths,
            explicit_locale=namespace.lang,
        )
        service = I18nService(
            locale=locale,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from captioner.gui.composition import build_gui_controllers
        from captioner.gui.main_window import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(["captioner-gui", *arguments])
        controllers = build_gui_controllers(
            service,
            paths=paths,
            startup_issue=startup_issue,
        )
        window = MainWindow(service, controllers)
        window.show()
        window.start()
        if namespace.smoke_test:
            _run_smoke_invariants(window)

            def _finish() -> None:
                window.close()

            def _quit() -> None:
                app.quit()

            QTimer.singleShot(100, _finish)
            QTimer.singleShot(250, _quit)
        return int(app.exec())
    except AppError as exc:
        print(exc.to_dict(), file=sys.stderr)
        return 2
    except Exception as exc:
        # Smoke mode must not swallow unexpected failures as success.
        print(
            {"code": "gui.smoke_failed", "params": {"reason": type(exc).__name__}}, file=sys.stderr
        )
        return 2


__all__ = ["build_parser", "main"]
