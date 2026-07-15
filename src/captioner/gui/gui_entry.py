"""Thin PySide6 GUI entry point with an offscreen smoke mode."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from captioner.core.domain.errors import AppError
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import resolve_app_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal GUI parser without importing Qt."""
    parser = argparse.ArgumentParser(prog="captioner-gui")
    parser.add_argument("--lang", default="en", help="Locale, for example zh-CN")
    parser.add_argument("--smoke-test", action="store_true", help="Open and close automatically")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create a window and optionally close it after one event-loop turn."""
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        namespace = build_parser().parse_args(arguments)
        paths = resolve_app_paths()
        service = I18nService(
            locale=namespace.lang,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from captioner.gui.main_window import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(["captioner-gui", *arguments])
        window = MainWindow(service)
        window.show()
        if namespace.smoke_test:
            QTimer.singleShot(50, window.close)
            QTimer.singleShot(75, app.quit)
        return int(app.exec())
    except AppError as exc:
        print(exc.to_dict(), file=sys.stderr)
        return 2
