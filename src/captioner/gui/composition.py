"""GUI composition root for Queue presentation components."""

from __future__ import annotations

from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.gui.application_runner import ApplicationRunnerBridge
from captioner.gui.batch_controller import BatchController
from captioner.gui.queue_table_model import QueueTableModel
from captioner.gui_bootstrap import build_gui_application_boundary
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import AppPaths


def build_batch_controller(
    service: I18nService,
    *,
    paths: AppPaths,
    recent_terminal_limit: int = 100,
    refresh_interval_ms: int = 1000,
) -> BatchController:
    """Compose the Queue model, runner bridge, and controller."""
    model = QueueTableModel(service)

    def boundary_factory() -> GuiApplicationBoundary:
        return build_gui_application_boundary(
            paths=paths,
            recent_terminal_limit=recent_terminal_limit,
        )

    runner = ApplicationRunnerBridge(boundary_factory)
    return BatchController(
        model,
        runner,
        refresh_interval_ms=refresh_interval_ms,
    )


__all__ = ["build_batch_controller"]
