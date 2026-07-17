"""Lightweight GUI composition root without ASR/LLM/Torch SDK imports."""

from __future__ import annotations

from captioner.adapters.persistence.filesystem_batch_catalog import FilesystemBatchCatalog
from captioner.core.application.queue_projection import QueueProjectionService
from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.infrastructure.app_paths import (
    AppPaths,
    ensure_runtime_layout,
    resolve_app_paths,
)


def build_gui_application_boundary(
    *,
    paths: AppPaths | None = None,
    recent_terminal_limit: int = 100,
) -> GuiApplicationBoundary:
    """Compose the read-only Queue boundary used by the GUI runner."""
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    catalog = FilesystemBatchCatalog(application_paths.batches_dir)
    return QueueProjectionService(
        catalog=catalog,
        recent_terminal_limit=recent_terminal_limit,
    )


__all__ = ["build_gui_application_boundary"]
