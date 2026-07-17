"""Unit tests for the lightweight GUI application composition boundary."""

from __future__ import annotations

import sys
from pathlib import Path

from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.gui_bootstrap import build_gui_application_boundary
from captioner.infrastructure.app_paths import resolve_app_paths

_PROHIBITED_MODULES = (
    "PySide6",
    "faster_whisper",
    "ctranslate2",
    "torch",
    "transformers",
    "openai",
)


def test_lightweight_import_does_not_load_heavy_sdks() -> None:
    before = {name for name in sys.modules if name.split(".", 1)[0] in _PROHIBITED_MODULES}
    module_names = (
        "captioner.gui.application_boundary",
        "captioner.gui_bootstrap",
    )
    for name in module_names:
        sys.modules.pop(name, None)
    import captioner.gui.application_boundary as application_boundary
    import captioner.gui_bootstrap as gui_bootstrap

    assert application_boundary.GuiApplicationBoundary is not None
    assert gui_bootstrap.build_gui_application_boundary is not None
    after = {name for name in sys.modules if name.split(".", 1)[0] in _PROHIBITED_MODULES}
    newly_loaded = after - before
    assert newly_loaded == set()


def test_structural_boundary_and_empty_snapshot(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    boundary: GuiApplicationBoundary = build_gui_application_boundary(paths=paths)
    first = boundary.get_queue_snapshot()
    second = boundary.refresh_queue()
    assert first.schema_version == 1
    assert first.revision == 1
    assert first.items == ()
    assert first.issues == ()
    assert first.omitted_terminal_jobs == 0
    assert second == first
    received: list[object] = []
    unsubscribe = boundary.subscribe_queue(received.append)
    unchanged = boundary.refresh_queue()
    assert unchanged.revision == 1
    assert received == []
    unsubscribe()
    unsubscribe()


def test_boundary_requires_no_qapplication(tmp_path: Path) -> None:
    """Composition and refresh must work without constructing QApplication."""
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    boundary: GuiApplicationBoundary = build_gui_application_boundary(paths=paths)
    snapshot = boundary.refresh_queue()
    assert snapshot.revision == 1
    assert snapshot.items == ()
