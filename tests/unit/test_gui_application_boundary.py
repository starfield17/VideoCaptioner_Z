"""Unit tests for the lightweight GUI application composition boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.gui_bootstrap import build_gui_application_boundary
from captioner.infrastructure.app_paths import resolve_app_paths

_ISOLATED_IMPORT_CHECK = """
import sys

import captioner.gui.application_boundary
import captioner.gui_bootstrap

prohibited = {
    "PySide6",
    "faster_whisper",
    "ctranslate2",
    "torch",
    "transformers",
    "openai",
}
loaded = {name.split(".", 1)[0] for name in sys.modules}
assert not prohibited & loaded, sorted(prohibited & loaded)
assert captioner.gui.application_boundary.GuiApplicationBoundary is not None
assert captioner.gui_bootstrap.build_gui_application_boundary is not None
"""


def test_lightweight_import_does_not_load_heavy_sdks() -> None:
    """Import boundary modules in a clean subprocess so dependency chains re-run."""
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATED_IMPORT_CHECK],
        check=False,
        capture_output=True,
        text=True,
        env=None,
    )
    assert result.returncode == 0, (
        f"isolated import check failed\\nstdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    )


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


def test_boundary_supports_input_preview_and_configuration(tmp_path: Path) -> None:
    from captioner.core.application.input_selection import InputSelectionRequest

    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    media = tmp_path / "clip.wav"
    media.write_bytes(b"")
    boundary: GuiApplicationBoundary = build_gui_application_boundary(paths=paths)
    preview = boundary.preview_inputs(
        InputSelectionRequest(entries=(str(media), str(media)), recursive=True)
    )
    assert preview.accepted_count == 2
    configuration = boundary.load_configuration()
    assert configuration.global_settings.locale == "en"
    assert [preset.name for preset in configuration.presets[:3]] == [
        "deterministic",
        "fast",
        "quality",
    ]


def test_boundary_import_still_excludes_heavy_sdks() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATED_IMPORT_CHECK],
        check=False,
        capture_output=True,
        text=True,
        env=None,
    )
    assert result.returncode == 0, result.stdout + result.stderr
