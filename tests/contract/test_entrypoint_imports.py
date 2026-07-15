from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{ROOT}"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_import_does_not_load_gui_or_qt() -> None:
    result = _run_import_probe(
        "import main; main.main(['--cli', 'doctor', '--json']); "
        "assert 'captioner.gui' not in __import__('sys').modules; "
        "assert 'PySide6' not in __import__('sys').modules"
    )
    assert result.returncode == 0, result.stderr


def test_core_import_does_not_load_gui() -> None:
    result = _run_import_probe(
        "import captioner.core.domain.errors; "
        "assert 'captioner.gui' not in __import__('sys').modules"
    )
    assert result.returncode == 0, result.stderr


def test_gui_entry_module_is_lazy_about_qt_and_sdk() -> None:
    result = _run_import_probe(
        "import captioner.cli.cli_entry, captioner.gui.gui_entry; "
        "import sys; "
        "assert 'PySide6' not in sys.modules; "
        "assert not any(name in sys.modules for name in "
        "('openai', 'torch', 'transformers', 'faster_whisper'))"
    )
    assert result.returncode == 0, result.stderr


def test_package_module_entrypoint_works_without_repository_root(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "captioner", "--cli", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: captioner" in result.stdout
