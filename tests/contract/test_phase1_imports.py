from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"


def _isolated(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_does_not_import_faster_whisper(tmp_path: Path) -> None:
    result = _isolated(
        "import captioner.cli.cli_entry; import sys; assert 'faster_whisper' not in sys.modules",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_gui_import_does_not_import_faster_whisper(tmp_path: Path) -> None:
    result = _isolated(
        "import captioner.gui.gui_entry; import sys; assert 'faster_whisper' not in sys.modules",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_package_module_help_works_with_only_src_on_pythonpath(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-m", "captioner", "--cli", "run", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert "faster_whisper" not in result.stderr
