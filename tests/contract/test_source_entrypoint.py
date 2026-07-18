from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ENTRYPOINT = ROOT / "main.py"


def _run_source_entrypoint(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    # Prove the wrapper itself, rather than the pytest pythonpath, supplies src/.
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(SOURCE_ENTRYPOINT), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_source_entrypoint_help_works_without_editable_install(tmp_path: Path) -> None:
    result = _run_source_entrypoint(ROOT, "--cli", "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: captioner" in result.stdout


def test_source_entrypoint_doctor_works_from_external_cwd(tmp_path: Path) -> None:
    result = _run_source_entrypoint(tmp_path, "--cli", "doctor", "--json")
    assert result.returncode == 0, result.stderr
    assert '"resource_root"' in result.stdout
    assert '"models_dir"' in result.stdout


def test_source_bootstrap_precedes_captioner_imports() -> None:
    tree = ast.parse(SOURCE_ENTRYPOINT.read_text(encoding="utf-8"))
    captioner_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("captioner")
    ]
    insert_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
        and node.func.value.attr == "path"
    ]
    assert captioner_import_lines
    assert insert_lines
    assert min(insert_lines) < min(captioner_import_lines)
