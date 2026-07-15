"""Unified quick and full quality gate."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str) -> list[str]:
    executable = shutil.which(name)
    return [executable] if executable is not None else [sys.executable, "-m", name]


def _python_script(path: str, *arguments: str) -> list[str]:
    return [sys.executable, path, *arguments]


def build_steps(mode: str) -> list[tuple[str, list[str]]]:
    """Return subprocess argument lists for one quality-gate mode."""
    pytest_quick = ["tests/unit", "tests/contract", "-q"]
    pytest_full = ["tests/unit", "tests/property", "tests/contract", "tests/packaging", "-q"]
    if mode == "quick":
        return [
            ("ruff format", [*_tool("ruff"), "format", "--check", "."]),
            ("ruff lint", [*_tool("ruff"), "check", "."]),
            ("pyright", [*_tool("pyright")]),
            ("pytest quick", [*_tool("pytest"), *pytest_quick]),
        ]
    if mode != "full":
        raise ValueError
    return [
        ("uv lock", ["uv", "lock", "--check"]),
        ("ruff format", [*_tool("ruff"), "format", "--check", "."]),
        ("ruff lint", [*_tool("ruff"), "check", "."]),
        ("pyright", [*_tool("pyright")]),
        ("import linter", [*_tool("lint-imports")]),
        ("i18n", _python_script("scripts/check_i18n.py")),
        ("forbidden patterns", _python_script("scripts/check_forbidden_patterns.py")),
        ("coverage erase", [*_tool("coverage"), "erase"]),
        (
            "pytest with branch coverage",
            [*_tool("coverage"), "run", "--branch", "-m", "pytest", *pytest_full],
        ),
        ("coverage report", [*_tool("coverage"), "report", "--fail-under", "85"]),
    ]


def run_steps(steps: Sequence[tuple[str, list[str]]]) -> int:
    """Run steps in order and stop at the first non-zero exit."""
    for name, command in steps:
        print(f"\n==> {name}: {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"quality gate failed at {name} with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode or 1
        except OSError as exc:
            print(f"quality gate could not start {name}: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested gate."""
    parser = argparse.ArgumentParser(description="Captioner quality gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    namespace = parser.parse_args(None if argv is None else list(argv))
    mode = "full" if namespace.full else "quick"
    return run_steps(build_steps(mode))


if __name__ == "__main__":
    raise SystemExit(main())
