"""Captioner package metadata."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import cast

_FALLBACK_VERSION = "0.0.0"


def _source_tree_version() -> str | None:
    """Read the project version when distribution metadata is unavailable."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    project_values = cast(dict[str, object], project)
    value = project_values.get("version")
    return value if isinstance(value, str) else None


def get_version() -> str:
    """Return the canonical version for installed and source-tree execution."""
    try:
        return installed_version("captioner")
    except PackageNotFoundError:
        return _source_tree_version() or _FALLBACK_VERSION


__version__ = get_version()

__all__ = ["__version__", "get_version"]
