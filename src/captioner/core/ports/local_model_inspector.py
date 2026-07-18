"""Core boundary for inspecting a local model directory before import."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.model import LocalModelInspection


class LocalModelInspector(Protocol):
    """Describe a local model without creating a durable model identity."""

    def inspect(
        self,
        model_directory: Path,
        backend_hint: str | None = None,
        model_format_hint: str | None = None,
    ) -> LocalModelInspection: ...


LocalModelInspectorPort = LocalModelInspector

__all__ = ["LocalModelInspector", "LocalModelInspectorPort"]
