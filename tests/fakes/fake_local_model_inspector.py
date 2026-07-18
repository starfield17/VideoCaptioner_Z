"""Deterministic local model inspection fake; it performs no filesystem I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from captioner.core.domain.model import LocalModelInspection


def _empty_calls() -> list[tuple[Path, str | None, str | None]]:
    return []


@dataclass(slots=True)
class FakeLocalModelInspector:
    inspection: LocalModelInspection
    calls: list[tuple[Path, str | None, str | None]] = field(default_factory=_empty_calls)

    def inspect(
        self,
        model_directory: Path,
        backend_hint: str | None = None,
        model_format_hint: str | None = None,
    ) -> LocalModelInspection:
        self.calls.append((model_directory, backend_hint, model_format_hint))
        return self.inspection


__all__ = ["FakeLocalModelInspector"]
