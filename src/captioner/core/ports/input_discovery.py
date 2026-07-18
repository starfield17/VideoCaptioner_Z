"""Port for lightweight media input discovery without FFprobe."""

from __future__ import annotations

from typing import Protocol

from captioner.core.application.input_selection import (
    InputPreview,
    InputSelectionRequest,
)


class InputDiscoveryPort(Protocol):
    def preview(self, request: InputSelectionRequest) -> InputPreview: ...


__all__ = ["InputDiscoveryPort"]
