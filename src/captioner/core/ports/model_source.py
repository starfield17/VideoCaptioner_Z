"""Non-networking Model Source lookup boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from captioner.core.domain.model import (
    ModelSourceCandidate,
    ModelSourceCapabilities,
    ModelSourceReference,
)
from captioner.core.domain.operation_progress import OperationProgress


class ModelSource(Protocol):
    def capabilities(self) -> ModelSourceCapabilities: ...

    def search(
        self, query: str, backend_id: str, limit: int
    ) -> tuple[ModelSourceCandidate, ...]: ...

    def resolve_exact(
        self,
        repository_id: str,
        revision: str | None,
        backend_id: str,
        model_format_hint: str | None = None,
    ) -> ModelSourceReference: ...


ProgressCallback = Callable[[OperationProgress], None]


class ModelMaterializer(Protocol):
    """Materialize one immutable source reference into caller-owned staging."""

    def materialize(
        self,
        reference: ModelSourceReference,
        destination: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> None: ...


ModelSourcePort = ModelSource

__all__ = ["ModelMaterializer", "ModelSource", "ModelSourcePort", "ProgressCallback"]
