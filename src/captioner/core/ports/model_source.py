"""Non-networking Model Source lookup boundary."""

from __future__ import annotations

from typing import Protocol

from captioner.core.domain.model import (
    ModelSourceCandidate,
    ModelSourceCapabilities,
    ModelSourceReference,
)


class ModelSource(Protocol):
    def capabilities(self) -> ModelSourceCapabilities: ...

    def search(
        self, query: str, backend_id: str, limit: int
    ) -> tuple[ModelSourceCandidate, ...]: ...

    def resolve_exact(
        self, repository_id: str, revision: str, backend_id: str
    ) -> ModelSourceReference | None: ...


ModelSourcePort = ModelSource

__all__ = ["ModelSource", "ModelSourcePort"]
