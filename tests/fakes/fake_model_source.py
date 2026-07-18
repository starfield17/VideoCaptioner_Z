"""Deterministic Hugging Face and ModelScope source fakes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelSourceCandidate,
    ModelSourceCapabilities,
    ModelSourceReference,
)


def _empty_exact() -> dict[tuple[str, str, str], ModelSourceReference]:
    return {}


@dataclass(slots=True)
class FakeModelSource:
    source_id: str
    search_supported: bool
    exact_supported: bool = True
    search_results: Iterable[ModelSourceCandidate] = ()
    exact_results: Iterable[ModelSourceReference] = ()
    _search: tuple[ModelSourceCandidate, ...] = field(default=(), init=False)
    _exact: dict[tuple[str, str, str], ModelSourceReference] = field(
        default_factory=_empty_exact, init=False
    )

    def __post_init__(self) -> None:
        self._search = tuple(self.search_results)
        self._exact = {
            (
                reference.repository_id,
                reference.revision,
                reference.backend_id,
            ): reference
            for reference in self.exact_results
        }

    def capabilities(self) -> ModelSourceCapabilities:
        return ModelSourceCapabilities(
            search=self.search_supported,
            exact_repository=self.exact_supported,
            local_directory=self.source_id == "local-import",
            unmanaged_local_directory=self.source_id == "external-path",
        )

    def search(self, query: str, backend_id: str, limit: int) -> tuple[ModelSourceCandidate, ...]:
        if not self.search_supported:
            raise AppError("model.source_search_unsupported")
        if limit <= 0:
            raise AppError("model.source_query_invalid")
        normalized = query.casefold()
        return tuple(
            candidate
            for candidate in self._search
            if candidate.backend_id == backend_id
            and (not normalized or normalized in candidate.repository_id.casefold())
        )[:limit]

    def resolve_exact(
        self, repository_id: str, revision: str, backend_id: str
    ) -> ModelSourceReference | None:
        if not self.exact_supported:
            raise AppError("model.source_exact_unsupported")
        return self._exact.get((repository_id, revision, backend_id))


class FakeHuggingFaceSource(FakeModelSource):
    def __init__(
        self,
        *,
        search_results: Iterable[ModelSourceCandidate] = (),
        exact_results: Iterable[ModelSourceReference] = (),
    ) -> None:
        super().__init__(
            source_id="huggingface",
            search_supported=True,
            exact_supported=True,
            search_results=search_results,
            exact_results=exact_results,
        )


class FakeModelScopeSource(FakeModelSource):
    def __init__(
        self,
        *,
        exact_results: Iterable[ModelSourceReference] = (),
    ) -> None:
        super().__init__(
            source_id="modelscope",
            search_supported=False,
            exact_supported=True,
            exact_results=exact_results,
        )


__all__ = ["FakeHuggingFaceSource", "FakeModelScopeSource", "FakeModelSource"]
