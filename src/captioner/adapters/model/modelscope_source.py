"""ModelScope exact-repository adapter; search is intentionally unsupported."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from typing import Protocol, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelSourceCandidate,
    ModelSourceCapabilities,
    ModelSourceReference,
)
from captioner.core.ports.model_source import ModelSource
from captioner.infrastructure.model_source_config import ModelSourceConfig

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


class _ModelScopeApi(Protocol):
    def get_model(self, *args: object, **kwargs: object) -> object: ...


class _ModelScopeApiFactory(Protocol):
    def __call__(self, **kwargs: object) -> _ModelScopeApi: ...


class ModelScopeModelSource(ModelSource):
    source_id = "modelscope"

    def __init__(self, config: ModelSourceConfig, *, api: object | None = None) -> None:
        if config.source_id != self.source_id:
            raise AppError("model.source_config_invalid", {"field": "source_id"})
        self.config = config
        self._api = api

    def capabilities(self) -> ModelSourceCapabilities:
        return ModelSourceCapabilities(search=False, exact_repository=True)

    def search(self, query: str, backend_id: str, limit: int) -> tuple[ModelSourceCandidate, ...]:
        del query, backend_id, limit
        raise AppError("model.source_search_unsupported")

    def resolve_exact(
        self,
        repository_id: str,
        revision: str | None,
        backend_id: str,
        model_format_hint: str | None = None,
    ) -> ModelSourceReference:
        _ensure_enabled(self.config.enabled)
        if (
            not repository_id.strip()
            or len(repository_id) > 512
            or "\\" in repository_id
            or ".." in repository_id
            or not backend_id.strip()
        ):
            raise AppError("model.source_query_invalid")
        if revision is None or not revision.strip():
            raise AppError("model.source_revision_required")
        api = self._client()
        try:
            info = api.get_model(repository_id, revision=revision)
        except AppError:
            raise
        except Exception as exc:
            raise _source_error(exc) from exc
        resolved = _resolved_revision(info)
        if resolved is None or resolved in {revision, "master", "main"}:
            raise AppError("model.source_revision_unresolved")
        return ModelSourceReference(
            source_id=self.source_id,
            repository_id=repository_id,
            revision=resolved,
            backend_id=backend_id,
            model_format_hint=model_format_hint,
        )

    def _client(self) -> _ModelScopeApi:
        if self._api is None:
            try:
                module = import_module("modelscope.hub.api")
                factory = cast(_ModelScopeApiFactory, module.__dict__["HubApi"])
            except (ImportError, KeyError) as exc:
                raise AppError("model.source_sdk_missing") from exc
            self._api = factory(endpoint=self.config.endpoint, token=self.config.token)
        return cast(_ModelScopeApi, self._api)


def _ensure_enabled(enabled: bool) -> None:
    if not enabled:
        raise AppError("model.source_disabled")


def _resolved_revision(value: object) -> str | None:
    for name in ("commit_id", "commit_hash", "sha", "revision", "id"):
        item: object = (
            cast(Mapping[object, object], value).get(name)
            if isinstance(value, Mapping)
            else getattr(value, name, None)
        )
        if isinstance(item, str) and item.strip():
            normalized = item.strip()
            if _SHA_RE.fullmatch(normalized):
                return normalized.lower()
    return None


def _source_error(error: Exception) -> AppError:
    error_name = type(error).__name__.casefold()
    if "revisionnotfound" in error_name:
        return AppError("model.source_revision_not_found")
    status = getattr(error, "status_code", None)
    if status == 401:
        return AppError("model.source_authentication_required")
    if status == 403:
        return AppError("model.source_access_denied")
    if status == 404:
        return AppError("model.source_repository_not_found")
    if status == 429:
        return AppError("model.source_rate_limited", retryable=True)
    return AppError("model.source_unavailable", retryable=True)


__all__ = ["ModelScopeModelSource"]
