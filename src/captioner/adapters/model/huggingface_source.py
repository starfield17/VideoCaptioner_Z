"""Lazy Hugging Face lookup adapter with immutable revision resolution."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
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

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MAX_QUERY = 256
_MAX_DESCRIPTION = 4096


class _HfApi(Protocol):
    def list_models(self, **kwargs: object) -> Iterable[object]: ...

    def model_info(self, **kwargs: object) -> object: ...


class _HfApiFactory(Protocol):
    def __call__(self, **kwargs: object) -> _HfApi: ...


class HuggingFaceModelSource(ModelSource):
    source_id = "huggingface"

    def __init__(self, config: ModelSourceConfig, *, api: object | None = None) -> None:
        if config.source_id != self.source_id:
            raise AppError("model.source_config_invalid", {"field": "source_id"})
        self.config = config
        self._api = api

    def capabilities(self) -> ModelSourceCapabilities:
        return ModelSourceCapabilities(search=True, exact_repository=True)

    def search(self, query: str, backend_id: str, limit: int) -> tuple[ModelSourceCandidate, ...]:
        _ensure_enabled(self.config.enabled)
        _validate_search(query, backend_id, limit)
        api = self._client()
        try:
            with redirect_stdout(sys.stderr):
                rows = api.list_models(search=query.strip(), limit=limit, token=self.config.token)
            candidates = tuple(_candidate(row, backend_id) for row in rows)
        except AppError:
            raise
        except Exception as exc:
            raise _source_error(exc, "model.source_unavailable") from exc
        return tuple(candidate for candidate in candidates if candidate is not None)[:limit]

    def resolve_exact(
        self,
        repository_id: str,
        revision: str | None,
        backend_id: str,
        model_format_hint: str | None = None,
    ) -> ModelSourceReference:
        _ensure_enabled(self.config.enabled)
        _validate_repository(repository_id)
        if not backend_id.strip():
            raise AppError("model.source_query_invalid", {"field": "backend_id"})
        api = self._client()
        try:
            with redirect_stdout(sys.stderr):
                info = api.model_info(
                    repo_id=repository_id,
                    revision=revision,
                    files_metadata=True,
                    token=self.config.token,
                )
        except AppError:
            raise
        except Exception as exc:
            raise _source_error(exc, "model.source_repository_not_found", exact=True) from exc
        resolved = _first_value(info, ("sha", "commit_hash", "commit_id"))
        if not isinstance(resolved, str) or _SHA_RE.fullmatch(resolved) is None:
            raise AppError("model.source_revision_unresolved")
        return ModelSourceReference(
            source_id=self.source_id,
            repository_id=repository_id,
            revision=resolved.lower(),
            backend_id=backend_id,
            model_format_hint=model_format_hint,
        )

    def _client(self) -> _HfApi:
        if self._api is None:
            try:
                module = import_module("huggingface_hub")
                factory = cast(_HfApiFactory, module.__dict__["HfApi"])
            except (ImportError, KeyError) as exc:
                raise AppError("model.source_sdk_missing") from exc
            self._api = factory(endpoint=self.config.endpoint, token=self.config.token)
        return cast(_HfApi, self._api)


def _candidate(value: object, backend_id: str) -> ModelSourceCandidate | None:
    repository_id = _first_value(value, ("id", "modelId", "model_id"))
    if not isinstance(repository_id, str) or not repository_id.strip():
        return None
    description_value = _first_value(value, ("description",))
    description = (
        description_value.strip()[:_MAX_DESCRIPTION] if isinstance(description_value, str) else ""
    )
    revision = _first_value(value, ("sha", "commit_hash"))
    normalized_revision = (
        revision.strip().lower()
        if isinstance(revision, str) and _SHA_RE.fullmatch(revision.strip())
        else None
    )
    return ModelSourceCandidate(
        source_id="huggingface",
        repository_id=repository_id.strip(),
        revision=normalized_revision,
        backend_id=backend_id,
        model_format_hint=_infer_format(value),
        display_name=repository_id.rsplit("/", 1)[-1],
        description=description,
    )


def _infer_format(value: object) -> str | None:
    siblings = _first_value(value, ("siblings", "files"))
    if not isinstance(siblings, Iterable) or isinstance(siblings, (str, bytes, Mapping)):
        return None
    names: set[str] = set()
    for item in cast(Iterable[object], siblings):
        name = _first_value(item, ("rfilename", "path", "name"))
        if isinstance(name, str):
            names.add(name)
    mlx = bool(names & {"model.safetensors", "weights.safetensors", "weights.npz"})
    ct2 = {"config.json", "model.bin", "tokenizer.json"} <= names
    if mlx and not ct2:
        return "mlx-whisper"
    if ct2 and not mlx:
        return "faster-whisper-ct2"
    return None


def _first_value(value: object, names: tuple[str, ...]) -> object:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        for name in names:
            if name in raw:
                item = raw[name]
                if item is not None:
                    return item
    object_value = cast(object, value)
    for name in names:
        item = getattr(object_value, name, None)
        if item is not None:
            return item
    return None


def _validate_search(query: str, backend_id: str, limit: int) -> None:
    if not query.strip() or len(query) > _MAX_QUERY:
        raise AppError("model.source_query_invalid", {"field": "query"})
    if not backend_id.strip():
        raise AppError("model.source_query_invalid", {"field": "backend_id"})
    if not 1 <= limit <= 100:
        raise AppError("model.source_query_invalid", {"field": "limit"})


def _validate_repository(repository_id: str) -> None:
    if (
        not repository_id.strip()
        or len(repository_id) > 512
        or "\\" in repository_id
        or ".." in repository_id
    ):
        raise AppError("model.source_query_invalid", {"field": "repository_id"})


def _ensure_enabled(enabled: bool) -> None:
    if not enabled:
        raise AppError("model.source_disabled")


def _source_error(error: Exception, fallback: str, *, exact: bool = False) -> AppError:
    error_name = type(error).__name__.casefold()
    if "revisionnotfound" in error_name or "entrynotfound" in error_name:
        return AppError("model.source_revision_not_found")
    if "repositorynotfound" in error_name or "repoaccess" in error_name:
        return AppError("model.source_repository_not_found")
    status = getattr(error, "status_code", None)
    if status == 401:
        return AppError("model.source_authentication_required")
    if status == 403:
        return AppError("model.source_access_denied")
    if status == 429:
        return AppError("model.source_rate_limited", retryable=True)
    if status == 404:
        return AppError("model.source_revision_not_found" if exact else fallback)
    return AppError(fallback, retryable=True)


__all__ = ["HuggingFaceModelSource"]
