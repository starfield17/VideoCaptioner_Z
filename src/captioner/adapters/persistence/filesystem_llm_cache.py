"""Atomic filesystem cache for schema-validated structured LLM responses."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm_cache import (
    LLM_CACHE_SCHEMA_VERSION,
    LLMCacheKey,
    canonical_cache_json,
)
from captioner.core.domain.result import JsonValue, thaw_json_value
from captioner.core.ports.llm_cache import LLMCachePort

T = TypeVar("T")


class _CacheResponse(Protocol):
    @classmethod
    def from_mapping(cls, value: object) -> object: ...

    def to_dict(self) -> JsonValue: ...


class FilesystemLLMCache(LLMCachePort):
    """Store each key under ``<cache_dir>/llm/sha256/``."""

    def __init__(self, cache_dir: Path) -> None:
        self._root = (cache_dir.expanduser().resolve() / "llm" / "sha256").resolve()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: LLMCacheKey) -> Path:
        digest = key.digest.removeprefix("sha256:")
        return self._root / digest[:2] / f"{digest}.json"

    def get(self, key: LLMCacheKey, response_schema: type[T]) -> T | None:
        path = self.path_for(key)
        try:
            raw = json.loads(
                path.read_bytes(),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, AppError, TypeError):
            # A corrupt or stale cache is a miss; it must never become a result.
            return None
        try:
            entry = _valid_entry(raw, key)
            if entry is None:
                return None
            return _decode_response(entry, response_schema)
        except (ValueError, AppError, TypeError):
            return None

    def put(
        self,
        key: LLMCacheKey,
        response: T,
        response_schema: type[T] | None = None,
    ) -> None:
        schema = response_schema or type(response)
        encoded_response = _encode_response(response, schema)
        entry: dict[str, JsonValue] = {
            "schema_version": LLM_CACHE_SCHEMA_VERSION,
            "cache_key": key.digest,
            "key_payload": thaw_json_value(key.payload),
            "response": encoded_response,
        }
        data = canonical_cache_json(entry)
        target = self.path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
            _fsync_directory(target.parent)
        except OSError as exc:
            raise AppError("llm.cache_write_failed") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def remove(self, key: LLMCacheKey) -> None:
        try:
            self.path_for(key).unlink(missing_ok=True)
        except OSError as exc:
            raise AppError("llm.cache_cleanup_failed") from exc


def _encode_response[T](response: T, response_schema: type[T]) -> JsonValue:
    candidate = cast(object, response)
    to_dict = getattr(candidate, "to_dict", None)
    if not callable(to_dict):
        raise AppError("llm.cache_value_invalid", {"reason": "to_dict"})
    value_object = to_dict()
    if not isinstance(value_object, (dict, list)):
        raise AppError("llm.cache_value_invalid", {"reason": "to_dict"})
    value = cast(JsonValue, value_object)
    try:
        parsed = cast(type[_CacheResponse], response_schema).from_mapping(value)
    except (AppError, TypeError, ValueError) as exc:
        raise AppError("llm.cache_value_invalid", {"reason": "schema"}) from exc
    if not isinstance(parsed, response_schema):
        raise AppError("llm.cache_value_invalid", {"reason": "schema"})
    return value


def _decode_response[T](entry: Mapping[str, object], response_schema: type[T]) -> T:
    value = entry.get("response")
    parsed = cast(type[_CacheResponse], response_schema).from_mapping(value)
    if not isinstance(parsed, response_schema):
        raise AppError("llm.cache_value_invalid", {"reason": "schema"})
    return parsed


def _valid_entry(raw: object, key: LLMCacheKey) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    entry = cast(Mapping[str, object], raw)
    if set(entry) != {"schema_version", "cache_key", "key_payload", "response"}:
        return None
    if entry["schema_version"] != LLM_CACHE_SCHEMA_VERSION:
        return None
    if entry["cache_key"] != key.digest:
        return None
    if entry["key_payload"] != thaw_json_value(key.payload):
        return None
    return entry


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_value:{value}")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
