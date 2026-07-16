"""Immutable cache-key identities for validated structured LLM results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLM_RESPONSE_SCHEMA_VERSION, LLMItem
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)

LLM_CACHE_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SECRET_KEYS = frozenset({"api_key", "authorization", "access_token", "token"})


@dataclass(frozen=True, slots=True)
class LLMCacheKey:
    digest: str
    payload: Mapping[str, FrozenJsonValue]

    def __post_init__(self) -> None:
        if _DIGEST_RE.fullmatch(self.digest) is None:
            raise AppError("llm.cache_key_invalid", {"reason": "digest"})
        frozen = freeze_json_value(self.payload)
        if not isinstance(frozen, Mapping):
            raise AppError("llm.cache_key_invalid", {"reason": "payload"})
        if _contains_secret_key(frozen):
            raise AppError("llm.cache_key_invalid", {"reason": "secret"})
        expected = _digest_payload(frozen)
        if expected != self.digest:
            raise AppError("llm.cache_key_invalid", {"reason": "digest_mismatch"})
        object.__setattr__(self, "payload", cast(Mapping[str, FrozenJsonValue], frozen))

    @property
    def value(self) -> str:
        return self.digest

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "digest": self.digest,
            "payload": thaw_json_value(self.payload),
        }


def build_llm_cache_key(
    *,
    task_kind: str,
    provider_kind: str,
    provider_identity: str,
    base_url_identity: str,
    model: str,
    temperature: float,
    source_language: str | None,
    target_language: str | None,
    profile: str,
    prompt_id: str,
    prompt_version: str,
    prompt_content_sha256: str,
    items: Sequence[LLMItem],
    context: Sequence[LLMItem] = (),
    chunk_config: Mapping[str, object] | None = None,
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
) -> LLMCacheKey:
    """Build a canonical key with no credential, request ID, or clock input."""
    identities = (
        task_kind,
        provider_kind,
        provider_identity,
        base_url_identity,
        model,
        profile,
        prompt_id,
        prompt_version,
        prompt_content_sha256,
    )
    if not all(value.strip() for value in identities):
        raise AppError("llm.cache_key_invalid", {"reason": "identity"})
    raw_temperature: object = cast(object, temperature)
    if not isinstance(raw_temperature, (int, float)) or isinstance(raw_temperature, bool):
        raise AppError("llm.cache_key_invalid", {"reason": "temperature"})
    if not math.isfinite(float(raw_temperature)):
        raise AppError("llm.cache_key_invalid", {"reason": "temperature"})
    if type(response_schema_version) is not int or response_schema_version < 1:
        raise AppError("llm.cache_key_invalid", {"reason": "response_schema"})
    item_values = tuple(items)
    context_values = tuple(context)
    if not item_values or len({item.id for item in item_values}) != len(item_values):
        raise AppError("llm.cache_key_invalid", {"reason": "items"})
    if len({item.id for item in context_values}) != len(context_values):
        raise AppError("llm.cache_key_invalid", {"reason": "context"})
    if {item.id for item in item_values} & {item.id for item in context_values}:
        raise AppError("llm.cache_key_invalid", {"reason": "context_output_overlap"})
    config = {} if chunk_config is None else dict(chunk_config)
    try:
        frozen_config = freeze_json_value(config)
    except (TypeError, ValueError) as exc:
        raise AppError("llm.cache_key_invalid", {"reason": "chunk_config"}) from exc
    if _contains_secret_key(frozen_config):
        raise AppError("llm.cache_key_invalid", {"reason": "secret"})
    payload: dict[str, JsonValue] = {
        "cache_schema_version": LLM_CACHE_SCHEMA_VERSION,
        "task_kind": task_kind,
        "provider": {
            "kind": provider_kind,
            "identity": provider_identity,
            "base_url": base_url_identity,
        },
        "model": model,
        "temperature": float(raw_temperature),
        "source_language": source_language,
        "target_language": target_language,
        "profile": profile,
        "prompt": {
            "id": prompt_id,
            "version": prompt_version,
            "content_sha256": prompt_content_sha256,
        },
        "response_schema_version": response_schema_version,
        "items": [item.to_dict() for item in item_values],
        "context": [item.to_dict() for item in context_values],
        "chunk_config": thaw_json_value(frozen_config),
    }
    frozen_payload = freeze_json_value(payload)
    if not isinstance(frozen_payload, Mapping):
        raise AppError("llm.cache_key_invalid", {"reason": "payload"})
    typed_payload = cast(Mapping[str, FrozenJsonValue], frozen_payload)
    return LLMCacheKey(_digest_payload(typed_payload), typed_payload)


def derive_llm_cache_key(
    *,
    task_kind: str,
    provider_kind: str,
    provider_identity: str,
    base_url_identity: str,
    model: str,
    temperature: float,
    source_language: str | None,
    target_language: str | None,
    profile: str,
    prompt_id: str,
    prompt_version: str,
    prompt_content_sha256: str,
    items: Sequence[LLMItem],
    context: Sequence[LLMItem] = (),
    chunk_config: Mapping[str, object] | None = None,
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
) -> str:
    """Return only the stable digest for callers that do not need the payload."""
    return build_llm_cache_key(
        task_kind=task_kind,
        provider_kind=provider_kind,
        provider_identity=provider_identity,
        base_url_identity=base_url_identity,
        model=model,
        temperature=temperature,
        source_language=source_language,
        target_language=target_language,
        profile=profile,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_content_sha256=prompt_content_sha256,
        items=items,
        context=context,
        chunk_config=chunk_config,
        response_schema_version=response_schema_version,
    ).digest


def canonical_cache_json(value: object) -> bytes:
    """Serialize cache metadata with stable ordering and finite JSON only."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AppError("llm.cache_value_invalid", {"reason": "json"}) from exc


def _digest_payload(payload: Mapping[str, FrozenJsonValue]) -> str:
    return f"sha256:{hashlib.sha256(canonical_cache_json(thaw_json_value(payload))).hexdigest()}"


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(
            (isinstance(key, str) and key.lower().replace("-", "_") in _SECRET_KEYS)
            or _contains_secret_key(item)
            for key, item in mapping.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret_key(item) for item in cast(Sequence[object], value))
    return False
