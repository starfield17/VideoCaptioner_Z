"""Strict, credential-free LLM configuration snapshots for durable Jobs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)
from captioner.core.domain.stage import PipelineProfile

LLM_JOB_SNAPSHOT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_TARGET_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
PUBLIC_PROVIDER_FIELDS = (
    "kind",
    "provider_profile",
    "base_url",
    "model",
    "max_concurrency",
    "request_timeout_sec",
    "max_retries",
    "temperature",
)

_FAST_PROMPTS = ("translate_fast", "repair_structured")
_QUALITY_PROMPTS = (
    "terminology",
    "correct_source",
    "translate_quality",
    "review_anomalies",
    "repair_structured",
)
_CHUNK_FIELDS = frozenset(
    {
        "max_items",
        "max_input_tokens",
        "context_before_items",
        "context_after_items",
        "max_audio_context_duration_ms",
    }
)


def required_prompts_for(profile: PipelineProfile | str) -> tuple[str, ...]:
    """Return the exact versioned prompt identities required by a profile."""
    selected = PipelineProfile(profile)
    if selected is PipelineProfile.DETERMINISTIC:
        return ()
    return _FAST_PROMPTS if selected is PipelineProfile.FAST else _QUALITY_PROMPTS


@dataclass(frozen=True, slots=True)
class ProviderPublicSnapshot:
    """The provider fields that define result identity, never including a key."""

    kind: str
    provider_profile: str
    base_url: str
    model: str
    max_concurrency: int
    request_timeout_sec: float
    max_retries: int
    temperature: float

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.kind, "kind"),
            (self.provider_profile, "provider_profile"),
            (self.base_url, "base_url"),
            (self.model, "model"),
        ):
            _nonempty_string(value, field_name)
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise AppError("llm.config_invalid", {"field": "max_concurrency"})
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise AppError("llm.config_invalid", {"field": "max_retries"})
        if not _finite_positive(self.request_timeout_sec):
            raise AppError("llm.config_invalid", {"field": "request_timeout_sec"})
        if not _finite_nonnegative(self.temperature):
            raise AppError("llm.config_invalid", {"field": "temperature"})
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "provider_profile", self.provider_profile.strip())
        object.__setattr__(self, "base_url", self.base_url.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "request_timeout_sec", float(self.request_timeout_sec))
        object.__setattr__(self, "temperature", float(self.temperature))

    @classmethod
    def from_mapping(cls, value: object) -> ProviderPublicSnapshot:
        if not isinstance(value, Mapping):
            raise AppError("llm.provider_snapshot_invalid", {"reason": "fields"})
        raw = cast(Mapping[str, object], value)
        if set(raw) != set(PUBLIC_PROVIDER_FIELDS):
            raise AppError("llm.provider_snapshot_invalid", {"reason": "fields"})
        kind = _required_string(raw, "kind")
        profile = _required_string(raw, "provider_profile")
        base_url = _required_string(raw, "base_url")
        model = _required_string(raw, "model")
        max_concurrency = _required_int(raw, "max_concurrency")
        timeout = _required_number(raw, "request_timeout_sec")
        retries = _required_int(raw, "max_retries", minimum=0)
        temperature = _required_number(raw, "temperature", minimum=0)
        return cls(kind, profile, base_url, model, max_concurrency, timeout, retries, temperature)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "provider_profile": self.provider_profile,
            "base_url": self.base_url,
            "model": self.model,
            "max_concurrency": self.max_concurrency,
            "request_timeout_sec": self.request_timeout_sec,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
        }

    def changed_fields(self, other: ProviderPublicSnapshot) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in PUBLIC_PROVIDER_FIELDS
            if getattr(self, field_name) != getattr(other, field_name)
        )


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    """A durable prompt identity; prompt content itself remains in resources."""

    prompt_id: str
    prompt_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        _prompt_component(self.prompt_id, "prompt_id")
        _prompt_component(self.prompt_version, "prompt_version")
        content_sha256 = cast(object, self.content_sha256)
        if not isinstance(content_sha256, str) or _SHA256_RE.fullmatch(content_sha256) is None:
            raise AppError("prompt.identity_invalid", {"prompt_id": self.prompt_id})

    @classmethod
    def from_mapping(cls, value: object) -> PromptSnapshot:
        if not isinstance(value, Mapping):
            raise AppError("prompt.identity_invalid", {"reason": "fields"})
        raw = cast(Mapping[str, object], value)
        if set(raw) != {
            "prompt_id",
            "prompt_version",
            "content_sha256",
        }:
            raise AppError("prompt.identity_invalid", {"reason": "fields"})
        values = tuple(
            _required_string(raw, name)
            for name in ("prompt_id", "prompt_version", "content_sha256")
        )
        return cls(*values)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class LLMJobSnapshot:
    """Complete profile-specific public LLM configuration persisted in a Job."""

    profile: PipelineProfile
    provider: ProviderPublicSnapshot
    source_language: str | None
    target_language: str
    chunk: Mapping[str, FrozenJsonValue]
    prompts: Mapping[str, PromptSnapshot]
    response_schema_version: int
    schema_version: int = LLM_JOB_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            profile = PipelineProfile(self.profile)
        except ValueError as exc:
            raise AppError("llm.snapshot_invalid", {"reason": "profile"}) from exc
        object.__setattr__(self, "profile", profile)
        if (
            type(self.schema_version) is not int
            or self.schema_version != LLM_JOB_SNAPSHOT_SCHEMA_VERSION
        ):
            raise AppError("llm.snapshot_invalid", {"reason": "schema_version"})
        if not isinstance(cast(object, self.provider), ProviderPublicSnapshot):
            raise AppError("llm.snapshot_invalid", {"reason": "provider"})
        _target_language(self.target_language)
        if self.source_language is not None:
            _nonempty_string(self.source_language, "source_language")
        if type(self.response_schema_version) is not int or self.response_schema_version < 1:
            raise AppError("llm.snapshot_invalid", {"reason": "response_schema_version"})
        try:
            chunk = freeze_json_value(self.chunk)
        except (TypeError, ValueError) as exc:
            raise AppError("llm.snapshot_invalid", {"reason": "chunk"}) from exc
        if not isinstance(chunk, Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "chunk"})
        _validate_chunk(chunk)
        object.__setattr__(self, "chunk", cast(Mapping[str, FrozenJsonValue], chunk))
        required = required_prompts_for(self.profile)
        if not isinstance(cast(object, self.prompts), Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "prompts"})
        prompts = dict(self.prompts)
        if set(prompts) != set(required):
            raise AppError("llm.snapshot_invalid", {"reason": "prompts"})
        for prompt_id in required:
            prompt = prompts.get(prompt_id)
            if not isinstance(prompt, PromptSnapshot) or prompt.prompt_id != prompt_id:
                raise AppError("llm.snapshot_invalid", {"reason": "prompt_identity"})
        object.__setattr__(self, "prompts", MappingProxyType(prompts))

    @classmethod
    def from_mapping(cls, value: object) -> LLMJobSnapshot:
        if not isinstance(value, Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "object"})
        raw = cast(Mapping[str, object], value)
        expected = {
            "snapshot_schema_version",
            *PUBLIC_PROVIDER_FIELDS,
            "profile",
            "source_language",
            "target_language",
            "chunk",
            "prompts",
            "response_schema_version",
        }
        if set(raw) != expected:
            raise AppError("llm.snapshot_invalid", {"reason": "fields"})
        try:
            profile = PipelineProfile(_required_string(raw, "profile"))
        except ValueError as exc:
            raise AppError("llm.snapshot_invalid", {"reason": "profile"}) from exc
        snapshot_version = raw.get("snapshot_schema_version")
        if type(snapshot_version) is not int or snapshot_version != LLM_JOB_SNAPSHOT_SCHEMA_VERSION:
            raise AppError("llm.snapshot_invalid", {"reason": "schema_version"})
        raw_prompts = raw.get("prompts")
        if not isinstance(raw_prompts, Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "prompts"})
        typed_prompts = cast(Mapping[object, object], raw_prompts)
        prompts: dict[str, PromptSnapshot] = {}
        for prompt_id, prompt_value in typed_prompts.items():
            if not isinstance(prompt_id, str):
                raise AppError("llm.snapshot_invalid", {"reason": "prompts"})
            prompts[prompt_id] = PromptSnapshot.from_mapping(prompt_value)
        if len(prompts) != len(typed_prompts):
            raise AppError("llm.snapshot_invalid", {"reason": "prompts"})
        raw_chunk = raw.get("chunk")
        if not isinstance(raw_chunk, Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "chunk"})
        source = raw.get("source_language")
        if source is not None and not isinstance(source, str):
            raise AppError("llm.snapshot_invalid", {"reason": "source_language"})
        return cls(
            profile,
            ProviderPublicSnapshot.from_mapping(
                {name: raw[name] for name in PUBLIC_PROVIDER_FIELDS}
            ),
            source,
            _required_string(raw, "target_language"),
            cast(Mapping[str, FrozenJsonValue], raw_chunk),
            prompts,
            _required_int(raw, "response_schema_version"),
            cast(int, snapshot_version),
        )

    def to_mapping(self) -> Mapping[str, FrozenJsonValue]:
        value: dict[str, JsonValue] = {
            "snapshot_schema_version": self.schema_version,
            **self.provider.to_mapping(),
            "profile": self.profile.value,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "chunk": thaw_json_value(self.chunk),
            "prompts": {key: prompt.to_mapping() for key, prompt in self.prompts.items()},
            "response_schema_version": self.response_schema_version,
        }
        frozen = freeze_json_value(value)
        if not isinstance(frozen, Mapping):
            raise AppError("llm.snapshot_invalid", {"reason": "mapping"})
        return cast(Mapping[str, FrozenJsonValue], frozen)


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AppError("llm.config_invalid", {"field": field_name})
    return value


def _prompt_component(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _PROMPT_RE.fullmatch(value) is None or ".." in value:
        raise AppError("prompt.identity_invalid", {"field": field_name})
    return value


def _target_language(value: object) -> str:
    if not isinstance(value, str) or _TARGET_LANGUAGE_RE.fullmatch(value) is None:
        raise AppError("llm.target_language_invalid")
    return value


def _finite_positive(value: object) -> bool:
    return (
        type(value) in {int, float}
        and float(cast(int | float, value)) > 0
        and math.isfinite(float(cast(int | float, value)))
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        type(value) in {int, float}
        and float(cast(int | float, value)) >= 0
        and math.isfinite(float(cast(int | float, value)))
    )


def _required_string(raw: Mapping[str, object], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str):
        raise AppError("llm.snapshot_invalid", {"reason": field_name})
    return value


def _required_int(raw: Mapping[str, object], field_name: str, *, minimum: int = 1) -> int:
    value = raw.get(field_name)
    if type(value) is not int or value < minimum:
        raise AppError("llm.snapshot_invalid", {"reason": field_name})
    return value


def _required_number(
    raw: Mapping[str, object], field_name: str, *, minimum: float | None = None
) -> float:
    value = raw.get(field_name)
    if type(value) not in {int, float}:
        raise AppError("llm.snapshot_invalid", {"reason": field_name})
    number = float(cast(int | float, value))
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise AppError("llm.snapshot_invalid", {"reason": field_name})
    return number


def _validate_chunk(chunk: Mapping[str, FrozenJsonValue]) -> None:
    if frozenset(chunk) != _CHUNK_FIELDS:
        raise AppError("llm.snapshot_invalid", {"reason": "chunk_fields"})
    positive_fields = ("max_items", "max_input_tokens")
    nonnegative_fields = ("context_before_items", "context_after_items")
    for field_name in (*positive_fields, *nonnegative_fields):
        value = chunk.get(field_name)
        minimum = 1 if field_name in positive_fields else 0
        if type(value) is not int or value < minimum:
            raise AppError("llm.snapshot_invalid", {"reason": field_name})
    duration = chunk.get("max_audio_context_duration_ms")
    if duration is not None and (type(duration) is not int or duration < 1):
        raise AppError("llm.snapshot_invalid", {"reason": "max_audio_context_duration_ms"})
