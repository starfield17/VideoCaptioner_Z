"""Redacted Model Source configuration loaded from the OS config directory."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from captioner.core.domain.errors import AppError

MODEL_SOURCES_FILENAME = "model-sources.toml"
_DEFAULT_ENDPOINTS = {
    "huggingface": "https://huggingface.co",
    "modelscope": "https://modelscope.cn",
}
_ENVIRONMENT_NAMES = {
    "huggingface": ("CAPTIONER_HF_TOKEN", "HF_TOKEN"),
    "modelscope": ("CAPTIONER_MODELSCOPE_TOKEN", "MODELSCOPE_API_TOKEN"),
}


@dataclass(frozen=True, slots=True, repr=False)
class ModelSourceCredential:
    """A source token that is intentionally absent from public projections."""

    value: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        raw_value: object = cast(object, self.value)
        if raw_value is not None and not isinstance(raw_value, str):
            raise AppError("model.source_config_invalid", {"field": "token"})
        if isinstance(self.value, str):
            object.__setattr__(self, "value", self.value.strip() or None)

    def __repr__(self) -> str:
        return "ModelSourceCredential(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ModelSourceConfig:
    source_id: str
    enabled: bool = True
    endpoint: str = ""
    credential: ModelSourceCredential = field(default_factory=ModelSourceCredential)
    max_workers: int = 4

    def __post_init__(self) -> None:
        raw_source_id: object = cast(object, self.source_id)
        if not isinstance(raw_source_id, str):
            raise AppError("model.source_config_invalid", {"field": "source_id"})
        source_id = raw_source_id.strip()
        if not source_id or self.source_id != source_id:
            raise AppError("model.source_config_invalid", {"field": "source_id"})
        if type(self.enabled) is not bool:
            raise AppError("model.source_config_invalid", {"field": "enabled"})
        raw_endpoint: object = cast(object, self.endpoint)
        if not isinstance(raw_endpoint, str):
            raise AppError("model.source_config_invalid", {"field": "endpoint"})
        endpoint = raw_endpoint.strip() or _DEFAULT_ENDPOINTS.get(source_id, "")
        _validate_endpoint(endpoint)
        if type(self.max_workers) is not int or not 1 <= self.max_workers <= 32:
            raise AppError("model.source_config_invalid", {"field": "max_workers"})
        raw_credential: object = cast(object, self.credential)
        if not isinstance(raw_credential, ModelSourceCredential):
            raise AppError("model.source_config_invalid", {"field": "token"})
        object.__setattr__(self, "endpoint", endpoint.rstrip("/"))

    @property
    def token(self) -> str | None:
        """Expose the token only at the adapter request boundary."""
        return self.credential.value

    def __repr__(self) -> str:
        return (
            "ModelSourceConfig("
            f"source_id={self.source_id!r}, enabled={self.enabled!r}, "
            f"endpoint={self.endpoint!r}, max_workers={self.max_workers!r}, "
            "credential=<redacted>)"
        )


def model_sources_config_path(config_dir_or_file: Path) -> Path:
    return (
        config_dir_or_file / MODEL_SOURCES_FILENAME
        if config_dir_or_file.suffix != ".toml"
        else config_dir_or_file
    )


def load_model_source_config(
    config_dir_or_file: Path,
    source_id: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> ModelSourceConfig:
    """Load one source without creating a sample file or logging its token."""
    path = model_sources_config_path(config_dir_or_file)
    raw_source: Mapping[object, object] = {}
    if path.exists():
        try:
            loaded = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise AppError("model.source_config_invalid", {"reason": "toml"}) from exc
        loaded_object: object = cast(object, loaded)
        if not isinstance(loaded_object, dict):
            raise AppError("model.source_config_invalid", {"reason": "root"})
        root = cast(dict[str, object], loaded_object)
        sources = root.get("sources")
        if not isinstance(sources, dict):
            raise AppError("model.source_config_invalid", {"field": "sources"})
        source_map = cast(dict[str, object], sources)
        source_value = source_map.get(source_id, {})
        if not isinstance(source_value, dict):
            raise AppError("model.source_config_invalid", {"field": source_id})
        raw_source = cast(Mapping[object, object], source_value)
        if set(raw_source) - {"enabled", "endpoint", "token", "max_workers"}:
            raise AppError("model.source_config_invalid", {"reason": "unknown_field"})
    enabled = raw_source.get("enabled", True)
    endpoint = raw_source.get("endpoint", _DEFAULT_ENDPOINTS.get(source_id, ""))
    max_workers = raw_source.get("max_workers", 4)
    configured_token = raw_source.get("token")
    if configured_token is not None and not isinstance(configured_token, str):
        raise AppError("model.source_config_invalid", {"field": "token"})
    token = configured_token.strip() if isinstance(configured_token, str) else ""
    if not token:
        env = os.environ if environment is None else environment
        for name in _ENVIRONMENT_NAMES.get(source_id, ()):
            candidate = env.get(name)
            if isinstance(candidate, str) and candidate.strip():
                token = candidate.strip()
                break
    if (
        not isinstance(enabled, bool)
        or not isinstance(endpoint, str)
        or type(max_workers) is not int
    ):
        raise AppError("model.source_config_invalid", {"reason": "types"})
    return ModelSourceConfig(
        source_id=source_id,
        enabled=enabled,
        endpoint=endpoint,
        credential=ModelSourceCredential(token or None),
        max_workers=max_workers,
    )


def load_all_model_source_configs(
    config_dir_or_file: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, ModelSourceConfig]:
    return {
        source_id: load_model_source_config(
            config_dir_or_file,
            source_id,
            environment=environment,
        )
        for source_id in _DEFAULT_ENDPOINTS
    }


def _validate_endpoint(value: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise AppError("model.source_endpoint_invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AppError("model.source_endpoint_invalid")


__all__ = [
    "MODEL_SOURCES_FILENAME",
    "ModelSourceConfig",
    "ModelSourceCredential",
    "load_all_model_source_configs",
    "load_model_source_config",
    "model_sources_config_path",
]
