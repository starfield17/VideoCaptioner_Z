"""Runtime-only LLM provider configuration loaded from the OS config area."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier
from captioner.core.domain.result import JsonValue

LLM_CONFIG_FILENAME = "llm.toml"
OPENAI_COMPATIBLE_KIND = "openai-compatible"
_PROVIDER_FIELDS = frozenset(
    {
        "kind",
        "base_url",
        "api_key",
        "model",
        "max_concurrency",
        "request_timeout_sec",
        "max_retries",
        "temperature",
    }
)


@dataclass(frozen=True, slots=True, repr=False)
class ProviderCredential:
    """An API credential that is intentionally not a serializable config value."""

    api_key: str = field(repr=False)

    def __post_init__(self) -> None:
        key: object = cast(object, self.api_key)
        if not isinstance(key, str) or not key.strip():
            raise AppError("llm.config_invalid", {"field": "api_key"})
        object.__setattr__(self, "api_key", key.strip())

    def __repr__(self) -> str:
        return "ProviderCredential(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class OpenAICompatibleProvider:
    """Validated runtime settings for one OpenAI-compatible provider profile."""

    profile_name: str
    base_url: str
    model: str
    credential: ProviderCredential
    max_concurrency: int = 4
    request_timeout_sec: float = 120.0
    max_retries: int = 5
    temperature: float = 0.1
    kind: str = OPENAI_COMPATIBLE_KIND

    def __post_init__(self) -> None:
        profile_name = validate_identifier(self.profile_name, field="provider_profile")
        if self.kind != OPENAI_COMPATIBLE_KIND:
            raise AppError("llm.config_invalid", {"field": "kind"})
        base_url = normalize_base_url_identity(self.base_url)
        model: object = cast(object, self.model)
        if not isinstance(model, str) or not model.strip():
            raise AppError("llm.config_invalid", {"field": "model"})
        credential: object = cast(object, self.credential)
        if not isinstance(credential, ProviderCredential):
            raise AppError("llm.config_invalid", {"field": "credential"})
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise AppError("llm.config_invalid", {"field": "max_concurrency"})
        timeout: object = cast(object, self.request_timeout_sec)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise AppError("llm.config_invalid", {"field": "request_timeout_sec"})
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise AppError("llm.config_invalid", {"field": "max_retries"})
        temperature: object = cast(object, self.temperature)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise AppError("llm.config_invalid", {"field": "temperature"})
        object.__setattr__(self, "profile_name", profile_name)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model.strip())
        object.__setattr__(self, "request_timeout_sec", float(timeout))
        object.__setattr__(self, "temperature", float(temperature))

    @property
    def api_key(self) -> str:
        """Return the credential only to the runtime request boundary."""
        return self.credential.api_key

    @property
    def normalized_base_url_identity(self) -> str:
        return self.base_url

    def to_snapshot(self) -> dict[str, JsonValue]:
        """Return the complete public config snapshot without credentials."""
        return {
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "max_concurrency": self.max_concurrency,
            "request_timeout_sec": self.request_timeout_sec,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "provider_profile": self.profile_name,
        }

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleProvider("
            f"profile_name={self.profile_name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, max_concurrency={self.max_concurrency!r}, "
            f"request_timeout_sec={self.request_timeout_sec!r}, "
            f"max_retries={self.max_retries!r}, temperature={self.temperature!r}, "
            "credential=<redacted>)"
        )


ProviderConfig = OpenAICompatibleProvider
LLMProviderConfig = OpenAICompatibleProvider


def load_provider_config(
    config_dir_or_file: Path,
    provider_profile: str = "default",
) -> OpenAICompatibleProvider:
    """Load one provider profile from ``llm.toml`` without creating files."""
    path = _config_path(config_dir_or_file)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AppError("llm.config_missing", {"path": str(path)}) from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AppError("llm.config_invalid", {"reason": "toml"}) from exc
    raw_object: object = cast(object, raw)
    if not isinstance(raw_object, dict):
        raise AppError("llm.config_invalid", {"reason": "root"})
    raw_mapping = cast(dict[str, object], raw_object)
    providers_value = raw_mapping.get("providers")
    if not isinstance(providers_value, dict):
        raise AppError("llm.config_invalid", {"field": "providers"})
    providers = cast(dict[str, object], providers_value)
    normalized_profile = validate_identifier(provider_profile, field="provider_profile")
    value = providers.get(normalized_profile)
    if not isinstance(value, dict):
        raise AppError("llm.provider_not_found", {"provider_profile": normalized_profile})
    provider = cast(dict[str, object], value)
    if set(provider) - _PROVIDER_FIELDS:
        raise AppError("llm.config_invalid", {"reason": "provider_fields"})
    try:
        kind = provider["kind"]
        base_url = provider["base_url"]
        api_key = provider["api_key"]
        model = provider["model"]
        max_concurrency = provider.get("max_concurrency", 4)
        request_timeout_sec = provider.get("request_timeout_sec", 120.0)
        max_retries = provider.get("max_retries", 5)
        temperature = provider.get("temperature", 0.1)
    except KeyError as exc:
        raise AppError("llm.config_invalid", {"field": str(exc.args[0])}) from exc
    if not all(isinstance(item, str) for item in (kind, base_url, api_key, model)):
        raise AppError("llm.config_invalid", {"reason": "provider_types"})
    if type(max_concurrency) is not int or type(max_retries) is not int:
        raise AppError("llm.config_invalid", {"reason": "retry_types"})
    if isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float)):
        raise AppError("llm.config_invalid", {"field": "request_timeout_sec"})
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise AppError("llm.config_invalid", {"field": "temperature"})
    return OpenAICompatibleProvider(
        normalized_profile,
        cast(str, base_url),
        cast(str, model),
        ProviderCredential(cast(str, api_key)),
        max_concurrency,
        float(request_timeout_sec),
        max_retries,
        float(temperature),
        cast(str, kind),
    )


def load_llm_config(
    config_dir_or_file: Path,
    provider_profile: str = "default",
) -> OpenAICompatibleProvider:
    """Compatibility spelling for callers that load the selected profile."""
    return load_provider_config(config_dir_or_file, provider_profile)


def normalize_base_url_identity(value: str) -> str:
    """Normalize a provider URL for both requests and cache identities."""
    if not value.strip():
        raise AppError("llm.config_invalid", {"field": "base_url"})
    normalized = value.strip()
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError as exc:
        raise AppError("llm.config_invalid", {"field": "base_url"}) from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
    ):
        raise AppError("llm.config_invalid", {"field": "base_url"})
    hostname = parsed.hostname
    netloc = hostname.lower()
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    is_default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    if port is not None and not is_default_port:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit(SplitResult(parsed.scheme.lower(), netloc, path, "", ""))


def _config_path(config_dir_or_file: Path) -> Path:
    path = config_dir_or_file.expanduser()
    if path.suffix.lower() != ".toml":
        path = path / LLM_CONFIG_FILENAME
    return path.resolve()


def write_llm_config(path: Path, content: str) -> None:
    """Write caller-provided TOML with restrictive POSIX permissions.

    This helper never invents a credential or writes a sample file.  It exists
    for explicit configuration UIs and tests that already hold the TOML text.
    """
    if not content.strip():
        raise AppError("llm.config_invalid", {"reason": "empty"})
    target = _config_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = config_file_creation_mode(os.name)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if should_chmod_private(os.name):
            os.chmod(target, 0o600)
    except OSError as exc:
        raise AppError("llm.config_write_failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def config_file_creation_mode(platform_name: str) -> int:
    """Return the portable creation mode for an explicit platform seam."""
    return 0o666 if platform_name == "nt" else 0o600


def should_chmod_private(platform_name: str) -> bool:
    """POSIX supports the explicit post-write private mode adjustment."""
    return platform_name != "nt"
