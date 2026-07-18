"""Application configuration models and service for GUI settings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from captioner.core.application.input_selection import OutputCollisionPolicy
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile

if TYPE_CHECKING:
    from captioner.core.ports.configuration_store import ConfigurationStorePort
    from captioner.core.ports.provider_probe import ProviderProbePort

CredentialSource = Literal[
    "config",
    "environment",
    "missing",
]

_LOCALE_VALUES = frozenset({"en", "zh-CN"})
_DEVICES = frozenset({"auto", "cpu", "cuda"})
_TOKENIZERS = frozenset({"cl100k_base", "o200k_base", "auto"})
_COLLISION_POLICIES = frozenset({"unique_subdir", "fail", "overwrite"})
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_BUILTIN_NAMES = frozenset({"deterministic", "fast", "quality"})


def built_in_presets() -> tuple[ExecutionPreset, ...]:
    return (
        ExecutionPreset(
            name="deterministic",
            display_name="Deterministic",
            built_in=True,
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language=None,
            provider_profile="default",
        ),
        ExecutionPreset(
            name="fast",
            display_name="Fast",
            built_in=True,
            pipeline_profile=PipelineProfile.FAST,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language="zh-CN",
            provider_profile="default",
        ),
        ExecutionPreset(
            name="quality",
            display_name="Quality",
            built_in=True,
            pipeline_profile=PipelineProfile.QUALITY,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language="zh-CN",
            provider_profile="default",
        ),
    )


def default_global_settings() -> GlobalSettings:
    return GlobalSettings()


def default_provider_public_settings() -> ProviderPublicSettings:
    return ProviderPublicSettings(
        profile_name="default",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        max_concurrency=4,
        request_timeout_sec=120.0,
        max_retries=5,
        temperature=0.1,
        tokenizer="cl100k_base",
        credential_source="missing",
    )


def default_configuration_snapshot(
    *,
    issues: tuple[ConfigurationIssue, ...] = (),
) -> ConfigurationSnapshot:
    return ConfigurationSnapshot(
        global_settings=default_global_settings(),
        presets=built_in_presets(),
        provider=default_provider_public_settings(),
        issues=issues,
    )


def validate_user_preset_name(name: str) -> str:
    trimmed = name.strip()
    if not trimmed or len(trimmed) > 64:
        raise AppError("config.preset_name_invalid")
    if _CONTROL_CHAR_RE.search(trimmed) is not None:
        raise AppError("config.preset_name_invalid")
    if trimmed.casefold() in {item.casefold() for item in _BUILTIN_NAMES}:
        raise AppError("config.preset_builtin_immutable")
    return trimmed


@dataclass(frozen=True, slots=True)
class GlobalSettings:
    locale: Literal["en", "zh-CN"] = "en"
    default_output_root: str | None = None
    recursive_input: bool = True
    default_preset_name: str = "deterministic"
    collision_policy: OutputCollisionPolicy = "unique_subdir"

    def __post_init__(self) -> None:
        if self.locale not in _LOCALE_VALUES:
            raise AppError("config.settings_invalid", {"field": "locale"})
        root = self.default_output_root
        if root is not None:
            cleaned = root.strip()
            object.__setattr__(self, "default_output_root", cleaned or None)
        if not self.default_preset_name.strip():
            raise AppError("config.settings_invalid", {"field": "default_preset_name"})
        if self.collision_policy not in _COLLISION_POLICIES:
            raise AppError("config.settings_invalid", {"field": "collision_policy"})
        object.__setattr__(self, "default_preset_name", self.default_preset_name.strip())


@dataclass(frozen=True, slots=True)
class ExecutionPreset:
    name: str
    display_name: str
    built_in: bool
    pipeline_profile: PipelineProfile
    model_ref: str
    device: Literal["auto", "cpu", "cuda"]
    compute_type: str
    source_language: str | None
    target_language: str | None
    provider_profile: str
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    def __post_init__(self) -> None:
        name = self.name.strip() if self.built_in else validate_user_preset_name(self.name)
        if not self.display_name.strip():
            raise AppError("config.preset_invalid", {"field": "display_name"})
        profile = PipelineProfile(self.pipeline_profile)
        if not self.model_ref.strip():
            raise AppError("config.preset_invalid", {"field": "model_ref"})
        device = str(self.device)
        if device not in _DEVICES:
            raise AppError("config.preset_invalid", {"field": "device"})
        if not self.compute_type.strip():
            raise AppError("config.preset_invalid", {"field": "compute_type"})
        if self.source_language is not None and not self.source_language.strip():
            raise AppError("config.preset_invalid", {"field": "source_language"})
        if profile is PipelineProfile.DETERMINISTIC:
            if self.target_language is not None:
                raise AppError("config.preset_invalid", {"field": "target_language"})
        elif self.target_language is None or not self.target_language.strip():
            raise AppError("config.preset_invalid", {"field": "target_language"})
        if not self.provider_profile.strip():
            raise AppError("config.preset_invalid", {"field": "provider_profile"})
        if not self.ffmpeg_bin.strip():
            raise AppError("config.preset_invalid", {"field": "ffmpeg_bin"})
        if not self.ffprobe_bin.strip():
            raise AppError("config.preset_invalid", {"field": "ffprobe_bin"})
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_name", self.display_name.strip())
        object.__setattr__(self, "pipeline_profile", profile)
        object.__setattr__(self, "device", device)  # type: ignore[arg-type]
        object.__setattr__(self, "model_ref", self.model_ref.strip())
        object.__setattr__(self, "compute_type", self.compute_type.strip())
        object.__setattr__(
            self,
            "source_language",
            None if self.source_language is None else self.source_language.strip(),
        )
        object.__setattr__(
            self,
            "target_language",
            None if self.target_language is None else self.target_language.strip(),
        )
        object.__setattr__(self, "provider_profile", self.provider_profile.strip())
        object.__setattr__(self, "ffmpeg_bin", self.ffmpeg_bin.strip())
        object.__setattr__(self, "ffprobe_bin", self.ffprobe_bin.strip())


@dataclass(frozen=True, slots=True)
class ProviderPublicSettings:
    profile_name: str
    base_url: str
    model: str
    max_concurrency: int
    request_timeout_sec: float
    max_retries: int
    temperature: float
    tokenizer: Literal["cl100k_base", "o200k_base", "auto"]
    credential_source: CredentialSource

    def __post_init__(self) -> None:
        if not self.profile_name.strip():
            raise AppError("config.provider_invalid", {"field": "profile_name"})
        if not self.base_url.strip():
            raise AppError("config.provider_invalid", {"field": "base_url"})
        if not self.model.strip():
            raise AppError("config.provider_invalid", {"field": "model"})
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise AppError("config.provider_invalid", {"field": "max_concurrency"})
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise AppError("config.provider_invalid", {"field": "max_retries"})
        timeout = float(self.request_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0:
            raise AppError("config.provider_invalid", {"field": "request_timeout_sec"})
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise AppError("config.provider_invalid", {"field": "temperature"})
        if self.tokenizer not in _TOKENIZERS:
            raise AppError("config.provider_invalid", {"field": "tokenizer"})
        if self.credential_source not in {"config", "environment", "missing"}:
            raise AppError("config.provider_invalid", {"field": "credential_source"})
        object.__setattr__(self, "request_timeout_sec", timeout)
        object.__setattr__(self, "temperature", temperature)


@dataclass(frozen=True, slots=True, repr=False)
class ProviderSettingsUpdate:
    profile_name: str
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    max_concurrency: int = 4
    request_timeout_sec: float = 120.0
    max_retries: int = 5
    temperature: float = 0.1
    tokenizer: str = "cl100k_base"

    def __post_init__(self) -> None:
        if not self.profile_name.strip():
            raise AppError("config.provider_invalid", {"field": "profile_name"})
        if not self.base_url.strip():
            raise AppError("config.provider_invalid", {"field": "base_url"})
        if not self.model.strip():
            raise AppError("config.provider_invalid", {"field": "model"})
        if self.api_key is not None and not self.api_key.strip():
            raise AppError("config.provider_invalid", {"field": "api_key"})
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise AppError("config.provider_invalid", {"field": "max_concurrency"})
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise AppError("config.provider_invalid", {"field": "max_retries"})
        timeout = float(self.request_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0:
            raise AppError("config.provider_invalid", {"field": "request_timeout_sec"})
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise AppError("config.provider_invalid", {"field": "temperature"})
        if self.tokenizer not in _TOKENIZERS:
            raise AppError("config.provider_invalid", {"field": "tokenizer"})
        object.__setattr__(self, "profile_name", self.profile_name.strip())
        object.__setattr__(self, "base_url", self.base_url.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "request_timeout_sec", timeout)
        object.__setattr__(self, "temperature", temperature)
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip())

    def __repr__(self) -> str:
        return "ProviderSettingsUpdate(<redacted>)"


@dataclass(frozen=True, slots=True)
class ConfigurationIssue:
    code: str


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    global_settings: GlobalSettings
    presets: tuple[ExecutionPreset, ...]
    provider: ProviderPublicSettings
    issues: tuple[ConfigurationIssue, ...]


@dataclass(frozen=True, slots=True)
class ProviderConnectionResult:
    ok: bool
    code: str


@dataclass(slots=True)
class ConfigurationService:
    store: ConfigurationStorePort
    provider_probe: ProviderProbePort

    def load(self) -> ConfigurationSnapshot:
        return self.store.load_snapshot()

    def save_global(self, settings: GlobalSettings) -> ConfigurationSnapshot:
        self.store.save_global(settings)
        return self.store.load_snapshot()

    def save_provider(self, update: ProviderSettingsUpdate) -> ConfigurationSnapshot:
        self.store.save_provider(update)
        return self.store.load_snapshot()

    def save_user_preset(self, preset: ExecutionPreset) -> ConfigurationSnapshot:
        if preset.built_in:
            raise AppError("config.preset_builtin_immutable")
        validated = ExecutionPreset(
            name=preset.name,
            display_name=preset.display_name,
            built_in=False,
            pipeline_profile=preset.pipeline_profile,
            model_ref=preset.model_ref,
            device=preset.device,
            compute_type=preset.compute_type,
            source_language=preset.source_language,
            target_language=preset.target_language,
            provider_profile=preset.provider_profile,
            ffmpeg_bin=preset.ffmpeg_bin,
            ffprobe_bin=preset.ffprobe_bin,
        )
        self.store.save_user_preset(validated)
        return self.store.load_snapshot()

    def delete_user_preset(self, name: str) -> ConfigurationSnapshot:
        cleaned = name.strip()
        if cleaned.casefold() in {item.casefold() for item in _BUILTIN_NAMES}:
            raise AppError("config.preset_builtin_immutable")
        self.store.delete_user_preset(cleaned)
        return self.store.load_snapshot()

    def test_provider(self, update: ProviderSettingsUpdate) -> ProviderConnectionResult:
        probe_settings = self.store.resolve_provider_for_test(update)
        return self.provider_probe.test(probe_settings)


__all__ = [
    "ConfigurationIssue",
    "ConfigurationService",
    "ConfigurationSnapshot",
    "CredentialSource",
    "ExecutionPreset",
    "GlobalSettings",
    "ProviderConnectionResult",
    "ProviderPublicSettings",
    "ProviderSettingsUpdate",
    "built_in_presets",
    "default_configuration_snapshot",
    "default_global_settings",
    "default_provider_public_settings",
    "validate_user_preset_name",
]
