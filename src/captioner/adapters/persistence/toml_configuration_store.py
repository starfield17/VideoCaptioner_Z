"""Strict TOML configuration store for settings.toml and llm.toml."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from captioner.core.application.configuration import (
    ConfigurationIssue,
    ConfigurationSnapshot,
    CredentialSource,
    ExecutionPreset,
    GlobalSettings,
    ProviderPublicSettings,
    ProviderSettingsUpdate,
    built_in_presets,
    default_global_settings,
    default_provider_public_settings,
    validate_user_preset_name,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier
from captioner.core.domain.stage import PipelineProfile
from captioner.core.ports.configuration_store import ProviderRuntimeProbeSettings
from captioner.infrastructure.config import (
    LLM_CONFIG_FILENAME,
    OPENAI_COMPATIBLE_KIND,
    PROVIDER_FIELDS,
    config_file_creation_mode,
    normalize_base_url_identity,
    resolve_provider_credential,
    should_chmod_private,
)

SETTINGS_FILENAME = "settings.toml"
SETTINGS_SCHEMA_VERSION = 1

_GLOBAL_FIELDS = frozenset(
    {
        "locale",
        "default_output_root",
        "recursive_input",
        "default_preset_name",
        "collision_policy",
    }
)
_PRESET_FIELDS = frozenset(
    {
        "display_name",
        "pipeline_profile",
        "model_ref",
        "device",
        "compute_type",
        "source_language",
        "target_language",
        "provider_profile",
        "ffmpeg_bin",
        "ffprobe_bin",
    }
)
_ROOT_SECTIONS = frozenset({"schema_version", "global", "presets"})
_ENV_PROFILE_RE = re.compile(r"[^A-Za-z0-9]+")


def normalize_profile_env_suffix(profile_name: str) -> str:
    upper = profile_name.upper()
    return _ENV_PROFILE_RE.sub("_", upper)


def escape_toml_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _toml_string(value: str) -> str:
    return f'"{escape_toml_string(value)}"'


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = config_file_creation_mode(os.name)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if should_chmod_private(os.name):
            os.chmod(path, 0o600)
    except OSError as exc:
        raise AppError("config.write_failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def serialize_settings_toml(
    settings: GlobalSettings,
    user_presets: tuple[ExecutionPreset, ...],
) -> str:
    lines = [
        f"schema_version = {SETTINGS_SCHEMA_VERSION}",
        "",
        "[global]",
        f"locale = {_toml_string(settings.locale)}",
        f"default_output_root = {_toml_string(settings.default_output_root or '')}",
        f"recursive_input = {_toml_bool(settings.recursive_input)}",
        f"default_preset_name = {_toml_string(settings.default_preset_name)}",
        f"collision_policy = {_toml_string(settings.collision_policy)}",
    ]
    for preset in user_presets:
        if preset.built_in:
            continue
        lines.append("")
        lines.append(f"[presets.{_toml_key(preset.name)}]")
        lines.append(f"display_name = {_toml_string(preset.display_name)}")
        lines.append(f"pipeline_profile = {_toml_string(preset.pipeline_profile.value)}")
        lines.append(f"model_ref = {_toml_string(preset.model_ref)}")
        lines.append(f"device = {_toml_string(preset.device)}")
        lines.append(f"compute_type = {_toml_string(preset.compute_type)}")
        lines.append(f"source_language = {_toml_string(preset.source_language or '')}")
        lines.append(f"target_language = {_toml_string(preset.target_language or '')}")
        lines.append(f"provider_profile = {_toml_string(preset.provider_profile)}")
        lines.append(f"ffmpeg_bin = {_toml_string(preset.ffmpeg_bin)}")
        lines.append(f"ffprobe_bin = {_toml_string(preset.ffprobe_bin)}")
    lines.append("")
    return "\n".join(lines)


def _toml_key(name: str) -> str:
    # Bare keys for simple identifiers; quoted keys otherwise.
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None:
        return name
    return _toml_string(name)


def serialize_llm_providers(providers: Mapping[str, Mapping[str, object]]) -> str:
    lines: list[str] = []
    for profile_name in sorted(providers):
        provider = providers[profile_name]
        if lines:
            lines.append("")
        lines.append(f"[providers.{_toml_key(profile_name)}]")
        for field_name in (
            "kind",
            "base_url",
            "api_key",
            "model",
            "max_concurrency",
            "request_timeout_sec",
            "max_retries",
            "temperature",
            "tokenizer",
        ):
            if field_name not in provider:
                continue
            value = provider[field_name]
            if isinstance(value, bool):
                rendered = _toml_bool(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                rendered = str(value)
            else:
                rendered = _toml_string(str(value))
            lines.append(f"{field_name} = {rendered}")
    lines.append("")
    return "\n".join(lines)


@dataclass(slots=True)
class TomlConfigurationStore:
    config_dir: Path
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.config_dir = self.config_dir.expanduser().resolve()

    @property
    def settings_path(self) -> Path:
        return self.config_dir / SETTINGS_FILENAME

    @property
    def llm_path(self) -> Path:
        return self.config_dir / LLM_CONFIG_FILENAME

    def _env(self) -> Mapping[str, str]:
        return os.environ if self.environment is None else self.environment

    def load_snapshot(self) -> ConfigurationSnapshot:
        issues: list[ConfigurationIssue] = []
        try:
            global_settings, user_presets = self._load_settings_file()
        except AppError as exc:
            if exc.code in {"config.settings_invalid", "config.settings_missing"}:
                if exc.code == "config.settings_invalid":
                    issues.append(ConfigurationIssue(code="config.settings_invalid"))
                global_settings = default_global_settings()
                user_presets = ()
            else:
                raise
        try:
            provider = self._load_provider_public("default")
        except AppError as exc:
            if exc.code.startswith("llm.") or exc.code.startswith("config."):
                issues.append(ConfigurationIssue(code=exc.code))
                provider = default_provider_public_settings()
            else:
                raise
        presets = built_in_presets() + user_presets
        return ConfigurationSnapshot(
            global_settings=global_settings,
            presets=presets,
            provider=provider,
            issues=tuple(issues),
        )

    def save_global(self, settings: GlobalSettings) -> None:
        _, user_presets = self._load_settings_or_defaults()
        content = serialize_settings_toml(settings, user_presets)
        _write_atomic(self.settings_path, content)

    def save_provider(self, update: ProviderSettingsUpdate) -> None:
        providers = self._read_raw_providers()
        profile = validate_identifier(update.profile_name, field="provider_profile")
        existing = dict(providers.get(profile, {}))
        existing_key = existing.get("api_key")
        api_key: str | None
        if update.api_key is None:
            if isinstance(existing_key, str) and existing_key.strip():
                api_key = existing_key.strip()
            else:
                # Preserve absence; runtime may still resolve environment.
                api_key = None
        else:
            api_key = update.api_key.strip()
        base_url = normalize_base_url_identity(update.base_url)
        payload: dict[str, object] = {
            "kind": OPENAI_COMPATIBLE_KIND,
            "base_url": base_url,
            "model": update.model,
            "max_concurrency": update.max_concurrency,
            "request_timeout_sec": float(update.request_timeout_sec),
            "max_retries": update.max_retries,
            "temperature": float(update.temperature),
            "tokenizer": update.tokenizer,
        }
        if api_key is not None:
            payload["api_key"] = api_key
        providers[profile] = payload
        _write_atomic(self.llm_path, serialize_llm_providers(providers))

    def save_user_preset(self, preset: ExecutionPreset) -> None:
        if preset.built_in:
            raise AppError("config.preset_builtin_immutable")
        settings, user_presets = self._load_settings_or_defaults()
        name = validate_user_preset_name(preset.name)
        remaining = tuple(item for item in user_presets if item.name.casefold() != name.casefold())
        saved = ExecutionPreset(
            name=name,
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
        updated = (*remaining, saved)
        # Stable case-insensitive display-name order for persistence.
        ordered = tuple(
            sorted(updated, key=lambda item: (item.display_name.casefold(), item.name.casefold()))
        )
        _write_atomic(self.settings_path, serialize_settings_toml(settings, ordered))

    def delete_user_preset(self, name: str) -> None:
        cleaned = name.strip()
        if cleaned.casefold() in {item.name.casefold() for item in built_in_presets()}:
            raise AppError("config.preset_builtin_immutable")
        settings, user_presets = self._load_settings_or_defaults()
        remaining = tuple(
            item for item in user_presets if item.name.casefold() != cleaned.casefold()
        )
        if len(remaining) == len(user_presets):
            raise AppError("config.preset_not_found")
        _write_atomic(self.settings_path, serialize_settings_toml(settings, remaining))

    def resolve_provider_for_test(
        self,
        update: ProviderSettingsUpdate,
    ) -> ProviderRuntimeProbeSettings:
        profile = validate_identifier(update.profile_name, field="provider_profile")
        api_key = update.api_key
        if api_key is None:
            providers = self._read_raw_providers()
            existing = providers.get(profile, {})
            existing_key = existing.get("api_key")
            config_key = (
                existing_key.strip()
                if isinstance(existing_key, str) and existing_key.strip()
                else None
            )
            resolved = resolve_provider_credential(
                profile_name=profile,
                config_api_key=config_key,
                environment=self._env(),
            )
            if resolved is None:
                raise AppError("llm.config_invalid", {"field": "api_key"})
            api_key = resolved
        base_url = normalize_base_url_identity(update.base_url)
        return ProviderRuntimeProbeSettings(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=float(update.request_timeout_sec),
        )

    def _load_settings_or_defaults(
        self,
    ) -> tuple[GlobalSettings, tuple[ExecutionPreset, ...]]:
        try:
            return self._load_settings_file()
        except AppError as exc:
            if exc.code in {"config.settings_invalid", "config.settings_missing"}:
                return default_global_settings(), ()
            raise

    def _load_settings_file(self) -> tuple[GlobalSettings, tuple[ExecutionPreset, ...]]:
        path = self.settings_path
        if not path.exists():
            raise AppError("config.settings_missing")
        try:
            loaded = cast(object, tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise AppError("config.settings_invalid") from exc
        if not isinstance(loaded, dict):
            raise AppError("config.settings_invalid")
        mapping = cast(dict[str, Any], loaded)
        unknown_roots = set(mapping) - _ROOT_SECTIONS
        if unknown_roots:
            raise AppError("config.settings_invalid")
        version = mapping.get("schema_version")
        if version != SETTINGS_SCHEMA_VERSION:
            raise AppError("config.settings_invalid")
        global_raw = mapping.get("global")
        if not isinstance(global_raw, dict):
            raise AppError("config.settings_invalid")
        global_map = cast(dict[str, Any], global_raw)
        if set(global_map) - _GLOBAL_FIELDS:
            raise AppError("config.settings_invalid")
        locale = global_map.get("locale", "en")
        default_output_root = global_map.get("default_output_root", "")
        recursive_input = global_map.get("recursive_input", True)
        default_preset_name = global_map.get("default_preset_name", "deterministic")
        collision_policy = global_map.get("collision_policy", "unique_subdir")
        if not isinstance(locale, str):
            raise AppError("config.settings_invalid")
        if not isinstance(default_output_root, str):
            raise AppError("config.settings_invalid")
        if not isinstance(recursive_input, bool):
            raise AppError("config.settings_invalid")
        if not isinstance(default_preset_name, str):
            raise AppError("config.settings_invalid")
        if not isinstance(collision_policy, str):
            raise AppError("config.settings_invalid")
        try:
            settings = GlobalSettings(
                locale=locale,  # type: ignore[arg-type]
                default_output_root=default_output_root or None,
                recursive_input=recursive_input,
                default_preset_name=default_preset_name,
                collision_policy=collision_policy,  # type: ignore[arg-type]
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError("config.settings_invalid") from exc

        presets_raw = mapping.get("presets", {})
        if presets_raw is None:
            presets_raw = {}
        if not isinstance(presets_raw, dict):
            raise AppError("config.settings_invalid")
        user_presets: list[ExecutionPreset] = []
        for name, value in cast(dict[str, Any], presets_raw).items():
            if not isinstance(value, dict):
                raise AppError("config.settings_invalid")
            preset_map = cast(dict[str, Any], value)
            if set(preset_map) - _PRESET_FIELDS:
                raise AppError("config.settings_invalid")
            try:
                display_name = preset_map.get("display_name", name)
                pipeline_profile = preset_map["pipeline_profile"]
                model_ref = preset_map["model_ref"]
                device = preset_map.get("device", "auto")
                compute_type = preset_map.get("compute_type", "default")
                source = preset_map.get("source_language", "")
                target = preset_map.get("target_language", "")
                provider_profile = preset_map.get("provider_profile", "default")
                ffmpeg_bin = preset_map.get("ffmpeg_bin", "ffmpeg")
                ffprobe_bin = preset_map.get("ffprobe_bin", "ffprobe")
            except KeyError as exc:
                raise AppError("config.settings_invalid") from exc
            # TOML table keys are strings; values must be exact runtime types.
            name_text = cast(object, name)
            if not isinstance(name_text, str) or not name_text.strip():
                raise AppError("config.settings_invalid")
            display_obj = cast(object, display_name)
            pipeline_obj = cast(object, pipeline_profile)
            model_obj = cast(object, model_ref)
            device_obj = cast(object, device)
            compute_obj = cast(object, compute_type)
            source_obj = cast(object, source)
            target_obj = cast(object, target)
            provider_obj = cast(object, provider_profile)
            ffmpeg_obj = cast(object, ffmpeg_bin)
            ffprobe_obj = cast(object, ffprobe_bin)
            if not isinstance(display_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(pipeline_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(model_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(device_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(compute_obj, str):
                raise AppError("config.settings_invalid")
            if source_obj not in ("", None) and not isinstance(source_obj, str):
                raise AppError("config.settings_invalid")
            if target_obj not in ("", None) and not isinstance(target_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(provider_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(ffmpeg_obj, str):
                raise AppError("config.settings_invalid")
            if not isinstance(ffprobe_obj, str):
                raise AppError("config.settings_invalid")
            if source_obj in ("", None):
                source_language: str | None = None
            else:
                assert isinstance(source_obj, str)
                source_language = source_obj
            if target_obj in ("", None):
                target_language: str | None = None
            else:
                assert isinstance(target_obj, str)
                target_language = target_obj
            try:
                user_presets.append(
                    ExecutionPreset(
                        name=name_text,
                        display_name=display_obj,
                        built_in=False,
                        pipeline_profile=PipelineProfile(pipeline_obj),
                        model_ref=model_obj,
                        device=device_obj,  # type: ignore[arg-type]
                        compute_type=compute_obj,
                        source_language=source_language,
                        target_language=target_language,
                        provider_profile=provider_obj,
                        ffmpeg_bin=ffmpeg_obj,
                        ffprobe_bin=ffprobe_obj,
                    )
                )
            except (AppError, TypeError, ValueError) as exc:
                raise AppError("config.settings_invalid") from exc
        ordered = tuple(
            sorted(
                user_presets,
                key=lambda item: (item.display_name.casefold(), item.name.casefold()),
            )
        )
        return settings, ordered

    def _load_provider_public(self, profile_name: str) -> ProviderPublicSettings:
        path = self.llm_path
        if not path.exists():
            return default_provider_public_settings()
        providers = self._read_raw_providers()
        profile = validate_identifier(profile_name, field="provider_profile")
        raw = providers.get(profile)
        if raw is None:
            return default_provider_public_settings()
        if set(raw) - PROVIDER_FIELDS:
            raise AppError("llm.config_invalid", {"reason": "provider_fields"})
        try:
            kind = raw.get("kind", OPENAI_COMPATIBLE_KIND)
            base_url_value = raw["base_url"]
            model_value = raw["model"]
            max_concurrency_raw = raw.get("max_concurrency", 4)
            request_timeout_raw = raw.get("request_timeout_sec", 120.0)
            max_retries_raw = raw.get("max_retries", 5)
            temperature_raw = raw.get("temperature", 0.1)
            tokenizer_value = raw.get("tokenizer", "cl100k_base")
        except KeyError as exc:
            raise AppError("llm.config_invalid", {"reason": "provider"}) from exc
        if not isinstance(kind, str):
            raise AppError("llm.config_invalid", {"field": "kind"})
        if not isinstance(base_url_value, str):
            raise AppError("llm.config_invalid", {"field": "base_url"})
        if not isinstance(model_value, str):
            raise AppError("llm.config_invalid", {"field": "model"})
        if type(max_concurrency_raw) is not int:
            raise AppError("llm.config_invalid", {"field": "max_concurrency"})
        if type(max_retries_raw) is not int:
            raise AppError("llm.config_invalid", {"field": "max_retries"})
        if isinstance(request_timeout_raw, bool) or not isinstance(
            request_timeout_raw, (int, float)
        ):
            raise AppError("llm.config_invalid", {"field": "request_timeout_sec"})
        if isinstance(temperature_raw, bool) or not isinstance(temperature_raw, (int, float)):
            raise AppError("llm.config_invalid", {"field": "temperature"})
        if not isinstance(tokenizer_value, str):
            raise AppError("llm.config_invalid", {"field": "tokenizer"})
        try:
            # ProviderPublicSettings enforces finite timeout/temperature parity.
            public = ProviderPublicSettings(
                profile_name=profile,
                base_url=normalize_base_url_identity(base_url_value),
                model=model_value,
                max_concurrency=max_concurrency_raw,
                request_timeout_sec=float(request_timeout_raw),
                max_retries=max_retries_raw,
                temperature=float(temperature_raw),
                tokenizer=tokenizer_value,  # type: ignore[arg-type]
                credential_source="missing",
            )
        except AppError as exc:
            if exc.code == "config.provider_invalid":
                raise AppError(
                    "llm.config_invalid",
                    {"field": str(exc.params.get("field", "provider"))},
                ) from exc
            raise AppError("llm.config_invalid", {"reason": "provider"}) from exc
        if kind != OPENAI_COMPATIBLE_KIND:
            raise AppError("llm.config_invalid", {"field": "kind"})
        config_key = raw.get("api_key")
        config_api_key = (
            config_key.strip() if isinstance(config_key, str) and config_key.strip() else None
        )
        source = self._credential_source(profile, config_api_key)
        return ProviderPublicSettings(
            profile_name=public.profile_name,
            base_url=public.base_url,
            model=public.model,
            max_concurrency=public.max_concurrency,
            request_timeout_sec=public.request_timeout_sec,
            max_retries=public.max_retries,
            temperature=public.temperature,
            tokenizer=public.tokenizer,
            credential_source=source,
        )

    def _credential_source(
        self,
        profile_name: str,
        config_api_key: str | None,
    ) -> CredentialSource:
        if config_api_key is not None:
            return "config"
        env_key = resolve_provider_credential(
            profile_name=profile_name,
            config_api_key=None,
            environment=self._env(),
        )
        if env_key is not None:
            return "environment"
        return "missing"

    def _read_raw_providers(self) -> dict[str, dict[str, object]]:
        path = self.llm_path
        if not path.exists():
            return {}
        try:
            loaded = cast(object, tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise AppError("llm.config_invalid", {"reason": "toml"}) from exc
        if not isinstance(loaded, dict):
            raise AppError("llm.config_invalid", {"reason": "root"})
        root = cast(dict[str, object], loaded)
        providers_value = root.get("providers")
        if providers_value is None:
            return {}
        if not isinstance(providers_value, dict):
            raise AppError("llm.config_invalid", {"field": "providers"})
        result: dict[str, dict[str, object]] = {}
        for name, value in cast(dict[str, object], providers_value).items():
            if not isinstance(value, dict):
                raise AppError("llm.config_invalid", {"reason": "provider"})
            result[str(name)] = {str(k): v for k, v in cast(dict[object, object], value).items()}
        return result


def load_startup_locale_from_settings(
    settings_path: Path,
) -> tuple[str, str | None]:
    """Return (locale, optional issue code) without rewriting files."""
    if not settings_path.exists():
        return "en", None
    try:
        loaded = cast(object, tomllib.loads(settings_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return "en", "config.settings_invalid"
    if not isinstance(loaded, dict):
        return "en", "config.settings_invalid"
    mapping = cast(dict[str, Any], loaded)
    if set(mapping) - _ROOT_SECTIONS:
        return "en", "config.settings_invalid"
    if mapping.get("schema_version") != SETTINGS_SCHEMA_VERSION:
        return "en", "config.settings_invalid"
    global_raw = mapping.get("global")
    if not isinstance(global_raw, dict):
        return "en", "config.settings_invalid"
    locale = cast(dict[str, Any], global_raw).get("locale", "en")
    if locale not in {"en", "zh-CN"}:
        return "en", "config.settings_invalid"
    return str(locale), None


# Re-export load_provider_config for adapter-side tests that need full runtime.
__all__ = [
    "SETTINGS_FILENAME",
    "TomlConfigurationStore",
    "escape_toml_string",
    "load_startup_locale_from_settings",
    "normalize_profile_env_suffix",
    "serialize_llm_providers",
    "serialize_settings_toml",
]
