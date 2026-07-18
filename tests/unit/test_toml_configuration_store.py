"""Unit tests for TomlConfigurationStore."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from captioner.adapters.persistence.toml_configuration_store import (
    TomlConfigurationStore,
    serialize_settings_toml,
)
from captioner.core.application.configuration import (
    ExecutionPreset,
    GlobalSettings,
    ProviderSettingsUpdate,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile
from captioner.infrastructure.config import write_llm_config


def test_missing_files_use_defaults(tmp_path: Path) -> None:
    store = TomlConfigurationStore(tmp_path)
    snapshot = store.load_snapshot()
    assert snapshot.global_settings.locale == "en"
    assert [p.name for p in snapshot.presets[:3]] == ["deterministic", "fast", "quality"]
    assert snapshot.provider.credential_source == "missing"
    assert snapshot.issues == ()


def test_valid_settings_and_unknown_fields(tmp_path: Path) -> None:
    settings = GlobalSettings(locale="zh-CN", recursive_input=False)
    (tmp_path / "settings.toml").write_text(
        serialize_settings_toml(settings, ()),
        encoding="utf-8",
    )
    store = TomlConfigurationStore(tmp_path)
    snapshot = store.load_snapshot()
    assert snapshot.global_settings.locale == "zh-CN"
    assert snapshot.global_settings.recursive_input is False

    bad = tmp_path / "settings.toml"
    bad.write_text(
        'schema_version = 1\n[global]\nlocale = "en"\nunknown = 1\n',
        encoding="utf-8",
    )
    snapshot = store.load_snapshot()
    assert any(issue.code == "config.settings_invalid" for issue in snapshot.issues)
    assert bad.read_text(encoding="utf-8").startswith("schema_version")


def test_unsupported_schema_and_invalid_toml_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 99\n[global]\nlocale = "en"\n', encoding="utf-8")
    original = path.read_text(encoding="utf-8")
    store = TomlConfigurationStore(tmp_path)
    snapshot = store.load_snapshot()
    assert any(issue.code == "config.settings_invalid" for issue in snapshot.issues)
    assert path.read_text(encoding="utf-8") == original

    path.write_text("not = [valid", encoding="utf-8")
    original = path.read_text(encoding="utf-8")
    snapshot = store.load_snapshot()
    assert any(issue.code == "config.settings_invalid" for issue in snapshot.issues)
    assert path.read_text(encoding="utf-8") == original


def test_atomic_save_and_posix_mode(tmp_path: Path) -> None:
    store = TomlConfigurationStore(tmp_path)
    store.save_global(GlobalSettings(locale="zh-CN"))
    path = tmp_path / "settings.toml"
    assert path.exists()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text(encoding="utf-8")
    assert "deterministic" not in text or "[presets.deterministic]" not in text
    assert 'locale = "zh-CN"' in text


def test_user_preset_round_trip(tmp_path: Path) -> None:
    store = TomlConfigurationStore(tmp_path)
    preset = ExecutionPreset(
        name="studio mix",
        display_name="Studio Mix",
        built_in=False,
        pipeline_profile=PipelineProfile.QUALITY,
        model_ref="small",
        device="cuda",
        compute_type="float16",
        source_language=None,
        target_language="en",
        provider_profile="default",
    )
    store.save_user_preset(preset)
    snapshot = store.load_snapshot()
    loaded = next(item for item in snapshot.presets if item.name == "studio mix")
    assert loaded.display_name == "Studio Mix"
    assert loaded.pipeline_profile is PipelineProfile.QUALITY
    store.delete_user_preset("studio mix")
    snapshot = store.load_snapshot()
    assert all(item.name != "studio mix" for item in snapshot.presets)


def test_provider_save_preserves_other_profiles_and_blank_key(tmp_path: Path) -> None:
    write_llm_config(
        tmp_path / "llm.toml",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "keep-me"
model = "old"

[providers.other]
kind = "openai-compatible"
base_url = "https://other.example/v1"
api_key = "other-key"
model = "other-model"
""",
    )
    store = TomlConfigurationStore(tmp_path)
    store.save_provider(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="new-model",
            api_key=None,
        )
    )
    text = (tmp_path / "llm.toml").read_text(encoding="utf-8")
    assert "keep-me" in text
    assert "other-key" in text
    assert "new-model" in text
    snapshot = store.load_snapshot()
    assert snapshot.provider.model == "new-model"
    assert snapshot.provider.credential_source == "config"
    assert "keep-me" not in repr(snapshot)


def test_credential_precedence(tmp_path: Path) -> None:
    write_llm_config(
        tmp_path / "llm.toml",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "config-key"
model = "m"
""",
    )
    store = TomlConfigurationStore(
        tmp_path,
        environment={
            "CAPTIONER_LLM_API_KEY_DEFAULT": "env-profile",
            "CAPTIONER_LLM_API_KEY": "env-global",
            "OPENAI_API_KEY": "openai",
        },
    )
    snapshot = store.load_snapshot()
    assert snapshot.provider.credential_source == "config"
    probe = store.resolve_provider_for_test(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
            api_key=None,
            tokenizer="o200k_base",
        )
    )
    assert probe.api_key == "config-key"
    assert probe.model == "m"
    assert probe.tokenizer == "o200k_base"


def test_environment_fallbacks(tmp_path: Path) -> None:
    write_llm_config(
        tmp_path / "llm.toml",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
model = "m"
""",
    )
    store = TomlConfigurationStore(
        tmp_path,
        environment={"CAPTIONER_LLM_API_KEY_DEFAULT": "profile-env"},
    )
    assert store.load_snapshot().provider.credential_source == "environment"
    store = TomlConfigurationStore(
        tmp_path,
        environment={"CAPTIONER_LLM_API_KEY": "global-env"},
    )
    assert store.load_snapshot().provider.credential_source == "environment"
    store = TomlConfigurationStore(
        tmp_path,
        environment={"OPENAI_API_KEY": "openai-env"},
    )
    assert store.load_snapshot().provider.credential_source == "environment"
    store = TomlConfigurationStore(tmp_path, environment={})
    assert store.load_snapshot().provider.credential_source == "missing"
    with pytest.raises(AppError, match=r"llm\.config_invalid"):
        store.resolve_provider_for_test(
            ProviderSettingsUpdate(
                profile_name="default",
                base_url="https://example.com/v1",
                model="m",
                api_key=None,
            )
        )


def test_explicit_api_key_save(tmp_path: Path) -> None:
    store = TomlConfigurationStore(tmp_path)
    store.save_provider(
        ProviderSettingsUpdate(
            profile_name="default",
            base_url="https://example.com/v1",
            model="m",
            api_key="written-key",
        )
    )
    text = (tmp_path / "llm.toml").read_text(encoding="utf-8")
    assert "written-key" in text
    snapshot = store.load_snapshot()
    assert "written-key" not in repr(snapshot)
    assert snapshot.provider.credential_source == "config"


@pytest.mark.parametrize(
    "content",
    [
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "k"
model = "m"
request_timeout_sec = -1
""",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "k"
model = "m"
request_timeout_sec = nan
""",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "k"
model = "m"
temperature = -5
""",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "k"
model = "m"
temperature = inf
""",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "k"
model = 123
""",
    ],
)
def test_invalid_provider_values_yield_issue_and_safe_defaults(
    tmp_path: Path, content: str
) -> None:
    (tmp_path / "llm.toml").write_text(content, encoding="utf-8")
    original = (tmp_path / "llm.toml").read_text(encoding="utf-8")
    store = TomlConfigurationStore(tmp_path)
    snapshot = store.load_snapshot()
    assert any(issue.code.startswith("llm.") for issue in snapshot.issues)
    assert snapshot.provider.model == "gpt-4o-mini"  # safe default
    assert (tmp_path / "llm.toml").read_text(encoding="utf-8") == original


def test_numeric_preset_fields_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "settings.toml").write_text(
        """
schema_version = 1
[global]
locale = "en"
default_output_root = ""
recursive_input = true
default_preset_name = "deterministic"
collision_policy = "unique_subdir"

[presets.bad]
display_name = "Bad"
pipeline_profile = "fast"
model_ref = 123
device = "auto"
compute_type = 456
source_language = ""
target_language = "zh-CN"
provider_profile = "default"
ffmpeg_bin = "ffmpeg"
ffprobe_bin = "ffprobe"
""",
        encoding="utf-8",
    )
    original = (tmp_path / "settings.toml").read_text(encoding="utf-8")
    store = TomlConfigurationStore(tmp_path)
    snapshot = store.load_snapshot()
    assert any(issue.code == "config.settings_invalid" for issue in snapshot.issues)
    assert all(preset.built_in for preset in snapshot.presets)
    assert (tmp_path / "settings.toml").read_text(encoding="utf-8") == original
