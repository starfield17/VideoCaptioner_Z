from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from tests.support import llm_snapshot

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm_job_config import (
    LLMJobSnapshot,
    PromptSnapshot,
    ProviderPublicSnapshot,
    required_prompts_for,
)
from captioner.core.domain.result import JsonValue, thaw_json_value
from captioner.core.domain.stage import PipelineProfile
from captioner.infrastructure.config import (
    ProviderCredential,
    config_file_creation_mode,
    load_provider_config,
    normalize_base_url_identity,
    should_chmod_private,
    write_llm_config,
)
from captioner.infrastructure.prompts import PromptIdentity

_TOML = """
[providers.default]
kind = "openai-compatible"
base_url = "HTTPS://Example.COM:443/v1/"
api_key = "unit-test-key"
model = "unit-model"
max_concurrency = 3
request_timeout_sec = 12
max_retries = 2
temperature = 0.2
"""


def test_provider_config_is_loaded_and_snapshot_is_redacted(tmp_path: Path) -> None:
    write_llm_config(tmp_path / "llm.toml", _TOML)
    provider = load_provider_config(tmp_path)

    assert provider.base_url == "https://example.com/v1"
    assert provider.normalized_base_url_identity == provider.base_url
    assert provider.api_key == "unit-test-key"
    assert provider.max_concurrency == 3
    assert provider.request_timeout_sec == 12.0
    snapshot = provider.to_snapshot()
    assert "api_key" not in snapshot
    assert "unit-test-key" not in repr(provider)
    assert "unit-test-key" not in repr(provider.credential)

    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "llm.toml").stat().st_mode) == 0o600


def test_config_uses_file_path_and_rejects_invalid_profiles(tmp_path: Path) -> None:
    path = tmp_path / "provider.toml"
    write_llm_config(path, _TOML)
    assert load_provider_config(path, "default").model == "unit-model"
    with pytest.raises(AppError, match=r"llm\.provider_not_found"):
        load_provider_config(path, "missing")
    with pytest.raises(AppError, match=r"llm\.config_missing"):
        load_provider_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ftp://example.com/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?token=secret",
        "https://example.com:bad/v1",
    ],
)
def test_base_url_validation_is_provider_neutral(value: str) -> None:
    with pytest.raises(AppError, match=r"llm\.config_invalid"):
        normalize_base_url_identity(value)


def test_credential_rejects_empty_value() -> None:
    with pytest.raises(AppError, match=r"llm\.config_invalid"):
        ProviderCredential(" ")


@pytest.mark.parametrize(
    "content",
    [
        "not = [valid",
        "[other]\nvalue = 1",
        '[providers.default]\nkind = "openai-compatible"\nunknown = 1',
        (
            '[providers.default]\nkind = "openai-compatible"\n'
            'base_url = "https://example.com/v1"\nmodel = "model"'
        ),
    ],
)
def test_provider_file_rejects_malformed_or_incomplete_profiles(
    tmp_path: Path, content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "llm.toml"
    write_llm_config(path, content)
    monkeypatch.delenv("CAPTIONER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("CAPTIONER_LLM_API_KEY_DEFAULT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AppError, match=r"llm\.(config_invalid|provider_not_found)"):
        load_provider_config(path, environment={})


def test_provider_config_accepts_environment_fallback(
    tmp_path: Path,
) -> None:
    write_llm_config(
        tmp_path / "llm.toml",
        (
            '[providers.default]\nkind = "openai-compatible"\n'
            'base_url = "https://example.com/v1"\nmodel = "model"\n'
        ),
    )
    provider = load_provider_config(
        tmp_path,
        environment={"CAPTIONER_LLM_API_KEY": "from-env"},
    )
    assert provider.api_key == "from-env"
    assert "from-env" not in repr(provider)


def test_config_api_key_wins_over_environment(tmp_path: Path) -> None:
    write_llm_config(tmp_path / "llm.toml", _TOML)
    provider = load_provider_config(
        tmp_path,
        environment={
            "CAPTIONER_LLM_API_KEY": "env-key",
            "OPENAI_API_KEY": "openai-key",
        },
    )
    assert provider.api_key == "unit-test-key"


def test_empty_config_write_is_rejected_and_url_identity_is_normalized() -> None:
    with pytest.raises(AppError, match=r"llm\.config_invalid"):
        write_llm_config(Path("/tmp/unused-llm.toml"), " ")
    assert normalize_base_url_identity("http://[2001:DB8::1]:80/v1/") == ("http://[2001:db8::1]/v1")


def test_config_file_platform_seams_cover_posix_and_windows() -> None:
    assert config_file_creation_mode("posix") == 0o600
    assert should_chmod_private("posix") is True
    assert config_file_creation_mode("nt") == 0o666
    assert should_chmod_private("nt") is False


def test_public_snapshot_diff_is_field_exact_and_credential_free() -> None:
    snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(llm_snapshot(PipelineProfile.QUALITY)))
    changed = replace(snapshot.provider, temperature=0.9)
    assert snapshot.provider.changed_fields(changed) == ("temperature",)
    assert set(snapshot.provider.to_mapping()) == {
        "kind",
        "provider_profile",
        "base_url",
        "model",
        "max_concurrency",
        "request_timeout_sec",
        "max_retries",
        "temperature",
        "tokenizer",
    }
    assert "api_key" not in repr(snapshot)


def test_required_prompt_sets_are_profile_exact() -> None:
    assert required_prompts_for(PipelineProfile.DETERMINISTIC) == ()
    assert required_prompts_for(PipelineProfile.FAST) == (
        "translate_fast",
        "repair_structured",
    )
    assert required_prompts_for(PipelineProfile.QUALITY) == (
        "terminology",
        "correct_source",
        "translate_quality",
        "review_anomalies",
        "repair_structured",
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "extra_field",
        "profile",
        "target_language",
        "response_schema",
        "chunk",
        "prompt_hash",
    ],
)
def test_snapshot_validator_rejects_semantically_incomplete_values(invalid_case: str) -> None:
    raw_value = thaw_json_value(llm_snapshot(PipelineProfile.FAST))
    raw = cast(dict[str, JsonValue], raw_value)
    if invalid_case == "extra_field":
        raw["flat_prompt_id"] = "translate_fast"
    elif invalid_case == "profile":
        raw["profile"] = "unknown"
    elif invalid_case == "target_language":
        raw["target_language"] = "zh CN"
    elif invalid_case == "response_schema":
        raw["response_schema_version"] = 0
    elif invalid_case == "chunk":
        raw["chunk"] = {"max_items": 1}
    else:
        prompts = cast(dict[str, JsonValue], raw["prompts"])
        translate = cast(dict[str, JsonValue], prompts["translate_fast"])
        translate["content_sha256"] = "not-a-sha"
    changed: object = raw
    with pytest.raises(AppError):
        LLMJobSnapshot.from_mapping(changed)


@pytest.mark.parametrize("value", ["../v1", "/v1", "prompt/name", ""])
def test_prompt_snapshot_rejects_path_like_identity(value: str) -> None:
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptSnapshot(value, "v1", "a" * 64)


def test_provider_snapshot_rejects_invalid_numeric_public_fields() -> None:
    base = {
        "kind": "openai-compatible",
        "provider_profile": "default",
        "base_url": "https://provider.example/v1",
        "model": "unit-model",
        "max_concurrency": 2,
        "request_timeout_sec": 30.0,
        "max_retries": 1,
        "temperature": 0.1,
        "tokenizer": "cl100k_base",
    }
    for field, value in (
        ("max_concurrency", 0),
        ("request_timeout_sec", 0),
        ("request_timeout_sec", float("nan")),
        ("temperature", -1),
        ("max_retries", -1),
    ):
        with pytest.raises(AppError):
            ProviderPublicSnapshot.from_mapping({**base, field: value})


@pytest.mark.parametrize(
    "field",
    ["kind", "provider_profile", "base_url", "model"],
)
def test_provider_snapshot_requires_each_public_string(field: str) -> None:
    base = {
        "kind": "openai-compatible",
        "provider_profile": "default",
        "base_url": "https://provider.example/v1",
        "model": "unit-model",
        "max_concurrency": 2,
        "request_timeout_sec": 30.0,
        "max_retries": 1,
        "temperature": 0.1,
        "tokenizer": "cl100k_base",
    }
    with pytest.raises(AppError, match=r"llm\.snapshot_invalid"):
        ProviderPublicSnapshot.from_mapping({**base, field: None})


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {"prompt_id": "id"},
        {"prompt_id": "id", "prompt_version": "v1", "content_sha256": "bad"},
    ],
)
def test_prompt_snapshot_mapping_is_strict(invalid: object) -> None:
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptSnapshot.from_mapping(invalid)


def test_prompt_identity_rejects_non_string_and_path_like_values() -> None:
    content = "Return a JSON object."
    digest = hashlib.sha256(content.encode()).hexdigest()
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptIdentity("prompt", "v1", cast(str, None), content)
    with pytest.raises(AppError, match=r"prompt\.invalid"):
        PromptIdentity("prompt", "v1", digest, cast(str, None))
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptIdentity(cast(str, None), "v1", digest, content)
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptIdentity("../prompt", "v1", digest, content)
    with pytest.raises(AppError, match=r"prompt\.identity_invalid"):
        PromptIdentity("prompt", "../v1", digest, content)
