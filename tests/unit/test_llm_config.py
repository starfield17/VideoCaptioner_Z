from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from captioner.core.domain.errors import AppError
from captioner.infrastructure.config import (
    ProviderCredential,
    load_provider_config,
    normalize_base_url_identity,
    write_llm_config,
)

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
