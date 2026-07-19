from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from captioner.core.domain.errors import AppError
from captioner.infrastructure.model_source_config import (
    ModelSourceConfig,
    ModelSourceCredential,
    load_all_model_source_configs,
    load_model_source_config,
)


def test_source_config_uses_environment_fallback_without_creating_file(tmp_path: Path) -> None:
    config = load_model_source_config(
        tmp_path,
        "huggingface",
        environment={"CAPTIONER_HF_TOKEN": "  env-token  "},
    )

    assert config.endpoint == "https://huggingface.co"
    assert config.token == "env-token"
    assert not (tmp_path / "model-sources.toml").exists()
    assert "env-token" not in repr(config)


def test_file_token_has_priority_over_environment(tmp_path: Path) -> None:
    (tmp_path / "model-sources.toml").write_text(
        "[sources.huggingface]\n"
        "enabled = true\n"
        "endpoint = 'https://mirror.example.test/'\n"
        "token = 'file-token'\n"
        "max_workers = 3\n",
        encoding="utf-8",
    )

    config = load_model_source_config(
        tmp_path,
        "huggingface",
        environment={"CAPTIONER_HF_TOKEN": "env-token"},
    )

    assert config.token == "file-token"
    assert config.endpoint == "https://mirror.example.test"
    assert config.max_workers == 3


def test_all_source_defaults_are_available_without_a_config_file(tmp_path: Path) -> None:
    configs = load_all_model_source_configs(tmp_path, environment={})

    assert set(configs) == {"huggingface", "modelscope"}
    assert all(config.token is None for config in configs.values())


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.test",
        "https://user:password@example.test",
        "https://example.test/?token=secret",
        "https://example.test/#fragment",
    ],
)
def test_source_config_rejects_unsafe_endpoint(endpoint: str) -> None:
    with pytest.raises(AppError, match=r"model\.source_endpoint_invalid"):
        ModelSourceConfig("huggingface", endpoint=endpoint)


def test_source_config_rejects_unknown_fields_and_non_string_token(tmp_path: Path) -> None:
    (tmp_path / "model-sources.toml").write_text(
        "[sources.huggingface]\ntoken = 42\n",
        encoding="utf-8",
    )
    with pytest.raises(AppError, match=r"model\.source_config_invalid"):
        load_model_source_config(tmp_path, "huggingface", environment={})

    with pytest.raises(AppError, match=r"model\.source_config_invalid"):
        ModelSourceConfig("huggingface", credential=cast(ModelSourceCredential, object()))


def test_credential_wrapper_redacts_repr() -> None:
    credential = ModelSourceCredential("secret")
    assert repr(credential) == "ModelSourceCredential(<redacted>)"
    assert "secret" not in repr(credential)
