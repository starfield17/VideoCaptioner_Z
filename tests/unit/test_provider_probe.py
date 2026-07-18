"""Unit tests for HTTP provider probe without importing network modules in tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from captioner.adapters.llm.http_provider_probe import HTTPProviderProbe, models_url
from captioner.core.domain.errors import AppError
from captioner.core.ports.configuration_store import ProviderRuntimeProbeSettings


def _settings(
    *,
    base_url: str = "https://api.example.com/v1",
    api_key: str = "probe-secret",
    timeout_sec: float = 5.0,
    model: str = "gpt-4o-mini",
    tokenizer: str = "cl100k_base",
) -> ProviderRuntimeProbeSettings:
    return ProviderRuntimeProbeSettings(
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
        model=model,
        tokenizer=tokenizer,
    )


def _client_with_response(status: int, body: bytes = b"{}") -> MagicMock:
    response = SimpleNamespace(status_code=status, content=body)
    client = MagicMock()
    client.get.return_value = response
    client.close = MagicMock()
    return client


def test_models_url_normalization() -> None:
    assert models_url("https://api.example.com/v1/") == "https://api.example.com/v1/models"
    assert models_url("https://api.example.com/v1/models") == "https://api.example.com/v1/models"


def test_success_sends_auth_and_models_path() -> None:
    client = _client_with_response(200, b'{"data":[]}')
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ) as client_cls:
        probe = HTTPProviderProbe()
        result = probe.test(_settings())
        assert result.ok is True
        assert result.code == "llm.connection_ok"
        assert result.model_listing_supported is True
        assert result.available_models == ()
        assert result.configured_model_found is False
        assert result.resolved_tokenizer == "cl100k_base"
        assert result.tokenizer_valid is True
        kwargs = client_cls.call_args.kwargs
        assert kwargs["follow_redirects"] is False
        call = client.get.call_args
        assert call.args[0] == "https://api.example.com/v1/models"
        assert call.kwargs["headers"]["Authorization"] == "Bearer probe-secret"
        assert call.kwargs["headers"]["Accept"] == "application/json"
        client.close.assert_called_once()
        assert "probe-secret" not in repr(result)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures(status: int) -> None:
    client = _client_with_response(status, b"secret body must not leak")
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        probe = HTTPProviderProbe()
        with pytest.raises(AppError, match=r"llm\.connection_auth_failed") as exc_info:
            probe.test(_settings())
    assert "secret body" not in repr(exc_info.value)
    assert "probe-secret" not in repr(exc_info.value)
    client.close.assert_called_once()


@pytest.mark.parametrize("status", [429, 500])
def test_rejected_status(status: int) -> None:
    client = _client_with_response(status, b"nope")
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        probe = HTTPProviderProbe()
        with pytest.raises(AppError, match=r"llm\.connection_rejected") as exc_info:
            probe.test(_settings())
    assert exc_info.value.params.get("status") == status
    assert "nope" not in repr(exc_info.value)


def test_timeout() -> None:
    import captioner.adapters.llm.http_provider_probe as module

    client = MagicMock()
    client.get.side_effect = module.httpx.TimeoutException("timeout")
    client.close = MagicMock()
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        probe = HTTPProviderProbe()
        with pytest.raises(AppError, match=r"llm\.connection_timeout") as exc_info:
            probe.test(_settings())
    assert exc_info.value.retryable is True
    client.close.assert_called_once()


def test_transport_failure() -> None:
    import captioner.adapters.llm.http_provider_probe as module

    client = MagicMock()
    client.get.side_effect = module.httpx.ConnectError("boom")
    client.close = MagicMock()
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        probe = HTTPProviderProbe()
        with pytest.raises(AppError, match=r"llm\.connection_failed") as exc_info:
            probe.test(_settings())
    assert exc_info.value.retryable is True
    assert "boom" not in repr(exc_info.value)
    client.close.assert_called_once()


def test_settings_repr_redacted() -> None:
    settings = _settings()
    assert "probe-secret" not in repr(settings)


def test_models_success_returns_sorted_deduplicated_ids_and_keeps_manual_model() -> None:
    client = _client_with_response(
        200,
        b'{"data":[{"id":"z-model"},{"id":"a-model"},{"id":"z-model"}]}',
    )
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        result = HTTPProviderProbe().test(_settings(model="manual-model"))
    assert result.ok is True
    assert result.available_models == ("a-model", "z-model")
    assert result.configured_model_found is False


def test_models_endpoint_unsupported_does_not_fail_valid_provider() -> None:
    client = _client_with_response(404, b'{"error":"not supported"}')
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        result = HTTPProviderProbe().test(_settings(model="manual-model"))
    assert result.ok is True
    assert result.model_listing_supported is False
    assert result.configured_model_found is None
    assert result.tokenizer_valid is True


def test_explicit_unsupported_models_response_does_not_fail_valid_provider() -> None:
    client = _client_with_response(400, b'{"error":"endpoint unsupported"}')
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        result = HTTPProviderProbe().test(_settings(model="manual-model"))
    assert result.ok is True
    assert result.model_listing_supported is False


def test_auto_tokenizer_unknown_model_returns_actionable_structured_code() -> None:
    client = _client_with_response(404)
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        result = HTTPProviderProbe().test(
            _settings(model="provider-specific-model", tokenizer="auto")
        )
    assert result.ok is False
    assert result.code == "llm.tokenizer_unknown"
    assert result.tokenizer_valid is False
    assert result.resolved_tokenizer is None


def test_auto_tokenizer_resolves_known_model() -> None:
    client = _client_with_response(404)
    with patch(
        "captioner.adapters.llm.http_provider_probe.httpx.Client",
        return_value=client,
    ):
        result = HTTPProviderProbe().test(_settings(model="gpt-4o-mini", tokenizer="auto"))
    assert result.ok is True
    assert result.resolved_tokenizer == "o200k_base"
    assert result.tokenizer_valid is True
