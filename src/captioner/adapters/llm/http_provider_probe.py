"""Synchronous HTTP provider connectivity probe using httpx."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import httpx

from captioner.adapters.llm.token_counter import ModelTokenCounter, resolve_tokenizer_id
from captioner.core.application.configuration import ProviderConnectionResult
from captioner.core.domain.errors import AppError
from captioner.core.ports.configuration_store import ProviderRuntimeProbeSettings
from captioner.infrastructure.config import normalize_base_url_identity


def models_url(base_url: str) -> str:
    normalized = normalize_base_url_identity(base_url)
    if normalized.endswith("/models"):
        return normalized
    return f"{normalized}/models"


@dataclass(frozen=True, slots=True)
class HTTPProviderProbe:
    """Probe ``GET /models`` without returning response bodies or credentials."""

    transport: httpx.BaseTransport | None = None

    def test(self, settings: ProviderRuntimeProbeSettings) -> ProviderConnectionResult:
        url = models_url(settings.base_url)
        timeout = httpx.Timeout(settings.timeout_sec)
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Accept": "application/json",
        }
        client = httpx.Client(
            transport=self.transport,
            timeout=timeout,
            follow_redirects=False,
        )
        try:
            try:
                response = client.get(url, headers=headers)
                status = response.status_code
                body = response.content
            except httpx.TimeoutException as exc:
                raise AppError("llm.connection_timeout", retryable=True) from exc
            except httpx.HTTPError as exc:
                raise AppError("llm.connection_failed", retryable=True) from exc
        finally:
            client.close()
        if status in {401, 403}:
            raise AppError("llm.connection_auth_failed")
        if _models_endpoint_unsupported(status, body):
            model_listing_supported = False
            available_models: tuple[str, ...] = ()
        elif 200 <= status < 300:
            model_listing_supported = True
            available_models = _model_ids_from_body(body)
        else:
            raise AppError("llm.connection_rejected", {"status": status})

        configured_model_found = (
            settings.model in available_models if model_listing_supported else None
        )
        resolved_tokenizer, tokenizer_valid, tokenizer_code = _check_tokenizer(settings)
        return ProviderConnectionResult(
            ok=tokenizer_valid,
            code=tokenizer_code,
            model_listing_supported=model_listing_supported,
            available_models=available_models,
            configured_model_found=configured_model_found,
            resolved_tokenizer=resolved_tokenizer,
            tokenizer_valid=tokenizer_valid,
        )


def _model_ids_from_body(body: bytes) -> tuple[str, ...]:
    """Read only model IDs from a successful OpenAI-compatible response."""
    try:
        decoded: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(decoded, dict):
        return ()
    decoded_mapping = cast(dict[object, object], decoded)
    data = decoded_mapping.get("data")
    if not isinstance(data, list):
        return ()
    model_entries = cast(list[object], data)
    model_ids: set[str] = set()
    for item in model_entries:
        if not isinstance(item, dict):
            continue
        item_mapping = cast(dict[object, object], item)
        model_id = item_mapping.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.add(model_id.strip())
    return tuple(sorted(model_ids))


def _models_endpoint_unsupported(status: int, body: bytes) -> bool:
    """Recognize explicit OpenAI-compatible providers without ``/models``."""
    if status in {404, 405, 501}:
        return True
    if status not in {400, 422}:
        return False
    try:
        message = body.decode("utf-8").casefold()
    except UnicodeDecodeError:
        return False
    return "not supported" in message or "unsupported" in message


def _check_tokenizer(
    settings: ProviderRuntimeProbeSettings,
) -> tuple[str | None, bool, str]:
    try:
        resolved = resolve_tokenizer_id(settings.tokenizer, settings.model)
        count = ModelTokenCounter(resolved).count("Captioner tokenizer smoke")
        if count <= 0:
            return None, False, "llm.tokenizer_unknown"
    except AppError as exc:
        return None, False, exc.code
    return resolved, True, "llm.connection_ok"


__all__ = ["HTTPProviderProbe", "models_url"]
