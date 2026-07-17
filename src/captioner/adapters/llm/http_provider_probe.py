"""Synchronous HTTP provider connectivity probe using httpx."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

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
                # Drop body immediately; never surface it.
                _ = response.content
            except httpx.TimeoutException as exc:
                raise AppError("llm.connection_timeout", retryable=True) from exc
            except httpx.HTTPError as exc:
                raise AppError("llm.connection_failed", retryable=True) from exc
        finally:
            client.close()
        if 200 <= status < 300:
            return ProviderConnectionResult(ok=True, code="llm.connection_ok")
        if status in {401, 403}:
            raise AppError("llm.connection_auth_failed")
        raise AppError("llm.connection_rejected", {"status": status})


__all__ = ["HTTPProviderProbe", "models_url"]
