"""Async OpenAI-compatible structured-generation adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

from captioner.adapters.llm.http_transport import (
    HTTPTimeout,
    HTTPTransport,
    HTTPTransportError,
    HttpxTransport,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRequest, response_schema_for
from captioner.core.domain.result import JsonValue
from captioner.core.ports.llm import LLMClient
from captioner.infrastructure.config import OpenAICompatibleProvider

T = TypeVar("T")
_DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class _ResponseFactory(Protocol):
    @classmethod
    def from_json(cls, value: str | bytes) -> object: ...

    @classmethod
    def from_mapping(cls, value: object) -> object: ...


class OpenAICompatibleClient(LLMClient):
    """Provider adapter; retries and repair remain in the application layer."""

    def __init__(
        self,
        provider: OpenAICompatibleProvider,
        *,
        transport: HTTPTransport | None = None,
        semaphore: asyncio.Semaphore | None = None,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError
        self._provider = provider
        self._semaphore = (
            asyncio.Semaphore(provider.max_concurrency) if semaphore is None else semaphore
        )
        self._transport = (
            HttpxTransport(
                timeout=HTTPTimeout.all(provider.request_timeout_sec),
                max_response_bytes=max_response_bytes,
            )
            if transport is None
            else transport
        )
        self._max_response_bytes = max_response_bytes

    @property
    def provider(self) -> OpenAICompatibleProvider:
        return self._provider

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def transport(self) -> HTTPTransport:
        return self._transport

    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        context.raise_if_cancelled()
        schema = response_schema_for(cast(type[object], response_schema))
        body = _encode_request(request, self._provider.model, self._provider.temperature, schema)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._provider.api_key}",
        }
        await _acquire_semaphore(self._semaphore, context)
        try:
            context.raise_if_cancelled()
            try:
                response = await self._transport.request(
                    "POST",
                    f"{self._provider.base_url}/chat/completions",
                    headers,
                    body,
                    HTTPTimeout.all(self._provider.request_timeout_sec),
                    self._max_response_bytes,
                )
            except HTTPTransportError as exc:
                raise _transport_app_error(exc) from exc
            except TimeoutError as exc:
                raise AppError("llm.timeout", retryable=True) from exc
            except OSError as exc:
                raise AppError("llm.network_error", retryable=True) from exc
        finally:
            self._semaphore.release()
        if len(response.body) > self._max_response_bytes:
            raise AppError("llm.response_too_large")
        return _decode_response(response.status_code, response.body, request, response_schema)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleClient("
            f"provider={self._provider!r}, max_response_bytes={self._max_response_bytes!r})"
        )


OpenAICompatibleLLMClient = OpenAICompatibleClient
OpenAICompatibleAdapter = OpenAICompatibleClient


def _encode_request(
    request: LLMRequest,
    model: str,
    temperature: float,
    response_schema: Mapping[str, JsonValue],
) -> bytes:
    payload: dict[str, JsonValue] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": request.prompt_content or "Return only the requested JSON object.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    request.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.task_kind.replace("-", "_"),
                "strict": True,
                "schema": dict(response_schema),
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _decode_response[T](
    status_code: int,
    body: bytes,
    request: LLMRequest,
    response_schema: type[T],
) -> T:
    error = _http_status_error(status_code)
    if error is not None:
        raise error
    parsed = _strict_json(body, "envelope")
    if not isinstance(parsed, Mapping):
        raise AppError("llm.schema_invalid", {"reason": "envelope_object"})
    raw_choices = cast(Mapping[object, object], parsed).get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise AppError("llm.schema_invalid", {"reason": "choices"})
    choices = cast(list[object], raw_choices)
    first = choices[0]
    if not isinstance(first, Mapping):
        raise AppError("llm.schema_invalid", {"reason": "choice"})
    message = cast(Mapping[object, object], first).get("message")
    if not isinstance(message, Mapping):
        raise AppError("llm.schema_invalid", {"reason": "message"})
    content = cast(Mapping[object, object], message).get("content")
    if not isinstance(content, str):
        raise AppError("llm.schema_invalid", {"reason": "content"})
    factory = cast(type[_ResponseFactory], response_schema)
    try:
        result = factory.from_json(content)
    except AppError as exc:
        raise AppError("llm.schema_invalid", {"reason": "structured_content"}) from exc
    except Exception as exc:
        raise AppError("llm.schema_invalid", {"reason": "structured_content"}) from exc
    if not isinstance(result, response_schema):
        raise AppError("llm.schema_invalid", {"reason": "response_type"})
    _validate_single_response_id(result, request)
    return result


def _strict_json(value: bytes | str, reason: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AppError("llm.schema_invalid", {"reason": reason}) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_value:{value}")


def _validate_single_response_id(result: object, request: LLMRequest) -> None:
    if len(request.items) != 1:
        return
    response_id = getattr(result, "id", None)
    if response_id == request.items[0].id:
        return
    if response_id in request.context_ids:
        raise AppError("llm.context_id_returned", {"expected": request.items[0].id})
    raise AppError("llm.id_mismatch", {"expected": request.items[0].id})


def _transport_app_error(error: HTTPTransportError) -> AppError:
    if error.kind == "timeout":
        return AppError("llm.timeout", retryable=True)
    if error.kind == "network":
        return AppError("llm.network_error", retryable=True)
    if error.kind == "response_too_large":
        return AppError("llm.response_too_large")
    return AppError("llm.network_error", retryable=True)


async def _acquire_semaphore(semaphore: asyncio.Semaphore, context: ExecutionContext) -> None:
    """Poll the cooperative token while waiting for a shared concurrency slot."""
    while True:
        context.raise_if_cancelled()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
        except TimeoutError:
            continue
        else:
            return


def _http_status_error(status_code: int) -> AppError | None:
    if 200 <= status_code < 300:
        return None
    if status_code == 429:
        return AppError("llm.rate_limited", {"status": status_code}, retryable=True)
    if status_code in {502, 503, 504}:
        return AppError("llm.upstream_unavailable", {"status": status_code}, retryable=True)
    if status_code in {401, 403}:
        return AppError("llm.auth_failed", {"status": status_code})
    if status_code == 400:
        return AppError("llm.request_rejected", {"status": status_code})
    return AppError("llm.http_error", {"status": status_code})
