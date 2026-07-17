from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

import pytest

from captioner.adapters.llm.http_transport import (
    HTTPResponse,
    HTTPStreamResponse,
    HTTPTimeout,
    HTTPTransportError,
    HttpxTransport,
)
from captioner.adapters.llm.openai_compatible import OpenAICompatibleClient
from captioner.core.application.structured_llm_service import StructuredLLMService
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMItem, LLMRequest, SourceCorrectionResponse
from captioner.infrastructure.config import OpenAICompatibleProvider, ProviderCredential


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "default",
        "https://fake.local/v1",
        "fake-model",
        ProviderCredential("unit-test-key"),
        max_concurrency=2,
        request_timeout_sec=1,
        max_retries=3,
    )


def _request() -> LLMRequest:
    prompt = "unit prompt"
    return LLMRequest(
        "correct_source",
        (LLMItem("item-1", "source text"),),
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
    )


def _success_body() -> bytes:
    content = json.dumps({"id": "item-1", "corrected_source": "corrected"})
    return json.dumps(
        {"choices": [{"message": {"content": content}}]}, separators=(",", ":")
    ).encode()


@dataclass(frozen=True, slots=True)
class FakeRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    content: bytes


@dataclass(frozen=True, slots=True)
class FakeStreamResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield self.body


_SCHEMA_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _validate_provider_schema_name(body: bytes) -> HTTPResponse | None:
    """Simulate real OpenAI-compatible rejection of illegal json_schema.name."""
    try:
        parsed: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HTTPResponse(400, {}, b'{"error":"invalid_json"}')
    if not isinstance(parsed, dict):
        return HTTPResponse(400, {}, b'{"error":"invalid_body"}')
    payload = cast(dict[str, object], parsed)
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return HTTPResponse(400, {}, b'{"error":"missing_response_format"}')
    format_map = cast(dict[str, object], response_format)
    json_schema = format_map.get("json_schema")
    if not isinstance(json_schema, dict):
        return HTTPResponse(400, {}, b'{"error":"missing_json_schema"}')
    schema_map = cast(dict[str, object], json_schema)
    if "name" not in schema_map:
        return HTTPResponse(400, {}, b'{"error":"missing_schema_name"}')
    name = schema_map.get("name")
    if not isinstance(name, str) or not name:
        return HTTPResponse(400, {}, b'{"error":"empty_schema_name"}')
    if _SCHEMA_NAME_RE.fullmatch(name) is None:
        return HTTPResponse(400, {}, b'{"error":"invalid_schema_name"}')
    return None


@dataclass(slots=True)
class FakeServer:
    responses: list[HTTPResponse | Exception]
    requests: list[FakeRequest] = field(default_factory=lambda: list[FakeRequest]())
    validate_schema_name: bool = True

    async def __call__(self, request: FakeRequest) -> HTTPResponse:
        self.requests.append(request)
        if self.validate_schema_name:
            rejected = _validate_provider_schema_name(request.content)
            if rejected is not None:
                return rejected
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass(slots=True)
class FakeClient:
    server: FakeServer
    closed: bool = False

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        content: bytes,
        timeout: object,
    ) -> AbstractAsyncContextManager[HTTPStreamResponse]:
        return self._stream(method, url, headers, content, timeout)

    @asynccontextmanager
    async def _stream(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: object,
    ) -> AsyncGenerator[HTTPStreamResponse]:
        del timeout
        response = await self.server(FakeRequest(method, url, headers, content))
        yield FakeStreamResponse(response.status_code, response.headers, response.body)

    async def aclose(self) -> None:
        self.closed = True


def _client(server: FakeServer) -> tuple[OpenAICompatibleClient, FakeClient]:
    raw_client = FakeClient(server)
    transport = HttpxTransport(timeout=HTTPTimeout.all(1), client=raw_client)
    return OpenAICompatibleClient(_provider(), transport=transport), raw_client


def test_fake_server_success_and_request_contract() -> None:
    server = FakeServer([HTTPResponse(200, {"content-type": "application/json"}, _success_body())])
    client, raw_client = _client(server)
    try:
        result = asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    finally:
        asyncio.run(client.close())
        asyncio.run(raw_client.aclose())
    assert result == SourceCorrectionResponse("item-1", "corrected")
    assert server.requests[0].headers["Authorization"] == "Bearer unit-test-key"
    assert b"unit-test-key" not in server.requests[0].content
    assert raw_client.closed


@pytest.mark.parametrize("status", [429, 503, 504, 401, 400])
def test_fake_server_statuses_are_classified(status: int) -> None:
    server = FakeServer([HTTPResponse(status, {}, b"provider body")])
    client, raw_client = _client(server)
    try:
        with pytest.raises(AppError):
            asyncio.run(
                client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
            )
    finally:
        asyncio.run(client.close())
        asyncio.run(raw_client.aclose())


def test_fake_server_retry_is_application_owned() -> None:
    server = FakeServer(
        [
            HTTPResponse(429, {}, b"rate limited"),
            HTTPResponse(503, {}, b"busy"),
            HTTPResponse(200, {}, _success_body()),
        ]
    )
    client, raw_client = _client(server)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    service = StructuredLLMService(client, sleep=sleep)
    try:
        result = asyncio.run(
            service.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    finally:
        asyncio.run(client.close())
        asyncio.run(raw_client.aclose())
    assert result == SourceCorrectionResponse("item-1", "corrected")
    assert delays == [1.0, 2.0]
    assert len(server.requests) == 3
    assert all(
        request.headers["Authorization"] == "Bearer unit-test-key" for request in server.requests
    )


def test_fake_server_timeout_is_classified_without_secret() -> None:
    server = FakeServer([HTTPTransportError("timeout")])
    client, raw_client = _client(server)
    try:
        with pytest.raises(AppError, match=r"llm\.timeout") as raised:
            asyncio.run(
                client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
            )
    finally:
        asyncio.run(client.close())
        asyncio.run(raw_client.aclose())
    assert "unit-test-key" not in str(raised.value)


def test_fake_server_rejects_invalid_schema_name() -> None:
    server = FakeServer([HTTPResponse(200, {}, _success_body())])
    # Directly exercise schema-name validation used by FakeServer.
    missing = _validate_provider_schema_name(b'{"response_format":{"json_schema":{}}}')
    assert missing is not None and missing.status_code == 400
    empty = _validate_provider_schema_name(b'{"response_format":{"json_schema":{"name":""}}}')
    assert empty is not None and empty.status_code == 400
    illegal = _validate_provider_schema_name(
        b'{"response_format":{"json_schema":{"name":"bad.<locals>.Name"}}}'
    )
    assert illegal is not None and illegal.status_code == 400
    too_long = _validate_provider_schema_name(
        json.dumps({"response_format": {"json_schema": {"name": "a" * 65}}}).encode()
    )
    assert too_long is not None and too_long.status_code == 400
    ok = _validate_provider_schema_name(
        b'{"response_format":{"json_schema":{"name":"captioner_correct_source_batch_v1"}}}'
    )
    assert ok is None

    client, raw_client = _client(server)
    try:
        result = asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    finally:
        asyncio.run(client.close())
        asyncio.run(raw_client.aclose())
    assert result.id == "item-1"
    body = json.loads(server.requests[0].content)
    name = body["response_format"]["json_schema"]["name"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
    assert "<locals>" not in name
