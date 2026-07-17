from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import cast

import pytest

from captioner.adapters.llm.http_transport import (
    HTTPResponse,
    HTTPTimeout,
    HTTPTransportError,
    HttpxTransport,
)
from captioner.adapters.llm.openai_compatible import OpenAICompatibleClient
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMItem, LLMRequest, SourceCorrectionResponse
from captioner.infrastructure.config import OpenAICompatibleProvider, ProviderCredential


def _provider(*, max_concurrency: int = 2) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        "default",
        "https://provider.example/v1/",
        "unit-model",
        ProviderCredential("unit-test-key"),
        max_concurrency=max_concurrency,
        request_timeout_sec=2,
        max_retries=2,
        temperature=0.2,
    )


def _request(item_id: str = "item-1") -> LLMRequest:
    prompt = "Return JSON."
    return LLMRequest(
        "correct_source",
        (LLMItem(item_id, "source text"),),
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
    )


def _response(item_id: str = "item-1", text: str = "corrected") -> HTTPResponse:
    content = json.dumps({"id": item_id, "corrected_source": text})
    body = json.dumps(
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
        separators=(",", ":"),
    ).encode()
    return HTTPResponse(200, {"content-type": "application/json"}, body)


@dataclass
class QueueTransport:
    outcomes: list[HTTPResponse | Exception]
    calls: list[tuple[str, str, Mapping[str, str], bytes]] = field(
        default_factory=lambda: list[tuple[str, str, Mapping[str, str], bytes]]()
    )
    closed: bool = False

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: HTTPTimeout,
        max_response_bytes: int,
    ) -> HTTPResponse:
        del timeout, max_response_bytes
        self.calls.append((method, url, headers, content))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


@dataclass
class HangingTransport:
    started: bool = False
    cancelled: bool = False

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: HTTPTimeout,
        max_response_bytes: int,
    ) -> HTTPResponse:
        del method, url, headers, content, timeout, max_response_bytes
        self.started = True
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")

    async def close(self) -> None:
        return None


def test_success_uses_schema_and_redacts_runtime_credential() -> None:
    transport = QueueTransport([_response()])
    client = OpenAICompatibleClient(_provider(), transport=transport)
    result = asyncio.run(
        client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
    )

    assert result == SourceCorrectionResponse("item-1", "corrected")
    method, url, headers, body = transport.calls[0]
    assert method == "POST"
    assert url == "https://provider.example/v1/chat/completions"
    assert headers["Authorization"] == "Bearer unit-test-key"
    assert b"unit-test-key" not in body
    assert "unit-test-key" not in repr(client)
    asyncio.run(client.close())
    assert transport.closed


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "llm.rate_limited", True),
        (502, "llm.upstream_unavailable", True),
        (503, "llm.upstream_unavailable", True),
        (504, "llm.upstream_unavailable", True),
        (401, "llm.auth_failed", False),
        (403, "llm.auth_failed", False),
        (400, "llm.request_rejected", False),
    ],
)
def test_http_statuses_are_classified_without_response_body(
    status: int, code: str, retryable: bool
) -> None:
    transport = QueueTransport([HTTPResponse(status, {}, b"unit-test-key")])
    client = OpenAICompatibleClient(_provider(), transport=transport)
    with pytest.raises(AppError) as raised:
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert "unit-test-key" not in str(raised.value)


def test_network_timeout_and_response_shape_failures_are_structured() -> None:
    for outcome, expected in (
        (HTTPTransportError("timeout"), "llm.timeout"),
        (HTTPTransportError("network"), "llm.network_error"),
        (HTTPTransportError("response_too_large"), "llm.response_too_large"),
        (TimeoutError("timeout"), "llm.timeout"),
        (OSError("network"), "llm.network_error"),
    ):
        transport = QueueTransport([outcome])
        client = OpenAICompatibleClient(_provider(), transport=transport)
        with pytest.raises(AppError, match=expected.replace(".", r"\.")):
            asyncio.run(
                client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
            )

    malformed = HTTPResponse(
        200,
        {},
        b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"id\\":\\"item-1\\",\\"id\\":\\"item-1\\"}"}}]}',
    )
    client = OpenAICompatibleClient(_provider(), transport=QueueTransport([malformed]))
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )


def test_id_mismatch_is_not_retried_by_the_adapter() -> None:
    client = OpenAICompatibleClient(_provider(), transport=QueueTransport([_response("other")]))
    with pytest.raises(AppError, match=r"llm\.id_mismatch"):
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"[]", "envelope_object"),
        (b'{"choices":[]}', "choices"),
        (b'{"choices":[1]}', "choice"),
        (b'{"choices":[{"finish_reason":"stop","message":1}]}', "message"),
        (b'{"choices":[{"finish_reason":"stop","message":{"content":1}}]}', "content"),
    ],
)
def test_response_envelope_shape_is_strict(body: bytes, reason: str) -> None:
    client = OpenAICompatibleClient(
        _provider(),
        transport=QueueTransport([HTTPResponse(200, {}, body)]),
    )
    with pytest.raises(AppError) as raised:
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert raised.value.code == "llm.provider_response_invalid"
    assert raised.value.params["reason"] == reason


@pytest.mark.parametrize(
    ("body", "code", "reason"),
    [
        (
            {"choices": [{"finish_reason": "stop", "message": {"refusal": "private refusal"}}]},
            "llm.refused",
            None,
        ),
        (
            {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]},
            "llm.content_filtered",
            None,
        ),
        (
            {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
            "llm.output_truncated",
            None,
        ),
        (
            {"choices": [{"finish_reason": "unknown", "message": {"content": "{}"}}]},
            "llm.provider_response_invalid",
            "finish_reason",
        ),
        (
            {"choices": [{"message": {"content": "{}"}}]},
            "llm.provider_response_invalid",
            "finish_reason",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            "llm.provider_response_invalid",
            "content",
        ),
    ],
)
def test_provider_outcomes_are_classified_before_structured_decoding(
    body: dict[str, object], code: str, reason: str | None
) -> None:
    raw_body = json.dumps(body).encode()
    client = OpenAICompatibleClient(
        _provider(),
        transport=QueueTransport([HTTPResponse(200, {}, raw_body)]),
    )
    with pytest.raises(AppError) as raised:
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert raised.value.code == code
    assert raised.value.retryable is False
    if reason is not None:
        assert raised.value.params["reason"] == reason
    assert "private refusal" not in str(raised.value)


def test_unknown_http_status_is_classified_without_retry() -> None:
    client = OpenAICompatibleClient(
        _provider(),
        transport=QueueTransport([HTTPResponse(418, {}, b"provider body")]),
    )
    with pytest.raises(AppError) as raised:
        asyncio.run(
            client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
        )
    assert raised.value.code == "llm.http_error"
    assert raised.value.retryable is False


def test_cancellation_while_waiting_for_global_semaphore_does_not_leave_a_waiter() -> None:
    async def run() -> None:
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        transport = QueueTransport([_response()])
        client = OpenAICompatibleClient(_provider(), transport=transport, semaphore=semaphore)
        context = ExecutionContext()
        task = asyncio.create_task(
            client.generate_structured(_request(), SourceCorrectionResponse, context)
        )
        await asyncio.sleep(0.06)
        context.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await asyncio.wait_for(task, timeout=1)
        assert transport.calls == []
        semaphore.release()
        result = await client.generate_structured(
            _request(), SourceCorrectionResponse, ExecutionContext()
        )
        assert result == SourceCorrectionResponse("item-1", "corrected")
        await client.close()

    asyncio.run(run())


def test_shared_semaphore_limits_active_requests() -> None:
    @dataclass
    class ConcurrentTransport:
        active: int = 0
        maximum: int = 0

        async def request(
            self,
            method: str,
            url: str,
            headers: Mapping[str, str],
            content: bytes,
            timeout: HTTPTimeout,
            max_response_bytes: int,
        ) -> HTTPResponse:
            del method, url, headers, content, timeout, max_response_bytes
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.001)
            self.active -= 1
            return _response()

        async def close(self) -> None:
            return None

    async def run_calls() -> int:
        transport = ConcurrentTransport()
        client = OpenAICompatibleClient(_provider(max_concurrency=2), transport=transport)
        await asyncio.gather(
            *(
                client.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
                for _ in range(8)
            )
        )
        return transport.maximum

    assert asyncio.run(run_calls()) <= 2


def test_cancelled_context_never_sends_request() -> None:
    context = ExecutionContext()
    context.cancel()
    transport = QueueTransport([_response()])
    client = OpenAICompatibleClient(_provider(), transport=transport)
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(client.generate_structured(_request(), SourceCorrectionResponse, context))
    assert transport.calls == []


def test_cancel_interrupts_hanging_transport_and_releases_semaphore() -> None:
    async def run() -> None:
        context = ExecutionContext()
        transport = HangingTransport()
        semaphore = asyncio.Semaphore(1)
        client = OpenAICompatibleClient(_provider(), transport=transport, semaphore=semaphore)
        task = asyncio.create_task(
            client.generate_structured(_request(), SourceCorrectionResponse, context)
        )
        for _ in range(100):
            if transport.started:
                break
            await asyncio.sleep(0)
        assert transport.started
        context.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await asyncio.wait_for(task, timeout=1)
        assert transport.cancelled
        assert semaphore._value == 1
        current = asyncio.current_task()
        assert all(candidate is current or candidate.done() for candidate in asyncio.all_tasks())
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("value", [True, "1", 0, -1, math.nan, math.inf, -math.inf])
def test_http_timeout_rejects_non_positive_or_non_finite_values(value: object) -> None:
    with pytest.raises(ValueError):
        HTTPTimeout(cast(float, value), 1, 1, 1)


@dataclass
class StreamResponse:
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    chunks: tuple[bytes, ...] = (b"ok",)

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


class StreamContext:
    def __init__(self, response: StreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> StreamResponse:
        return self.response

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeHTTPXClient:
    def __init__(self, outcome: StreamResponse | Exception) -> None:
        self.outcome = outcome
        self.closed = False

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        content: bytes,
        timeout: object,
    ) -> StreamContext:
        del method, url, headers, content, timeout
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return StreamContext(self.outcome)

    async def aclose(self) -> None:
        self.closed = True


def test_httpx_transport_bounds_reads_and_closes_injected_client() -> None:
    raw = FakeHTTPXClient(StreamResponse(headers={"content-type": "text/plain"}))
    transport = HttpxTransport(timeout=HTTPTimeout.all(1), client=raw)
    result = asyncio.run(
        transport.request("POST", "https://provider.example/v1", {}, b"{}", HTTPTimeout.all(1), 10)
    )
    assert result.body == b"ok"
    asyncio.run(transport.close())
    assert raw.closed is False

    oversized = FakeHTTPXClient(StreamResponse(chunks=(b"too", b"large")))
    bounded = HttpxTransport(timeout=HTTPTimeout.all(1), client=oversized)
    with pytest.raises(HTTPTransportError, match="response_too_large"):
        asyncio.run(
            bounded.request("POST", "https://provider.example/v1", {}, b"{}", HTTPTimeout.all(1), 4)
        )
