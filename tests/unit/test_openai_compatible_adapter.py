from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from captioner.adapters.llm.http_transport import (
    HTTPResponse,
    HTTPTimeout,
    HTTPTransportError,
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
    return LLMRequest(
        "correct_source",
        (LLMItem(item_id, "source text"),),
        prompt_content="Return JSON.",
    )


def _response(item_id: str = "item-1", text: str = "corrected") -> HTTPResponse:
    content = json.dumps({"id": item_id, "corrected_source": text})
    body = json.dumps(
        {"choices": [{"message": {"content": content}}]},
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
        b'{"choices":[{"message":{"content":"{\\"id\\":\\"item-1\\",\\"id\\":\\"item-1\\"}"}}]}',
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
