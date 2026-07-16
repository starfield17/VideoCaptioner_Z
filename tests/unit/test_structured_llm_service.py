from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import pytest

from captioner.core.application.structured_llm_service import StructuredLLMService
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMItem, LLMRequest, SourceCorrectionResponse


def _request() -> LLMRequest:
    return LLMRequest("correct_source", (LLMItem("item-1", "source"),))


@dataclass
class ScriptedClient:
    outcomes: list[object | AppError]
    calls: list[LLMRequest] = field(default_factory=lambda: [])

    async def generate_structured[T](
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        context.raise_if_cancelled()
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, AppError):
            raise outcome
        return cast(T, outcome)


def _service(
    client: ScriptedClient,
    delays: list[float],
    *,
    max_retries: int = 5,
    repair_request_factory: Callable[[LLMRequest], LLMRequest] | None = None,
) -> StructuredLLMService:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return StructuredLLMService(
        client,
        max_retries=max_retries,
        sleep=sleep,
        repair_request_factory=repair_request_factory,
    )


def test_retryable_failures_use_bounded_exponential_backoff() -> None:
    client = ScriptedClient(
        [
            AppError("llm.rate_limited", retryable=True),
            AppError("llm.upstream_unavailable", retryable=True),
            SourceCorrectionResponse("item-1", "corrected"),
        ]
    )
    delays: list[float] = []
    result = asyncio.run(
        _service(client, delays).generate_structured(
            _request(), SourceCorrectionResponse, ExecutionContext()
        )
    )
    assert result == SourceCorrectionResponse("item-1", "corrected")
    assert delays == [1.0, 2.0]
    assert len(client.calls) == 3


def test_permanent_failure_and_id_mismatch_are_not_retried() -> None:
    for code in ("llm.auth_failed", "llm.request_rejected", "llm.id_mismatch"):
        client = ScriptedClient([AppError(code, retryable=True)])
        delays: list[float] = []
        with pytest.raises(AppError, match=code.replace(".", r"\.")):
            asyncio.run(
                _service(client, delays).generate_structured(
                    _request(), SourceCorrectionResponse, ExecutionContext()
                )
            )
        assert delays == []
        assert len(client.calls) == 1


def test_schema_failure_gets_exactly_one_repair_request() -> None:
    client = ScriptedClient(
        [
            AppError("llm.schema_invalid"),
            SourceCorrectionResponse("item-1", "corrected"),
        ]
    )
    delays: list[float] = []
    result = asyncio.run(
        _service(client, delays).generate_structured(
            _request(), SourceCorrectionResponse, ExecutionContext()
        )
    )
    assert result == SourceCorrectionResponse("item-1", "corrected")
    assert [call.task_kind for call in client.calls] == ["correct_source", "repair_structured"]
    assert delays == []

    failing = ScriptedClient([AppError("llm.schema_invalid"), AppError("llm.schema_invalid")])
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        asyncio.run(
            _service(failing, []).generate_structured(
                _request(), SourceCorrectionResponse, ExecutionContext()
            )
        )
    assert len(failing.calls) == 2


def test_cancellation_after_backoff_is_not_retried() -> None:
    context = ExecutionContext()
    client = ScriptedClient([AppError("llm.timeout", retryable=True)])

    async def cancel_sleep(delay: float) -> None:
        del delay
        context.cancel()

    service = StructuredLLMService(client, sleep=cancel_sleep)
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(service.generate_structured(_request(), SourceCorrectionResponse, context))
