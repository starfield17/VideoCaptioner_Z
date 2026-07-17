from __future__ import annotations

import asyncio
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
) -> StructuredLLMService:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return StructuredLLMService(
        client,
        max_retries=max_retries,
        sleep=sleep,
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


def test_schema_failure_belongs_to_chunk_executor_not_transport_service() -> None:
    client = ScriptedClient([AppError("llm.schema_invalid")])
    delays: list[float] = []
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        asyncio.run(
            _service(client, delays).generate_structured(
                _request(), SourceCorrectionResponse, ExecutionContext()
            )
        )
    assert [call.task_kind for call in client.calls] == ["correct_source"]
    assert delays == []


def test_cancellation_after_backoff_is_not_retried() -> None:
    context = ExecutionContext()
    client = ScriptedClient([AppError("llm.timeout", retryable=True)])

    async def cancel_sleep(delay: float) -> None:
        del delay
        context.cancel()

    service = StructuredLLMService(client, sleep=cancel_sleep)
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(service.generate_structured(_request(), SourceCorrectionResponse, context))


def test_cancel_interrupts_retry_backoff() -> None:
    client = ScriptedClient([AppError("llm.rate_limited", retryable=True)])
    context = ExecutionContext()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def sleep(delay: float) -> None:
        del delay
        started.set()
        await asyncio.sleep(10)

    service = StructuredLLMService(client, max_retries=5, sleep=sleep)

    async def run() -> None:
        try:
            await service.generate_structured(_request(), SourceCorrectionResponse, context)
        except AppError as exc:
            assert exc.code == "operation.cancelled"
            finished.set()

    async def scenario() -> None:
        task = asyncio.create_task(run())
        await started.wait()
        context.cancel()
        await asyncio.wait_for(finished.wait(), timeout=1.0)
        await task
        pending = [item for item in asyncio.all_tasks() if item is not asyncio.current_task()]
        assert not any("sleep" in repr(item.get_coro()) for item in pending)

    asyncio.run(scenario())
    assert len(client.calls) == 1


def test_cancelled_backoff_does_not_retry() -> None:
    client = ScriptedClient([AppError("llm.timeout", retryable=True)])
    context = ExecutionContext()

    async def sleep(delay: float) -> None:
        del delay
        context.cancel()
        await asyncio.sleep(0.01)

    service = StructuredLLMService(client, max_retries=5, sleep=sleep)
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(service.generate_structured(_request(), SourceCorrectionResponse, context))
    assert len(client.calls) == 1


def test_cancelled_backoff_leaves_no_pending_tasks() -> None:
    client = ScriptedClient([AppError("llm.network_error", retryable=True)])
    context = ExecutionContext()

    async def sleep(delay: float) -> None:
        del delay
        context.cancel()
        await asyncio.Event().wait()

    service = StructuredLLMService(client, max_retries=3, sleep=sleep)

    async def scenario() -> None:
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await service.generate_structured(_request(), SourceCorrectionResponse, context)
        await asyncio.sleep(0)
        current = asyncio.current_task()
        leftovers = [
            task for task in asyncio.all_tasks() if task is not current and not task.done()
        ]
        assert leftovers == []

    asyncio.run(scenario())


def test_completed_backoff_still_retries() -> None:
    client = ScriptedClient(
        [
            AppError("llm.rate_limited", retryable=True),
            SourceCorrectionResponse("item-1", "ok"),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    service = StructuredLLMService(client, max_retries=2, sleep=sleep)
    result = asyncio.run(
        service.generate_structured(_request(), SourceCorrectionResponse, ExecutionContext())
    )
    assert result.corrected_source == "ok"
    assert delays == [1.0]
    assert len(client.calls) == 2
