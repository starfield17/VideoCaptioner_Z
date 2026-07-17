"""Application-level retry and structured-repair policy for LLM calls."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRepairContext, LLMRepairDiagnostic, LLMRequest
from captioner.core.ports.llm import LLMClient

type Sleep = Callable[[float], Awaitable[None]]
_RETRYABLE_ERRORS = frozenset(
    {
        "llm.rate_limited",
        "llm.upstream_unavailable",
        "llm.network_error",
        "llm.timeout",
    }
)


@dataclass(slots=True)
class StructuredLLMService(LLMClient):
    """Retry one structured request without retrying permanent or cancelled work."""

    client: LLMClient
    max_retries: int = 5
    backoff_base_sec: float = 1.0
    sleep: Sleep = asyncio.sleep

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError
        if (
            isinstance(self.backoff_base_sec, bool)
            or not math.isfinite(self.backoff_base_sec)
            or self.backoff_base_sec < 0
        ):
            raise ValueError

    async def generate_structured[T](
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        current_request = request
        retry_index = 0
        while True:
            context.raise_if_cancelled()
            try:
                return await self.client.generate_structured(
                    current_request, response_schema, context
                )
            except AppError as exc:
                if not _is_retryable(exc) or retry_index >= self.max_retries:
                    raise
                delay = self.backoff_base_sec * (2**retry_index)
                retry_index += 1
                await sleep_with_cancellation(delay, context, self.sleep)


async def sleep_with_cancellation(
    delay: float,
    context: ExecutionContext,
    sleep: Sleep,
) -> None:
    """Sleep until delay elapses or the execution context is cancelled.

    Cancel wins over sleep: the sleep task is cancelled and cleaned up, then
    ``operation.cancelled`` is raised so the caller never enters another retry.
    """
    context.raise_if_cancelled()
    if delay <= 0:
        return
    sleep_task = asyncio.create_task(_run_sleep(sleep, delay))
    cancel_task = asyncio.create_task(context.wait_cancelled())
    try:
        done, pending = await asyncio.wait(
            (sleep_task, cancel_task), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if cancel_task in done:
            if not sleep_task.done():
                sleep_task.cancel()
                await asyncio.gather(sleep_task, return_exceptions=True)
            raise AppError("operation.cancelled")
        # Sleep completed first; surface any sleep failure.
        sleep_task.result()
    except asyncio.CancelledError:
        for task in (sleep_task, cancel_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleep_task, cancel_task, return_exceptions=True)
        raise


async def _run_sleep(sleep: Sleep, delay: float) -> None:
    await sleep(delay)


def structured_repair_request(
    request: LLMRequest,
    *,
    invalid_response: str,
    diagnostics: tuple[LLMRepairDiagnostic, ...],
    repair_prompt_id: str,
    repair_prompt_version: str,
    repair_prompt_content_sha256: str,
    repair_prompt_content: str,
) -> LLMRequest:
    """Create one contextual repair request owned by the executor."""
    if not all(
        (
            repair_prompt_id,
            repair_prompt_version,
            repair_prompt_content_sha256,
            repair_prompt_content,
        )
    ):
        raise AppError("prompt.identity_missing", {"prompt_id": "repair_structured"})
    if request.repair_context is not None:
        raise AppError("llm.repair_already_attempted")
    return LLMRequest(
        task_kind=request.task_kind,
        items=request.items,
        context=request.context,
        source_language=request.source_language,
        target_language=request.target_language,
        prompt_id=request.prompt_id,
        prompt_version=request.prompt_version,
        prompt_content_sha256=request.prompt_content_sha256,
        prompt_content=request.prompt_content,
        context_payload=request.context_payload,
        repair_prompt_id=repair_prompt_id,
        repair_prompt_version=repair_prompt_version,
        repair_prompt_content_sha256=repair_prompt_content_sha256,
        repair_prompt_content=repair_prompt_content,
        repair_context=LLMRepairContext(
            original_task_kind=request.task_kind,
            invalid_response=invalid_response,
            diagnostics=diagnostics,
        ),
    )


def _is_retryable(error: AppError) -> bool:
    if error.code == "operation.cancelled":
        return False
    return error.code in _RETRYABLE_ERRORS
