"""Application-level retry and structured-repair policy for LLM calls."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRequest
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
                context.raise_if_cancelled()
                await self.sleep(delay)


def structured_repair_request(
    request: LLMRequest,
    *,
    repair_prompt_id: str,
    repair_prompt_version: str,
    repair_prompt_content_sha256: str,
    repair_prompt_content: str,
) -> LLMRequest:
    """Create the single resource-backed repair request owned by the executor."""
    if not all(
        (
            repair_prompt_id,
            repair_prompt_version,
            repair_prompt_content_sha256,
            repair_prompt_content,
        )
    ):
        raise AppError("prompt.identity_missing", {"prompt_id": "repair_structured"})
    return LLMRequest(
        task_kind="repair_structured",
        items=request.items,
        context=request.context,
        source_language=request.source_language,
        target_language=request.target_language,
        prompt_id=repair_prompt_id,
        prompt_version=repair_prompt_version,
        prompt_content_sha256=repair_prompt_content_sha256,
        prompt_content=repair_prompt_content,
        context_payload=request.context_payload,
        repair_prompt_id=request.repair_prompt_id,
        repair_prompt_version=request.repair_prompt_version,
        repair_prompt_content_sha256=request.repair_prompt_content_sha256,
        repair_prompt_content=request.repair_prompt_content,
    )


def _is_retryable(error: AppError) -> bool:
    if error.code == "operation.cancelled":
        return False
    if error.code in {
        "llm.auth_failed",
        "llm.request_rejected",
        "llm.http_error",
        "llm.schema_invalid",
        "llm.response_invalid",
        "llm.id_mismatch",
        "llm.context_id_returned",
    }:
        return False
    return error.code in _RETRYABLE_ERRORS or error.retryable
