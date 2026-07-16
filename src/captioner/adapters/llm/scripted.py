"""Ordered probe results for deterministic future LLM tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from captioner.adapters.llm.fake import StructuredOutcome, resolve_structured_outcome
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRequest
from captioner.core.ports import CapabilityProbe
from captioner.core.ports.llm import LLMClient


@dataclass(slots=True)
class ScriptedLLMAdapter(LLMClient):
    responses: Sequence[CapabilityProbe | AppError] = field(default_factory=tuple)
    delay_seconds: float = 0.0
    structured_responses: Sequence[StructuredOutcome] = field(default_factory=tuple)
    structured_calls: list[LLMRequest] = field(default_factory=lambda: list[LLMRequest]())
    _index: int = field(default=0, init=False)
    _structured_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.responses = tuple(self.responses)
        self.structured_responses = tuple(self.structured_responses)

    async def probe(self) -> CapabilityProbe:
        from captioner.adapters._probe import probe_result

        if self.delay_seconds < 0:
            raise ValueError
        if self._index >= len(self.responses):
            raise AppError("llm.script_exhausted", retryable=False)
        response = self.responses[self._index]
        self._index += 1
        if isinstance(response, AppError):
            raise response
        return await probe_result(
            available=response.available,
            details=response.details,
            delay_seconds=self.delay_seconds,
            failure=None,
        )

    async def generate_structured[T](
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        if self._structured_index >= len(self.structured_responses):
            raise AppError("llm.script_exhausted", {"kind": "structured"})
        outcome = self.structured_responses[self._structured_index]
        self._structured_index += 1
        self.structured_calls.append(request)
        return cast(
            T,
            resolve_structured_outcome(outcome, request, response_schema, context),
        )
