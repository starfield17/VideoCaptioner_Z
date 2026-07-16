"""Dependency-free LLM capability fake."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from captioner.adapters._probe import empty_details, probe_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRequest
from captioner.core.domain.result import JsonValue
from captioner.core.ports import CapabilityProbe
from captioner.core.ports.llm import LLMClient


@dataclass(frozen=True, slots=True)
class ScriptedJSON:
    payload: str | bytes


@dataclass(frozen=True, slots=True)
class ScriptedCancellation:
    checkpoint: str = "before_request"


@dataclass(frozen=True, slots=True)
class ScriptedCrash:
    checkpoint: str = "before_request"


class ScriptedCrashError(RuntimeError):
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        super().__init__(checkpoint)


type StructuredOutcome = (
    object
    | AppError
    | ScriptedJSON
    | ScriptedCancellation
    | ScriptedCrash
    | Callable[[LLMRequest, type[object], ExecutionContext], object]
)


@dataclass(slots=True)
class FakeLLMAdapter(LLMClient):
    available: bool = True
    details: Mapping[str, JsonValue] = field(default_factory=empty_details)
    delay_seconds: float = 0.0
    failure: AppError | None = None
    structured_response: StructuredOutcome | None = None
    structured_calls: list[LLMRequest] = field(default_factory=lambda: list[LLMRequest]())

    async def probe(self) -> CapabilityProbe:
        return await probe_result(
            available=self.available,
            details=self.details,
            delay_seconds=self.delay_seconds,
            failure=self.failure,
        )

    async def generate_structured[T](
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        context.raise_if_cancelled()
        self.structured_calls.append(request)
        if self.structured_response is None:
            raise AppError("llm.fake_unconfigured")
        return cast(
            T,
            resolve_structured_outcome(
                self.structured_response,
                request,
                cast(type[object], response_schema),
                context,
            ),
        )


def resolve_structured_outcome(
    outcome: StructuredOutcome,
    request: LLMRequest,
    response_schema: type[object],
    context: ExecutionContext,
) -> object:
    if isinstance(outcome, AppError):
        raise outcome
    if isinstance(outcome, ScriptedJSON):
        parser = getattr(response_schema, "from_json", None)
        if not callable(parser):
            raise AppError("llm.schema_invalid", {"reason": "scripted_schema"})
        return parser(outcome.payload)
    if isinstance(outcome, ScriptedCancellation):
        context.cancel()
        context.checkpoint(outcome.checkpoint)
        raise AppError("operation.cancelled")
    if isinstance(outcome, ScriptedCrash):
        raise ScriptedCrashError(outcome.checkpoint)
    if callable(outcome):
        return outcome(request, response_schema, context)
    return outcome
