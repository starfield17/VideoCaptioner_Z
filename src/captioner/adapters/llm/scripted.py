"""Ordered probe results for deterministic future LLM tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from captioner.core.domain.errors import AppError
from captioner.core.ports import CapabilityProbe


@dataclass(slots=True)
class ScriptedLLMAdapter:
    responses: Sequence[CapabilityProbe | AppError] = field(default_factory=tuple)
    delay_seconds: float = 0.0
    _index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.responses = tuple(self.responses)

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
