"""LLM ports with provider-neutral structured generation contracts."""

from typing import Protocol, TypeVar

from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import LLMRequest
from captioner.core.ports import CapabilityProbe


class LLMPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether an LLM implementation is available."""
        ...


T = TypeVar("T")


class LLMClient(Protocol):
    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
        context: ExecutionContext,
    ) -> T:
        """Generate one provider-neutral structured response."""
        ...
