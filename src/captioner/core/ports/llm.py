"""LLM port without a provider request/response model."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class LLMPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether an LLM implementation is available."""
        ...
