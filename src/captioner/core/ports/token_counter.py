"""Token counting port used by deterministic LLM chunk planning."""

from typing import Protocol

from captioner.core.domain.llm import LLMRequest


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Return a deterministic token estimate for one text value."""
        ...


class LLMRequestEstimator(Protocol):
    def estimate_input_tokens(
        self,
        request: LLMRequest,
        response_schema: type[object],
    ) -> int:
        """Estimate the complete serialized request before network access."""
        ...
