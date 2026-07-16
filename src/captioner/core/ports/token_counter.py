"""Token counting port used by deterministic LLM chunk planning."""

from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Return a deterministic token estimate for one text value."""
        ...
