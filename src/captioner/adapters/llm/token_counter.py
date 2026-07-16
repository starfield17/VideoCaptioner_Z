"""Small deterministic token-count approximation for the default runtime."""

from __future__ import annotations


class CharacterTokenCounter:
    """Count Unicode code points without importing a provider tokenizer."""

    def count(self, text: str) -> int:
        return len(text)
