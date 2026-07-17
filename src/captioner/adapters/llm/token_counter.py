"""Small deterministic token-count approximation for the default runtime."""

from __future__ import annotations

from captioner.core.domain.llm import LLMRequest, encode_llm_request


class CharacterTokenCounter:
    """Count Unicode code points without importing a provider tokenizer."""

    def count(self, text: str) -> int:
        return len(text)


class SerializedRequestTokenCounter:
    """Estimate the full wire request with the same serialization as the adapter."""

    def __init__(
        self, token_counter: CharacterTokenCounter, model: str, temperature: float
    ) -> None:
        self._token_counter = token_counter
        self._model = model
        self._temperature = temperature

    def estimate_input_tokens(self, request: LLMRequest, response_schema: type[object]) -> int:
        return self._token_counter.count(
            encode_llm_request(request, self._model, self._temperature, response_schema).decode(
                "utf-8"
            )
        )
