"""Production and test-double token counters for LLM request budgeting."""

from __future__ import annotations

from typing import Final

import tiktoken

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLMRequest, encode_llm_request

SUPPORTED_TOKENIZERS: Final[frozenset[str]] = frozenset({"cl100k_base", "o200k_base"})

# Explicit model → tokenizer map for tokenizer="auto". Unknown models fail closed.
_MODEL_TOKENIZER_MAP: Final[dict[str, str]] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}


def resolve_tokenizer_id(tokenizer: str, model: str) -> str:
    """Resolve a configured tokenizer id; unknown values fail closed."""
    if not tokenizer.strip():
        raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer})
    selected = tokenizer.strip()
    if selected == "auto":
        if not model.strip():
            raise AppError("llm.tokenizer_unknown", {"model": model})
        model_name = model.strip()
        mapped = _MODEL_TOKENIZER_MAP.get(model_name)
        if mapped is None:
            # Also try prefix match for dated model ids like gpt-4o-2024-11-20.
            for prefix, encoding_id in _MODEL_TOKENIZER_MAP.items():
                if model_name.startswith(prefix):
                    return encoding_id
            raise AppError("llm.tokenizer_unknown", {"model": model_name})
        return mapped
    if selected not in SUPPORTED_TOKENIZERS:
        raise AppError("llm.tokenizer_unknown", {"tokenizer": selected})
    return selected


class CharacterTokenCounter:
    """Test-double counter that counts Unicode code points, not model tokens.

    Production bootstrap must not use this class. It exists only so unit and
    recovery tests can inject a deterministic, dependency-free estimate.
    """

    def count(self, text: str) -> int:
        return len(text)


class ModelTokenCounter:
    """Production counter backed by a configured tiktoken encoding."""

    def __init__(self, tokenizer_id: str) -> None:
        if tokenizer_id not in SUPPORTED_TOKENIZERS:
            raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer_id})
        self._tokenizer_id = tokenizer_id
        try:
            self._encoding = tiktoken.get_encoding(tokenizer_id)
        except Exception as exc:
            raise AppError("llm.tokenizer_unknown", {"tokenizer": tokenizer_id}) from exc

    @property
    def tokenizer_id(self) -> str:
        return self._tokenizer_id

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


class SerializedRequestTokenCounter:
    """Estimate the full wire request with the same serialization as the adapter."""

    def __init__(
        self,
        token_counter: CharacterTokenCounter | ModelTokenCounter,
        model: str,
        temperature: float,
        *,
        response_schema_version: int = 1,
    ) -> None:
        self._token_counter = token_counter
        self._model = model
        self._temperature = temperature
        self._response_schema_version = response_schema_version

    def estimate_input_tokens(self, request: LLMRequest, response_schema: type[object]) -> int:
        encoded = encode_llm_request(
            request,
            self._model,
            self._temperature,
            response_schema,
            response_schema_version=self._response_schema_version,
        )
        return self._token_counter.count(encoded.decode("utf-8"))
