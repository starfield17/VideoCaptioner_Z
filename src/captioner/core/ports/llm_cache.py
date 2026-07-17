"""Provider-neutral validated LLM cache port."""

from typing import Protocol, TypeVar

from captioner.core.domain.llm_cache import LLMCacheKey

T = TypeVar("T")


class LLMCachePort(Protocol):
    def get(self, key: LLMCacheKey, response_schema: type[T]) -> T | None:
        """Return a freshly decoded valid response or a cache miss."""
        ...

    def put(
        self,
        key: LLMCacheKey,
        response: T,
        response_schema: type[T] | None = None,
    ) -> None:
        """Atomically store a response that the application fully validated."""
        ...

    def remove(self, key: LLMCacheKey) -> None:
        """Delete an entry that failed the complete validator."""
        ...


LLMCache = LLMCachePort
