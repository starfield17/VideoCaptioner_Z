"""Concurrent, cache-aware execution of independently validated LLM Chunks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    LLM_RESPONSE_SCHEMA_VERSION,
    LLMItem,
    LLMRequest,
    StructuredResponseBatch,
    response_batch_schema,
)
from captioner.core.domain.llm_cache import LLMCacheKey, build_llm_cache_key
from captioner.core.domain.result import JsonValue
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner, LLMChunk
from captioner.core.policies.llm_validation import validate_responses
from captioner.core.ports.llm import LLMClient
from captioner.core.ports.llm_cache import LLMCachePort

type ChunkRequestFactory = Callable[[LLMChunk], LLMRequest]

_ID_MISMATCH_ERRORS = frozenset(
    {
        "llm.id_mismatch",
        "llm.missing_id",
        "llm.extra_id",
        "llm.duplicate_id",
        "llm.context_id_returned",
    }
)


@dataclass(frozen=True, slots=True)
class LLMChunkExecutionConfig:
    task_kind: str
    provider_kind: str
    provider_identity: str
    base_url_identity: str
    model: str
    temperature: float
    source_language: str | None
    target_language: str | None
    profile: str
    prompt_id: str
    prompt_version: str
    prompt_content_sha256: str
    prompt_content: str = ""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION

    def chunk_config(self) -> dict[str, JsonValue]:
        return {
            "max_items": self.chunking.max_items,
            "max_input_tokens": self.chunking.max_input_tokens,
            "context_before_items": self.chunking.context_before_items,
            "context_after_items": self.chunking.context_after_items,
            "max_audio_context_duration_ms": self.chunking.max_audio_context_duration_ms,
        }


@dataclass(slots=True)
class LLMChunkExecutor:
    client: LLMClient
    cache: LLMCachePort
    planner: ChunkPlanner
    config: LLMChunkExecutionConfig

    async def execute[T](
        self,
        items: Sequence[ChunkItem],
        response_schema: type[T],
        context: ExecutionContext | None = None,
        request_factory: ChunkRequestFactory | None = None,
    ) -> tuple[T, ...]:
        execution = ExecutionContext() if context is None else context
        ordered = tuple(items)
        chunks = self.planner.plan(ordered, self.config.chunking)
        if not chunks:
            return ()
        factory = request_factory or self._default_request
        batch_schema = response_batch_schema(response_schema)
        tasks = [
            asyncio.create_task(
                self._execute_with_shrink(
                    chunk,
                    ordered,
                    batch_schema,
                    factory,
                    execution,
                )
            )
            for chunk in chunks
        ]
        try:
            chunk_results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await _cancel_tasks(tasks)
            raise
        except Exception:
            await _cancel_tasks(tasks)
            raise
        by_id: dict[str, object] = {}
        for result in chunk_results:
            for response in result:
                response_id = _response_id(response)
                if response_id in by_id:
                    raise AppError("llm.duplicate_id", {"id": response_id})
                by_id[response_id] = response
        expected_ids = tuple(item.id for item in ordered)
        if set(by_id) != set(expected_ids):
            raise AppError("llm.missing_id", {"ids": list(expected_ids)})
        return tuple(cast(T, by_id[item_id]) for item_id in expected_ids)

    async def _execute_with_shrink(
        self,
        chunk: LLMChunk,
        all_items: tuple[ChunkItem, ...],
        batch_schema: type[StructuredResponseBatch],
        request_factory: ChunkRequestFactory,
        context: ExecutionContext,
    ) -> tuple[object, ...]:
        try:
            return await self._execute_chunk(chunk, batch_schema, request_factory, context)
        except AppError as exc:
            if exc.code not in _ID_MISMATCH_ERRORS or len(chunk.items) < 2:
                raise
            positions = {item.id: index for index, item in enumerate(all_items)}
            indexes = tuple(positions[item.id] for item in chunk.items)
            if indexes != tuple(range(indexes[0], indexes[-1] + 1)):
                raise AppError("llm.id_mismatch", {"reason": "non_contiguous_chunk"}) from exc
            midpoint = indexes[0] + len(indexes) // 2
            left = self.planner.plan_range(
                all_items, indexes[0], midpoint, self.config.chunking, index=chunk.index * 2
            )
            right = self.planner.plan_range(
                all_items,
                midpoint,
                indexes[-1] + 1,
                self.config.chunking,
                index=chunk.index * 2 + 1,
            )
            left_result = await self._execute_with_shrink(
                left, all_items, batch_schema, request_factory, context
            )
            right_result = await self._execute_with_shrink(
                right, all_items, batch_schema, request_factory, context
            )
            return (*left_result, *right_result)

    async def _execute_chunk(
        self,
        chunk: LLMChunk,
        batch_schema: type[StructuredResponseBatch],
        request_factory: ChunkRequestFactory,
        context: ExecutionContext,
    ) -> tuple[object, ...]:
        request = request_factory(chunk)
        key = self._cache_key(chunk)
        cached = self.cache.get(key, batch_schema)
        if cached is not None:
            try:
                return self._validate_and_order(cached, chunk)
            except AppError:
                pass
        generated = await self.client.generate_structured(request, batch_schema, context)
        responses = self._validate_and_order(generated, chunk)
        cache_value = _batch_from_responses(batch_schema, responses)
        self.cache.put(key, cache_value, batch_schema)
        return responses

    def _cache_key(self, chunk: LLMChunk) -> LLMCacheKey:
        return build_llm_cache_key(
            task_kind=self.config.task_kind,
            provider_kind=self.config.provider_kind,
            provider_identity=self.config.provider_identity,
            base_url_identity=self.config.base_url_identity,
            model=self.config.model,
            temperature=self.config.temperature,
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            profile=self.config.profile,
            prompt_id=self.config.prompt_id,
            prompt_version=self.config.prompt_version,
            prompt_content_sha256=self.config.prompt_content_sha256,
            items=tuple(LLMItem(item.id, item.text) for item in chunk.items),
            context=tuple(LLMItem(item.id, item.text) for item in chunk.context),
            chunk_config=self.config.chunk_config(),
            response_schema_version=self.config.response_schema_version,
        )

    def _default_request(self, chunk: LLMChunk) -> LLMRequest:
        return LLMRequest(
            self.config.task_kind,
            tuple(LLMItem(item.id, item.text) for item in chunk.items),
            tuple(LLMItem(item.id, item.text) for item in chunk.context),
            self.config.source_language,
            self.config.target_language,
            self.config.prompt_id,
            self.config.prompt_version,
            self.config.prompt_content_sha256,
            self.config.prompt_content,
        )

    def _validate_and_order(
        self,
        value: object,
        chunk: LLMChunk,
    ) -> tuple[object, ...]:
        responses = _responses_from_batch(value)
        expected_ids = chunk.item_ids
        source_texts = {item.id: item.text for item in chunk.items}
        validated = validate_responses(
            responses,
            expected_ids,
            context_ids=chunk.context_ids,
            source_texts=source_texts,
            target_language=self.config.target_language,
        )
        by_id = {_response_id(response): response for response in validated}
        return tuple(by_id[item_id] for item_id in expected_ids)


def _responses_from_batch(value: object) -> tuple[object, ...]:
    if isinstance(value, StructuredResponseBatch):
        return value.responses
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(cast(Sequence[object], value))
    return (value,)


def _batch_from_responses(
    batch_schema: type[StructuredResponseBatch], responses: Sequence[object]
) -> StructuredResponseBatch:
    values: list[JsonValue] = []
    for response in responses:
        to_dict = getattr(response, "to_dict", None)
        if not callable(to_dict):
            raise AppError("llm.cache_value_invalid", {"reason": "response"})
        value = to_dict()
        if not isinstance(value, dict):
            raise AppError("llm.cache_value_invalid", {"reason": "response"})
        values.append(cast(dict[str, JsonValue], value))
    parser = getattr(batch_schema, "from_mapping", None)
    if not callable(parser):
        raise AppError("llm.schema_invalid", {"reason": "batch_schema"})
    result = parser(values)
    if not isinstance(result, StructuredResponseBatch):
        raise AppError("llm.cache_value_invalid", {"reason": "batch"})
    return result


def _response_id(response: object) -> str:
    value = getattr(response, "id", None)
    if not isinstance(value, str) or not value:
        if isinstance(response, Mapping):
            raw = cast(Mapping[str, object], response).get("id")
            if isinstance(raw, str) and raw:
                return raw
        raise AppError("llm.response_invalid", {"reason": "id"})
    return value


async def _cancel_tasks(tasks: Sequence[asyncio.Task[object]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
