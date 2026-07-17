"""Concurrent, cache-aware execution of independently validated LLM Chunks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from captioner.core.application.structured_llm_service import structured_repair_request
from captioner.core.domain.errors import AppError, LLMStructuredDecodeError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    LLM_RESPONSE_SCHEMA_VERSION,
    LLMItem,
    LLMRepairDiagnostic,
    LLMRequest,
    StructuredResponseBatch,
    response_batch_schema,
)
from captioner.core.domain.llm_cache import (
    LLMCacheKey,
    build_llm_cache_key_for_request,
)
from captioner.core.domain.result import JsonValue, thaw_json_value
from captioner.core.policies.llm_chunking import (
    ChunkingConfig,
    ChunkItem,
    LLMChunk,
    SerializedRequestTokenEstimator,
    validate_request_budget,
)
from captioner.core.policies.llm_validation import validate_response_schema, validate_responses
from captioner.core.ports.llm import LLMClient
from captioner.core.ports.llm_cache import LLMCachePort

type ChunkRequestFactory = Callable[[LLMChunk], LLMRequest]
type ChunkSemanticValidator[T] = Callable[[LLMChunk, tuple[T, ...]], tuple[T, ...]]
type AggregateSemanticValidator[T] = Callable[[tuple[T, ...]], tuple[T, ...]]


class ChunkPlannerPort(Protocol):
    def plan(
        self,
        items: Sequence[ChunkItem],
        config: ChunkingConfig | None = None,
    ) -> tuple[LLMChunk, ...]: ...

    def plan_range(
        self,
        items: Sequence[ChunkItem],
        core_start: int,
        core_end: int,
        config: ChunkingConfig | None = None,
        index: int = 0,
    ) -> LLMChunk: ...


_ID_MISMATCH_ERRORS = frozenset(
    {
        "llm.id_mismatch",
        "llm.missing_id",
        "llm.extra_id",
        "llm.duplicate_id",
        "llm.context_id_returned",
    }
)
_SHRINK_ERRORS = _ID_MISMATCH_ERRORS | frozenset({"llm.output_truncated"})
_VALIDATION_REPAIR_ERRORS = frozenset(
    {
        "llm.schema_invalid",
        "llm.response_invalid",
        "llm.empty_text",
        "llm.non_canonical_text",
        "llm.wrong_language",
        "llm.protected_token_lost",
        "llm.terminology_invalid",
        "llm.terminology_conflict",
        "llm.correction_units_invalid",
    }
)
_LLM_STAGE_VERSIONS = {
    "correct_source": "correct-source-v2",
    "terminology": "correct-source-v2",
    "translate_fast": "translate-v2",
    "translate_quality": "translate-quality-v2",
    "review": "review-v2",
}
_PROTECTED_FACT_POLICY_VERSION = "protected-facts-v2"


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
    repair_prompt_id: str = ""
    repair_prompt_version: str = ""
    repair_prompt_content_sha256: str = ""
    repair_prompt_content: str = ""
    tokenizer: str = "cl100k_base"
    stage_version: str = ""
    protected_fact_policy_version: str = _PROTECTED_FACT_POLICY_VERSION
    context_payload_factory: Callable[[LLMChunk], Mapping[str, JsonValue]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.stage_version:
            object.__setattr__(
                self,
                "stage_version",
                _LLM_STAGE_VERSIONS.get(self.task_kind, f"llm-{self.task_kind}-v2"),
            )
        if not self.protected_fact_policy_version:
            object.__setattr__(
                self,
                "protected_fact_policy_version",
                _PROTECTED_FACT_POLICY_VERSION,
            )

    def chunk_config(self) -> dict[str, JsonValue]:
        return {
            "max_items": self.chunking.max_items,
            "max_input_tokens": self.chunking.max_input_tokens,
            "context_before_items": self.chunking.context_before_items,
            "context_after_items": self.chunking.context_after_items,
            "max_audio_context_duration_ms": self.chunking.max_audio_context_duration_ms,
            "stage_version": self.stage_version,
            "protected_fact_policy_version": self.protected_fact_policy_version,
        }


@dataclass(slots=True)
class LLMChunkExecutor:
    client: LLMClient
    cache: LLMCachePort
    planner: ChunkPlannerPort
    config: LLMChunkExecutionConfig
    # Keys written during this execute() call; used for aggregate cleanup.
    _written_keys: list[LLMCacheKey] = field(
        default_factory=lambda: list[LLMCacheKey](), init=False, repr=False
    )

    async def execute[T](
        self,
        items: Sequence[ChunkItem],
        response_schema: type[T],
        context: ExecutionContext | None = None,
        request_factory: ChunkRequestFactory | None = None,
        validation_source_texts: Mapping[str, str] | None = None,
        semantic_validator: ChunkSemanticValidator[T] | None = None,
        aggregate_validator: AggregateSemanticValidator[T] | None = None,
    ) -> tuple[T, ...]:
        execution = ExecutionContext() if context is None else context
        ordered = tuple(items)
        factory = request_factory or self._default_request
        batch_schema = response_batch_schema(response_schema)
        chunks = self._plan(ordered, factory, batch_schema)
        if not chunks:
            return ()
        self._written_keys = []
        object_semantic = cast(
            ChunkSemanticValidator[object] | None,
            semantic_validator,
        )
        tasks = [
            asyncio.create_task(
                self._execute_with_shrink(
                    chunk,
                    ordered,
                    batch_schema,
                    response_schema,
                    factory,
                    execution,
                    validation_source_texts,
                    object_semantic,
                )
            )
            for chunk in chunks
        ]
        try:
            chunk_results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await _cancel_tasks(tasks)
            self._cleanup_written_keys()
            raise
        except Exception:
            await _cancel_tasks(tasks)
            self._cleanup_written_keys()
            raise
        by_id: dict[str, object] = {}
        for result in chunk_results:
            for response in result:
                response_id = _response_id(response)
                if response_id in by_id:
                    self._cleanup_written_keys()
                    raise AppError("llm.duplicate_id", {"id": response_id})
                by_id[response_id] = response
        expected_ids = tuple(item.id for item in ordered)
        if set(by_id) != set(expected_ids):
            self._cleanup_written_keys()
            raise AppError("llm.missing_id", {"ids": list(expected_ids)})
        ordered_results = tuple(cast(T, by_id[item_id]) for item_id in expected_ids)
        if aggregate_validator is not None:
            try:
                ordered_results = aggregate_validator(ordered_results)
            except AppError:
                self._cleanup_written_keys()
                raise
        return ordered_results

    def _cleanup_written_keys(self) -> None:
        """Remove only cache entries written during this execute() call."""
        failures: list[str] = []
        for key in self._written_keys:
            try:
                self.cache.remove(key)
            except Exception as exc:
                failures.append(key.digest)
                del exc
        self._written_keys = []
        if failures:
            raise AppError(
                "llm.cache_cleanup_failed",
                {"keys": cast(list[JsonValue], failures)},
            )

    async def _execute_with_shrink(
        self,
        chunk: LLMChunk,
        all_items: tuple[ChunkItem, ...],
        batch_schema: type[StructuredResponseBatch],
        item_schema: type[object],
        request_factory: ChunkRequestFactory,
        context: ExecutionContext,
        validation_source_texts: Mapping[str, str] | None,
        semantic_validator: ChunkSemanticValidator[object] | None,
    ) -> tuple[object, ...]:
        try:
            return await self._execute_chunk(
                chunk,
                batch_schema,
                item_schema,
                request_factory,
                context,
                validation_source_texts,
                semantic_validator,
            )
        except AppError as exc:
            if exc.code not in _SHRINK_ERRORS or len(chunk.items) < 2:
                raise
            positions = {item.id: index for index, item in enumerate(all_items)}
            indexes = tuple(positions[item.id] for item in chunk.items)
            if indexes != tuple(range(indexes[0], indexes[-1] + 1)):
                raise AppError("llm.id_mismatch", {"reason": "non_contiguous_chunk"}) from exc
            midpoint = indexes[0] + len(indexes) // 2
            left = self._plan_range(
                all_items,
                indexes[0],
                midpoint,
                request_factory,
                batch_schema,
                chunk.index * 2,
            )
            right = self._plan_range(
                all_items,
                midpoint,
                indexes[-1] + 1,
                request_factory,
                batch_schema,
                chunk.index * 2 + 1,
            )
            left_result = await self._execute_with_shrink(
                left,
                all_items,
                batch_schema,
                item_schema,
                request_factory,
                context,
                validation_source_texts,
                semantic_validator,
            )
            right_result = await self._execute_with_shrink(
                right,
                all_items,
                batch_schema,
                item_schema,
                request_factory,
                context,
                validation_source_texts,
                semantic_validator,
            )
            return (*left_result, *right_result)

    async def _execute_chunk(
        self,
        chunk: LLMChunk,
        batch_schema: type[StructuredResponseBatch],
        item_schema: type[object],
        request_factory: ChunkRequestFactory,
        context: ExecutionContext,
        validation_source_texts: Mapping[str, str] | None,
        semantic_validator: ChunkSemanticValidator[object] | None,
    ) -> tuple[object, ...]:
        request = request_factory(chunk)
        if request.item_ids != chunk.item_ids or request.context_ids != chunk.context_ids:
            raise AppError("llm.request_factory_invalid", {"reason": "ids"})
        context.raise_if_cancelled()
        key = self._cache_key(request, batch_schema)
        cached = self.cache.get(key, batch_schema)
        if cached is not None:
            try:
                return self._validate_and_order(
                    cached,
                    item_schema,
                    chunk,
                    validation_source_texts,
                    semantic_validator,
                )
            except AppError:
                self.cache.remove(key)
        estimator = self._estimator()
        validate_request_budget(
            request,
            batch_schema,
            estimator,
            self.config.chunking.max_input_tokens,
            request_kind=request.task_kind,
        )
        generated: object | None = None
        try:
            generated = await self.client.generate_structured(request, batch_schema, context)
            responses = self._validate_and_order(
                generated,
                item_schema,
                chunk,
                validation_source_texts,
                semantic_validator,
            )
        except AppError as exc:
            if exc.code not in _VALIDATION_REPAIR_ERRORS:
                raise
            context.raise_if_cancelled()
            invalid_response = _invalid_response_candidate(exc, generated)
            repair_request = structured_repair_request(
                request,
                invalid_response=invalid_response,
                diagnostics=(_repair_diagnostic(exc),),
                repair_prompt_id=self.config.repair_prompt_id,
                repair_prompt_version=self.config.repair_prompt_version,
                repair_prompt_content_sha256=self.config.repair_prompt_content_sha256,
                repair_prompt_content=self.config.repair_prompt_content,
            )
            try:
                validate_request_budget(
                    repair_request,
                    batch_schema,
                    estimator,
                    self.config.chunking.max_input_tokens,
                    request_kind=repair_request.task_kind,
                )
            except AppError as budget_error:
                if budget_error.code != "llm.item_too_large":
                    raise
                budget_params: dict[str, JsonValue] = {}
                for key in ("estimated_tokens", "max_input_tokens"):
                    if key in budget_error.params:
                        budget_params[key] = thaw_json_value(budget_error.params[key])
                raise AppError("llm.repair_budget_exceeded", budget_params) from budget_error
            repaired = await self.client.generate_structured(
                repair_request,
                batch_schema,
                context,
            )
            responses = self._validate_and_order(
                repaired,
                item_schema,
                chunk,
                validation_source_texts,
                semantic_validator,
            )
        cache_value = _batch_from_responses(batch_schema, responses)
        context.raise_if_cancelled()
        self.cache.put(key, cache_value, batch_schema)
        self._written_keys.append(key)
        return responses

    def _estimator(self) -> SerializedRequestTokenEstimator:
        token_counter = getattr(self.planner, "token_counter", None)
        if token_counter is None:
            raise AppError("llm.tokenizer_unknown")
        return SerializedRequestTokenEstimator(
            token_counter,
            self.config.model,
            self.config.temperature,
            response_schema_version=self.config.response_schema_version,
        )

    def _cache_key(
        self, request: LLMRequest, response_schema: type[StructuredResponseBatch]
    ) -> LLMCacheKey:
        return build_llm_cache_key_for_request(
            request,
            provider_kind=self.config.provider_kind,
            provider_identity=self.config.provider_identity,
            base_url_identity=self.config.base_url_identity,
            model=self.config.model,
            temperature=self.config.temperature,
            profile=self.config.profile,
            chunk_config=self.config.chunk_config(),
            response_schema_version=self.config.response_schema_version,
            response_schema=response_schema,
            tokenizer=self.config.tokenizer,
        )

    def _default_request(self, chunk: LLMChunk) -> LLMRequest:
        context_payload = (
            None
            if self.config.context_payload_factory is None
            else self.config.context_payload_factory(chunk)
        )
        return LLMRequest(
            task_kind=self.config.task_kind,
            items=tuple(LLMItem(item.id, item.text) for item in chunk.items),
            context=tuple(LLMItem(item.id, item.text) for item in chunk.context),
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            prompt_id=self.config.prompt_id,
            prompt_version=self.config.prompt_version,
            prompt_content_sha256=self.config.prompt_content_sha256,
            prompt_content=self.config.prompt_content,
            context_payload=context_payload,
            repair_prompt_id=self.config.repair_prompt_id,
            repair_prompt_version=self.config.repair_prompt_version,
            repair_prompt_content_sha256=self.config.repair_prompt_content_sha256,
            repair_prompt_content=self.config.repair_prompt_content,
        )

    def _plan(
        self,
        items: Sequence[ChunkItem],
        request_factory: ChunkRequestFactory,
        response_schema: type[StructuredResponseBatch],
    ) -> tuple[LLMChunk, ...]:
        estimator = self._estimator()
        method = getattr(self.planner, "plan_for_request", None)
        if callable(method):
            typed_method = cast(
                Callable[
                    [
                        Sequence[ChunkItem],
                        ChunkingConfig,
                        ChunkRequestFactory,
                        type[object],
                        SerializedRequestTokenEstimator,
                    ],
                    tuple[LLMChunk, ...],
                ],
                method,
            )
            return typed_method(
                items,
                self.config.chunking,
                request_factory,
                response_schema,
                estimator,
            )
        return self.planner.plan(items, self.config.chunking)

    def _plan_range(
        self,
        items: Sequence[ChunkItem],
        core_start: int,
        core_end: int,
        request_factory: ChunkRequestFactory,
        response_schema: type[StructuredResponseBatch],
        index: int,
    ) -> LLMChunk:
        estimator = self._estimator()
        method = getattr(self.planner, "plan_range_for_request", None)
        if callable(method):
            typed_method = cast(
                Callable[..., LLMChunk],
                method,
            )
            return typed_method(
                items,
                core_start,
                core_end,
                self.config.chunking,
                request_factory,
                response_schema,
                estimator,
                index,
            )
        return self.planner.plan_range(
            items, core_start, core_end, self.config.chunking, index=index
        )

    def _validate_and_order(
        self,
        value: object,
        item_schema: type[object],
        chunk: LLMChunk,
        validation_source_texts: Mapping[str, str] | None,
        semantic_validator: ChunkSemanticValidator[object] | None,
    ) -> tuple[object, ...]:
        responses = tuple(
            validate_response_schema(_response_mapping(response), item_schema)
            for response in _responses_from_batch(value)
        )
        expected_ids = chunk.item_ids
        source_texts = {
            item.id: (
                item.text
                if validation_source_texts is None
                else validation_source_texts.get(item.id, item.text)
            )
            for item in chunk.items
        }
        validated = validate_responses(
            responses,
            expected_ids,
            context_ids=chunk.context_ids,
            source_texts=source_texts,
            target_language=self.config.target_language,
        )
        by_id = {_response_id(response): response for response in validated}
        ordered = tuple(by_id[item_id] for item_id in expected_ids)
        if semantic_validator is not None:
            ordered = semantic_validator(chunk, ordered)
        return ordered


def _responses_from_batch(value: object) -> tuple[object, ...]:
    if isinstance(value, StructuredResponseBatch):
        return value.responses
    raise AppError("llm.schema_invalid", {"reason": "batch_object"})


def _response_mapping(response: object) -> object:
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return response


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
    result = parser({"responses": values})
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


def _invalid_response_candidate(error: AppError, generated: object | None) -> str:
    if isinstance(error, LLMStructuredDecodeError):
        return error.raw_content
    if generated is None:
        raise error
    to_dict = getattr(generated, "to_dict", None)
    if not callable(to_dict):
        raise error
    candidate = to_dict()
    if not isinstance(candidate, Mapping):
        raise error
    try:
        return json.dumps(
            candidate,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as serialization_error:
        raise error from serialization_error


def _repair_diagnostic(error: AppError) -> LLMRepairDiagnostic:
    params = error.params
    item_id = _safe_string(params.get("id")) or _safe_string(params.get("item_id"))
    field = _safe_string(params.get("field"))
    reason = _safe_string(params.get("reason"))
    expected_kind = _safe_string(params.get("expected_kind"))
    actual_kind = _safe_string(params.get("actual_kind"))
    raw_position = params.get("position")
    position = raw_position if type(raw_position) is int and raw_position >= 0 else None
    return LLMRepairDiagnostic(
        code=error.code,
        item_id=item_id,
        field=field,
        reason=reason,
        expected_kind=expected_kind,
        actual_kind=actual_kind,
        position=position,
    )


def _safe_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
