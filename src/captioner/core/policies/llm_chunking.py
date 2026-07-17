"""Deterministic bounded chunk planning for structured LLM tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLMRequest, encode_llm_request
from captioner.core.policies.unicode_metrics import normalize_text
from captioner.core.ports.token_counter import LLMRequestEstimator, TokenCounter


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_items: int = 32
    max_input_tokens: int = 4096
    context_before_items: int = 0
    context_after_items: int = 0
    max_audio_context_duration_ms: int | None = None

    def __post_init__(self) -> None:
        if self.max_items < 1 or self.max_input_tokens < 1:
            raise AppError("llm.chunk_config_invalid", {"reason": "positive_budget"})
        if self.context_before_items < 0 or self.context_after_items < 0:
            raise AppError("llm.chunk_config_invalid", {"reason": "context"})
        if (
            self.max_audio_context_duration_ms is not None
            and self.max_audio_context_duration_ms < 1
        ):
            raise AppError("llm.chunk_config_invalid", {"reason": "audio_duration"})


@dataclass(frozen=True, slots=True)
class ChunkItem:
    id: str
    text: str
    start_ms: int = 0
    end_ms: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip() or self.id != normalize_text(self.id):
            raise AppError("llm.chunk_item_invalid", {"field": "id"})
        if not self.text.strip() or self.text != normalize_text(self.text):
            raise AppError("llm.chunk_item_invalid", {"field": "text"})
        if (
            type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or self.start_ms < 0
            or self.end_ms < self.start_ms
        ):
            raise AppError("llm.chunk_item_invalid", {"field": "time"})

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class LLMChunk:
    index: int
    items: tuple[ChunkItem, ...]
    context: tuple[ChunkItem, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        context = tuple(self.context)
        if not items:
            raise AppError("llm.chunk_invalid", {"reason": "empty"})
        item_ids = tuple(item.id for item in items)
        context_ids = tuple(item.id for item in context)
        if len(set(item_ids)) != len(item_ids) or set(item_ids) & set(context_ids):
            raise AppError("llm.chunk_invalid", {"reason": "context_output_overlap"})
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "context", context)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)

    @property
    def output_ids(self) -> tuple[str, ...]:
        return self.item_ids

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.context)


ChunkPlan = LLMChunk


@dataclass(frozen=True, slots=True)
class SerializedRequestTokenEstimator:
    """Count the complete serialized request using an injected token counter."""

    token_counter: TokenCounter
    model: str
    temperature: float
    response_schema_version: int = 1

    def estimate_input_tokens(
        self,
        request: LLMRequest,
        response_schema: type[object],
    ) -> int:
        encoded = encode_llm_request(
            request,
            self.model,
            self.temperature,
            response_schema,
            response_schema_version=self.response_schema_version,
        )
        count = self.token_counter.count(encoded.decode("utf-8"))
        if type(count) is not int or count < 0:
            raise AppError("llm.token_count_invalid", {"item_id": request.item_ids[0]})
        return count


def validate_request_budget(
    request: LLMRequest,
    response_schema: type[object],
    estimator: LLMRequestEstimator,
    max_input_tokens: int,
    *,
    request_kind: str | None = None,
) -> int:
    """Fail closed before network access when a complete request exceeds budget.

    Parameters on the raised error are limited to ids, counts, and kind so that
    prompt content and source text never enter structured error payloads.
    """
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise AppError("llm.chunk_config_invalid", {"reason": "max_input_tokens"})
    estimated = estimator.estimate_input_tokens(request, response_schema)
    if type(estimated) is not int or estimated < 0:
        raise AppError(
            "llm.token_count_invalid",
            {"item_id": request.item_ids[0]},
        )
    if estimated > max_input_tokens:
        raise AppError(
            "llm.item_too_large",
            {
                "item_id": request.item_ids[0],
                "estimated_tokens": estimated,
                "max_input_tokens": max_input_tokens,
                "request_kind": request_kind or request.task_kind,
            },
        )
    return estimated


class ChunkPlanner:
    """Greedy, forward-only planner with deterministic context trimming."""

    def __init__(self, token_counter: TokenCounter, config: ChunkingConfig | None = None) -> None:
        self._token_counter = token_counter
        self._config = config or ChunkingConfig()

    @property
    def token_counter(self) -> TokenCounter:
        return self._token_counter

    def plan(
        self,
        items: Sequence[ChunkItem],
        config: ChunkingConfig | None = None,
    ) -> tuple[LLMChunk, ...]:
        selected = self._config if config is None else config
        return plan_chunks(items, selected, self._token_counter)

    def plan_for_request(
        self,
        items: Sequence[ChunkItem],
        config: ChunkingConfig,
        request_factory: Callable[[LLMChunk], LLMRequest],
        response_schema: type[object],
        estimator: LLMRequestEstimator,
    ) -> tuple[LLMChunk, ...]:
        return plan_chunks(
            items,
            config,
            self._token_counter,
            request_factory=request_factory,
            response_schema=response_schema,
            request_estimator=estimator,
        )

    def plan_range(
        self,
        items: Sequence[ChunkItem],
        core_start: int,
        core_end: int,
        config: ChunkingConfig | None = None,
        index: int = 0,
    ) -> LLMChunk:
        selected = self._config if config is None else config
        return plan_chunk_range(
            items, core_start, core_end, selected, self._token_counter, index=index
        )

    def plan_range_for_request(
        self,
        items: Sequence[ChunkItem],
        core_start: int,
        core_end: int,
        config: ChunkingConfig,
        request_factory: Callable[[LLMChunk], LLMRequest],
        response_schema: type[object],
        estimator: LLMRequestEstimator,
        index: int = 0,
    ) -> LLMChunk:
        return plan_chunk_range(
            items,
            core_start,
            core_end,
            config,
            self._token_counter,
            index=index,
            request_factory=request_factory,
            response_schema=response_schema,
            request_estimator=estimator,
        )


def plan_chunks(
    items: Sequence[ChunkItem],
    config: ChunkingConfig,
    token_counter: TokenCounter,
    *,
    request_factory: Callable[[LLMChunk], LLMRequest] | None = None,
    response_schema: type[object] | None = None,
    request_estimator: LLMRequestEstimator | None = None,
) -> tuple[LLMChunk, ...]:
    """Plan chunks without ever placing context IDs in the output set."""
    ordered = tuple(items)
    if not ordered:
        return ()
    ids = tuple(item.id for item in ordered)
    if len(set(ids)) != len(ids):
        raise AppError("llm.chunk_items_invalid", {"reason": "duplicate_ids"})
    token_counts = tuple(_count(token_counter, item.text, item.id) for item in ordered)
    for item, count in zip(ordered, token_counts, strict=True):
        if count > config.max_input_tokens:
            raise AppError("llm.item_too_large", {"item_id": item.id, "tokens": count})
        if (
            config.max_audio_context_duration_ms is not None
            and item.duration_ms > config.max_audio_context_duration_ms
        ):
            raise AppError(
                "llm.item_too_large",
                {"item_id": item.id, "duration_ms": item.duration_ms},
            )

    chunks: list[LLMChunk] = []
    start = 0
    while start < len(ordered):
        end = start + 1
        best: tuple[tuple[ChunkItem, ...], tuple[ChunkItem, ...]] | None = None
        while end <= len(ordered) and end - start <= config.max_items:
            core = ordered[start:end]
            candidate = _fit_window(
                ordered,
                core_start=start,
                core_end=end,
                token_counts=token_counts,
                config=config,
                request_factory=request_factory,
                response_schema=response_schema,
                request_estimator=request_estimator,
            )
            if candidate is None:
                break
            best = candidate
            end += 1
        if best is None:
            # A single item was checked above, so this branch denotes an
            # impossible planner state rather than a recoverable split.
            raise AppError("llm.item_too_large", {"item_id": ordered[start].id})
        core, context = best
        chunks.append(LLMChunk(len(chunks), core, context))
        start += len(core)
    return tuple(chunks)


def plan_chunk_range(
    items: Sequence[ChunkItem],
    core_start: int,
    core_end: int,
    config: ChunkingConfig,
    token_counter: TokenCounter,
    *,
    index: int = 0,
    request_factory: Callable[[LLMChunk], LLMRequest] | None = None,
    response_schema: type[object] | None = None,
    request_estimator: LLMRequestEstimator | None = None,
) -> LLMChunk:
    """Replan one contiguous core range with freshly computed context."""
    ordered = tuple(items)
    if core_start < 0 or core_end <= core_start or core_end > len(ordered) or index < 0:
        raise AppError("llm.chunk_invalid", {"reason": "range"})
    ids = tuple(item.id for item in ordered)
    if len(set(ids)) != len(ids):
        raise AppError("llm.chunk_items_invalid", {"reason": "duplicate_ids"})
    token_counts = tuple(_count(token_counter, item.text, item.id) for item in ordered)
    _validate_item_budgets(ordered, token_counts, config)
    candidate = _fit_window(
        ordered,
        core_start=core_start,
        core_end=core_end,
        token_counts=token_counts,
        config=config,
        request_factory=request_factory,
        response_schema=response_schema,
        request_estimator=request_estimator,
    )
    if candidate is None:
        raise AppError("llm.item_too_large", {"item_id": ordered[core_start].id})
    core, context = candidate
    return LLMChunk(index, core, context)


def _validate_item_budgets(
    ordered: tuple[ChunkItem, ...],
    token_counts: tuple[int, ...],
    config: ChunkingConfig,
) -> None:
    for item, count in zip(ordered, token_counts, strict=True):
        if count > config.max_input_tokens:
            raise AppError("llm.item_too_large", {"item_id": item.id, "tokens": count})
        if (
            config.max_audio_context_duration_ms is not None
            and item.duration_ms > config.max_audio_context_duration_ms
        ):
            raise AppError(
                "llm.item_too_large",
                {"item_id": item.id, "duration_ms": item.duration_ms},
            )


def _fit_window(
    ordered: tuple[ChunkItem, ...],
    *,
    core_start: int,
    core_end: int,
    token_counts: tuple[int, ...],
    config: ChunkingConfig,
    request_factory: Callable[[LLMChunk], LLMRequest] | None = None,
    response_schema: type[object] | None = None,
    request_estimator: LLMRequestEstimator | None = None,
) -> tuple[tuple[ChunkItem, ...], tuple[ChunkItem, ...]] | None:
    core = ordered[core_start:core_end]
    if sum(token_counts[core_start:core_end]) > config.max_input_tokens:
        return None
    before = list(ordered[max(0, core_start - config.context_before_items) : core_start])
    after = list(ordered[core_end : core_end + config.context_after_items])
    while True:
        context = tuple(before + after)
        total_tokens = sum(token_counts[core_start:core_end]) + sum(
            token_counts[index]
            for index in _context_indexes(ordered, core_start, core_end, context, token_counts)
        )
        complete_tokens = total_tokens
        if request_factory is not None:
            if response_schema is None or request_estimator is None:
                raise AppError("llm.chunk_config_invalid", {"reason": "request_budget"})
            candidate_chunk = LLMChunk(0, tuple(core), context)
            request = request_factory(candidate_chunk)
            complete_tokens = request_estimator.estimate_input_tokens(request, response_schema)
        if complete_tokens <= config.max_input_tokens and _within_audio_budget(
            core, context, config
        ):
            return tuple(core), context
        if not before and not after:
            return None
        # Remove the farthest context item first, retaining the nearest
        # context when a budget cannot hold the complete requested window.
        if before:
            before.pop(0)
        else:
            after.pop()


def _context_indexes(
    ordered: tuple[ChunkItem, ...],
    core_start: int,
    core_end: int,
    context: tuple[ChunkItem, ...],
    token_counts: tuple[int, ...],
) -> tuple[int, ...]:
    del token_counts
    positions = {item.id: index for index, item in enumerate(ordered)}
    indexes = tuple(positions[item.id] for item in context)
    if any(core_start <= index < core_end for index in indexes):
        raise AppError("llm.chunk_invalid", {"reason": "context_output_overlap"})
    return indexes


def _within_audio_budget(
    core: tuple[ChunkItem, ...], context: tuple[ChunkItem, ...], config: ChunkingConfig
) -> bool:
    limit = config.max_audio_context_duration_ms
    if limit is None:
        return True
    window = (*context, *core)
    if not window:
        return True
    return max(item.end_ms for item in window) - min(item.start_ms for item in window) <= limit


def _count(token_counter: TokenCounter, text: str, item_id: str) -> int:
    count = token_counter.count(text)
    if type(count) is not int or count < 0:
        raise AppError("llm.token_count_invalid", {"item_id": item_id})
    return count
