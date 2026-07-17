"""Deterministic anomaly selection for quality subtitle review."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLMRequest
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.terminology import Terminology, contains_term
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner, LLMChunk
from captioner.core.policies.llm_validation import (
    is_obvious_wrong_language,
)
from captioner.core.policies.protected_spans import protected_tokens_preserved
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import join_token_texts, measure_text, normalize_text
from captioner.core.ports.token_counter import LLMRequestEstimator, TokenCounter


@dataclass(frozen=True, slots=True)
class SubtitleAnomaly:
    cue_id: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.cue_id.strip() or not self.reasons:
            raise ValueError
        reasons = tuple(dict.fromkeys(self.reasons))
        object.__setattr__(self, "reasons", reasons)


class AnomalyChunkPlanner:
    """Plan anomaly outputs while retaining adjacent cues as context only."""

    def __init__(self, token_counter: TokenCounter, all_items: Sequence[ChunkItem]) -> None:
        self._token_counter = token_counter
        self._all_items = tuple(all_items)
        self._positions = {item.id: index for index, item in enumerate(self._all_items)}

    @property
    def token_counter(self) -> TokenCounter:
        return self._token_counter

    def plan(
        self,
        items: Sequence[ChunkItem],
        config: ChunkingConfig | None = None,
    ) -> tuple[LLMChunk, ...]:
        selected = ChunkingConfig() if config is None else config
        core_config = replace(selected, context_before_items=0, context_after_items=0)
        core_chunks = ChunkPlanner(self._token_counter, core_config).plan(items, core_config)
        return tuple(self._with_context(chunk, selected) for chunk in core_chunks)

    def plan_range(
        self,
        items: Sequence[ChunkItem],
        core_start: int,
        core_end: int,
        config: ChunkingConfig | None = None,
        index: int = 0,
    ) -> LLMChunk:
        selected = ChunkingConfig() if config is None else config
        core_config = replace(selected, context_before_items=0, context_after_items=0)
        chunk = ChunkPlanner(self._token_counter, core_config).plan_range(
            items, core_start, core_end, core_config, index
        )
        return self._with_context(chunk, selected)

    def plan_for_request(
        self,
        items: Sequence[ChunkItem],
        config: ChunkingConfig,
        request_factory: Callable[[LLMChunk], LLMRequest],
        response_schema: type[object],
        estimator: LLMRequestEstimator,
    ) -> tuple[LLMChunk, ...]:
        core_config = replace(config, context_before_items=0, context_after_items=0)
        core_planner = ChunkPlanner(self._token_counter, core_config)
        core_chunks = core_planner.plan_for_request(
            items,
            core_config,
            lambda chunk: request_factory(
                self._with_context(chunk, config, request_factory, response_schema, estimator)
            ),
            response_schema,
            estimator,
        )
        return tuple(
            self._with_context(chunk, config, request_factory, response_schema, estimator)
            for chunk in core_chunks
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
        core_config = replace(config, context_before_items=0, context_after_items=0)
        core_planner = ChunkPlanner(self._token_counter, core_config)
        core_chunk = core_planner.plan_range_for_request(
            items,
            core_start,
            core_end,
            core_config,
            lambda chunk: request_factory(
                self._with_context(chunk, config, request_factory, response_schema, estimator)
            ),
            response_schema,
            estimator,
            index,
        )
        return self._with_context(core_chunk, config, request_factory, response_schema, estimator)

    def _with_context(
        self,
        chunk: LLMChunk,
        config: ChunkingConfig,
        request_factory: Callable[[LLMChunk], LLMRequest] | None = None,
        response_schema: type[object] | None = None,
        estimator: LLMRequestEstimator | None = None,
    ) -> LLMChunk:
        positions = [self._positions[item.id] for item in chunk.items]
        first = min(positions)
        last = max(positions) + 1
        before = list(self._all_items[max(0, first - config.context_before_items) : first])
        after = list(self._all_items[last : last + config.context_after_items])
        core_tokens = sum(self._token_counter.count(item.text) for item in chunk.items)
        while True:
            context = tuple(before + after)
            total_tokens = core_tokens + sum(
                self._token_counter.count(item.text) for item in context
            )
            complete_tokens = total_tokens
            if request_factory is not None:
                if response_schema is None or estimator is None:
                    raise AppError("llm.chunk_config_invalid", {"reason": "request_budget"})
                complete_tokens = estimator.estimate_input_tokens(
                    request_factory(LLMChunk(chunk.index, chunk.items, context)), response_schema
                )
            if complete_tokens <= config.max_input_tokens and self._within_audio_budget(
                chunk.items, context, config
            ):
                return LLMChunk(chunk.index, chunk.items, context)
            if not before and not after:
                raise AppError("llm.item_too_large", {"item_id": chunk.items[0].id})
            if before:
                before.pop(0)
            else:
                after.pop()

    @staticmethod
    def _within_audio_budget(
        core: Sequence[ChunkItem], context: Sequence[ChunkItem], config: ChunkingConfig
    ) -> bool:
        if config.max_audio_context_duration_ms is None:
            return True
        window = (*context, *core)
        if not window:
            return True
        return max(item.end_ms for item in window) - min(item.start_ms for item in window) <= (
            config.max_audio_context_duration_ms
        )


def detect_anomalies(
    track: SubtitleTrack,
    transcript: Transcript,
    target_language: str,
    config: SegmentationPolicyConfig,
    terminology: Terminology | None = None,
) -> tuple[SubtitleAnomaly, ...]:
    """Return only cues needing a review request, in track order."""
    terms = () if terminology is None else terminology.entries
    words = {word.id: word for word in transcript.words}
    anomalies: list[SubtitleAnomaly] = []
    for cue in track.cues:
        reasons: list[str] = []
        translated = cue.translated_text
        if translated is None or not translated.strip():
            reasons.append("empty_translation")
        else:
            if is_obvious_wrong_language(translated, target_language):
                reasons.append("wrong_language")
            original_source = join_token_texts(
                words[word_id].text for word_id in cue.source_word_ids if word_id in words
            )
            if not protected_tokens_preserved(original_source or cue.source_text, translated):
                reasons.append("protected_token_loss")
            try:
                if normalize_text(translated) != translated:
                    reasons.append("non_canonical_text")
            except AppError:
                # The domain normally rejects controls before this policy runs;
                # malformed recovered data is still selected for review.
                reasons.append("invalid_control_character")
            if _terminology_inconsistent(cue.source_text, translated, terms):
                reasons.append("terminology_inconsistent")
            speed = reading_speed(
                translated,
                cue.end_ms - cue.start_ms,
                target_cps_milli=config.target_cps_milli,
                max_cps_milli=config.max_cps_milli,
            )
            if speed.status in {"warning", "error"}:
                reasons.append("abnormal_reading_speed")
            if any(
                measure_text(line).display_columns > config.max_line_width for line in cue.lines
            ):
                reasons.append("abnormal_line_width")
        if reasons:
            anomalies.append(SubtitleAnomaly(cue.id, tuple(reasons)))
    return tuple(anomalies)


def _terminology_inconsistent(source: str, translated: str, terminology: Sequence[object]) -> bool:
    for entry in terminology:
        entry_source = getattr(entry, "source", "")
        entry_target = getattr(entry, "target", "")
        if (
            entry_source
            and contains_term(source, entry_source)
            and entry_target
            and not contains_term(translated, entry_target)
        ):
            return True
    return False
