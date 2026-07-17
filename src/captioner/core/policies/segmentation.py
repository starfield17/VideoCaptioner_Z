"""Bounded deterministic dynamic-programming subtitle segmentation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.domain.transcript import Transcript, WordToken
from captioner.core.policies.line_breaking import break_lines
from captioner.core.policies.protected_spans import (
    ProtectedSpan,
    find_protected_spans,
    protected_break_cost,
)
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import join_token_texts, measure_text, normalize_text

_SENTENCE_END = frozenset(".!?…\u3002\uff01\uff1f")
_CLAUSE_END = frozenset(",;:\u3001\uff0c\uff1b\uff1a")
_MAX_CANDIDATE_WORDS = 96
_Cost = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    end: int
    source_text: str
    duration_ms: int
    lines: tuple[str, ...]
    cost: _Cost


@dataclass(frozen=True, slots=True)
class _Path:
    cost: _Cost
    ends: tuple[int, ...]


def canonical_words(words: Sequence[WordToken]) -> tuple[WordToken, ...]:
    """Sort valid Words independently of input tuple order."""
    identifiers = [word.id for word in words]
    if len(set(identifiers)) != len(identifiers):
        raise AppError("subtitle.segmentation_failed", {"reason": "duplicate_word_id"})
    for word in words:
        normalize_text(word.text)
    return tuple(sorted(words, key=lambda word: (word.start_ms, word.end_ms, word.id)))


def segment_transcript_dp(
    transcript: Transcript,
    config: SegmentationPolicyConfig | None = None,
    progress: Callable[[], None] | None = None,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> SubtitleTrack:
    settings = SegmentationPolicyConfig() if config is None else config
    corrected = None if corrected_text_by_word_id is None else dict(corrected_text_by_word_id)
    if corrected is not None:
        expected_ids = {word.id for word in transcript.words}
        if set(corrected) != expected_ids:
            raise AppError("subtitle.segmentation_failed", {"reason": "corrected_mapping"})
        words_for_segmentation = tuple(
            replace(word, text=corrected[word.id]) for word in transcript.words
        )
    else:
        words_for_segmentation = transcript.words
    words = canonical_words(words_for_segmentation)
    partitions = _hard_partitions(words, settings.hard_gap_ms)
    spans: list[tuple[WordToken, ...]] = []
    progress_emitted = False
    for partition in partitions:
        selected, emitted = _solve_partition(
            partition,
            settings,
            progress if progress is not None and not progress_emitted else None,
        )
        spans.extend(selected)
        progress_emitted = progress_emitted or emitted

    cues: list[SubtitleCue] = []
    previous_end = 0
    for number, span in enumerate(spans, start=1):
        source_text = _join_words(span)
        lines = break_lines(source_text, settings)
        initial_start = min(word.start_ms for word in span)
        initial_end = max(word.end_ms for word in span)
        start_ms = max(0, initial_start, previous_end)
        end_ms = max(initial_end, start_ms + 1)
        cue = SubtitleCue(
            id=f"cue-{number:06d}",
            start_ms=start_ms,
            end_ms=end_ms,
            source_word_ids=tuple(word.id for word in span),
            source_text=source_text,
            translated_text=None,
            lines=lines,
        )
        cues.append(cue)
        previous_end = end_ms

    track_id = derive_subtitle_track_id(
        transcript.id,
        transcript.language,
        cues,
        settings.to_mapping(),
    )
    track = SubtitleTrack(
        id=track_id,
        source_transcript_id=transcript.id,
        language=transcript.language,
        cues=tuple(cues),
        revision=0,
        policy_signature=settings.signature,
    )
    report = validate_subtitle_track(
        track,
        transcript,
        settings,
        corrected_text_by_word_id=corrected,
    )
    if not report.is_valid:
        first = next(issue for issue in report.issues if issue.severity.value == "error")
        raise AppError("subtitle.validation_failed", {"reason": first.code})
    return track


def _hard_partitions(
    words: Sequence[WordToken], hard_gap_ms: int
) -> tuple[tuple[WordToken, ...], ...]:
    if not words:
        raise AppError("subtitle.segmentation_failed", {"reason": "empty_words"})
    partitions: list[tuple[WordToken, ...]] = []
    start = 0
    for index in range(1, len(words)):
        gap = words[index].start_ms - words[index - 1].end_ms
        if gap >= hard_gap_ms:
            partitions.append(tuple(words[start:index]))
            start = index
    partitions.append(tuple(words[start:]))
    return tuple(partitions)


def _solve_partition(
    words: tuple[WordToken, ...],
    config: SegmentationPolicyConfig,
    progress: Callable[[], None] | None,
) -> tuple[tuple[tuple[WordToken, ...], ...], bool]:
    count = len(words)
    full_text, boundary_offsets = _canonical_text_and_boundaries(words)
    protected_spans = find_protected_spans(full_text)
    best: list[_Path | None] = [None] * (count + 1)
    best[0] = _Path(_zero_cost(), ())
    progress_emitted = False
    for start in range(count):
        prior = best[start]
        if prior is None:
            continue
        for end in range(start + 1, min(count, start + _MAX_CANDIDATE_WORDS) + 1):
            candidate = _evaluate_candidate(
                words,
                start,
                end,
                full_text,
                boundary_offsets,
                protected_spans,
                config,
            )
            if progress is not None and not progress_emitted and end == start + 1 and count > 1:
                progress()
                progress_emitted = True
            proposed = _Path(_add_cost(prior.cost, candidate.cost), (*prior.ends, end))
            current = best[end]
            if current is None or (proposed.cost, proposed.ends) < (current.cost, current.ends):
                best[end] = proposed
            if _candidate_window_is_exceeded(start, end, candidate, config):
                break
    final = best[count]
    if final is None:
        raise AppError("subtitle.segmentation_failed", {"reason": "no_path"})
    spans: list[tuple[WordToken, ...]] = []
    start = 0
    for end in final.ends:
        spans.append(words[start:end])
        start = end
    return tuple(spans), progress_emitted


def _evaluate_candidate(
    words: tuple[WordToken, ...],
    start: int,
    end: int,
    full_text: str,
    boundary_offsets: tuple[int, ...],
    protected_spans: tuple[ProtectedSpan, ...],
    config: SegmentationPolicyConfig,
) -> _Candidate:
    selected = words[start:end]
    source_text = _join_words(selected)
    duration_ms = max(word.end_ms for word in selected) - min(word.start_ms for word in selected)
    duration_ms = max(duration_ms, 1)
    lines = break_lines(source_text, config)
    metrics = measure_text(source_text)
    reading_speed(
        source_text,
        duration_ms,
        target_cps_milli=config.target_cps_milli,
        max_cps_milli=config.max_cps_milli,
    )
    line_width_over = sum(
        max(0, measure_text(line).display_columns - config.max_line_width) for line in lines
    )
    duration_over = max(0, duration_ms - config.max_duration_ms)
    cps_over = max(
        0,
        metrics.reading_characters * 1_000_000 - config.max_cps_milli * duration_ms,
    )
    width_over = max(0, metrics.display_columns - config.max_cue_width) + line_width_over
    duration_under = max(0, config.min_duration_ms - duration_ms)
    boundary = boundary_offsets[end]
    protected = (
        protected_break_cost(full_text, boundary, protected_spans) if end < len(words) else 0
    )
    equal_timestamp_break = int(
        end < len(words)
        and words[end - 1].start_ms == words[end].start_ms
        and words[end - 1].end_ms == words[end].end_ms
        and metrics.display_columns <= config.max_cue_width
    )
    boundary_quality = 0
    silence_quality = 0
    if end < len(words):
        last_text = words[end - 1].text.rstrip()
        if last_text.endswith(tuple(_SENTENCE_END)):
            boundary_quality = 2
        elif last_text.endswith(tuple(_CLAUSE_END)):
            boundary_quality = 1
        gap = words[end].start_ms - words[end - 1].end_ms
        if gap >= config.preferred_gap_ms:
            silence_quality = 1
    line_balance = 0
    if len(lines) == 2:
        line_balance = abs(
            measure_text(lines[0]).display_columns - measure_text(lines[1]).display_columns
        )
    cost = (
        0,
        width_over * config.overflow_penalty + duration_over * config.overflow_penalty,
        equal_timestamp_break,
        protected * config.protected_break_penalty,
        cps_over,
        duration_over,
        width_over,
        -boundary_quality * config.punctuation_bonus - silence_quality * config.silence_bonus,
        abs(duration_ms - config.target_duration_ms),
        duration_under,
        line_balance,
        1,
    )
    return _Candidate(end, source_text, duration_ms, lines, cost)


def _candidate_window_is_exceeded(
    start: int,
    end: int,
    candidate: _Candidate,
    config: SegmentationPolicyConfig,
) -> bool:
    if end <= start + 1:
        return False
    if candidate.duration_ms > config.max_duration_ms:
        return True
    return measure_text(candidate.source_text).display_columns > config.max_cue_width


def _join_words(words: Sequence[WordToken]) -> str:
    return join_token_texts(word.text for word in words)


def _canonical_text_and_boundaries(
    words: Sequence[WordToken],
) -> tuple[str, tuple[int, ...]]:
    """Build canonical source text and word-end offsets in one pass.

    Word text is already a valid token, so normalizing each token and adding
    one separator when either adjacent source token contains whitespace is
    equivalent to normalizing their concatenation for the supported transcript
    representation. Keeping offsets here avoids a whole-prefix join for every
    dynamic-programming candidate.
    """
    pieces: list[str] = []
    offsets = [0]
    previous_raw: str | None = None
    for word in words:
        piece = normalize_text(word.text)
        separator = previous_raw is not None and (
            previous_raw[-1].isspace() or word.text[0].isspace()
        )
        if separator:
            pieces.append(" ")
        pieces.append(piece)
        offsets.append(offsets[-1] + len(piece) + int(separator))
        previous_raw = word.text
    return "".join(pieces), tuple(offsets)


def _zero_cost() -> _Cost:
    return (0,) * 12


def _add_cost(left: _Cost, right: _Cost) -> _Cost:
    return tuple(a + b for a, b in zip(left, right, strict=True))
