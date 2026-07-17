"""Pure assembly and deterministic merge rules for quality source correction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import SourceCorrectionResponse, TerminologyResponse
from captioner.core.domain.terminology import Terminology, TerminologyEntry, normalize_term
from captioner.core.domain.transcript import CorrectedSpan, CorrectedTranscript, Transcript
from captioner.core.policies.protected_spans import protected_tokens_preserved
from captioner.core.policies.segmentation import canonical_words
from captioner.core.policies.unicode_metrics import join_token_texts


@dataclass(frozen=True, slots=True)
class TerminologyUnit:
    id: str
    word_ids: tuple[str, ...]
    word_texts: tuple[str, ...]
    text: str
    start_ms: int
    end_ms: int


def build_corrected_transcript(
    transcript: Transcript,
    responses: Sequence[object],
) -> CorrectedTranscript:
    """Copy validated model text onto stable, one-Word correction units."""
    response_by_id = _source_responses_by_id(responses)
    words = canonical_words(transcript.words)
    expected_ids = tuple(word.id for word in words)
    if tuple(response_by_id) != expected_ids:
        raise AppError("llm.correction_units_invalid", {"reason": "ids"})
    spans: list[CorrectedSpan] = []
    for word in words:
        response = response_by_id[word.id]
        corrected_text = response.corrected_source
        _ensure_protected_numbers(word.text, corrected_text, word.id)
        spans.append(CorrectedSpan((word.id,), corrected_text))
    return CorrectedTranscript(transcript.id, tuple(spans), expected_ids)


def merge_terminology(
    transcript: Transcript,
    source_language: str,
    target_language: str,
    responses: Sequence[object],
    units: Sequence[TerminologyUnit] | None = None,
) -> Terminology:
    """Merge sparse unit responses using application-owned token matching."""
    response_by_id = _terminology_responses_by_id(responses)
    selected_units = tuple(build_terminology_units(transcript) if units is None else units)
    expected_ids = tuple(unit.id for unit in selected_units)
    if tuple(response_by_id) != expected_ids:
        raise AppError("llm.terminology_units_invalid", {"reason": "ids"})
    entries_by_source: dict[str, TerminologyEntry] = {}
    first_position: dict[str, int] = {}
    word_positions = {
        word.id: index for index, word in enumerate(canonical_words(transcript.words))
    }
    for unit in selected_units:
        response = response_by_id[unit.id]
        for term in response.terms:
            matched_ids = _match_term_to_word_ids(term.source_term, unit)
            if not matched_ids:
                raise AppError("llm.terminology_invalid", {"reason": "term_not_in_unit"})
            _ensure_protected_numbers(term.source_term, term.target_term, unit.id)
            normalized_source = normalize_term(term.source_term)
            existing = entries_by_source.get(normalized_source)
            if existing is None:
                entries_by_source[normalized_source] = TerminologyEntry(
                    term.source_term,
                    normalized_source,
                    term.target_term,
                    matched_ids,
                )
                first_position[normalized_source] = min(
                    word_positions[word_id] for word_id in matched_ids
                )
                continue
            if normalize_term(existing.target) != normalize_term(term.target_term):
                raise AppError("llm.terminology_conflict", {"source": existing.source})
            merged_ids = tuple(dict.fromkeys((*existing.source_word_ids, *matched_ids)))
            entries_by_source[normalized_source] = TerminologyEntry(
                existing.source,
                existing.normalized_source,
                existing.target,
                merged_ids,
            )
    entries = tuple(
        entry
        for _, entry in sorted(entries_by_source.items(), key=lambda pair: first_position[pair[0]])
    )
    return Terminology(transcript.id, source_language, target_language, entries)


def build_terminology_units(transcript: Transcript) -> tuple[TerminologyUnit, ...]:
    """Build stable consecutive units from transcript segments."""
    words = {word.id: word for word in canonical_words(transcript.words)}
    units: list[TerminologyUnit] = []
    for index, segment in enumerate(transcript.segments, start=1):
        segment_words = tuple(words[word_id] for word_id in segment.word_ids if word_id in words)
        if not segment_words:
            continue
        units.append(
            TerminologyUnit(
                f"term-unit-{index:06d}",
                tuple(word.id for word in segment_words),
                tuple(word.text for word in segment_words),
                join_token_texts(word.text for word in segment_words),
                segment_words[0].start_ms,
                segment_words[-1].end_ms,
            )
        )
    if units:
        return tuple(units)
    ordered = tuple(words.values())
    if not ordered:
        return ()
    return (
        TerminologyUnit(
            "term-unit-000001",
            tuple(word.id for word in ordered),
            tuple(word.text for word in ordered),
            join_token_texts(word.text for word in ordered),
            ordered[0].start_ms,
            ordered[-1].end_ms,
        ),
    )


def _match_term_to_word_ids(term: str, unit: TerminologyUnit) -> tuple[str, ...]:
    term_tokens = _lexical_tokens(term)
    if not term_tokens:
        return ()
    expanded: list[tuple[str, str]] = []
    for word_id, word_text in zip(unit.word_ids, unit.word_texts, strict=True):
        expanded.extend((token, word_id) for token in _lexical_tokens(word_text))
    matches: list[str] = []
    width = len(term_tokens)
    for start in range(0, len(expanded) - width + 1):
        if tuple(token for token, _ in expanded[start : start + width]) == term_tokens:
            matches.extend(word_id for _, word_id in expanded[start : start + width])
    return tuple(dict.fromkeys(matches))


def _lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def _source_responses_by_id(
    responses: Sequence[object],
) -> dict[str, SourceCorrectionResponse]:
    result: dict[str, SourceCorrectionResponse] = {}
    for response in responses:
        if not isinstance(response, SourceCorrectionResponse):
            raise AppError("llm.response_invalid", {"reason": "response_type"})
        response_id = response.id
        if response_id in result:
            raise AppError("llm.duplicate_id", {"id": response_id})
        result[response_id] = response
    return result


def _terminology_responses_by_id(
    responses: Sequence[object],
) -> dict[str, TerminologyResponse]:
    result: dict[str, TerminologyResponse] = {}
    for response in responses:
        if not isinstance(response, TerminologyResponse):
            raise AppError("llm.response_invalid", {"reason": "response_type"})
        response_id = response.id
        if response_id in result:
            raise AppError("llm.duplicate_id", {"id": response_id})
        result[response_id] = response
    return result


def validate_terminology_chunk(
    units_by_id: Mapping[str, TerminologyUnit],
    chunk: object,
    responses: Sequence[object],
) -> tuple[object, ...]:
    """Chunk-local terminology checks that must pass before cache put."""
    del chunk
    ordered = tuple(responses)
    for response in ordered:
        if not isinstance(response, TerminologyResponse):
            raise AppError("llm.response_invalid", {"reason": "response_type"})
        unit = units_by_id.get(response.id)
        if unit is None:
            raise AppError("llm.terminology_invalid", {"reason": "unit_id", "id": response.id})
        for term in response.terms:
            if not term.source_term.strip() or not term.target_term.strip():
                raise AppError("llm.terminology_invalid", {"reason": "empty_term"})
            matched = _match_term_to_word_ids(term.source_term, unit)
            if not matched:
                raise AppError("llm.terminology_invalid", {"reason": "term_not_in_unit"})
            _ensure_protected_numbers(term.source_term, term.target_term, unit.id)
    return ordered


def validate_terminology_aggregate(
    units: Sequence[TerminologyUnit],
    responses: Sequence[object],
) -> tuple[object, ...]:
    """Cross-chunk conflict detection after all chunks validate locally."""
    units_by_id = {unit.id: unit for unit in units}
    targets_by_source: dict[str, str] = {}
    for response in responses:
        if not isinstance(response, TerminologyResponse):
            raise AppError("llm.response_invalid", {"reason": "response_type"})
        unit = units_by_id.get(response.id)
        if unit is None:
            raise AppError("llm.terminology_invalid", {"reason": "unit_id", "id": response.id})
        for term in response.terms:
            normalized_source = normalize_term(term.source_term)
            existing = targets_by_source.get(normalized_source)
            if existing is None:
                targets_by_source[normalized_source] = normalize_term(term.target_term)
                continue
            if existing != normalize_term(term.target_term):
                raise AppError("llm.terminology_conflict", {"source": term.source_term})
    return tuple(responses)


def _ensure_protected_numbers(source: str, output: str, word_id: str) -> None:
    if not protected_tokens_preserved(source, output):
        raise AppError("llm.protected_token_lost", {"id": word_id, "token": "protected"})
