"""Pure assembly and deterministic merge rules for quality source correction."""

from __future__ import annotations

from collections.abc import Sequence

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import SourceCorrectionResponse, TerminologyResponse
from captioner.core.domain.terminology import Terminology, TerminologyEntry, normalize_term
from captioner.core.domain.transcript import CorrectedSpan, CorrectedTranscript, Transcript
from captioner.core.policies.llm_validation import protected_numeric_tokens
from captioner.core.policies.segmentation import canonical_words


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
) -> Terminology:
    """Merge per-Word terminology responses by first source occurrence."""
    response_by_id = _terminology_responses_by_id(responses)
    words = canonical_words(transcript.words)
    expected_ids = tuple(word.id for word in words)
    if tuple(response_by_id) != expected_ids:
        raise AppError("llm.terminology_units_invalid", {"reason": "ids"})
    entries: list[TerminologyEntry] = []
    by_normalized: dict[str, int] = {}
    for word in words:
        response = response_by_id[word.id]
        normalized_source = normalize_term(response.source_term)
        if not normalized_source:
            raise AppError("llm.terminology_invalid", {"reason": "empty_source"})
        _ensure_protected_numbers(word.text, response.source_term, word.id)
        _ensure_protected_numbers(word.text, response.target_term, word.id)
        existing_index = by_normalized.get(normalized_source)
        if existing_index is None:
            by_normalized[normalized_source] = len(entries)
            entries.append(
                TerminologyEntry(
                    response.source_term,
                    normalized_source,
                    response.target_term,
                    (word.id,),
                )
            )
            continue
        existing = entries[existing_index]
        if normalize_term(existing.target) != normalize_term(response.target_term):
            raise AppError(
                "llm.terminology_conflict",
                {"source": existing.source, "word_id": word.id},
            )
        entries[existing_index] = TerminologyEntry(
            existing.source,
            existing.normalized_source,
            existing.target,
            (*existing.source_word_ids, word.id),
        )
    return Terminology(transcript.id, source_language, target_language, tuple(entries))


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


def _ensure_protected_numbers(source: str, output: str, word_id: str) -> None:
    output_digits = "".join(character for character in output if character.isdigit())
    cursor = 0
    for token in protected_numeric_tokens(source):
        position = output_digits.find(token.digits, cursor)
        if position < 0 or (token.percent and "%" not in output):
            raise AppError("llm.protected_token_lost", {"id": word_id, "token": token.text})
        cursor = position + len(token.digits)
