from __future__ import annotations

from typing import cast

import pytest
from tests.support import make_transcript

from captioner.adapters.exporters.transcript_json import serialize
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.domain.transcript import (
    Transcript,
    TranscriptSegment,
    WordToken,
    derive_transcript_id,
)


def test_valid_transcript_and_deterministic_id() -> None:
    transcript = make_transcript()
    assert transcript.words[0].start_ms == 0
    assert transcript.segments[0].word_ids == ("word-000001", "word-000002")


def test_duplicate_word_ids_are_rejected() -> None:
    word = WordToken("word-1", "one", 0, 100)
    segment = TranscriptSegment("segment-1", ("word-1",), "one", 0, 100, None)
    with pytest.raises(AppError, match="duplicate_ids"):
        Transcript("transcript-1", "en", (word, word), (segment,), "fake", "model", {})


def test_overlapping_words_and_missing_references_are_rejected() -> None:
    first = WordToken("word-1", "one", 0, 100)
    second = WordToken("word-2", "two", 50, 150)
    segment = TranscriptSegment("segment-1", ("word-1", "word-2"), "one two", 0, 150, None)
    with pytest.raises(AppError, match="overlap_or_order"):
        Transcript("transcript-1", "en", (first, second), (segment,), "fake", "model", {})

    missing = TranscriptSegment("segment-1", ("word-missing",), "one", 0, 100, None)
    with pytest.raises(AppError, match="missing_reference"):
        Transcript("transcript-1", "en", (first,), (missing,), "fake", "model", {})


def test_transcript_rejects_float_milliseconds() -> None:
    with pytest.raises(AppError, match="integer_ms"):
        WordToken("word-1", "hello", 0.5, 1_000)  # type: ignore[arg-type]


def test_unassigned_word_is_rejected() -> None:
    first = WordToken("word-1", "one", 0, 100)
    second = WordToken("word-2", "two", 100, 200)
    segment = TranscriptSegment("segment-1", ("word-1",), "one", 0, 100, None)
    with pytest.raises(AppError, match="unassigned_word"):
        Transcript("transcript-1", "en", (first, second), (segment,), "fake", "model", {})


@pytest.mark.parametrize("segment_start,segment_end", [(10, 100), (0, 90)])
def test_segment_must_contain_referenced_words(segment_start: int, segment_end: int) -> None:
    word = WordToken("word-1", "one", 0, 100)
    segment = TranscriptSegment("segment-1", ("word-1",), "one", segment_start, segment_end, None)
    with pytest.raises(AppError, match="word_outside_segment"):
        Transcript("transcript-1", "en", (word,), (segment,), "fake", "model", {})


def test_transcript_metadata_freeze_preserves_deterministic_identity() -> None:
    items = ["stable"]
    metadata = cast(dict[str, JsonValue], {"nested": {"items": items}})
    original = make_transcript(metadata=metadata)
    derived_before = derive_transcript_id(
        language=original.language,
        words=original.words,
        segments=original.segments,
        engine_id=original.engine_id,
        model_id=original.model_id,
        metadata=original.metadata,
    )
    serialized_before = serialize(original)
    items.append("mutated")
    assert serialize(original) == serialized_before
    with pytest.raises(TypeError):
        original.metadata["nested"]["items"] = []  # type: ignore[index]
    derived_after = derive_transcript_id(
        language=original.language,
        words=original.words,
        segments=original.segments,
        engine_id=original.engine_id,
        model_id=original.model_id,
        metadata=original.metadata,
    )
    assert derived_after == derived_before
