from __future__ import annotations

import pytest
from tests.support import make_transcript

from captioner.core.domain.errors import AppError
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken


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
