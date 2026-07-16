from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


def _cue(number: int, start: int, end: int, word_id: str) -> SubtitleCue:
    return SubtitleCue(
        id=f"cue-{number:06d}",
        start_ms=start,
        end_ms=end,
        source_word_ids=(word_id,),
        source_text=word_id,
        translated_text=None,
        lines=(word_id,),
    )


def test_track_accepts_ordered_non_overlapping_cues() -> None:
    track = SubtitleTrack("track-1", "transcript-1", "en", (_cue(1, 0, 100, "word-1"),), 0)
    assert track.revision == 0


def test_track_rejects_duplicate_word_assignment_and_overlap() -> None:
    first = _cue(1, 0, 100, "word-1")
    duplicate = _cue(2, 100, 200, "word-1")
    with pytest.raises(AppError, match="multiple_assignment"):
        SubtitleTrack("track-1", "transcript-1", "en", (first, duplicate), 0)

    overlap = _cue(2, 50, 200, "word-2")
    with pytest.raises(AppError, match="overlap_or_order"):
        SubtitleTrack("track-1", "transcript-1", "en", (first, overlap), 0)


def test_translated_text_and_nonzero_revision_are_phase1_invalid() -> None:
    with pytest.raises(AppError, match="phase1_forbidden"):
        SubtitleCue("cue-1", 0, 100, ("word-1",), "one", "一", ("one",))
    with pytest.raises(AppError, match="phase1_must_be_zero"):
        SubtitleTrack("track-1", "transcript-1", "en", (), 1)


def test_blank_source_word_ids_are_rejected() -> None:
    with pytest.raises(AppError, match="duplicate_or_empty"):
        SubtitleCue("cue-1", 0, 100, (" ",), "one", None, ("one",))


def test_subtitle_rejects_float_milliseconds() -> None:
    with pytest.raises(AppError, match="integer_ms"):
        SubtitleCue("cue-1", 0.5, 1_000, ("word-1",), "one", None, ("one",))  # type: ignore[arg-type]


def test_subtitle_source_text_must_be_canonical_nfc() -> None:
    with pytest.raises(AppError, match="not_canonical"):
        SubtitleCue("cue-1", 0, 100, ("word-1",), "e\u0301", None, ("é",))
