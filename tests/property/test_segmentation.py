from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support import make_transcript

from captioner.core.policies.segmentation import canonical_words
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript


@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=18), min_size=1, max_size=32))
def test_dp_segmentation_assigns_every_word_once(lengths: list[int]) -> None:
    transcript = make_transcript(tuple("x" * length + " " for length in lengths))
    track = segment_transcript(transcript, SegmentationPolicyConfig())
    assigned = [word_id for cue in track.cues for word_id in cue.source_word_ids]
    assert assigned == [word.id for word in canonical_words(transcript.words)]
    assert len(assigned) == len(set(assigned))
    assert all(
        left.end_ms <= right.start_ms
        for left, right in zip(track.cues, track.cues[1:], strict=False)
    )


def test_unordered_words_produce_identical_track() -> None:
    transcript = make_transcript(("one ", "two ", "three"))
    shuffled = type(transcript)(
        transcript.id,
        transcript.language,
        tuple(reversed(transcript.words)),
        transcript.segments,
        transcript.engine_id,
        transcript.model_id,
        transcript.metadata,
    )
    assert segment_transcript(transcript) == segment_transcript(shuffled)


@settings(deadline=None)
@given(st.permutations(tuple(range(6))))
def test_random_word_permutations_produce_identical_track(
    order: tuple[int, ...],
) -> None:
    transcript = make_transcript(tuple(f"word-{index} " for index in range(6)))
    shuffled = replace(transcript, words=tuple(transcript.words[index] for index in order))
    assert segment_transcript(shuffled) == segment_transcript(transcript)


def test_overlap_and_equal_timestamps_are_normalized_to_legal_cues() -> None:
    from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken

    words = (
        WordToken("word-1", "A ", 0, 100),
        WordToken("word-2", "B ", 50, 150),
        WordToken("word-3", "C", 150, 150 + 1),
    )
    transcript = Transcript(
        "overlap",
        "en",
        words,
        (TranscriptSegment("segment", tuple(word.id for word in words), "A B C", 0, 151, None),),
        "fake",
        "model",
        {},
    )
    track = segment_transcript(transcript)
    assert all(cue.end_ms > cue.start_ms for cue in track.cues)
    assert all(
        left.end_ms <= right.start_ms
        for left, right in zip(track.cues, track.cues[1:], strict=False)
    )
