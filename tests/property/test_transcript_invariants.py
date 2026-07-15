from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from tests.support import make_transcript

from captioner.core.domain.transcript import Transcript


@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1).filter(
            lambda value: bool(value.strip())
        ),
        min_size=1,
        max_size=12,
    )
)
def test_generated_word_sequences_have_unique_ordered_timestamps(texts: list[str]) -> None:
    normalized = tuple(f"{text} " for text in texts)
    transcript = make_transcript(normalized)
    assert isinstance(transcript, Transcript)
    assert all(
        left.end_ms <= right.start_ms
        for left, right in zip(transcript.words, transcript.words[1:], strict=False)
    )
    assert len({word.id for word in transcript.words}) == len(transcript.words)
