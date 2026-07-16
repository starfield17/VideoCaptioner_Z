from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.policies.unicode_metrics import measure_text


@given(
    st.lists(
        st.sampled_from(("a", "é", "中", "日", "👩🏽‍💻", "🇨🇳", " ")),
        min_size=1,
        max_size=40,
    )
)
def test_unicode_metrics_are_repeatable_for_grapheme_sequences(parts: list[str]) -> None:
    text = "".join(parts)
    first = measure_text(text)
    assert first == measure_text(text)
    assert first.reading_characters <= first.graphemes
    assert first.display_columns >= 0
