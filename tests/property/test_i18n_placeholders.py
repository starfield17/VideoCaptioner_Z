from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from captioner.i18n.validation import placeholder_names, validate_placeholder_pair


@given(st.lists(st.sampled_from(["current", "total", "name"]), min_size=0, max_size=5))
def test_placeholder_set_is_stable_for_reordered_messages(names: list[str]) -> None:
    english = " ".join(f"{{{name}}}" for name in names)
    translation = " ".join(f"{{{name}}}" for name in reversed(names))
    assert placeholder_names(english) == placeholder_names(translation)
    validate_placeholder_pair("test", english, translation)
