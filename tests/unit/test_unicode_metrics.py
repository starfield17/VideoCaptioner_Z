from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.policies.unicode_metrics import measure_text, normalize_text


@pytest.mark.parametrize(
    ("text", "graphemes", "reading", "columns"),
    [
        ("ASCII 123", 9, 8, 8),
        ("e\u0301", 1, 1, 1),
        ("中文", 2, 2, 4),
        ("日本語", 3, 3, 6),
        ("한국어", 3, 3, 6),
        ("\uff21", 1, 1, 2),
        ("👩🏽‍💻", 1, 1, 2),
        ("👨‍👩‍👧‍👦", 1, 1, 2),
        ("🇨🇳", 1, 1, 2),
    ],
)
def test_metrics_use_graphemes_and_display_columns(
    text: str, graphemes: int, reading: int, columns: int
) -> None:
    result = measure_text(text)
    assert (result.graphemes, result.reading_characters, result.display_columns) == (
        graphemes,
        reading,
        columns,
    )


def test_normalization_is_nfc_and_collapses_permitted_whitespace() -> None:
    assert normalize_text("  cafe\u0301\t\n  世界  ") == "café 世界"


def test_invalid_control_character_is_structured() -> None:
    with pytest.raises(AppError, match=r"subtitle\.control_character"):
        measure_text("hello\x00world")
