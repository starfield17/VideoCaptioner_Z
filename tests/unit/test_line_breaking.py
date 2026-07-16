from __future__ import annotations

from captioner.core.policies.line_breaking import break_lines, join_rendered_lines
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import measure_text


def _config(width: int) -> SegmentationPolicyConfig:
    return SegmentationPolicyConfig(max_line_width=width, max_cue_width=width * 2)


def test_two_line_break_minimizes_width_difference_for_latin() -> None:
    lines = break_lines("one two three four five six", _config(10))
    assert len(lines) == 2
    assert abs(measure_text(lines[0]).display_columns - measure_text(lines[1]).display_columns) <= 2
    assert join_rendered_lines(lines) == "one two three four five six"


def test_cjk_breaks_on_grapheme_boundaries_and_preserves_text() -> None:
    text = "这是一个很长的中文字幕句子用于测试分行平衡"
    lines = break_lines(text, _config(24))
    assert len(lines) == 2
    assert join_rendered_lines(lines) == text
    assert all(measure_text(line).display_columns <= 24 for line in lines)


def test_emoji_zwj_sequence_is_never_split() -> None:
    family = "👨‍👩‍👧‍👦"
    lines = break_lines(f"{family} family words", _config(10))
    assert family in lines[0] or family in lines[1]
    assert join_rendered_lines(lines) == f"{family} family words"


def test_long_atomic_token_uses_grapheme_safe_fallback() -> None:
    lines = break_lines("abcdefghijklmnop", _config(8))
    assert len(lines) == 1
    assert join_rendered_lines(lines) == "abcdefghijklmnop"
    assert measure_text(lines[0]).display_columns > 8


def test_currency_and_unit_remain_together_when_feasible() -> None:
    lines = break_lines("start $100 10 kg", _config(8))
    assert "$100" in "".join(lines)
    assert "10 kg" in join_rendered_lines(lines)
