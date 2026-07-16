from __future__ import annotations

from captioner.core.policies.reading_speed import reading_speed


def test_cps_exact_limit_is_valid_and_one_unit_over_is_error() -> None:
    exact = reading_speed("12345678901234567890", 1_000, max_cps_milli=20_000)
    over = reading_speed("123456789012345678901", 1_000, max_cps_milli=20_000)
    assert exact.status != "error"
    assert over.status == "error"


def test_whitespace_does_not_inflate_reading_characters_and_zero_duration_is_error() -> None:
    result = reading_speed("a   b\n", 1_000, max_cps_milli=20_000)
    assert result.characters == 2
    assert reading_speed("abc", 0).status == "error"
