from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
from captioner.i18n.locale import SUPPORTED_LOCALES, normalize_locale


@given(st.text(min_size=0, max_size=24))
def test_non_strict_locale_normalization_never_returns_empty(value: str) -> None:
    normalized = normalize_locale(value, strict=False)
    assert normalized in SUPPORTED_LOCALES


@given(st.sampled_from(["", "fr-FR", "en-US", "zh-Hant", "--"]))
def test_invalid_locale_strict_failure_is_stable(value: str) -> None:
    with pytest.raises(AppError) as first:
        normalize_locale(value, strict=True)
    with pytest.raises(AppError) as second:
        normalize_locale(value, strict=True)
    assert first.value.code == second.value.code
