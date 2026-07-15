from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.i18n.locale import available_locales, normalize_locale


@pytest.mark.parametrize(
    ("value", "expected"),
    [("en", "en"), ("EN", "en"), ("zh_cn", "zh-CN"), ("ZH-cn", "zh-CN")],
)
def test_locale_normalization(value: str, expected: str) -> None:
    assert normalize_locale(value) == expected


def test_unsupported_locale_strict_and_non_strict() -> None:
    with pytest.raises(AppError, match="locale_unsupported"):
        normalize_locale("fr-FR", strict=True)
    assert normalize_locale("fr-FR", strict=False) == "en"


def test_empty_locale_strict_and_available_order() -> None:
    with pytest.raises(AppError, match="locale_invalid"):
        normalize_locale("", strict=True)
    assert available_locales(["zh_cn", "en", "en"]) == ("zh-CN", "en")
