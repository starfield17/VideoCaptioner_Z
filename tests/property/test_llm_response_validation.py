from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import FastTranslationResponse
from captioner.core.policies.llm_validation import (
    is_obvious_wrong_language,
    protected_numeric_tokens,
    validate_responses,
)
from captioner.core.policies.protected_spans import protected_tokens_preserved


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=30))
def test_validated_response_rejects_noncanonical_model_text(text: str) -> None:
    if text != text.strip() or text != text.replace("\n", " "):
        with pytest.raises(AppError):
            validate_responses(
                ({"id": "unit-1", "corrected_source": "source", "translated_text": text},),
                ("unit-1",),
            )


def test_response_validator_covers_id_context_language_and_numbers() -> None:
    with pytest.raises(AppError, match=r"llm\.missing_id"):
        validate_responses((), ("unit-1",))
    with pytest.raises(AppError, match=r"llm\.duplicate_id"):
        validate_responses(
            (
                FastTranslationResponse("unit-1", "one", "一"),
                FastTranslationResponse("unit-1", "one", "一"),
            ),
            ("unit-1", "unit-2"),
        )
    with pytest.raises(AppError, match=r"llm\.context_id_returned"):
        validate_responses(
            (FastTranslationResponse("context-1", "one", "一"),),
            ("unit-1",),
            context_ids=("context-1",),
        )
    with pytest.raises(AppError, match=r"llm\.wrong_language"):
        validate_responses(
            (FastTranslationResponse("unit-1", "one", "hello"),),
            ("unit-1",),
            target_language="zh-CN",
        )
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        validate_responses(
            (FastTranslationResponse("unit-1", "Price 20%", "价格"),),
            ("unit-1",),
            source_texts={"unit-1": "Price 20%"},
        )


def test_script_heuristic_and_protected_token_extraction_are_deterministic() -> None:
    assert is_obvious_wrong_language("hello", "zh-CN")
    assert not is_obvious_wrong_language("你好", "zh-CN")
    assert protected_numeric_tokens("USD 12.50 and 20%")


@pytest.mark.parametrize(
    ("source", "output", "preserved"),
    [
        ("-5", "5", False),
        ("-5", "-5", True),
        ("12.30", "1230", False),
        ("12.30", "12,3", True),
        ("$100", "100", False),
        ("$100", "$100", True),
        ("-$100", "$100", False),
        ("-$100", "-$100", True),
        ("10 kg", "10", False),
        ("10 kg", "10 kg", True),
        ("1,000", "1000", True),
        ("10%", "10", False),
        ("10%", "10 %", True),
        ("2024-01-02", "2024/1/2", True),
        ("2024-01-02", "2024/1/3", False),
        ("١٢\u066b٣", "12.3", True),
        ("5 and -2.5", "5 -2.5", True),
        ("5 and -2.5", "-2.5 5", False),
    ],
)
def test_protected_spans_preserve_numeric_semantics(
    source: str, output: str, preserved: bool
) -> None:
    assert protected_tokens_preserved(source, output) is preserved


@pytest.mark.parametrize(
    ("source", "output", "preserved"),
    [
        ("Total is 10 kg", "total 10 kg and 20 more", False),
        ("5", "5 5", False),
        ("5 and 10", "5 10", True),
        ("5 and 10", "10 5", False),
        ("January 5, 2024", "February 5, 2024", False),
        ("January 5, 2024", "2024-01-05", True),
        ("10:00 AM", "10:00 PM", False),
        ("10:00 AM", "22:00", False),
        ("\u22125", "5", False),
        ("\u22125", "-5", True),
        ("100元", "100円", False),
        ("$100", "US$100", False),
        ("$100", "$100", True),
    ],
)
def test_protected_token_exact_sequence_and_semantic_facts(
    source: str, output: str, preserved: bool
) -> None:
    assert protected_tokens_preserved(source, output) is preserved


def test_fast_fields_are_validated_independently() -> None:
    # corrected_source lost the percent while translated_text kept it.
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        validate_responses(
            (FastTranslationResponse("unit-1", "Price", "价格 20%"),),
            ("unit-1",),
            source_texts={"unit-1": "Price 20%"},
        )
    # translated_text lost the percent while corrected_source kept it.
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        validate_responses(
            (FastTranslationResponse("unit-1", "Price 20%", "价格"),),
            ("unit-1",),
            source_texts={"unit-1": "Price 20%"},
        )
    validate_responses(
        (FastTranslationResponse("unit-1", "Price 20%", "价格 20%"),),
        ("unit-1",),
        source_texts={"unit-1": "Price 20%"},
        target_language="zh-CN",
    )
