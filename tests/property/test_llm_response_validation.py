from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    FastTranslationResponse,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
)
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
        ("10:00", "10:00 AM", False),
        ("\u22125", "5", False),
        ("\u22125", "-5", True),
        ("100元", "100円", False),
        ("$100", "US$100", False),
        ("$100", "$100", True),
        ("See you soon", "See you in 2027", False),
        ("There are 100", "There are 100", True),
        ("There are 100", "It costs $100", False),
        ("Value 100", "Weight 100 kg", False),
        ("Score 10", "Score 10%", False),
        ("100 kg", "100 kg", True),
        ("100 kg", "100 g", False),
        ("10 kg", "10 kg and 20 pieces", False),
        ("100", "100 dollars", False),
        ("100", "100 euros", False),
        ("10", "10 percent", False),
        ("10", "10 per cent", False),
        ("100", "100 kilograms", False),
        ("100", "100 meters", False),
        ("100 dollars", "100", False),
        ("10 percent", "10", False),
        ("100 kilograms", "100", False),
        ("100 dollars", "100 dollars", True),
        ("10 percent", "10 percent", True),
        ("100 kilograms", "100 kg", True),
        ("1 meter", "1 metre", True),
        ("1 gigabyte", "1 GB", True),
        ("100 dollars", "100 euros", False),
        ("10 kg", "10 g", False),
    ],
)
def test_protected_token_exact_sequence_and_semantic_facts(
    source: str, output: str, preserved: bool
) -> None:
    assert protected_tokens_preserved(source, output) is preserved


@given(
    numbers=st.lists(st.integers(min_value=-9999, max_value=9999), min_size=1, max_size=6),
    extra=st.integers(min_value=-9999, max_value=9999),
)
def test_protected_sequences_reject_added_and_removed_facts(numbers: list[int], extra: int) -> None:
    source = " ".join(str(number) for number in numbers)
    assert protected_tokens_preserved(source, f"{source} {extra}") is False
    assert (
        protected_tokens_preserved(source, " ".join(str(number) for number in numbers[:-1]))
        is False
    )


@given(
    first=st.integers(min_value=-9999, max_value=9999),
    second=st.integers(min_value=-9999, max_value=9999),
)
def test_protected_sequences_reject_reordering(first: int, second: int) -> None:
    if first == second:
        return
    assert protected_tokens_preserved(f"{first} {second}", f"{second} {first}") is False


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


def test_protected_diagnostic_identifies_fast_field_and_item() -> None:
    with pytest.raises(AppError) as raised:
        validate_responses(
            (FastTranslationResponse("cue-000123", "Price 100", "It costs 100 dollars"),),
            ("cue-000123",),
            source_texts={"cue-000123": "Price 100"},
        )
    assert raised.value.params["id"] == "cue-000123"
    assert raised.value.params["field"] == "translated_text"
    assert raised.value.params["reason"] == "protected_fact_kind_changed"
    assert "100 dollars" not in str(raised.value)
    assert "Price 100" not in str(raised.value)


def test_protected_diagnostic_identifies_source_correction_field() -> None:
    with pytest.raises(AppError) as raised:
        validate_responses(
            (FastTranslationResponse("cue-000123", "Weight 100 kilograms", "Weight 100"),),
            ("cue-000123",),
            source_texts={"cue-000123": "Weight 100"},
        )
    assert raised.value.params["id"] == "cue-000123"
    assert raised.value.params["field"] == "corrected_source"
    assert raised.value.params["expected_kind"] == "number"
    assert raised.value.params["actual_kind"] == "unit"


@pytest.mark.parametrize(
    ("response", "field"),
    [
        (SourceCorrectionResponse("cue-1", "100 kilograms"), "corrected_source"),
        (QualityTranslationResponse("cue-1", "100 dollars"), "translated_text"),
        (ReviewResponse("cue-1", "10"), "translated_text"),
    ],
)
def test_protected_diagnostic_uses_stage_text_field(response: object, field: str) -> None:
    source = "10 percent" if isinstance(response, ReviewResponse) else "100"
    with pytest.raises(AppError) as raised:
        validate_responses(
            (response,),
            ("cue-1",),
            source_texts={"cue-1": source},
        )
    assert raised.value.params["id"] == "cue-1"
    assert raised.value.params["field"] == field


@given(st.sampled_from(("dollars", "euros", "pounds", "yen", "yuan")))
def test_recognized_textual_currency_cannot_be_added_to_a_bare_number(marker: str) -> None:
    assert protected_tokens_preserved("100", f"100 {marker}") is False


@given(st.sampled_from(("percent", "percentage", "per cent")))
def test_recognized_textual_percentage_cannot_be_added_to_a_bare_number(marker: str) -> None:
    assert protected_tokens_preserved("10", f"10 {marker}") is False


@given(
    st.sampled_from(
        (
            "kilograms",
            "grams",
            "meters",
            "metres",
            "centimeters",
            "gigabytes",
            "hertz",
        )
    )
)
def test_recognized_textual_unit_cannot_be_added_to_a_bare_number(marker: str) -> None:
    assert protected_tokens_preserved("100", f"100 {marker}") is False


@pytest.mark.parametrize(
    "text", ["grammar", "moment", "itemization", "percentagewise", "dollarette"]
)
def test_unrelated_words_are_not_textual_markers(text: str) -> None:
    assert protected_numeric_tokens(text) == ()
