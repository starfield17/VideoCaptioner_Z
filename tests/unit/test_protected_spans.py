from __future__ import annotations

from captioner.core.policies.protected_spans import (
    find_protected_spans,
    protected_break_cost,
    protected_token_differences,
)


def test_numbers_currency_dates_and_units_are_protected() -> None:
    text = "1,000 1.25 $100 +65 1234 5678 10 kg 2026-07-16 July 16, 2026 10:30 AM Dr."
    spans = find_protected_spans(text)
    assert {span.kind for span in spans} >= {
        "number",
        "currency",
        "phone",
        "date-time",
        "unit",
        "abbreviation",
    }
    assert protected_break_cost(text, text.index("000")) == 1


def test_unprotected_word_boundary_has_no_protected_cost() -> None:
    text = "hello world"
    assert protected_break_cost(text, text.index(" world")) == 0


def test_protected_differences_are_safe_structured_metadata() -> None:
    differences = protected_token_differences("Value 100", "Value 100 kg")
    assert differences[0].code == "protected_fact_kind_changed"
    assert differences[0].position == 0
    assert differences[0].expected_kind == "number"
    assert differences[0].actual_kind == "unit"


def test_protected_fact_reordering_has_safe_order_diagnostic() -> None:
    differences = protected_token_differences("Value 1 and 2", "Value 2 and 1")
    assert differences[0].code == "protected_fact_order_changed"
    assert differences[0].position == 0
