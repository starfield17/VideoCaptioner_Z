from __future__ import annotations

from captioner.core.policies.protected_spans import find_protected_spans, protected_break_cost


def test_numbers_currency_dates_and_units_are_protected() -> None:
    text = "1,000 1.25 $100 10 kg 2026-07-16 10:30 AM"
    spans = find_protected_spans(text)
    assert {span.kind for span in spans} >= {"number", "currency", "date-time", "unit"}
    assert protected_break_cost(text, text.index("000")) == 1


def test_unprotected_word_boundary_has_no_protected_cost() -> None:
    text = "hello world"
    assert protected_break_cost(text, text.index(" world")) == 0
