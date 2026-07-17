from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from operator import attrgetter
from typing import Protocol, cast

import pytest

import captioner.core.policies.protected_spans as protected_spans_module
from captioner.core.policies.protected_spans import (
    ProtectedSpan,
    ProtectedToken,
    find_protected_spans,
    protected_break_cost,
    protected_token_differences,
    protected_tokens,
    protected_tokens_preserved,
)


class _NumberResult(Protocol):
    end: int
    raw: str
    normalized: str


class _ProtectedItemResult(Protocol):
    span: ProtectedSpan
    token: ProtectedToken


def _scan_number(text: str, position: int) -> _NumberResult | None:
    scanner = cast(
        Callable[[str, int], _NumberResult | None],
        attrgetter("_scan_number")(protected_spans_module),
    )
    return scanner(text, position)


def _scan_numeric_fact_at(text: str, position: int) -> _ProtectedItemResult | None:
    scanner = cast(
        Callable[[str, int], _ProtectedItemResult | None],
        attrgetter("_scan_numeric_fact_at")(protected_spans_module),
    )
    return scanner(text, position)


def _protected_items(text: str) -> tuple[_ProtectedItemResult, ...]:
    scanner = cast(
        Callable[[str], tuple[_ProtectedItemResult, ...]],
        attrgetter("_protected_items")(protected_spans_module),
    )
    return scanner(text)


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


@pytest.mark.parametrize(
    ("text", "expected_number"),
    [
        ("100 m/s", "100"),
        ("10 percent/year", "10"),
        ("100 dollars/kg", "100"),
        ("100 kg / m", "100"),
        ("١٢\u066b٣ kg/m", "12.3"),
    ],
)
def test_quantity_scanner_consumes_the_complete_number(text: str, expected_number: str) -> None:
    number = _scan_number(text, 0)
    assert number is not None
    assert number.raw == text[: number.end]
    assert number.normalized == expected_number
    item = _scan_numeric_fact_at(text, 0)
    assert item is not None
    assert item.token.numeric_value == expected_number
    assert item.span.start == 0
    assert item.span.end > number.end
    assert all(
        token.numeric_value not in {"1", "10"}
        for token in protected_tokens(text)
        if expected_number == "100"
    )


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("10", "100 m/s"),
        ("1", "10 percent/year"),
        ("10", "100 dollars/kg"),
        ("1", "100 kg/m"),
        ("100", "1000 kg/m"),
    ],
)
def test_quantity_scanner_cannot_backtrack_numeric_prefix(source: str, output: str) -> None:
    assert protected_tokens_preserved(source, output) is False


@pytest.mark.parametrize("spacing", [("", ""), (" ", ""), ("", " "), (" ", " ")])
def test_unsupported_slash_spacing_has_one_canonical_identity(
    spacing: tuple[str, str],
) -> None:
    left, right = spacing
    item = _scan_numeric_fact_at(f"100 kg{left}/{right}m", 0)
    assert item is not None
    assert item.token.kind == "unsupported-compound"
    assert item.token.numeric_value == "100"
    assert item.token.marker == "unsupported:unit:kg/m"
    assert protected_tokens_preserved("100 kg/m", f"100 kg{left}/{right}m")


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("100 kg", "100 kg."),
        ("100 kg", "100 kg,"),
        ("100 kg", "100 kg!"),
        ("100 kg", "100 kg)"),
        ("10 percent", "10 percent,"),
        ("100 dollars", "100 dollars!"),
    ],
)
def test_punctuation_after_complete_fact_is_not_part_of_marker(source: str, output: str) -> None:
    assert protected_tokens_preserved(source, output)


@pytest.mark.parametrize(
    "text",
    [
        "percentagewise",
        "dollarette",
        "metersomething",
        "itemization",
        "kilogrammatic",
        "megabytesomething",
        "myUSD",
        "predollar",
        "supermeter",
    ],
)
def test_alias_matching_does_not_consume_word_prefixes(text: str) -> None:
    assert protected_tokens(text) == ()


@pytest.mark.parametrize("text", ["$100", "US$100", "USD 100", "EUR 100", "€100", "-USD 100"])
def test_currency_prefix_scanner_consumes_one_complete_fact(text: str) -> None:
    item = _protected_items(text)[0]
    assert item.token.kind == "currency"
    assert item.token.numeric_value == "100"


@pytest.mark.parametrize("text", ["$100/kg", "USD 100/kg", "USD 100 / kg"])
def test_currency_prefix_scanner_preserves_unsupported_slash_suffix(text: str) -> None:
    item = _protected_items(text)[0]
    assert item.token.kind == "unsupported-compound"
    assert item.token.numeric_value == "100"
    assert "/kg" in item.token.marker


@pytest.mark.parametrize(
    ("text", "numeric_value"),
    [
        ("$100USD", "100"),
        ("$100kg", "100"),
        ("USD 100kg", "100"),
        ("10/20kg", "10|20"),
        ("100\u00d7200kg", "100|200"),
        ("100kgx", "100"),
    ],
)
def test_attached_quantity_tail_is_one_unsupported_fact(text: str, numeric_value: str) -> None:
    tokens = protected_tokens(text)

    assert len(tokens) == 1
    assert tokens[0].kind == "unsupported-attached"
    assert tokens[0].numeric_value == numeric_value
    assert tokens[0].text == text


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("$100", "$100USD"),
        ("$100", "$100kg"),
        ("USD 100", "USD 100kg"),
        ("10/20", "10/20kg"),
        ("100\u00d7200", "100\u00d7200kg"),
        ("100", "100kgx"),
    ],
)
def test_attached_quantity_tail_cannot_be_ignored(source: str, output: str) -> None:
    assert protected_tokens_preserved(source, output) is False


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("$100USD", "$100kg"),
        ("10/20kg", "10/20items"),
        ("100\u00d7200kg", "100\u00d7200cm"),
        ("100kgx", "100kgy"),
    ],
)
def test_attached_quantity_tail_identity_is_distinct(source: str, output: str) -> None:
    assert protected_tokens_preserved(source, output) is False


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("100 people", "number"),
        ("$100 total", "currency"),
        ("10/20 ratio", "number"),
        ("100\u00d7200 pixels", "unit"),
    ],
)
def test_separated_prose_is_not_an_attached_tail(text: str, expected_kind: str) -> None:
    tokens = protected_tokens(text)

    assert len(tokens) == 1
    assert tokens[0].kind == expected_kind
    assert tokens[0].kind != "unsupported-attached"


@pytest.mark.parametrize(
    "text",
    [
        "x$100",
        "foo€100",
        "price£100",
        "cost¥100",
        "约人民币100",
    ],
)
def test_free_currency_prefix_can_follow_ordinary_text(text: str) -> None:
    tokens = protected_tokens(text)

    assert len(tokens) == 1
    assert tokens[0].kind == "currency"
    assert tokens[0].numeric_value == "100"


@pytest.mark.parametrize(
    "output",
    [
        "x$100",
        "foo€100",
        "price£100",
        "cost¥100",
        "约人民币100",
    ],
)
def test_free_currency_prefix_is_not_reduced_to_bare_number(output: str) -> None:
    assert protected_tokens_preserved("100", output) is False


@pytest.mark.parametrize("text", ["xUSD 100", "myEUR 100", "abcUS$100"])
def test_currency_code_prefix_still_requires_word_boundary(text: str) -> None:
    tokens = protected_tokens(text)

    assert all(token.kind != "currency" for token in tokens)


def test_attached_tail_is_bounded() -> None:
    text = "$100" + ("a" * 256)
    tokens = protected_tokens(text)
    spans = find_protected_spans(text)

    assert len(tokens) == 1
    assert tokens[0].kind == "unsupported-attached"
    assert "overflow" in tokens[0].marker
    assert len(spans) == 1
    assert spans[0].end > spans[0].start
    assert spans[0].end <= len(text)


@pytest.mark.parametrize("text", ["10/20", "100/200", "١٢/٣"])
def test_numeric_ratio_consumes_both_numbers(text: str) -> None:
    token = protected_tokens(text)[0]
    assert token.kind == "number"
    assert token.numeric_value.count("|") == 1


@pytest.mark.parametrize("text", ["100\u00d7200", "100 \u00d7 200"])
def test_numeric_dimension_consumes_both_numbers(text: str) -> None:
    token = protected_tokens(text)[0]
    assert token.kind == "unit"
    assert token.numeric_value == "100|200"


def test_supported_compound_alias_is_not_an_unsupported_slash_fact() -> None:
    for text in ("10 km/h", "10 KM/H", "10 mph"):
        token = protected_tokens(text)[0]
        assert token.kind == "unit"
        assert token.marker in {"km/h", "mph"}


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("100 kg/m", "100 kg/s"),
        ("100 kg/m", "100 kg/hour"),
        ("10 percent/year", "10 percent/month"),
        ("100 dollars/kg", "100 dollars/item"),
    ],
)
def test_unsupported_compound_identity_is_not_interchangeable(source: str, output: str) -> None:
    assert protected_tokens_preserved(source, output) is False


@pytest.mark.parametrize(
    ("text", "expected_sign"),
    [("-100", "minus"), ("\u2212100", "minus"), ("\uff0d100", "minus"), ("+$100", "plus")],
)
def test_sign_scanning_is_part_of_the_fact(text: str, expected_sign: str) -> None:
    token = protected_tokens(text)[0]
    assert token.sign == expected_sign


def test_numeric_ratio_and_dimension_values_are_not_truncated() -> None:
    assert protected_tokens("10/200")[0].numeric_value == "10|200"
    assert protected_tokens("100/20")[0].numeric_value == "100|20"
    assert protected_tokens("100\u00d7200")[0].numeric_value == "100|200"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "/",
        "////",
        "$",
        "USD",
        "+",
        "-",
        ".",
        ",",
        "kg",
        "percent",
        "100 /",
        "100 ////",
        "100 kg ////",
    ],
)
def test_quantity_scanner_terminates_on_edge_inputs(text: str) -> None:
    spans = find_protected_spans(text)
    assert all(span.end > span.start for span in spans)
    assert all(left.end <= right.start for left, right in pairwise(spans))


def test_legacy_facts_are_occupied_before_quantity_scanning() -> None:
    text = "2024-01-02 2024/1/2 January 5, 2024 10:00 AM +1 555 123 4567 Dr."
    tokens = protected_tokens(text)
    assert [token.kind for token in tokens] == [
        "date",
        "date",
        "date",
        "time",
        "phone",
        "abbreviation",
    ]
