"""Deterministic protected-span detection for subtitle line and cue breaks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class ProtectedToken:
    """A protected semantic token shared by LLM and subtitle validators."""

    text: str
    kind: str
    numeric_value: str
    sign: str
    marker: str
    date_components: str = ""
    time_components: str = ""
    am_pm: str = ""
    phone_components: str = ""

    @property
    def digits(self) -> str:
        return "".join(character for character in self.text if character.isdigit())

    @property
    def percent(self) -> bool:
        return self.kind == "percentage"

    def identity(self) -> tuple[str, ...]:
        """Canonical identity used for exact ordered sequence comparison."""
        return (
            self.kind,
            self.numeric_value,
            self.sign,
            self.marker,
            self.date_components,
            self.time_components,
            self.am_pm,
            self.phone_components,
        )


@dataclass(frozen=True, slots=True)
class ProtectedTokenDifference:
    """Safe metadata describing one protected-fact sequence difference."""

    code: str
    position: int
    expected_kind: str | None = None
    actual_kind: str | None = None


_MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

# These are deliberately explicit.  In particular, "$", "US$", and
# textual dollar aliases are not inferred to be interchangeable merely
# because their text overlaps.
_CURRENCY_SYMBOL_ALIASES = MappingProxyType(
    {
        "$": "symbol:$",
        "US$": "symbol:US$",
        "A$": "symbol:A$",
        "C$": "symbol:C$",
        "NZ$": "symbol:NZ$",
        "€": "symbol:€",
        "£": "symbol:£",
        "¥": "symbol:¥",
        "₹": "symbol:₹",
        "USD": "code:USD",
        "EUR": "code:EUR",
        "GBP": "code:GBP",
        "JPY": "code:JPY",
        "CNY": "code:CNY",
        "RMB": "code:CNY",
        "元": "code:CNY",
        "人民币": "code:CNY",
        "円": "code:JPY",
        "ドル": "word:USD",
    }
)
_CURRENCY_WORD_ALIASES = MappingProxyType(
    {
        "dollar": "word:USD",
        "dollars": "word:USD",
        "US dollar": "word:USD",
        "US dollars": "word:USD",
        "euro": "word:EUR",
        "euros": "word:EUR",
        "pound": "word:GBP",
        "pounds": "word:GBP",
        "British pound": "word:GBP",
        "British pounds": "word:GBP",
        "yen": "word:JPY",
        "yuan": "word:CNY",
        "renminbi": "word:CNY",
    }
)
_CURRENCY_ALIASES = MappingProxyType({**_CURRENCY_SYMBOL_ALIASES, **_CURRENCY_WORD_ALIASES})
_PERCENTAGE_WORD_ALIASES = MappingProxyType({"percent": "%", "percentage": "%", "per cent": "%"})
_UNIT_COMPACT_ALIASES = MappingProxyType(
    {
        "kg": "kg",
        "g": "g",
        "km/h": "km/h",
        "km": "km",
        "m": "m",
        "cm": "cm",
        "mm": "mm",
        "mph": "mph",
        "°c": "°c",
        "°f": "°f",
        "hz": "hz",
        "khz": "khz",
        "gb": "gb",
        "mb": "mb",
    }
)
_UNIT_WORD_ALIASES = MappingProxyType(
    {
        "kilogram": "kg",
        "kilograms": "kg",
        "gram": "g",
        "grams": "g",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "mile": "mile",
        "miles": "mile",
        "hour": "hour",
        "hours": "hour",
        "minute": "minute",
        "minutes": "minute",
        "second": "second",
        "seconds": "second",
        "piece": "piece",
        "pieces": "piece",
        "item": "item",
        "items": "item",
        "gigabyte": "gb",
        "gigabytes": "gb",
        "megabyte": "mb",
        "megabytes": "mb",
        "hertz": "hz",
        "kilohertz": "khz",
    }
)
_UNIT_ALIASES = MappingProxyType({**_UNIT_COMPACT_ALIASES, **_UNIT_WORD_ALIASES})


def _alternation(aliases: Sequence[str]) -> str:
    return "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))


_CURRENCY_SYMBOL_MARKERS = _alternation(tuple(_CURRENCY_SYMBOL_ALIASES))
_CURRENCY_WORD_MARKERS = _alternation(tuple(_CURRENCY_WORD_ALIASES))
_PERCENTAGE_WORD_MARKERS = _alternation(tuple(_PERCENTAGE_WORD_ALIASES))
_UNIT_COMPACT_MARKERS = _alternation(tuple(_UNIT_COMPACT_ALIASES))
_UNIT_WORD_MARKERS = _alternation(tuple(_UNIT_WORD_ALIASES))
_NUMBER = r"\d[\d,\.\u066B\u066C]*"
_SIGN = r"[+\-\u2212\uff0d]\s*"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "currency",
        re.compile(
            rf"(?:{_SIGN})?(?:{_CURRENCY_SYMBOL_MARKERS})\s?{_NUMBER}"
            rf"|(?:{_SIGN})?{_NUMBER}\s?(?:{_CURRENCY_SYMBOL_MARKERS})(?!\w)"
            rf"|(?:{_SIGN})?(?:{_CURRENCY_WORD_MARKERS})(?!\w)\s+{_NUMBER}"
            rf"|(?:{_SIGN})?{_NUMBER}\s+(?:{_CURRENCY_WORD_MARKERS})(?!\w)",
            re.IGNORECASE,
        ),
    ),
    (
        "phone",
        re.compile(r"\+\d{1,3}(?:\s+\d{3,4}){2,3}"),
    ),
    (
        "date-time",
        re.compile(
            r"(?:"
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
            r"|\d{4}年\d{1,2}月\d{1,2}日"
            r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
            r"|\d{1,2}:\d{2}(?:\s?[AP]M)?"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "percentage",
        re.compile(
            rf"(?:{_SIGN})?{_NUMBER}\s?%"
            rf"|(?:{_SIGN})?{_NUMBER}\s+(?:{_PERCENTAGE_WORD_MARKERS})(?!\w)",
            re.IGNORECASE,
        ),
    ),
    (
        "unit",
        re.compile(
            rf"(?:{_SIGN})?{_NUMBER}\s?(?:{_UNIT_COMPACT_MARKERS})(?!\w)"
            rf"|(?:{_SIGN})?{_NUMBER}\s+(?:{_UNIT_WORD_MARKERS})(?!\w)"
            rf"|(?:{_SIGN})?{_NUMBER}\s?\u00d7\s?{_NUMBER}",
            re.IGNORECASE,
        ),
    ),
    (
        "number",
        re.compile(rf"(?:{_SIGN})?{_NUMBER}(?:[:/]{_NUMBER})?"),
    ),
    (
        "abbreviation",
        re.compile(r"(?<!\w)(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e)\.", re.IGNORECASE),
    ),
)


def find_protected_spans(text: str) -> tuple[ProtectedSpan, ...]:
    found: list[ProtectedSpan] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            candidate = ProtectedSpan(match.start(), match.end(), kind, match.group(0))
            if not any(_overlap(candidate, current) for current in found):
                found.append(candidate)
    found.sort(key=lambda span: (span.start, span.end, span.kind))
    return tuple(found)


def protected_tokens(text: str) -> tuple[ProtectedToken, ...]:
    """Classify protected spans without flattening their numeric semantics."""
    return tuple(_classify_span(span) for span in find_protected_spans(text))


def protected_token_differences(source: str, output: str) -> tuple[ProtectedTokenDifference, ...]:
    """Return safe, ordered differences between protected semantic sequences."""
    expected = protected_tokens(source)
    actual = protected_tokens(output)
    differences: list[ProtectedTokenDifference] = []
    if len(expected) != len(actual):
        if len(actual) > len(expected):
            for position in range(len(expected), len(actual)):
                differences.append(
                    ProtectedTokenDifference(
                        "protected_fact_added", position, None, actual[position].kind
                    )
                )
        else:
            for position in range(len(actual), len(expected)):
                differences.append(
                    ProtectedTokenDifference(
                        "protected_fact_removed", position, expected[position].kind, None
                    )
                )
    if len(expected) == len(actual) and _same_multiset(expected, actual):
        expected_identities = tuple(token.identity() for token in expected)
        actual_identities = tuple(token.identity() for token in actual)
        if expected_identities != actual_identities:
            position = next(
                index
                for index, (expected_identity, actual_identity) in enumerate(
                    zip(expected_identities, actual_identities, strict=True)
                )
                if expected_identity != actual_identity
            )
            return (
                ProtectedTokenDifference(
                    "protected_fact_order_changed",
                    position,
                    expected[position].kind,
                    actual[position].kind,
                ),
            )
    for position, (expected_token, actual_token) in enumerate(zip(expected, actual, strict=False)):
        if expected_token.kind != actual_token.kind:
            differences.append(
                ProtectedTokenDifference(
                    "protected_fact_kind_changed",
                    position,
                    expected_token.kind,
                    actual_token.kind,
                )
            )
        elif not _matches(expected_token, actual_token):
            value_changed = (
                expected_token.numeric_value != actual_token.numeric_value
                or expected_token.sign != actual_token.sign
                or expected_token.date_components != actual_token.date_components
                or expected_token.time_components != actual_token.time_components
                or expected_token.am_pm != actual_token.am_pm
            )
            code = (
                "protected_fact_value_changed" if value_changed else "protected_fact_marker_changed"
            )
            differences.append(
                ProtectedTokenDifference(code, position, expected_token.kind, actual_token.kind)
            )
    if len(expected) == len(actual) and not differences:
        return ()
    return tuple(differences)


def protected_tokens_preserved(source: str, output: str) -> bool:
    """Require exact ordered semantic token sequence equality.

    Formatting differences that normalize to the same identity are allowed.
    An empty source has an empty expected sequence and therefore still rejects
    any protected fact introduced by the output.
    """
    return not protected_token_differences(source, output)


def protected_break_cost(
    text: str, boundary: int, spans: Sequence[ProtectedSpan] | None = None
) -> int:
    """Return one when a boundary is protected and zero otherwise."""
    candidates = find_protected_spans(text) if spans is None else spans
    return int(any(span.start < boundary < span.end for span in candidates))


def punctuation_attachment_cost(text: str, boundary: int) -> int:
    if boundary <= 0 or boundary >= len(text):
        return 0
    left = text[:boundary].rstrip()
    right = text[boundary:].lstrip()
    if not left or not right:
        return 0
    opening = "([{\u201c\u2018「『【\uff08《〈"
    closing = ")]};:,.!?%\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201d\u2019」』】》\uff09〉"
    return int(left[-1] in opening or right[0] in closing)


def _overlap(left: ProtectedSpan, right: ProtectedSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _classify_span(span: ProtectedSpan) -> ProtectedToken:
    text = span.text
    sign = (
        "minus"
        if re.search(r"(?<!\d)[-\u2212\uff0d]\s*(?:(?:US\$|NZ\$|A\$|C\$|[$€£¥₹])\s*)?\d", text)
        else "plus"
        if re.search(r"(?<!\d)\+\s*\d", text)
        else "none"
    )
    number_matches = tuple(re.finditer(r"\d[\d,\.\u066b\u066c]*", text))
    numeric_value = "|".join(_normalize_numeric(match.group(0)) for match in number_matches)
    marker = _marker(span.kind, text, number_matches[0].group(0) if number_matches else "")
    date_components = ""
    time_components = ""
    am_pm = ""
    phone_components = ""
    kind = span.kind
    if span.kind == "date-time":
        date_components, time_components, am_pm = _date_time_identity(text)
        if date_components:
            kind = "date"
            numeric_value = date_components
        elif time_components:
            kind = "time"
            numeric_value = time_components
    if span.kind == "phone":
        phone_components = re.sub(r"\D+", "|", text).strip("|")
        numeric_value = phone_components
    return ProtectedToken(
        text,
        kind,
        numeric_value,
        sign,
        marker,
        date_components,
        time_components,
        am_pm,
        phone_components,
    )


def _date_time_identity(text: str) -> tuple[str, str, str]:
    month_match = re.search(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if month_match is not None:
        month = _MONTHS[month_match.group(1).casefold()]
        day = f"{int(month_match.group(2)):02d}"
        year = month_match.group(3)
        return f"{year}-{month}-{day}", "", ""
    iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso is not None:
        return (
            f"{iso.group(1)}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}",
            "",
            "",
        )
    slash = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", text)
    if slash is not None:
        year = slash.group(3)
        if len(year) == 2:
            year = f"20{year}"
        return (
            f"{year}-{int(slash.group(1)):02d}-{int(slash.group(2)):02d}",
            "",
            "",
        )
    japanese = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if japanese is not None:
        return (
            f"{japanese.group(1)}-{int(japanese.group(2)):02d}-{int(japanese.group(3)):02d}",
            "",
            "",
        )
    time_match = re.search(r"(\d{1,2}):(\d{2})(?:\s?([AP]M))?", text, re.IGNORECASE)
    if time_match is not None:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        period = (time_match.group(3) or "").upper()
        if period == "PM" and hour < 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0
        return "", f"{hour:02d}:{minute:02d}", period
    return "", "", ""


def _normalize_numeric(value: str) -> str:
    raw = (
        value.replace("\u2212", "-")
        .replace("\uff0d", "-")
        .replace("\u066b", ".")
        .replace("\u066c", ",")
        .replace(" ", "")
    )
    sign = "-" if raw.startswith("-") else ""
    raw = raw.lstrip("+-")
    raw = "".join(
        str(unicodedata.digit(character)) if character.isdigit() else character for character in raw
    )
    if "," in raw and "." in raw:
        decimal_separator = "." if raw.rfind(".") > raw.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        raw = raw.replace(grouping_separator, "").replace(decimal_separator, ".")
    elif "," in raw:
        parts = raw.split(",")
        raw = "".join(parts) if len(parts[-1]) == 3 and all(parts) else ".".join(parts)
    try:
        normalized = format(Decimal(raw), "f")
    except (InvalidOperation, ValueError):
        normalized = raw
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    normalized = normalized or "0"
    return sign + normalized


def _marker(kind: str, text: str, number: str) -> str:
    remainder = text.replace(number, "")
    if kind == "currency":
        marker_text = re.sub(r"^[+\-\u2212\uff0d\s]+|\s+$", "", remainder)
        marker_text = marker_text.strip().casefold()
        for alias, identity in _CURRENCY_ALIASES.items():
            if alias.casefold() == marker_text:
                return identity
        return "currency"
    if kind == "unit":
        unit = re.sub(r"[^%A-Za-z°/]+", "", remainder).casefold()
        return _UNIT_ALIASES.get(unit, unit)
    if kind == "percentage":
        return "%"
    if kind in {"date-time", "date", "time", "phone", "abbreviation"}:
        return kind
    return "number"


def _matches(expected: ProtectedToken, actual: ProtectedToken) -> bool:
    """Match only identical semantic kinds and identities."""
    if expected.kind != actual.kind:
        return False
    if expected.kind == "date":
        return expected.date_components == actual.date_components
    if expected.kind == "time":
        return expected.time_components == actual.time_components and expected.am_pm == actual.am_pm
    if expected.kind == "phone":
        return expected.phone_components == actual.phone_components
    return expected.identity() == actual.identity()


def _same_multiset(expected: Sequence[ProtectedToken], actual: Sequence[ProtectedToken]) -> bool:
    return sorted(token.identity() for token in expected) == sorted(
        token.identity() for token in actual
    )
