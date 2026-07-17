"""Deterministic protected-span detection for subtitle line and cue breaks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.policies.quantity_scanner import QuantityFact, scan_quantity_facts


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


@dataclass(frozen=True, slots=True)
class _ProtectedItem:
    span: ProtectedSpan
    token: ProtectedToken


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

# Quantity facts intentionally use the deterministic scanner below.  Regex
# remains only for the isolated legacy categories listed here.
_PHONE_PATTERN = re.compile(r"\+\d{1,3}(?:\s+\d{3,4}){2,3}")
_DATE_TIME_PATTERN = re.compile(
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
)
_ABBREVIATION_PATTERN = re.compile(
    r"(?<!\w)(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e)\.", re.IGNORECASE
)
_LEGACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("phone", _PHONE_PATTERN),
    ("date-time", _DATE_TIME_PATTERN),
    ("abbreviation", _ABBREVIATION_PATTERN),
)


def _legacy_items(text: str) -> tuple[_ProtectedItem, ...]:
    candidates: list[tuple[int, ProtectedSpan]] = []
    for priority, (kind, pattern) in enumerate(_LEGACY_PATTERNS):
        for match in pattern.finditer(text):
            candidates.append(
                (
                    priority,
                    ProtectedSpan(match.start(), match.end(), kind, match.group(0)),
                )
            )
    candidates.sort(key=lambda candidate: (candidate[1].start, -candidate[1].end, candidate[0]))
    found: list[_ProtectedItem] = []
    for _, span in candidates:
        if any(_overlap(span, item.span) for item in found):
            continue
        found.append(_ProtectedItem(span, _classify_span(span)))
    return tuple(found)


def _quantity_fact_to_item(
    text: str,
    fact: QuantityFact,
) -> _ProtectedItem:
    span = ProtectedSpan(
        fact.start,
        fact.end,
        fact.kind,
        text[fact.start : fact.end],
    )
    token = ProtectedToken(
        span.text,
        fact.kind,
        fact.numeric_value,
        fact.sign,
        fact.marker,
    )
    return _ProtectedItem(span, token)


def _protected_items(text: str) -> tuple[_ProtectedItem, ...]:
    legacy = _legacy_items(text)
    occupied_ranges = tuple((item.span.start, item.span.end) for item in legacy)
    quantity = tuple(
        _quantity_fact_to_item(text, fact)
        for fact in scan_quantity_facts(text, occupied_ranges=occupied_ranges)
    )
    items = sorted(
        (*legacy, *quantity),
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.kind,
        ),
    )
    previous_end = 0
    for item in items:
        if item.span.start < previous_end:
            raise AppError(
                "llm.protected_scanner_invalid",
                {
                    "reason": "overlap",
                    "position": item.span.start,
                    "category": "pipeline",
                },
            )
        previous_end = item.span.end
    return tuple(items)


def find_protected_spans(text: str) -> tuple[ProtectedSpan, ...]:
    return tuple(item.span for item in _protected_items(text))


def protected_tokens(text: str) -> tuple[ProtectedToken, ...]:
    """Return protected semantic tokens from one shared scan."""
    return tuple(item.token for item in _protected_items(text))


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
    sign = "plus" if text.startswith("+") else "none"
    numeric_value = ""
    marker = _legacy_marker(span.kind)
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
        phone_components = "|".join(
            "".join(character for character in part if character.isdigit()) for part in text.split()
        )
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


def _legacy_marker(kind: str) -> str:
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
