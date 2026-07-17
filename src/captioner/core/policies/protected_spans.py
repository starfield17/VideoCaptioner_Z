"""Deterministic protected-span detection for subtitle line and cue breaks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


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

    @property
    def digits(self) -> str:
        return "".join(character for character in self.text if character.isdigit())

    @property
    def percent(self) -> bool:
        return self.marker == "%"


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "currency",
        re.compile(
            r"[+-]?\s*(?:US\$|[$€£¥₹])\s?\d[\d,\.\u066B\u066C]*|[+-]?\d[\d,\.\u066B\u066C]*\s?(?:元|円|€|£|ドル)"
        ),
    ),
    (
        "phone",
        re.compile(r"\+\d{1,3}(?:\s+\d{3,4}){2,3}"),
    ),
    (
        "date-time",
        re.compile(
            r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}:\d{2}(?:\s?[AP]M)?|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})",
            re.IGNORECASE,
        ),
    ),
    (
        "unit",
        re.compile(
            r"[+-]?\d[\d,\.\u066B\u066C]*\s?(?:kg|g|km/h|km|m|cm|mm|mph|°C|°F|%|Hz|kHz|GB|MB|\u00d7\s?\d[\d,]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "number",
        re.compile(r"[+-]?\d[\d,\.\u066B\u066C]*(?:[:/]\d[\d,\.\u066B\u066C]*)?"),
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


def protected_tokens_preserved(source: str, output: str) -> bool:
    """Check ordered value/sign/percent/currency/unit preservation."""
    expected = protected_tokens(source)
    if not expected:
        return True
    actual = protected_tokens(output)
    cursor = 0
    for token in expected:
        match = next(
            (index for index in range(cursor, len(actual)) if _matches(token, actual[index])),
            None,
        )
        if match is None:
            return False
        cursor = match + 1
    return True


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
        if re.search(r"(?<!\d)[-\u2212]\s*(?:(?:US\$|[$€£¥₹])\s*)?\d", text)
        else "plus"
        if re.search(r"(?<!\d)\+\s*\d", text)
        else "none"
    )
    number_matches = tuple(re.finditer(r"\d[\d,\.\u066b\u066c]*", text))
    numeric_value = "|".join(_normalize_numeric(match.group(0)) for match in number_matches)
    marker = _marker(span.kind, text, number_matches[0].group(0) if number_matches else "")
    return ProtectedToken(text, span.kind, numeric_value, sign, marker)


def _normalize_numeric(value: str) -> str:
    raw = (
        value.replace("\u2212", "-").replace("\u066b", ".").replace("\u066c", ",").replace(" ", "")
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
        symbols = re.findall(r"US\$|[$€£¥₹]|[A-Z]{3}", text)
        return symbols[0].casefold() if symbols else "currency"
    if kind == "unit":
        unit = re.sub(r"[^%A-Za-z°/]+", "", remainder).casefold()
        return unit
    if kind in {"date-time", "phone", "abbreviation"}:
        return kind
    return "number"


def _matches(expected: ProtectedToken, actual: ProtectedToken) -> bool:
    if expected.kind in {"currency", "unit", "date-time", "phone", "abbreviation"}:
        if actual.kind != expected.kind or actual.marker != expected.marker:
            return False
    elif actual.kind not in {"number", "currency", "unit"}:
        return False
    return (
        expected.numeric_value == actual.numeric_value
        and expected.sign == actual.sign
        and (expected.marker == "%") == (actual.marker == "%")
    )
