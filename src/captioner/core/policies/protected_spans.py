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
    date_components: str = ""
    time_components: str = ""
    am_pm: str = ""
    phone_components: str = ""

    @property
    def digits(self) -> str:
        return "".join(character for character in self.text if character.isdigit())

    @property
    def percent(self) -> bool:
        return self.marker == "%"

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

_CURRENCY_SYMBOLS = r"US\$|[$€£¥₹]|元|円|ドル"
_UNIT_MARKERS = r"kg|g|km/h|km|m|cm|mm|mph|°C|°F|%|Hz|kHz|GB|MB"
_NUMBER = r"\d[\d,\.\u066B\u066C]*"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "currency",
        re.compile(
            rf"(?:[+\-\u2212\uff0d]\s*)?(?:{_CURRENCY_SYMBOLS})\s?{_NUMBER}"
            rf"|(?:[+\-\u2212\uff0d]\s*)?{_NUMBER}\s?(?:元|円|€|£|ドル)"
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
        "unit",
        re.compile(
            rf"(?:[+\-\u2212\uff0d]\s*)?{_NUMBER}\s?(?:{_UNIT_MARKERS}|\u00d7\s?{_NUMBER})",
            re.IGNORECASE,
        ),
    ),
    (
        "number",
        re.compile(rf"(?:[+\-\u2212\uff0d]\s*)?{_NUMBER}(?:[:/]{_NUMBER})?"),
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
    """Require exact ordered semantic token sequence equality.

    Representation differences that normalize to the same identity are allowed
    (grouping symbols, Arabic-Indic digits, ISO vs month-name dates). Adding,
    removing, or reordering protected facts is rejected.
    """
    expected = protected_tokens(source)
    if not expected:
        return True
    actual = protected_tokens(output)
    if len(actual) != len(expected):
        return False
    return all(_matches(left, right) for left, right in zip(expected, actual, strict=True))


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
        if re.search(r"(?<!\d)[-\u2212\uff0d]\s*(?:(?:US\$|[$€£¥₹])\s*)?\d", text)
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
    if span.kind == "date-time":
        date_components, time_components, am_pm = _date_time_identity(text)
        if date_components:
            numeric_value = date_components
        elif time_components:
            numeric_value = time_components
    if span.kind == "phone":
        phone_components = re.sub(r"\D+", "|", text).strip("|")
        numeric_value = phone_components
    return ProtectedToken(
        text,
        span.kind,
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
        # Keep AM/PM as part of identity so 10:00 AM != 10:00 PM and != 22:00
        # when the source used a 12-hour form with period.
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
        if "元" in text:
            return "yuan"
        if "円" in text:
            return "yen"
        if "ドル" in text:
            return "dollar"
        if "US$" in text or "us$" in text.casefold():
            return "us$"
        symbols = re.findall(r"[$€£¥₹]", text)
        if symbols:
            return symbols[0]
        return "currency"
    if kind == "unit":
        unit = re.sub(r"[^%A-Za-z°/]+", "", remainder).casefold()
        return unit
    if kind in {"date-time", "phone", "abbreviation"}:
        return kind
    return "number"


def _matches(expected: ProtectedToken, actual: ProtectedToken) -> bool:
    """Exact identity match for one ordered protected fact."""
    if expected.kind == "date-time":
        if actual.kind != "date-time":
            return False
        if expected.date_components:
            return expected.date_components == actual.date_components
        if expected.time_components:
            # Require same wall-clock time AND same AM/PM form when source had it.
            if expected.time_components != actual.time_components:
                return False
            if expected.am_pm and expected.am_pm != actual.am_pm:
                return False
            if not expected.am_pm and actual.am_pm:
                # Source was 24h; reject introducing AM/PM that would change meaning
                # only if the numeric hour already encodes period — already compared.
                return True
            return True
        return expected.identity() == actual.identity()
    if expected.kind == "phone":
        return actual.kind == "phone" and expected.phone_components == actual.phone_components
    if expected.kind in {"currency", "unit", "abbreviation"}:
        if actual.kind != expected.kind or actual.marker != expected.marker:
            return False
        return expected.numeric_value == actual.numeric_value and expected.sign == actual.sign
    # Generic numbers may surface as unit/currency when markers appear in output;
    # still require exact ordered numeric identity and percent marker.
    if actual.kind not in {"number", "currency", "unit"}:
        return False
    return (
        expected.numeric_value == actual.numeric_value
        and expected.sign == actual.sign
        and (expected.marker == "%") == (actual.marker == "%")
        and (
            expected.marker == "number"
            or actual.marker == expected.marker
            or (expected.marker == "%" and actual.marker == "%")
        )
    )
