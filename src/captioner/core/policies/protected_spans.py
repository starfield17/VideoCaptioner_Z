"""Deterministic protected-span detection for subtitle line and cue breaks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str
    text: str


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "currency",
        re.compile(
            r"(?:US\$|[$€£¥₹])\s?[+-]?\d[\d,]*(?:\.\d+)?|[+-]?\d[\d,]*(?:\.\d+)?\s?(?:元|円|€|£|ドル)"
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
            r"[+-]?\d[\d,]*(?:\.\d+)?\s?(?:kg|g|km/h|km|m|cm|mm|mph|°C|°F|%|Hz|kHz|GB|MB|\u00d7\s?\d[\d,]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "number",
        re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?(?:[:/]\d[\d,]*(?:\.\d+)?)?"),
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
