"""Deterministic protected-span detection for subtitle line and cue breaks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from types import MappingProxyType

from captioner.core.domain.errors import AppError


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
_CURRENCY_FREE_PREFIX_ALIASES = MappingProxyType(
    {
        "$": _CURRENCY_SYMBOL_ALIASES["$"],
        "€": _CURRENCY_SYMBOL_ALIASES["€"],
        "£": _CURRENCY_SYMBOL_ALIASES["£"],
        "¥": _CURRENCY_SYMBOL_ALIASES["¥"],
        "₹": _CURRENCY_SYMBOL_ALIASES["₹"],
        "元": _CURRENCY_SYMBOL_ALIASES["元"],
        "人民币": _CURRENCY_SYMBOL_ALIASES["人民币"],
        "円": _CURRENCY_SYMBOL_ALIASES["円"],
        "ドル": _CURRENCY_SYMBOL_ALIASES["ドル"],
    }
)
_CURRENCY_BOUNDARY_PREFIX_ALIASES = MappingProxyType(
    {
        "US$": _CURRENCY_SYMBOL_ALIASES["US$"],
        "A$": _CURRENCY_SYMBOL_ALIASES["A$"],
        "C$": _CURRENCY_SYMBOL_ALIASES["C$"],
        "NZ$": _CURRENCY_SYMBOL_ALIASES["NZ$"],
        "USD": _CURRENCY_SYMBOL_ALIASES["USD"],
        "EUR": _CURRENCY_SYMBOL_ALIASES["EUR"],
        "GBP": _CURRENCY_SYMBOL_ALIASES["GBP"],
        "JPY": _CURRENCY_SYMBOL_ALIASES["JPY"],
        "CNY": _CURRENCY_SYMBOL_ALIASES["CNY"],
        "RMB": _CURRENCY_SYMBOL_ALIASES["RMB"],
    }
)
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


@dataclass(frozen=True, slots=True)
class _AliasSpec:
    text: str
    identity: str
    boundary_before: bool
    boundary_after: bool
    requires_space_before: bool = False
    requires_space_after: bool = False


@dataclass(frozen=True, slots=True)
class _NumberMatch:
    start: int
    end: int
    raw: str
    normalized: str


@dataclass(frozen=True, slots=True)
class _SignMatch:
    end: int
    semantic: str


@dataclass(frozen=True, slots=True)
class _AliasMatch:
    start: int
    end: int
    text: str
    identity: str


@dataclass(frozen=True, slots=True)
class _SlashTailMatch:
    end: int
    identity: str


@dataclass(frozen=True, slots=True)
class _AttachedTailMatch:
    end: int
    identity: str


@dataclass(frozen=True, slots=True)
class _ProtectedItem:
    span: ProtectedSpan
    token: ProtectedToken


class _ContinuationKind(StrEnum):
    END = "end"
    PUNCTUATION = "punctuation"
    SEPARATED_WORD = "separated_word"
    ATTACHED_WORD_SUFFIX = "attached_word_suffix"
    SLASH_SUFFIX = "slash_suffix"


_INLINE_SPACES = " \t"
_MAX_UNSUPPORTED_TAIL_LENGTH = 128


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _skip_inline_spaces(text: str, position: int) -> int:
    while position < len(text) and text[position] in _INLINE_SPACES:
        position += 1
    return position


def _sorted_alias_specs(
    aliases: Mapping[str, str],
    *,
    boundary_before: bool,
    boundary_after: bool,
    requires_space_before: bool = False,
    requires_space_after: bool = False,
) -> tuple[_AliasSpec, ...]:
    return tuple(
        sorted(
            (
                _AliasSpec(
                    alias,
                    identity,
                    boundary_before,
                    boundary_after,
                    requires_space_before,
                    requires_space_after,
                )
                for alias, identity in aliases.items()
            ),
            key=lambda spec: len(spec.text),
            reverse=True,
        )
    )


_LONGEST_CURRENCY_PREFIX_ALIASES = tuple(
    sorted(
        (
            *_sorted_alias_specs(
                _CURRENCY_FREE_PREFIX_ALIASES,
                boundary_before=False,
                boundary_after=False,
            ),
            *_sorted_alias_specs(
                _CURRENCY_BOUNDARY_PREFIX_ALIASES,
                boundary_before=True,
                boundary_after=False,
            ),
            *_sorted_alias_specs(
                _CURRENCY_WORD_ALIASES,
                boundary_before=True,
                boundary_after=True,
                requires_space_after=True,
            ),
        ),
        key=lambda spec: len(spec.text),
        reverse=True,
    )
)
_LONGEST_CURRENCY_SUFFIX_ALIASES = tuple(
    sorted(
        (
            *_sorted_alias_specs(
                _CURRENCY_SYMBOL_ALIASES,
                boundary_before=False,
                boundary_after=True,
            ),
            *_sorted_alias_specs(
                _CURRENCY_WORD_ALIASES,
                boundary_before=False,
                boundary_after=True,
                requires_space_before=True,
            ),
        ),
        key=lambda spec: len(spec.text),
        reverse=True,
    )
)
_LONGEST_PERCENTAGE_ALIASES = tuple(
    sorted(
        (
            *_sorted_alias_specs(
                {"%": "%"},
                boundary_before=False,
                boundary_after=True,
            ),
            *_sorted_alias_specs(
                _PERCENTAGE_WORD_ALIASES,
                boundary_before=False,
                boundary_after=True,
                requires_space_before=True,
            ),
        ),
        key=lambda spec: len(spec.text),
        reverse=True,
    )
)
_LONGEST_UNIT_ALIASES = tuple(
    sorted(
        (
            *_sorted_alias_specs(
                _UNIT_COMPACT_ALIASES,
                boundary_before=False,
                boundary_after=True,
            ),
            *_sorted_alias_specs(
                _UNIT_WORD_ALIASES,
                boundary_before=False,
                boundary_after=True,
                requires_space_before=True,
            ),
        ),
        key=lambda spec: len(spec.text),
        reverse=True,
    )
)
_QUANTITY_START_CHARACTERS = frozenset(
    {
        "+",
        "-",
        "\u2212",
        "\uff0d",
        *(alias.text[0].casefold() for alias in _LONGEST_CURRENCY_PREFIX_ALIASES),
    }
)


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


def _match_alias(
    text: str,
    position: int,
    aliases: Sequence[_AliasSpec],
) -> _AliasMatch | None:
    for alias in aliases:
        if alias.boundary_before and position > 0 and _is_word_character(text[position - 1]):
            continue
        end = position + len(alias.text)
        if end > len(text) or text[position:end].casefold() != alias.text.casefold():
            continue
        if alias.boundary_after and end < len(text) and _is_word_character(text[end]):
            continue
        if alias.requires_space_after and (end >= len(text) or text[end] not in _INLINE_SPACES):
            continue
        return _AliasMatch(position, end, text[position:end], alias.identity)
    return None


def _is_embedded_currency_code_suffix(
    text: str,
    position: int,
    alias_text: str,
) -> bool:
    alias_end = position + len(alias_text)
    for boundary_alias in _CURRENCY_BOUNDARY_PREFIX_ALIASES:
        if not boundary_alias.endswith(alias_text):
            continue
        boundary_start = alias_end - len(boundary_alias)
        if boundary_start < 0:
            continue
        if text[boundary_start:alias_end].casefold() != boundary_alias.casefold():
            continue
        if boundary_start > 0 and _is_word_character(text[boundary_start - 1]):
            return True
    return False


def _match_suffix_alias(
    text: str,
    position: int,
    aliases: Sequence[_AliasSpec],
) -> _AliasMatch | None:
    for alias in aliases:
        candidate = position
        if alias.requires_space_before:
            if candidate >= len(text) or text[candidate] not in _INLINE_SPACES:
                continue
            candidate = _skip_inline_spaces(text, candidate)
        else:
            candidate = _skip_inline_spaces(text, candidate)
        match = _match_alias(text, candidate, (alias,))
        if match is not None:
            return match
    return None


def _scan_sign(text: str, position: int) -> _SignMatch:
    if position >= len(text) or text[position] not in "+-\u2212\uff0d":
        return _SignMatch(position, "none")
    semantic = "plus" if text[position] == "+" else "minus"
    return _SignMatch(_skip_inline_spaces(text, position + 1), semantic)


def _scan_number(text: str, position: int) -> _NumberMatch | None:
    if position >= len(text) or not text[position].isdigit():
        return None
    end = position + 1
    while end < len(text) and (text[end].isdigit() or text[end] in ",.\u066b\u066c"):
        end += 1
    raw = text[position:end]
    return _NumberMatch(position, end, raw, _normalize_numeric(raw))


def _classify_fact_continuation(
    text: str,
    position: int,
) -> tuple[_ContinuationKind, int]:
    if position >= len(text):
        return _ContinuationKind.END, position
    if _is_word_character(text[position]):
        return _ContinuationKind.ATTACHED_WORD_SUFFIX, position
    candidate = _skip_inline_spaces(text, position)
    if candidate >= len(text):
        return _ContinuationKind.END, candidate
    if text[candidate] == "/":
        return _ContinuationKind.SLASH_SUFFIX, candidate
    if _is_word_character(text[candidate]):
        return _ContinuationKind.SEPARATED_WORD, candidate
    return _ContinuationKind.PUNCTUATION, candidate


def _is_slash_component_character(character: str) -> bool:
    return _is_word_character(character) or character in "$€£¥₹%°+-"


def _scan_slash_tail(text: str, position: int) -> _SlashTailMatch | None:
    slash_position = _skip_inline_spaces(text, position)
    if slash_position >= len(text) or text[slash_position] != "/":
        return None
    limit = min(len(text), slash_position + _MAX_UNSUPPORTED_TAIL_LENGTH)
    cursor = slash_position
    components: list[str] = []
    last_end = slash_position + 1
    overflow = False
    while cursor < len(text) and text[cursor] == "/":
        if cursor >= limit:
            overflow = True
            break
        cursor += 1
        cursor = _skip_inline_spaces(text, cursor)
        component_start = cursor
        while cursor < len(text) and cursor < limit and _is_slash_component_character(text[cursor]):
            cursor += 1
        if cursor == limit and cursor < len(text) and _is_slash_component_character(text[cursor]):
            overflow = True
        component = text[component_start:cursor].casefold()
        components.append(component)
        last_end = cursor if component else max(cursor, slash_position + 1)
        probe = _skip_inline_spaces(text, cursor)
        if probe < len(text) and text[probe] == "/" and probe < limit:
            cursor = probe
            continue
        break
    identity = "overflow" if overflow else "/".join(components)
    return _SlashTailMatch(max(last_end, slash_position + 1), identity)


def _scan_attached_word_tail(text: str, position: int) -> _AttachedTailMatch | None:
    if position >= len(text) or not _is_word_character(text[position]):
        return None

    limit = min(len(text), position + _MAX_UNSUPPORTED_TAIL_LENGTH)
    cursor = position
    while cursor < limit and _is_word_character(text[cursor]):
        cursor += 1
    overflow = cursor == limit and cursor < len(text) and _is_word_character(text[cursor])
    identity = "overflow" if overflow else text[position:cursor].casefold()
    return _AttachedTailMatch(cursor, identity)


def _quantity_item(
    text: str,
    start: int,
    end: int,
    kind: str,
    numeric_value: str,
    sign: str,
    marker: str,
) -> _ProtectedItem:
    span = ProtectedSpan(start, end, kind, text[start:end])
    token = ProtectedToken(span.text, kind, numeric_value, sign, marker)
    return _ProtectedItem(span, token)


def _finish_quantity_fact(
    text: str,
    start: int,
    base_end: int,
    kind: str,
    numeric_value: str,
    sign: str,
    marker: str,
) -> _ProtectedItem:
    continuation, continuation_position = _classify_fact_continuation(text, base_end)
    if continuation is _ContinuationKind.SLASH_SUFFIX:
        tail = _scan_slash_tail(text, base_end)
        if tail is not None:
            base_marker = marker or "none"
            unsupported_marker = f"unsupported:{kind}:{base_marker}/{tail.identity}"
            return _quantity_item(
                text,
                start,
                tail.end,
                "unsupported-compound",
                numeric_value,
                sign,
                unsupported_marker,
            )
    if continuation is _ContinuationKind.ATTACHED_WORD_SUFFIX:
        tail = _scan_attached_word_tail(text, continuation_position)
        if tail is not None:
            base_marker = marker or "none"
            unsupported_marker = f"unsupported:{kind}:{base_marker}+{tail.identity}"
            return _quantity_item(
                text,
                start,
                tail.end,
                "unsupported-attached",
                numeric_value,
                sign,
                unsupported_marker,
            )
    return _quantity_item(text, start, base_end, kind, numeric_value, sign, marker)


def _scan_currency_prefix_fact_at(
    text: str,
    position: int,
) -> _ProtectedItem | None:
    sign = _scan_sign(text, position)
    alias = _match_alias(text, sign.end, _LONGEST_CURRENCY_PREFIX_ALIASES)
    if alias is None:
        return None
    if _is_embedded_currency_code_suffix(text, sign.end, alias.text):
        return None
    number_position = alias.end
    number_position = _skip_inline_spaces(text, number_position)
    number = _scan_number(text, number_position)
    if number is None:
        return None
    return _finish_quantity_fact(
        text,
        position,
        number.end,
        "currency",
        number.normalized,
        sign.semantic,
        alias.identity,
    )


def _scan_numeric_fact_at(text: str, position: int) -> _ProtectedItem | None:
    sign = _scan_sign(text, position)
    number = _scan_number(text, sign.end)
    if number is None:
        return None

    dimension_position = _skip_inline_spaces(text, number.end)
    if dimension_position < len(text) and text[dimension_position] == "\u00d7":
        second = _scan_number(text, _skip_inline_spaces(text, dimension_position + 1))
        if second is not None:
            return _finish_quantity_fact(
                text,
                position,
                second.end,
                "unit",
                f"{number.normalized}|{second.normalized}",
                sign.semantic,
                "",
            )

    if number.end < len(text) and text[number.end] == "/":
        second = _scan_number(text, number.end + 1)
        if second is not None:
            return _finish_quantity_fact(
                text,
                position,
                second.end,
                "number",
                f"{number.normalized}|{second.normalized}",
                sign.semantic,
                "number",
            )

    percentage = _match_suffix_alias(text, number.end, _LONGEST_PERCENTAGE_ALIASES)
    if percentage is not None:
        return _finish_quantity_fact(
            text,
            position,
            percentage.end,
            "percentage",
            number.normalized,
            sign.semantic,
            percentage.identity,
        )

    currency = _match_suffix_alias(text, number.end, _LONGEST_CURRENCY_SUFFIX_ALIASES)
    if currency is not None:
        return _finish_quantity_fact(
            text,
            position,
            currency.end,
            "currency",
            number.normalized,
            sign.semantic,
            currency.identity,
        )

    unit = _match_suffix_alias(text, number.end, _LONGEST_UNIT_ALIASES)
    if unit is not None:
        return _finish_quantity_fact(
            text,
            position,
            unit.end,
            "unit",
            number.normalized,
            sign.semantic,
            unit.identity,
        )

    return _finish_quantity_fact(
        text,
        position,
        number.end,
        "number",
        number.normalized,
        sign.semantic,
        "number",
    )


def _scan_quantity_fact_at(text: str, position: int) -> _ProtectedItem | None:
    prefix = _scan_currency_prefix_fact_at(text, position)
    if prefix is not None:
        return prefix
    return _scan_numeric_fact_at(text, position)


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


def _protected_items(text: str) -> tuple[_ProtectedItem, ...]:
    legacy = _legacy_items(text)
    quantity: list[_ProtectedItem] = []
    position = 0
    legacy_index = 0
    while position < len(text):
        while legacy_index < len(legacy) and legacy[legacy_index].span.end <= position:
            legacy_index += 1
        if legacy_index < len(legacy):
            occupied = legacy[legacy_index].span
            if occupied.start <= position < occupied.end:
                quantity.append(legacy[legacy_index])
                position = occupied.end
                legacy_index += 1
                continue
        if (
            not text[position].isdigit()
            and text[position].casefold() not in _QUANTITY_START_CHARACTERS
        ):
            position += 1
            continue
        item = _scan_quantity_fact_at(text, position)
        if item is None:
            position += 1
            continue
        if item.span.end <= position:
            raise AppError(
                "llm.protected_scanner_invalid",
                {"reason": "no_progress", "position": position, "category": "quantity"},
            )
        if legacy_index < len(legacy) and _overlap(item.span, legacy[legacy_index].span):
            position = max(position + 1, legacy[legacy_index].span.start)
            continue
        quantity.append(item)
        position = item.span.end
    quantity.sort(key=lambda item: (item.span.start, item.span.end, item.span.kind))
    return tuple(quantity)


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
