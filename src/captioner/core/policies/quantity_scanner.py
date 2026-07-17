"""Deterministic, regex-free scanning of protected numeric facts."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from captioner.core.domain.errors import AppError
from captioner.core.policies.quantity_aliases import (
    _CURRENCY_BOUNDARY_PREFIX_ALIASES,
    _LONGEST_CURRENCY_PREFIX_ALIASES,
    _LONGEST_CURRENCY_SUFFIX_ALIASES,
    _LONGEST_PERCENTAGE_ALIASES,
    _LONGEST_UNIT_ALIASES,
    _QUANTITY_START_CHARACTERS,
    _AliasSpec,
)

__all__ = ["QuantityFact", "scan_quantity_facts"]


@dataclass(frozen=True, slots=True)
class QuantityFact:
    start: int
    end: int
    kind: str
    numeric_value: str
    sign: str
    marker: str


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


class _ContinuationKind(StrEnum):
    END = "end"
    SAFE_PUNCTUATION = "safe_punctuation"
    SEPARATED_WORD = "separated_word"
    ATTACHED_WORD_SUFFIX = "attached_word_suffix"
    ATTACHED_SYMBOL_SUFFIX = "attached_symbol_suffix"
    SLASH_SUFFIX = "slash_suffix"


_INLINE_SPACES = " \t"
_MAX_INLINE_IDENTITY_LENGTH = 128
_HASH_HEX_LENGTH = 24
_SAFE_TERMINAL_PUNCTUATION = frozenset(
    {
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        ")",
        "]",
        "}",
        '"',
        "'",
        "…",
        "、",
        "。",
        "\uff0c",
        "\uff01",
        "\uff1f",
        "\uff1b",
        "\uff1a",
        "”",
        "\u2019",
        "」",
        "』",
        "】",
        "》",
        "〉",
        "\uff09",
    }
)


def _bounded_identity(canonical: str) -> str:
    if len(canonical) <= _MAX_INLINE_IDENTITY_LENGTH:
        return canonical
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_HASH_HEX_LENGTH]
    return f"overflow:{len(canonical)}:{digest}"


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _skip_inline_spaces(text: str, position: int) -> int:
    while position < len(text) and text[position] in _INLINE_SPACES:
        position += 1
    return position


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
    if text[position] in _SAFE_TERMINAL_PUNCTUATION:
        return _ContinuationKind.SAFE_PUNCTUATION, position
    candidate = _skip_inline_spaces(text, position)
    if candidate >= len(text):
        return _ContinuationKind.END, candidate
    if text[candidate] == "/":
        return _ContinuationKind.SLASH_SUFFIX, candidate
    if _is_word_character(text[candidate]):
        return _ContinuationKind.SEPARATED_WORD, candidate
    if text[candidate] in _SAFE_TERMINAL_PUNCTUATION:
        return _ContinuationKind.SAFE_PUNCTUATION, candidate
    return _ContinuationKind.ATTACHED_SYMBOL_SUFFIX, candidate


def _is_slash_component_character(character: str) -> bool:
    return (
        character != "/"
        and character not in _INLINE_SPACES
        and character not in "\r\n"
        and character not in _SAFE_TERMINAL_PUNCTUATION
    )


def _scan_slash_tail(text: str, position: int) -> _SlashTailMatch | None:
    slash_position = _skip_inline_spaces(text, position)
    if slash_position >= len(text) or text[slash_position] != "/":
        return None

    cursor = slash_position
    components: list[str] = []
    last_end = slash_position + 1
    while cursor < len(text) and text[cursor] == "/":
        slash_start = cursor
        cursor = _skip_inline_spaces(text, cursor + 1)
        component_start = cursor
        while cursor < len(text) and _is_slash_component_character(text[cursor]):
            cursor += 1
        if component_start < cursor:
            components.append(text[component_start:cursor].casefold())
            last_end = cursor
        else:
            last_end = max(last_end, slash_start + 1)
        probe = _skip_inline_spaces(text, cursor)
        if probe < len(text) and text[probe] == "/":
            cursor = probe
            continue
        break
    identity = _bounded_identity("/".join(components))
    return _SlashTailMatch(last_end, identity)


def _is_attached_tail_boundary(character: str) -> bool:
    return (
        character in _INLINE_SPACES
        or character in "\r\n"
        or character in _SAFE_TERMINAL_PUNCTUATION
    )


def _scan_attached_terminal_tail(
    text: str,
    position: int,
) -> _AttachedTailMatch | None:
    if position >= len(text) or _is_attached_tail_boundary(text[position]):
        return None
    cursor = position
    while cursor < len(text) and not _is_attached_tail_boundary(text[cursor]):
        cursor += 1
    canonical = text[position:cursor].casefold()
    return _AttachedTailMatch(cursor, _bounded_identity(canonical))


def _quantity_fact(
    start: int,
    end: int,
    kind: str,
    numeric_value: str,
    sign: str,
    marker: str,
) -> QuantityFact:
    return QuantityFact(start, end, kind, numeric_value, sign, marker)


def _finish_quantity_fact(
    text: str,
    start: int,
    base_end: int,
    kind: str,
    numeric_value: str,
    sign: str,
    marker: str,
) -> QuantityFact:
    continuation, continuation_position = _classify_fact_continuation(text, base_end)
    base_marker = marker or "none"
    if continuation is _ContinuationKind.SLASH_SUFFIX:
        tail = _scan_slash_tail(text, base_end)
        if tail is not None:
            return _quantity_fact(
                start,
                tail.end,
                "unsupported-compound",
                numeric_value,
                sign,
                f"unsupported:{kind}:{base_marker}/{tail.identity}",
            )
    if continuation in {
        _ContinuationKind.ATTACHED_WORD_SUFFIX,
        _ContinuationKind.ATTACHED_SYMBOL_SUFFIX,
    }:
        if continuation is _ContinuationKind.ATTACHED_SYMBOL_SUFFIX:
            separated = continuation_position > base_end
            starts_currency = _scan_currency_prefix_fact_at(text, continuation_position)
            starts_number = _scan_numeric_fact_at(text, continuation_position)
            if separated and (starts_currency is not None or starts_number is not None):
                return _quantity_fact(
                    start,
                    base_end,
                    kind,
                    numeric_value,
                    sign,
                    marker,
                )
        tail = _scan_attached_terminal_tail(text, continuation_position)
        if tail is not None:
            tail_category = (
                "word" if continuation is _ContinuationKind.ATTACHED_WORD_SUFFIX else "symbol"
            )
            return _quantity_fact(
                start,
                tail.end,
                "unsupported-attached",
                numeric_value,
                sign,
                f"unsupported:{kind}:{base_marker}:{tail_category}:{tail.identity}",
            )
    return _quantity_fact(start, base_end, kind, numeric_value, sign, marker)


def _scan_currency_prefix_fact_at(
    text: str,
    position: int,
) -> QuantityFact | None:
    sign = _scan_sign(text, position)
    alias = _match_alias(text, sign.end, _LONGEST_CURRENCY_PREFIX_ALIASES)
    if alias is None or _is_embedded_currency_code_suffix(text, sign.end, alias.text):
        return None
    number = _scan_number(text, _skip_inline_spaces(text, alias.end))
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


def _scan_numeric_fact_at(text: str, position: int) -> QuantityFact | None:
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


def _scan_quantity_fact_at(text: str, position: int) -> QuantityFact | None:
    prefix = _scan_currency_prefix_fact_at(text, position)
    if prefix is not None:
        return prefix
    return _scan_numeric_fact_at(text, position)


def scan_quantity_facts(
    text: str,
    occupied_ranges: Sequence[tuple[int, int]] = (),
) -> tuple[QuantityFact, ...]:
    facts: list[QuantityFact] = []
    occupied = tuple(occupied_ranges)
    occupied_index = 0
    position = 0
    while position < len(text):
        while occupied_index < len(occupied) and occupied[occupied_index][1] <= position:
            occupied_index += 1
        if occupied_index < len(occupied):
            occupied_start, occupied_end = occupied[occupied_index]
            if occupied_start <= position < occupied_end:
                position = occupied_end
                continue
        if (
            not text[position].isdigit()
            and text[position].casefold() not in _QUANTITY_START_CHARACTERS
        ):
            position += 1
            continue
        fact = _scan_quantity_fact_at(text, position)
        if fact is None:
            position += 1
            continue
        if fact.end <= position:
            raise AppError(
                "llm.protected_scanner_invalid",
                {"reason": "no_progress", "position": position, "category": "quantity"},
            )
        if occupied_index < len(occupied) and fact.end > occupied[occupied_index][0]:
            position = max(position + 1, occupied[occupied_index][0])
            continue
        facts.append(fact)
        position = fact.end
    return tuple(facts)


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
