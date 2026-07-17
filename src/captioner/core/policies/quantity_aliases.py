"""Immutable aliases used by the deterministic protected-quantity scanner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "_CURRENCY_BOUNDARY_PREFIX_ALIASES",
    "_LONGEST_CURRENCY_PREFIX_ALIASES",
    "_LONGEST_CURRENCY_SUFFIX_ALIASES",
    "_LONGEST_PERCENTAGE_ALIASES",
    "_LONGEST_UNIT_ALIASES",
    "_QUANTITY_START_CHARACTERS",
    "_AliasSpec",
]


# These are deliberately explicit. In particular, "$", "US$", and
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
_PERCENTAGE_SYMBOL_ALIASES = MappingProxyType({"%": "%"})
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
                _PERCENTAGE_SYMBOL_ALIASES,
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
