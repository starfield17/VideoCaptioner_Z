"""Strict JSON types and a small generic result container."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
type FrozenJsonPrimitive = None | bool | int | float | str
type FrozenJsonValue = (
    FrozenJsonPrimitive | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
)


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Recursively copy a JSON value into immutable containers.

    The returned mapping and tuple containers never retain mutable containers
    owned by the caller.  ``TypeError`` and ``ValueError`` are intentionally
    distinct so callers can report invalid shape and non-finite numbers.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        frozen: dict[str, FrozenJsonValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError
            frozen[key] = freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(freeze_json_value(item) for item in sequence)
    raise TypeError


def thaw_json_value(value: FrozenJsonValue) -> JsonValue:
    """Return a fresh mutable JSON-compatible copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def validate_json_value(value: object) -> None:
    """Reject values outside the finite, JSON-compatible value subset."""
    freeze_json_value(value)


class _Missing:
    """Sentinel type used to distinguish an omitted value from ``None``."""


_MISSING = _Missing()


@dataclass(frozen=True, slots=True, init=False)
class Result[T]:
    """Represent either a value or an application error."""

    value: T | None
    error: Exception | None
    _has_value: bool

    def __init__(self, value: T | _Missing = _MISSING, error: Exception | None = None) -> None:
        has_value = not isinstance(value, _Missing)
        if has_value == (error is not None):
            raise ValueError
        object.__setattr__(self, "value", None if isinstance(value, _Missing) else value)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "_has_value", has_value)

    @property
    def ok(self) -> bool:
        return self._has_value

    @classmethod
    def success(cls, value: T) -> Result[T]:
        return cls(value)

    @classmethod
    def failure(cls, error: Exception) -> Result[T]:
        return cls(error=error)
