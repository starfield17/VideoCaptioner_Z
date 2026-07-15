"""Strict JSON types and a small generic result container."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


def validate_json_value(value: object) -> None:
    """Reject values outside the finite, JSON-compatible value subset."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError
            validate_json_value(item)
        return
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        for item in sequence:
            validate_json_value(item)
        return
    raise TypeError


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
