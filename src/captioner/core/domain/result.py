"""Strict JSON types and a small generic result container."""

from __future__ import annotations

from dataclasses import dataclass

type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Result[T]:
    """Represent either a value or an application error."""

    value: T | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, value: T) -> Result[T]:
        return cls(value=value)

    @classmethod
    def failure(cls, error: Exception) -> Result[T]:
        return cls(error=error)
