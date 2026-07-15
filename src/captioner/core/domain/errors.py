"""Structured, localization-neutral application errors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from captioner.core.domain.result import JsonValue


class AppError(RuntimeError):
    """An error with a stable machine code and JSON-safe parameters."""

    code: str
    params: Mapping[str, JsonValue]
    retryable: bool

    def __init__(
        self,
        code: str,
        params: Mapping[str, JsonValue] | None = None,
        retryable: bool = False,
    ) -> None:
        normalized_code = code.strip()
        if not normalized_code:
            raise ValueError
        normalized_params = dict(params or {})
        try:
            json.dumps(normalized_params, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError from exc
        self.code = normalized_code
        self.params = MappingProxyType(normalized_params)
        self.retryable = retryable
        super().__init__(self._debug_message())

    def _debug_message(self) -> str:
        rendered_params = json.dumps(dict(self.params), ensure_ascii=False, sort_keys=True)
        return f"{self.code}: {rendered_params}"

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible error object."""
        return {
            "code": self.code,
            "params": dict(self.params),
            "retryable": self.retryable,
        }
