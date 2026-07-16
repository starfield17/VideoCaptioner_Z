"""Structured, localization-neutral application errors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)


class AppError(RuntimeError):
    """An error with a stable machine code and JSON-safe parameters."""

    code: str
    params: Mapping[str, FrozenJsonValue]
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
        normalized_params = {} if params is None else dict(params)
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(normalized_params))
        json.dumps(
            thaw_json_value(frozen),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
        self.code = normalized_code
        self.params = frozen
        self.retryable = retryable
        super().__init__(self._debug_message())

    def _debug_message(self) -> str:
        rendered_params = json.dumps(
            thaw_json_value(self.params), allow_nan=False, ensure_ascii=False, sort_keys=True
        )
        return f"{self.code}: {rendered_params}"

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible error object."""
        return {
            "code": self.code,
            "params": thaw_json_value(self.params),
            "retryable": self.retryable,
        }
