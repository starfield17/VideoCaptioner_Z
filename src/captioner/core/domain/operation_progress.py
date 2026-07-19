"""Stage-only progress values shared by Runtime, Model, and Worker flows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import FrozenJsonValue, JsonValue, freeze_json_value

type JSONScalar = None | bool | int | float | str

_FORBIDDEN_PROGRESS_FIELDS = frozenset(
    {
        "percent",
        "percentage",
        "progress_value",
        "completed_units",
        "total_units",
        "percent_milli",
        "eta",
    }
)


@dataclass(frozen=True, slots=True)
class OperationProgress:
    """Progress that names the current phase without a misleading percentage."""

    operation: str
    phase: str
    message_code: str
    detail_parameters: Mapping[str, JSONScalar]

    def __post_init__(self) -> None:
        for field, raw_value in (
            ("operation", cast(object, self.operation)),
            ("phase", cast(object, self.phase)),
            ("message_code", cast(object, self.message_code)),
        ):
            if (
                not isinstance(raw_value, str)
                or not raw_value.strip()
                or raw_value != raw_value.strip()
            ):
                raise AppError("worker.progress_invalid", {"field": field})
        raw_details = cast(object, self.detail_parameters)
        if not isinstance(raw_details, Mapping):
            raise AppError("worker.progress_invalid", {"field": "detail_parameters"})
        raw = dict(cast(Mapping[object, object], raw_details))
        if any(
            not isinstance(key, str) or key.lower() in _FORBIDDEN_PROGRESS_FIELDS for key in raw
        ):
            raise AppError("worker.progress_invalid", {"field": "detail_parameters"})
        if any(not _is_json_scalar(value) for value in raw.values()):
            raise AppError("worker.progress_invalid", {"field": "detail_parameters"})
        try:
            frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(raw))
        except (TypeError, ValueError) as exc:
            raise AppError("worker.progress_invalid", {"field": "detail_parameters"}) from exc
        object.__setattr__(
            self,
            "detail_parameters",
            cast(Mapping[str, JSONScalar], frozen),
        )

    def to_payload(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-compatible payload."""
        return {
            "operation": self.operation,
            "phase": self.phase,
            "message_code": self.message_code,
            "detail_parameters": dict(self.detail_parameters),
        }

    @classmethod
    def from_payload(cls, value: object) -> OperationProgress:
        if not isinstance(value, Mapping):
            raise AppError("worker.progress_invalid", {"field": "payload"})
        raw = cast(Mapping[object, object], value)
        operation = raw.get("operation")
        phase = raw.get("phase")
        message_code = raw.get("message_code")
        details = raw.get("detail_parameters")
        if (
            not isinstance(operation, str)
            or not isinstance(phase, str)
            or not isinstance(message_code, str)
            or not isinstance(details, Mapping)
        ):
            raise AppError("worker.progress_invalid", {"field": "payload"})
        return cls(
            operation,
            phase,
            message_code,
            cast(Mapping[str, JSONScalar], details),
        )


def _is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


__all__ = ["JSONScalar", "OperationProgress"]
