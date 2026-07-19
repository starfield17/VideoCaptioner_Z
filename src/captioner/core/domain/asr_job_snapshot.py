"""Durable effective ASR selection for schema-3 Jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelIdentity
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import RuntimeIdentity

ASR_JOB_SNAPSHOT_SCHEMA_VERSION = 1
_REQUESTED_DEVICE_KINDS = frozenset({"auto", "cpu", "cuda", "metal"})
_EFFECTIVE_DEVICE_KINDS = frozenset({"cpu", "cuda", "metal"})


@dataclass(frozen=True, slots=True)
class ASRJobSnapshot:
    schema_version: int
    requested_model_selector: str
    requested_device: str
    effective_backend_id: str
    effective_runtime_identity: RuntimeIdentity
    effective_model_identity: ModelIdentity
    effective_device_kind: str
    compute_type: str

    def __post_init__(self) -> None:
        if self.schema_version != ASR_JOB_SNAPSHOT_SCHEMA_VERSION:
            raise AppError("job.asr_snapshot_invalid", {"field": "schema_version"})
        for field, value in (
            ("requested_model_selector", self.requested_model_selector),
            ("requested_device", self.requested_device),
            ("effective_backend_id", self.effective_backend_id),
            ("effective_device_kind", self.effective_device_kind),
            ("compute_type", self.compute_type),
        ):
            if not value.strip() or value != value.strip():
                raise AppError("job.asr_snapshot_invalid", {"field": field})
        raw_runtime_identity: object = cast(object, self.effective_runtime_identity)
        if type(raw_runtime_identity) is not RuntimeIdentity:
            raise AppError("job.asr_snapshot_invalid", {"field": "runtime_identity"})
        raw_model_identity: object = cast(object, self.effective_model_identity)
        if type(raw_model_identity) is not ModelIdentity:
            raise AppError("job.asr_snapshot_invalid", {"field": "model_identity"})
        if self.requested_device not in _REQUESTED_DEVICE_KINDS:
            raise AppError("job.asr_snapshot_invalid", {"field": "requested_device"})
        if self.effective_device_kind not in _EFFECTIVE_DEVICE_KINDS:
            raise AppError("job.asr_snapshot_invalid", {"field": "effective_device_kind"})
        if self.effective_model_identity.backend_id != self.effective_backend_id:
            raise AppError("job.asr_snapshot_invalid", {"field": "effective_backend_id"})

    def to_mapping(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "requested_model_selector": self.requested_model_selector,
            "requested_device": self.requested_device,
            "effective_backend_id": self.effective_backend_id,
            "effective_runtime_identity": self.effective_runtime_identity.to_dict(),
            "effective_model_identity": self.effective_model_identity.to_dict(),
            "effective_device_kind": self.effective_device_kind,
            "compute_type": self.compute_type,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ASRJobSnapshot:
        if not isinstance(value, Mapping):
            raise AppError("job.asr_snapshot_invalid", {"field": "snapshot"})
        raw = cast(Mapping[object, object], value)
        required = (
            "schema_version",
            "requested_model_selector",
            "requested_device",
            "effective_backend_id",
            "effective_runtime_identity",
            "effective_model_identity",
            "effective_device_kind",
            "compute_type",
        )
        if set(raw) != set(required):
            raise AppError("job.asr_snapshot_invalid", {"field": "fields"})
        schema = raw["schema_version"]
        if type(schema) is not int:
            raise AppError("job.asr_snapshot_invalid", {"field": "schema_version"})
        return cls(
            schema_version=schema,
            requested_model_selector=_required_string(raw, "requested_model_selector"),
            requested_device=_required_string(raw, "requested_device"),
            effective_backend_id=_required_string(raw, "effective_backend_id"),
            effective_runtime_identity=RuntimeIdentity.from_dict(raw["effective_runtime_identity"]),
            effective_model_identity=ModelIdentity.from_dict(raw["effective_model_identity"]),
            effective_device_kind=_required_string(raw, "effective_device_kind"),
            compute_type=_required_string(raw, "compute_type"),
        )


def _required_string(value: Mapping[object, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str):
        raise AppError("job.asr_snapshot_invalid", {"field": field})
    return item


__all__ = ["ASR_JOB_SNAPSHOT_SCHEMA_VERSION", "ASRJobSnapshot"]
