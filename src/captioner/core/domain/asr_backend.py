"""Stable ASR backend, device, and model-format vocabulary.

The enums document the identifiers shipped by Captioner today.  Domain
objects intentionally accept strings for backend/device/format fields so a
future backend can be introduced without changing the Phase 6 contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.errors import AppError


class ASRBackend(StrEnum):
    """Known backend identifiers."""

    FASTER_WHISPER = "faster-whisper"
    MLX_WHISPER = "mlx-whisper"


class DeviceKind(StrEnum):
    """A requested or effective execution device."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"


class ModelFormat(StrEnum):
    """Model formats understood by the corresponding ASR backends."""

    FASTER_WHISPER_CT2 = "faster-whisper-ct2"
    MLX_WHISPER = "mlx-whisper"


# Compatibility aliases keep the vocabulary discoverable under the names used
# by the runtime/model contracts without making the domain closed to strings.
BackendId = ASRBackend
ASRBackendId = ASRBackend
Device = DeviceKind
ASRModelFormat = ModelFormat


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """Capabilities advertised by one backend/device combination."""

    backend_id: str
    device_kind: str
    supported_model_formats: tuple[str, ...]
    word_timestamps: bool
    language_detection: bool
    translation_task: bool

    def __post_init__(self) -> None:
        _require_text(self.backend_id, "backend_id")
        _require_text(self.device_kind, "device_kind")
        if self.device_kind == DeviceKind.AUTO.value:
            raise AppError("runtime.capability_invalid", {"field": "device_kind"})
        formats = tuple(self.supported_model_formats)
        if not formats or any(not value.strip() for value in formats):
            raise AppError("runtime.capability_invalid", {"field": "supported_model_formats"})
        if len(set(formats)) != len(formats):
            raise AppError(
                "runtime.capability_invalid",
                {"field": "supported_model_formats", "reason": "duplicate"},
            )
        object.__setattr__(self, "supported_model_formats", formats)


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AppError("runtime.capability_invalid", {"field": field})


FASTER_WHISPER_CPU_CAPABILITY = BackendCapability(
    backend_id=ASRBackend.FASTER_WHISPER.value,
    device_kind=DeviceKind.CPU.value,
    supported_model_formats=(ModelFormat.FASTER_WHISPER_CT2.value,),
    word_timestamps=True,
    language_detection=True,
    translation_task=True,
)

FASTER_WHISPER_CUDA_CAPABILITY = BackendCapability(
    backend_id=ASRBackend.FASTER_WHISPER.value,
    device_kind=DeviceKind.CUDA.value,
    supported_model_formats=(ModelFormat.FASTER_WHISPER_CT2.value,),
    word_timestamps=True,
    language_detection=True,
    translation_task=True,
)

MLX_WHISPER_METAL_CAPABILITY = BackendCapability(
    backend_id=ASRBackend.MLX_WHISPER.value,
    device_kind=DeviceKind.METAL.value,
    supported_model_formats=(ModelFormat.MLX_WHISPER.value,),
    word_timestamps=True,
    language_detection=True,
    translation_task=True,
)


__all__ = [
    "FASTER_WHISPER_CPU_CAPABILITY",
    "FASTER_WHISPER_CUDA_CAPABILITY",
    "MLX_WHISPER_METAL_CAPABILITY",
    "ASRBackend",
    "ASRBackendId",
    "ASRModelFormat",
    "BackendCapability",
    "BackendId",
    "Device",
    "DeviceKind",
    "ModelFormat",
]
