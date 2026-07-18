"""Pure Runtime/Model compatibility policy."""

from __future__ import annotations

from dataclasses import dataclass

from captioner.core.domain.asr_backend import DeviceKind
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation, ModelManifest
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeManifest


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Typed compatibility outcome rather than an uninformative bool."""

    compatible: bool
    error_code: str | None = None
    message_code: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.compatible) is not bool:
            raise AppError("runtime.compatibility_invalid", {"field": "compatible"})
        reasons = tuple(self.reasons)
        if any(not reason.strip() for reason in reasons):
            raise AppError("runtime.compatibility_invalid", {"field": "reasons"})
        if self.compatible and reasons:
            raise AppError("runtime.compatibility_invalid", {"field": "reasons"})
        if not self.compatible and not reasons:
            raise AppError("runtime.compatibility_invalid", {"field": "reasons"})
        object.__setattr__(self, "reasons", reasons)


def check_model_compatibility(
    runtime: RuntimeInstallation | RuntimeManifest,
    model: ModelInstallation | ModelManifest,
) -> CompatibilityResult:
    """Check backend, format, capability, and target constraints."""
    runtime_manifest = runtime.manifest if isinstance(runtime, RuntimeInstallation) else runtime
    model_manifest = model.manifest if isinstance(model, ModelInstallation) else model
    reasons: list[str] = []

    if runtime_manifest.backend_id != model_manifest.identity.backend_id:
        reasons.append("backend_mismatch")
    if runtime_manifest.backend_id not in model_manifest.compatible_runtime_backends:
        reasons.append("backend_not_supported_by_model")
    if model_manifest.model_format not in runtime_manifest.supported_model_formats:
        reasons.append("model_format_mismatch")
    if runtime_manifest.capabilities.backend_id != runtime_manifest.backend_id:
        reasons.append("runtime_capability_backend_mismatch")
    for required in model_manifest.required_capabilities:
        if not _capability_present(runtime_manifest, required):
            reasons.append(f"missing_capability:{required}")
    if (
        model_manifest.required_device_kind is not None
        and runtime_manifest.target.device_kind != model_manifest.required_device_kind
    ):
        reasons.append("device_mismatch")
    if (
        model_manifest.required_platform is not None
        and runtime_manifest.target.platform != model_manifest.required_platform
    ):
        reasons.append("platform_mismatch")
    if runtime_manifest.target.device_kind == DeviceKind.METAL.value:
        if runtime_manifest.target.platform != "macos":
            reasons.append("metal_non_macos")
        if runtime_manifest.backend_id != "mlx-whisper":
            reasons.append("metal_backend_mismatch")
    if model_manifest.model_format == "mlx-whisper":
        if runtime_manifest.backend_id != "mlx-whisper":
            reasons.append("mlx_model_requires_mlx_backend")
        if runtime_manifest.target.device_kind != DeviceKind.METAL.value:
            reasons.append("mlx_model_requires_metal")
        if runtime_manifest.target.platform != "macos":
            reasons.append("mlx_model_requires_macos")
    if (
        model_manifest.model_format == "faster-whisper-ct2"
        and runtime_manifest.backend_id != "faster-whisper"
    ):
        reasons.append("ct2_model_requires_faster_whisper_backend")

    if reasons:
        return CompatibilityResult(
            compatible=False,
            error_code="runtime.model_incompatible",
            message_code="runtime.model_incompatible",
            reasons=tuple(dict.fromkeys(reasons)),
        )
    return CompatibilityResult(compatible=True)


def ensure_model_compatibility(
    runtime: RuntimeInstallation | RuntimeManifest,
    model: ModelInstallation | ModelManifest,
) -> None:
    """Raise the stable preflight error used by orchestration callers."""
    result = check_model_compatibility(runtime, model)
    if not result.compatible:
        raise AppError(
            result.error_code or "runtime.model_incompatible",
            {"reasons": list(result.reasons)},
        )


def _capability_present(runtime: RuntimeManifest, required: str) -> bool:
    capability = runtime.capabilities
    values: dict[str, bool] = {
        "word_timestamps": capability.word_timestamps,
        "language_detection": capability.language_detection,
        "translation_task": capability.translation_task,
    }
    return values.get(required, required in capability.additional_capabilities)


__all__ = [
    "CompatibilityResult",
    "check_model_compatibility",
    "ensure_model_compatibility",
]
