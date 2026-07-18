"""Pure effective Runtime selection policy for one not-yet-persisted Job."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from captioner.core.application.model_compatibility import check_model_compatibility
from captioner.core.domain.asr_backend import DeviceKind
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelIdentity, ModelInstallation, ModelManifest, ModelState
from captioner.core.domain.runtime import (
    SUPPORTED_RUNTIME_ARCHITECTURES,
    SUPPORTED_RUNTIME_PLATFORMS,
    RuntimeIdentity,
    RuntimeInstallation,
)

_OS_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True, slots=True)
class HostFacts:
    """Already-normalized host facts supplied by an adapter/probe."""

    platform: str
    architecture: str
    os_version: str
    native_architecture: bool

    def __post_init__(self) -> None:
        for field, value in (
            ("platform", self.platform),
            ("architecture", self.architecture),
            ("os_version", self.os_version),
        ):
            raw_value = cast(object, value)
            if not isinstance(raw_value, str) or not value.strip() or value != value.strip():
                raise AppError("runtime.host_facts_invalid", {"field": field})
        if self.platform not in SUPPORTED_RUNTIME_PLATFORMS:
            raise AppError("runtime.host_facts_invalid", {"field": "platform"})
        if self.architecture not in SUPPORTED_RUNTIME_ARCHITECTURES:
            raise AppError("runtime.host_facts_invalid", {"field": "architecture"})
        if _OS_VERSION_RE.fullmatch(self.os_version) is None:
            raise AppError("runtime.host_facts_invalid", {"field": "os_version"})
        if type(self.native_architecture) is not bool:
            raise AppError("runtime.host_facts_invalid", {"field": "native_architecture"})

    @property
    def os_version_parts(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.os_version.split("."))


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Effective values that a future Job creator will persist."""

    effective_backend_id: str
    effective_runtime_identity: RuntimeIdentity
    effective_device: str
    effective_model_identity: ModelIdentity

    def __post_init__(self) -> None:
        for field, value in (
            ("effective_backend_id", self.effective_backend_id),
            ("effective_device", self.effective_device),
        ):
            raw_value = cast(object, value)
            if not isinstance(raw_value, str) or not value.strip():
                raise AppError("runtime.selection_invalid", {"field": field})


@dataclass(frozen=True, slots=True)
class RuntimeSelectionResult:
    """Typed preflight result for callers that do not want exceptions."""

    ok: bool
    selection: RuntimeSelection | None = None
    error_code: str | None = None
    message_code: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise AppError("runtime.selection_invalid", {"field": "ok"})
        if self.ok != (self.selection is not None):
            raise AppError("runtime.selection_invalid", {"field": "selection"})
        if self.ok and self.reasons:
            raise AppError("runtime.selection_invalid", {"field": "reasons"})
        if not self.ok and not self.reasons:
            raise AppError("runtime.selection_invalid", {"field": "reasons"})
        if any(
            not isinstance(cast(object, reason), str) or not reason.strip()
            for reason in self.reasons
        ):
            raise AppError("runtime.selection_invalid", {"field": "reasons"})


def select_runtime(
    *,
    requested_backend_id: str = "auto",
    requested_device: str = "auto",
    host: HostFacts,
    active_runtimes: Sequence[RuntimeInstallation],
    model: ModelInstallation | ModelManifest,
) -> RuntimeSelection:
    """Select a compatible available Runtime without side effects."""
    result = try_select_runtime(
        requested_backend_id=requested_backend_id,
        requested_device=requested_device,
        host=host,
        active_runtimes=active_runtimes,
        model=model,
    )
    if not result.ok or result.selection is None:
        raise AppError(
            result.error_code or "runtime.preflight_failed",
            {"reasons": list(result.reasons)},
        )
    return result.selection


def try_select_runtime(
    *,
    requested_backend_id: str = "auto",
    requested_device: str = "auto",
    host: HostFacts,
    active_runtimes: Sequence[RuntimeInstallation],
    model: ModelInstallation | ModelManifest,
) -> RuntimeSelectionResult:
    """Return a typed failure rather than installing, downloading, or mutating state."""
    raw_backend_id = cast(object, requested_backend_id)
    raw_device = cast(object, requested_device)
    if (
        not isinstance(raw_backend_id, str)
        or not isinstance(raw_device, str)
        or not requested_backend_id.strip()
        or not requested_device.strip()
    ):
        return _failure("requested_selection_empty")
    if requested_device not in {
        DeviceKind.AUTO.value,
        DeviceKind.CPU.value,
        DeviceKind.CUDA.value,
        DeviceKind.METAL.value,
    }:
        return _failure("requested_device_invalid")
    model_result = _validate_model_for_selection(model)
    if model_result is not None:
        return _failure(model_result)
    model_manifest = model.manifest if isinstance(model, ModelInstallation) else model
    model_backend = model_manifest.identity.backend_id
    if requested_backend_id != "auto" and requested_backend_id != model_backend:
        return _failure("explicit_backend_model_mismatch")

    available = tuple(
        runtime
        for runtime in active_runtimes
        if runtime.is_available and _runtime_matches_host(runtime, host)
    )
    if not available:
        return _failure("no_available_runtime")

    candidates = tuple(
        runtime
        for runtime in available
        if _backend_matches(runtime, requested_backend_id, model_backend)
        and _device_matches(runtime, requested_device)
        and check_model_compatibility(runtime, model).compatible
    )

    if _has_ambiguous_active_runtime(candidates):
        return _failure_with_code(
            "active_selection_ambiguous", "runtime.active_selection_ambiguous"
        )

    if requested_device == DeviceKind.AUTO.value:
        candidates = _apply_auto_preference(
            candidates, model_backend, model_manifest.model_format, host
        )
    elif model_manifest.model_format == "mlx-whisper" and not candidates:
        return _failure("mlx_model_requires_compatible_runtime")

    if not candidates:
        return _failure("no_compatible_runtime")
    selected = candidates[0]
    return RuntimeSelectionResult(
        ok=True,
        selection=RuntimeSelection(
            effective_backend_id=selected.manifest.backend_id,
            effective_runtime_identity=selected.identity,
            effective_device=selected.manifest.target.device_kind,
            effective_model_identity=model_manifest.identity,
        ),
    )


def _apply_auto_preference(
    candidates: Sequence[RuntimeInstallation],
    model_backend: str,
    model_format: str,
    host: HostFacts,
) -> tuple[RuntimeInstallation, ...]:
    if model_format == "mlx-whisper" or model_backend == "mlx-whisper":
        if _mlx_host_allowed(host):
            mlx = tuple(
                runtime
                for runtime in candidates
                if runtime.manifest.backend_id == "mlx-whisper"
                and runtime.manifest.target.device_kind == "metal"
            )
            return mlx
        return ()
    cuda = tuple(
        runtime
        for runtime in candidates
        if runtime.manifest.backend_id == "faster-whisper"
        and runtime.manifest.target.device_kind == DeviceKind.CUDA.value
    )
    if cuda:
        return cuda
    return tuple(
        runtime
        for runtime in candidates
        if runtime.manifest.backend_id == "faster-whisper"
        and runtime.manifest.target.device_kind == DeviceKind.CPU.value
    )


def _runtime_matches_host(runtime: RuntimeInstallation, host: HostFacts) -> bool:
    target = runtime.manifest.target
    if target.platform != host.platform:
        return False
    if target.architecture != host.architecture and not (
        host.platform == "macos"
        and host.architecture == "arm64"
        and not host.native_architecture
        and target.architecture == "x86_64"
    ):
        return False
    if target.minimum_os_version and not _version_at_least(
        host.os_version_parts, _version_tuple(target.minimum_os_version)
    ):
        return False
    return not (target.device_kind == DeviceKind.METAL.value and not _mlx_host_allowed(host))


def _backend_matches(runtime: RuntimeInstallation, requested: str, model_backend: str) -> bool:
    return runtime.manifest.backend_id == (model_backend if requested == "auto" else requested)


def _device_matches(runtime: RuntimeInstallation, requested: str) -> bool:
    return requested == "auto" or runtime.manifest.target.device_kind == requested


def _mlx_host_allowed(host: HostFacts) -> bool:
    return (
        host.platform == "macos"
        and host.architecture == "arm64"
        and host.native_architecture
        and host.os_version_parts >= (14,)
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _version_at_least(actual: tuple[int, ...], required: tuple[int, ...]) -> bool:
    width = max(len(actual), len(required))
    return (actual + (0,) * (width - len(actual))) >= (required + (0,) * (width - len(required)))


def _failure(reason: str) -> RuntimeSelectionResult:
    return _failure_with_code(reason, "runtime.preflight_failed")


def _failure_with_code(reason: str, error_code: str) -> RuntimeSelectionResult:
    return RuntimeSelectionResult(
        ok=False,
        error_code=error_code,
        message_code=error_code,
        reasons=(reason,),
    )


def _validate_model_for_selection(model: ModelInstallation | ModelManifest) -> str | None:
    if not isinstance(model, ModelInstallation):
        return None
    if model.state is ModelState.STAGED:
        return "model_not_installed"
    if model.state is ModelState.FAILED:
        return "model_failed"
    if model.state is ModelState.EXTERNAL_UNMANAGED and not model.is_validated:
        return "model_not_validated"
    if model.state not in {
        ModelState.INSTALLED,
        ModelState.LOAD_VERIFIED,
        ModelState.EXTERNAL_UNMANAGED,
    }:
        return "model_not_installed"
    return None


def _has_ambiguous_active_runtime(candidates: Sequence[RuntimeInstallation]) -> bool:
    seen: set[tuple[str, tuple[str, str, str]]] = set()
    for runtime in candidates:
        slot = (runtime.manifest.backend_id, runtime.manifest.target.key)
        if slot in seen:
            return True
        seen.add(slot)
    return False


__all__ = [
    "HostFacts",
    "RuntimeSelection",
    "RuntimeSelectionResult",
    "select_runtime",
    "try_select_runtime",
]
