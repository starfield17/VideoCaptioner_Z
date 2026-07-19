"""New-Job Runtime/Model preflight and effective ASR snapshot creation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from captioner.core.application.model_compatibility import check_model_compatibility
from captioner.core.application.model_selector import select_model
from captioner.core.application.runtime_selection import (
    HostFacts,
    RuntimeSelection,
    try_select_runtime,
)
from captioner.core.domain.asr_job_snapshot import ASRJobSnapshot
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation, ModelState
from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.ports.model_validator import ModelValidator


@dataclass(frozen=True, slots=True)
class ModelPreflightResult:
    ok: bool
    snapshot: ASRJobSnapshot | None = None
    selection: RuntimeSelection | None = None
    error_code: str | None = None
    message_code: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ok != (self.snapshot is not None and self.selection is not None):
            raise AppError("runtime.preflight_invalid")
        if not self.ok and not self.reasons:
            raise AppError("runtime.preflight_invalid")


def preflight_model(
    *,
    model: ModelInstallation,
    requested_model_selector: str,
    requested_device: str,
    compute_type: str,
    host: HostFacts,
    active_runtimes: Sequence[RuntimeInstallation],
    validator: ModelValidator | None = None,
    requested_backend_id: str = "auto",
) -> ModelPreflightResult:
    """Select only available resources and return values safe to persist."""
    if model.state in {ModelState.STAGED, ModelState.FAILED}:
        return _failure("model.not_installed", (f"model_{model.state.value}",))
    if not model.is_validated:
        return _failure("model.validation_required", ("validation_passed_required",))
    if validator is not None:
        report = validator.validate(model.manifest, model.model_directory)
        if not report.ok:
            return _failure(
                "model.external_content_changed"
                if model.state is ModelState.EXTERNAL_UNMANAGED
                else (report.error_code or "model.validation_failed"),
                ("external_validation_failed",)
                if model.state is ModelState.EXTERNAL_UNMANAGED
                else (report.error_code or "model.validation_failed",),
            )
    result = try_select_runtime(
        requested_backend_id=requested_backend_id,
        requested_device=requested_device,
        host=host,
        active_runtimes=active_runtimes,
        model=model,
    )
    if not result.ok or result.selection is None:
        return _failure(
            result.error_code or "runtime.preflight_failed",
            result.reasons,
        )
    compatibility = next(
        (
            check_model_compatibility(runtime, model)
            for runtime in active_runtimes
            if runtime.identity == result.selection.effective_runtime_identity
        ),
        None,
    )
    if compatibility is None or not compatibility.compatible:
        return _failure("runtime.model_incompatible", ("selected_runtime_missing",))
    compute_error = _compute_type_error(
        result.selection.effective_backend_id,
        result.selection.effective_device,
        compute_type,
    )
    if compute_error is not None:
        return _failure(compute_error, (compute_error,))
    snapshot = ASRJobSnapshot(
        schema_version=1,
        requested_model_selector=requested_model_selector,
        requested_device=requested_device,
        effective_backend_id=result.selection.effective_backend_id,
        effective_runtime_identity=result.selection.effective_runtime_identity,
        effective_model_identity=result.selection.effective_model_identity,
        effective_device_kind=result.selection.effective_device,
        compute_type=compute_type,
    )
    return ModelPreflightResult(ok=True, snapshot=snapshot, selection=result.selection)


def preflight_new_job(
    *,
    model_selector: str,
    requested_device: str,
    compute_type: str,
    host: HostFacts,
    installed_models: Sequence[ModelInstallation],
    active_runtimes: Sequence[RuntimeInstallation],
    validator: ModelValidator | None = None,
    requested_backend_id: str = "auto",
) -> ModelPreflightResult:
    """Resolve a new Job selector once and return its durable ASR snapshot."""
    try:
        model = select_model(model_selector, installed_models)
    except AppError as exc:
        return _failure(exc.code, (exc.code,))
    return preflight_model(
        model=model,
        requested_model_selector=model_selector,
        requested_device=requested_device,
        compute_type=compute_type,
        host=host,
        active_runtimes=active_runtimes,
        validator=validator,
        requested_backend_id=requested_backend_id,
    )


def _failure(code: str, reasons: tuple[str, ...]) -> ModelPreflightResult:
    return ModelPreflightResult(
        ok=False,
        error_code=code,
        message_code=code,
        reasons=reasons,
    )


def _compute_type_error(backend_id: str, device_kind: str, compute_type: str) -> str | None:
    if not compute_type.strip() or compute_type != compute_type.strip():
        return "asr.compute_type_invalid"
    if backend_id == "mlx-whisper" and compute_type != "default":
        return "asr.compute_type_invalid"
    if (
        backend_id == "faster-whisper"
        and device_kind == "cpu"
        and compute_type
        not in {
            "default",
            "int8",
            "float32",
        }
    ):
        return "asr.compute_type_invalid"
    return None


__all__ = ["ModelPreflightResult", "preflight_model", "preflight_new_job"]
