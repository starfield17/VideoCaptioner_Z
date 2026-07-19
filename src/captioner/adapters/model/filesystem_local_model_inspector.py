"""Concrete, offline LocalModelInspector for clean model directories."""

from __future__ import annotations

from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    LocalModelInspection,
    ModelFileEntry,
    ModelValidationCheck,
    ModelValidationReport,
    required_files_for_format,
)
from captioner.core.ports.local_model_inspector import LocalModelInspector

from .filesystem_model_validator import FilesystemModelValidator


class FilesystemLocalModelInspector(LocalModelInspector):
    """Inspect a directory without writing to it or creating an identity."""

    def __init__(self, validator: FilesystemModelValidator | None = None) -> None:
        self.validator = validator or FilesystemModelValidator()

    def inspect(
        self,
        model_directory: Path,
        backend_hint: str | None = None,
        model_format_hint: str | None = None,
    ) -> LocalModelInspection:
        try:
            inventory = self.validator.inventory(model_directory)
        except AppError as exc:
            return _inspection_failure(exc.code)
        paths = {entry.relative_path for entry in inventory}
        mlx = _has_mlx_signature(paths)
        ct2 = _has_ct2_signature(paths)
        if model_format_hint is not None:
            detected_format = model_format_hint
            detected_backend = backend_hint or _backend_for_format(model_format_hint)
            if model_format_hint not in {"mlx-whisper", "faster-whisper-ct2"}:
                return _inspection(
                    None,
                    None,
                    inventory,
                    _failure_report("model.format_unknown"),
                    model_directory,
                )
        elif mlx and ct2:
            return _inspection(
                None,
                None,
                inventory,
                _failure_report("model.format_ambiguous"),
                model_directory,
            )
        elif mlx:
            detected_format = "mlx-whisper"
            detected_backend = "mlx-whisper"
        elif ct2:
            detected_format = "faster-whisper-ct2"
            detected_backend = "faster-whisper"
        else:
            return _inspection(
                None,
                None,
                inventory,
                _failure_report("model.format_unknown"),
                model_directory,
            )
        if backend_hint is not None and backend_hint != detected_backend:
            return _inspection(
                detected_backend,
                detected_format,
                inventory,
                _failure_report("model.format_backend_mismatch"),
                model_directory,
            )
        required_ok, required_code = _required_ok(detected_format, paths)
        checks = [
            ModelValidationCheck("format", True, message_code="model.format_detected"),
            ModelValidationCheck(
                "required_files",
                required_ok,
                error_code=None if required_ok else required_code,
                message_code=None if required_ok else required_code,
            ),
        ]
        checks.extend(self.validator.json_checks(detected_format, model_directory, paths))
        report = ModelValidationReport(
            ok=all(check.ok for check in checks),
            checks=tuple(checks),
            error_code=None
            if all(check.ok for check in checks)
            else next(check.error_code for check in checks if not check.ok),
            message_code=None
            if all(check.ok for check in checks)
            else next(check.message_code for check in checks if not check.ok),
        )
        return _inspection(
            detected_backend,
            detected_format,
            inventory,
            report,
            model_directory,
        )


def _inspection(
    backend: str | None,
    model_format: str | None,
    inventory: tuple[ModelFileEntry, ...],
    report: ModelValidationReport,
    directory: Path,
) -> LocalModelInspection:
    return LocalModelInspection(
        detected_backend_id=backend,
        detected_model_format=model_format,
        required_files_present=report.ok,
        file_inventory=inventory,
        validation_report=report,
        display_name_suggestion=directory.name or None,
    )


def _inspection_failure(code: str) -> LocalModelInspection:
    return LocalModelInspection(
        detected_backend_id=None,
        detected_model_format=None,
        required_files_present=False,
        file_inventory=(),
        validation_report=_failure_report(code),
    )


def _failure_report(code: str) -> ModelValidationReport:
    check = ModelValidationCheck("format", False, error_code=code, message_code=code)
    return ModelValidationReport(False, (check,), error_code=code, message_code=code)


def _backend_for_format(model_format: str) -> str | None:
    return {
        "faster-whisper-ct2": "faster-whisper",
        "mlx-whisper": "mlx-whisper",
    }.get(model_format)


def _has_mlx_signature(paths: set[str]) -> bool:
    return "config.json" in paths and bool(
        paths & {"model.safetensors", "weights.safetensors", "weights.npz"}
    )


def _has_ct2_signature(paths: set[str]) -> bool:
    return {"config.json", "model.bin", "tokenizer.json"} <= paths


def _required_ok(model_format: str, paths: set[str]) -> tuple[bool, str]:
    groups = required_files_for_format(model_format)
    if model_format == "mlx-whisper" and not (
        "tokenizer.json" in paths or {"vocab.json", "merges.txt"} <= paths
    ):
        return False, "model.mlx_tokenizer_missing"
    if not groups or any(not group <= paths for group in groups[:1]):
        return False, "model.required_files_missing"
    if len(groups) > 1 and not groups[1] & paths:
        return False, "model.required_files_missing"
    return True, ""


__all__ = ["FilesystemLocalModelInspector"]
