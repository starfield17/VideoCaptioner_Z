"""Static local model validation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.model import ModelManifest, ModelValidationReport


class ModelValidator(Protocol):
    def validate(self, manifest: ModelManifest, model_directory: Path) -> ModelValidationReport: ...


ModelValidatorPort = ModelValidator

__all__ = ["ModelValidator", "ModelValidatorPort"]
