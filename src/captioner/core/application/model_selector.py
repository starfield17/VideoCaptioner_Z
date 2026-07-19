"""Selection of already-installed models for new Jobs."""

from __future__ import annotations

from collections.abc import Sequence

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation, ModelState


def select_model(selector: str, models: Sequence[ModelInstallation]) -> ModelInstallation:
    """Resolve a selector without triggering a source lookup or download."""
    normalized = selector.strip()
    if not normalized:
        raise AppError("model.not_installed")
    candidates = [
        model
        for model in models
        if model.state
        in {
            ModelState.INSTALLED,
            ModelState.LOAD_VERIFIED,
            ModelState.EXTERNAL_UNMANAGED,
        }
        and model.validation_passed
        and (
            model.identity.manifest_sha256 == normalized
            or (
                len(normalized) >= 12
                and model.identity.manifest_sha256.startswith(normalized.casefold())
            )
            or model.identity.repository_id == normalized
            or model.manifest.display_name == normalized
        )
    ]
    if not candidates:
        raise AppError("model.not_installed", {"selector": normalized})
    if len(candidates) > 1:
        raise AppError(
            "model.selector_ambiguous",
            {"selector": normalized, "matches": len(candidates)},
        )
    return candidates[0]


__all__ = ["select_model"]
