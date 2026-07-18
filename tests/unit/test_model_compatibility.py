from __future__ import annotations

import pytest
from tests.fakes.phase6_values import model_installation, runtime_installation

from captioner.core.application.model_compatibility import (
    check_model_compatibility,
    ensure_model_compatibility,
)
from captioner.core.domain.errors import AppError


def test_matching_faster_whisper_runtime_and_model_are_compatible() -> None:
    result = check_model_compatibility(runtime_installation(), model_installation())
    assert result.compatible
    assert result.reasons == ()


def test_faster_runtime_and_mlx_model_are_rejected() -> None:
    result = check_model_compatibility(
        runtime_installation(),
        model_installation(
            backend_id="mlx-whisper",
            model_format="mlx-whisper",
            repository_id="org/mlx-model",
        ),
    )
    assert not result.compatible
    assert "backend_mismatch" in result.reasons
    with pytest.raises(AppError, match=r"runtime\.model_incompatible"):
        ensure_model_compatibility(
            runtime_installation(),
            model_installation(
                backend_id="mlx-whisper",
                model_format="mlx-whisper",
                repository_id="org/mlx-model",
            ),
        )


def test_mlx_runtime_and_ct2_model_are_rejected() -> None:
    runtime = runtime_installation(
        backend_id="mlx-whisper",
        device_kind="metal",
        model_format="mlx-whisper",
        runtime_id="mlx-whisper-metal-macos-arm64",
    )
    result = check_model_compatibility(runtime, model_installation())
    assert not result.compatible
    assert "backend_mismatch" in result.reasons


def test_unknown_required_capability_is_not_inferred_from_model_format() -> None:
    result = check_model_compatibility(
        runtime_installation(),
        model_installation(required_capabilities=("future_capability",)),
    )
    assert not result.compatible
    assert "missing_capability:future_capability" in result.reasons


def test_explicit_additional_capability_satisfies_model_requirement() -> None:
    result = check_model_compatibility(
        runtime_installation(additional_capabilities=("future_capability",)),
        model_installation(required_capabilities=("future_capability",)),
    )
    assert result.compatible


def test_model_format_is_not_a_capability() -> None:
    result = check_model_compatibility(
        runtime_installation(),
        model_installation(required_capabilities=("faster-whisper-ct2",)),
    )
    assert not result.compatible
    assert "missing_capability:faster-whisper-ct2" in result.reasons
