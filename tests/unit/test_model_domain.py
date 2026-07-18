from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.phase6_values import model_installation, model_manifest

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelFileEntry,
    ModelIdentity,
    ModelManifest,
    ModelState,
    required_files_for_format,
)


def test_same_display_name_can_have_distinct_source_revision_identity() -> None:
    first = model_manifest(source_id="huggingface", revision="revision-a")
    second = model_manifest(source_id="modelscope", revision="revision-b")
    assert first.display_name == second.display_name
    assert first.identity != second.identity


def test_model_identity_rejects_absolute_local_path() -> None:
    with pytest.raises(AppError, match=r"model\.identity_invalid"):
        ModelIdentity(
            "faster-whisper",
            "external-path",
            "/private/model",
            "r1",
            "faster-whisper-ct2",
            "a" * 64,
        )


def test_model_identity_has_no_machine_local_path() -> None:
    identity = model_manifest().identity
    assert not Path(identity.repository_id).is_absolute()
    assert "/captioner" not in repr(identity)


def test_mlx_manifest_requires_config_and_one_supported_weight_file() -> None:
    assert required_files_for_format("mlx-whisper") == (
        frozenset({"config.json"}),
        frozenset({"model.safetensors", "weights.safetensors", "weights.npz"}),
    )
    with pytest.raises(AppError, match=r"model\.manifest_invalid"):
        identity = ModelIdentity(
            "mlx-whisper",
            "huggingface",
            "org/mlx-model",
            "revision-a",
            "mlx-whisper",
            "d" * 64,
        )
        ModelManifest(
            1,
            identity,
            "large-v3",
            (ModelFileEntry("config.json", 1, "b" * 64),),
            ("mlx-whisper",),
            "mlx-whisper",
        )


def test_faster_whisper_and_mlx_formats_are_not_interchangeable() -> None:
    faster = model_manifest(model_format="faster-whisper-ct2")
    mlx = model_manifest(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx-model",
    )
    assert faster.identity.model_format != mlx.identity.model_format
    assert faster.identity.backend_id != mlx.identity.backend_id


def test_managed_and_external_delete_semantics_are_separate() -> None:
    managed = model_installation(state=ModelState.INSTALLED)
    external = model_installation(
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
    )
    assert managed.can_delete_files
    assert not external.can_delete_files
    assert not managed.is_load_verified
    assert not external.is_load_verified
    verified = model_installation(state=ModelState.LOAD_VERIFIED)
    assert verified.is_load_verified


def test_model_file_rejects_path_traversal() -> None:
    with pytest.raises(AppError, match=r"model\.file_invalid"):
        ModelFileEntry("../weights.bin", 1, "a" * 64)
