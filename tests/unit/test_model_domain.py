from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from tests.fakes.phase6_values import model_installation, model_manifest

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    LocalModelInspection,
    ModelFileEntry,
    ModelIdentity,
    ModelManifest,
    ModelSourceCandidate,
    ModelSourceReference,
    ModelState,
    ModelValidationCheck,
    ModelValidationReport,
    compute_model_manifest_sha256,
    model_manifest_digest_payload,
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


@pytest.mark.parametrize(
    ("state", "load_verified", "validation_passed"),
    [
        (ModelState.INSTALLED, False, False),
        (ModelState.LOAD_VERIFIED, False, False),
        (ModelState.INSTALLED, True, None),
        (ModelState.STAGED, True, None),
        (ModelState.FAILED, False, True),
    ],
)
def test_model_installation_rejects_contradictory_state_flags(
    state: ModelState,
    load_verified: bool,
    validation_passed: bool | None,
) -> None:
    with pytest.raises(AppError, match=r"model\.installation_invalid"):
        model_installation(
            state=state,
            load_verified=load_verified,
            validation_passed=validation_passed,
        )


def test_validated_external_model_can_be_load_verified_without_managed_delete() -> None:
    external = model_installation(
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        load_verified=True,
        validation_passed=True,
    )
    assert external.is_validated
    assert external.is_load_verified
    assert not external.can_delete_files


def test_source_candidate_and_resolved_reference_do_not_contain_final_identity() -> None:
    candidate = ModelSourceCandidate(
        source_id="huggingface",
        repository_id="org/model",
        revision=None,
        backend_id="faster-whisper",
        model_format_hint="faster-whisper-ct2",
        display_name="large-v3",
    )
    reference = ModelSourceReference(
        source_id="huggingface",
        repository_id="org/model",
        revision="revision-a",
        backend_id="faster-whisper",
        model_format_hint="faster-whisper-ct2",
    )
    assert candidate.revision is None
    assert reference.revision == "revision-a"
    assert "manifest_sha256" not in candidate.to_dict()
    assert "manifest_sha256" not in reference.to_dict()
    assert "/" not in repr(reference).split("repository_id=")[0]


def test_local_inspection_keeps_validation_and_format_projection_separate() -> None:
    inspection = LocalModelInspection(
        detected_backend_id="mlx-whisper",
        detected_model_format="mlx-whisper",
        required_files_present=True,
        file_inventory=(ModelFileEntry("config.json", 1, "b" * 64),),
        validation_report=ModelValidationReport(
            False,
            (
                ModelValidationCheck(
                    "weights",
                    False,
                    error_code="model.mlx_required_files",
                    message_code="model.mlx_required_files",
                ),
            ),
            error_code="model.mlx_required_files",
            message_code="model.mlx_required_files",
        ),
        display_name_suggestion="large-v3",
    )
    assert inspection.files == inspection.file_inventory
    assert not inspection.validation_passed


def test_model_file_rejects_path_traversal() -> None:
    with pytest.raises(AppError, match=r"model\.file_invalid"):
        ModelFileEntry("../weights.bin", 1, "a" * 64)


def test_model_manifest_digest_is_canonical_and_excludes_itself() -> None:
    first = model_manifest(
        source_metadata={"z": "last", "a": "first"},
        required_capabilities=("translation_task", "word_timestamps"),
        files=(
            ModelFileEntry("model.bin", 1, "c" * 64),
            ModelFileEntry("config.json", 1, "b" * 64),
        ),
        compatible_runtime_backends=("future-backend", "faster-whisper"),
        required_device_kind="cpu",
        required_platform="linux",
    )
    second = model_manifest(
        source_metadata={"a": "first", "z": "last"},
        required_capabilities=("word_timestamps", "translation_task"),
        files=(
            ModelFileEntry("config.json", 1, "b" * 64),
            ModelFileEntry("model.bin", 1, "c" * 64),
        ),
        compatible_runtime_backends=("faster-whisper", "future-backend"),
        required_device_kind="cpu",
        required_platform="linux",
    )
    first_payload = model_manifest_digest_payload(first)
    assert "manifest_sha256" not in first_payload
    assert first.identity.manifest_sha256 == compute_model_manifest_sha256(first)
    assert first.identity.manifest_sha256 == second.identity.manifest_sha256


@pytest.mark.parametrize(
    "changed",
    [
        lambda: model_manifest(revision="revision-b"),
        lambda: model_manifest(source_metadata={"source": "changed"}),
        lambda: model_manifest(model_format="other-format"),
        lambda: model_manifest(
            files=(
                ModelFileEntry("config.json", 2, "b" * 64),
                ModelFileEntry("model.bin", 1, "c" * 64),
            )
        ),
    ],
)
def test_semantic_manifest_changes_change_digest(
    changed: Callable[[], ModelManifest],
) -> None:
    baseline = model_manifest()
    assert baseline.identity.manifest_sha256 != changed().identity.manifest_sha256


def test_model_manifest_rejects_a_self_or_stale_digest() -> None:
    identity = ModelIdentity(
        "faster-whisper",
        "huggingface",
        "org/model",
        "revision-a",
        "faster-whisper-ct2",
        "0" * 64,
    )
    with pytest.raises(AppError, match=r"model\.manifest_digest_mismatch"):
        ModelManifest(
            1,
            identity,
            "large-v3",
            (
                ModelFileEntry("config.json", 1, "b" * 64),
                ModelFileEntry("model.bin", 1, "c" * 64),
            ),
            ("faster-whisper",),
            "faster-whisper-ct2",
        )
