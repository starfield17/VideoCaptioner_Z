from __future__ import annotations

from dataclasses import replace

import pytest
from tests.fakes.phase6_values import runtime_installation, worker_handshake

from captioner.core.application.worker_handshake_validation import (
    validate_worker_handshake,
)
from captioner.core.domain.runtime import RuntimeManifest
from captioner.core.domain.worker_protocol import HandshakeRequest, WorkerHandshake


def _request() -> HandshakeRequest:
    return HandshakeRequest(
        required_capabilities=("word_timestamps",),
        required_backend_id="faster-whisper",
        required_result_schema_versions=(1,),
    )


def _valid() -> tuple[RuntimeManifest, HandshakeRequest, WorkerHandshake]:
    runtime = runtime_installation()
    return runtime.manifest, _request(), worker_handshake()


def test_matching_handshake_passes() -> None:
    runtime_manifest, request, handshake = _valid()
    result = validate_worker_handshake(runtime_manifest, request, handshake)
    assert result.ok
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("runtime_id", "other-runtime", "runtime_id_mismatch"),
        ("runtime_version", "2.0.0", "runtime_version_mismatch"),
        ("backend_id", "other-backend", "backend_id_mismatch"),
        ("platform", "linux", "platform_mismatch"),
        ("architecture", "x86_64", "architecture_mismatch"),
    ],
)
def test_handshake_exact_identity_fields_are_validated(field: str, value: str, reason: str) -> None:
    runtime_manifest, request, handshake = _valid()
    mismatched = replace(handshake, **{field: value})
    result = validate_worker_handshake(runtime_manifest, request, mismatched)
    assert not result.ok
    assert reason in result.reasons


def test_handshake_requires_compatible_backend_version() -> None:
    runtime_manifest, request, handshake = _valid()
    compatible = validate_worker_handshake(
        runtime_manifest, request, replace(handshake, backend_version="1.5.0")
    )
    assert compatible.ok
    incompatible = validate_worker_handshake(
        runtime_manifest, request, replace(handshake, backend_version="2.0.0")
    )
    assert not incompatible.ok
    assert "backend_version_incompatible" in incompatible.reasons


def test_handshake_protocol_major_mismatch_fails() -> None:
    runtime_manifest, request, handshake = _valid()
    result = validate_worker_handshake(
        runtime_manifest, request, replace(handshake, protocol_version="2.0")
    )
    assert not result.ok
    assert "protocol_version_mismatch" in result.reasons


def test_handshake_required_backend_is_checked() -> None:
    runtime_manifest, _, handshake = _valid()
    request = HandshakeRequest(required_backend_id="mlx-whisper")
    result = validate_worker_handshake(runtime_manifest, request, handshake)
    assert not result.ok
    assert "required_backend_mismatch" in result.reasons


def test_handshake_required_capability_and_schema_are_checked() -> None:
    runtime_manifest, _, handshake = _valid()
    request = HandshakeRequest(
        required_capabilities=("word_timestamps", "diarization"),
        required_result_schema_versions=(1, 2),
    )
    result = validate_worker_handshake(runtime_manifest, request, handshake)
    assert not result.ok
    assert "missing_capability:diarization" in result.reasons
    assert "missing_result_schema_version:2" in result.reasons


def test_handshake_advertised_device_and_model_format_are_required() -> None:
    runtime_manifest, request, handshake = _valid()
    result = validate_worker_handshake(
        runtime_manifest,
        request,
        replace(handshake, supported_devices=(), supported_model_formats=()),
    )
    assert not result.ok
    assert "device_not_advertised:cpu" in result.reasons
    assert "model_format_not_advertised:faster-whisper-ct2" in result.reasons


def test_handshake_result_is_typed_failure() -> None:
    runtime_manifest, request, handshake = _valid()
    result = validate_worker_handshake(
        runtime_manifest, request, replace(handshake, runtime_id="wrong")
    )
    assert result.error_code == "worker.handshake_invalid"
    assert result.message_code == "worker.handshake_invalid"
