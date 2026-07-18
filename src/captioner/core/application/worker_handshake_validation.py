"""Pure validation of the Runtime Worker activation handshake."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeManifest
from captioner.core.domain.worker_protocol import (
    HandshakeRequest,
    WorkerHandshake,
    check_protocol_compatibility,
)

_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")


@dataclass(frozen=True, slots=True)
class HandshakeValidationResult:
    """Typed activation handshake result without worker-specific exceptions."""

    ok: bool
    error_code: str | None = None
    message_code: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise AppError("worker.handshake_validation_invalid", {"field": "ok"})
        reasons = tuple(self.reasons)
        if any(
            not isinstance(cast(object, reason), str) or not reason.strip() for reason in reasons
        ):
            raise AppError("worker.handshake_validation_invalid", {"field": "reasons"})
        if self.ok and reasons:
            raise AppError("worker.handshake_validation_invalid", {"field": "reasons"})
        if not self.ok and not reasons:
            raise AppError("worker.handshake_validation_invalid", {"field": "reasons"})
        object.__setattr__(self, "reasons", reasons)


def validate_worker_handshake(
    runtime_manifest: RuntimeManifest,
    handshake_request: HandshakeRequest,
    worker_handshake: WorkerHandshake,
) -> HandshakeValidationResult:
    """Check that a Worker is the exact Runtime and capability set requested."""
    reasons: list[str] = []
    if not check_protocol_compatibility(
        runtime_manifest.worker_protocol_version,
        worker_handshake.protocol_version,
    ):
        reasons.append("protocol_version_mismatch")
    if worker_handshake.runtime_id != runtime_manifest.runtime_identity.runtime_id:
        reasons.append("runtime_id_mismatch")
    if worker_handshake.runtime_version != runtime_manifest.runtime_identity.version:
        reasons.append("runtime_version_mismatch")
    if worker_handshake.backend_id != runtime_manifest.backend_id:
        reasons.append("backend_id_mismatch")
    if not _backend_version_compatible(
        runtime_manifest.backend_version,
        worker_handshake.backend_version,
    ):
        reasons.append("backend_version_incompatible")
    if worker_handshake.platform != runtime_manifest.target.platform:
        reasons.append("platform_mismatch")
    if worker_handshake.architecture != runtime_manifest.target.architecture:
        reasons.append("architecture_mismatch")
    if (
        handshake_request.required_backend_id is not None
        and worker_handshake.backend_id != handshake_request.required_backend_id
    ):
        reasons.append("required_backend_mismatch")
    advertised_capabilities = set(worker_handshake.capabilities)
    required_capabilities = _manifest_capabilities(runtime_manifest)
    required_capabilities.update(handshake_request.required_capabilities)
    for required in sorted(required_capabilities):
        if required not in advertised_capabilities:
            reasons.append(f"missing_capability:{required}")
    advertised_schemas = set(worker_handshake.supported_result_schema_versions)
    for required in handshake_request.required_result_schema_versions:
        if required not in advertised_schemas:
            reasons.append(f"missing_result_schema_version:{required}")
    advertised_formats = set(worker_handshake.supported_model_formats)
    for supported in runtime_manifest.supported_model_formats:
        if supported not in advertised_formats:
            reasons.append(f"model_format_not_advertised:{supported}")
    if runtime_manifest.target.device_kind not in set(worker_handshake.supported_devices):
        reasons.append(f"device_not_advertised:{runtime_manifest.target.device_kind}")
    if reasons:
        unique_reasons = tuple(dict.fromkeys(reasons))
        return HandshakeValidationResult(
            ok=False,
            error_code="worker.handshake_invalid",
            message_code="worker.handshake_invalid",
            reasons=unique_reasons,
        )
    return HandshakeValidationResult(ok=True)


def _manifest_capabilities(runtime_manifest: RuntimeManifest) -> set[str]:
    return set(runtime_manifest.capabilities.advertised_capabilities)


def _backend_version_compatible(expected: str, actual: str) -> bool:
    expected_parts = _version_parts(expected)
    actual_parts = _version_parts(actual)
    return (
        expected_parts is not None
        and actual_parts is not None
        and expected_parts[0] == actual_parts[0]
    )


def _version_parts(value: str) -> tuple[int, ...] | None:
    if _VERSION_RE.fullmatch(value) is None:
        return None
    return tuple(int(part) for part in value.split("."))


__all__ = ["HandshakeValidationResult", "validate_worker_handshake"]
