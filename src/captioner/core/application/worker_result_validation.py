"""Core-side validation of Worker result descriptors."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.worker_protocol import ResultDescriptor

SupportedSchemaVersions = Mapping[str, Collection[int]] | Collection[int]


def validate_worker_result(
    descriptor: ResultDescriptor,
    attempt_workspace: Path,
    *,
    supported_schema_versions: SupportedSchemaVersions,
) -> Path:
    """Verify a descriptor's local file before any Artifact Store commit."""
    if not attempt_workspace.is_absolute():
        raise AppError("worker.result_workspace_invalid")
    if not _schema_supported(
        descriptor.schema_id, descriptor.schema_version, supported_schema_versions
    ):
        raise AppError("worker.result_schema_unsupported", {"schema_id": descriptor.schema_id})
    workspace = attempt_workspace.resolve()
    relative = PurePosixPath(descriptor.relative_path)
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise AppError("worker.result_path_invalid", {"reason": "outside_workspace"}) from exc
    if not candidate.is_file():
        raise AppError("worker.result_missing", {"reason": "file"})
    try:
        actual_size = candidate.stat().st_size
        actual_hash = _sha256(candidate)
    except OSError as exc:
        raise AppError("worker.result_read_failed", {"reason": "filesystem"}) from exc
    if actual_size != descriptor.size_bytes:
        raise AppError("worker.result_size_mismatch")
    if actual_hash != descriptor.sha256:
        raise AppError("worker.result_hash_mismatch")
    return candidate


validate_result_descriptor = validate_worker_result


def _schema_supported(
    schema_id: str,
    schema_version: int,
    supported: SupportedSchemaVersions,
) -> bool:
    if isinstance(supported, Mapping):
        versions = cast(Mapping[str, Collection[int]], supported).get(schema_id)
        return versions is not None and schema_version in versions
    return schema_version in supported


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "SupportedSchemaVersions",
    "validate_result_descriptor",
    "validate_worker_result",
]
