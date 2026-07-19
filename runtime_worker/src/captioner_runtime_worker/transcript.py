"""Transcript result serialization independent of the Core package."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


def result_model_id(model_identity: Mapping[str, object]) -> str:
    backend = model_identity.get("backend_id")
    digest = model_identity.get("manifest_sha256")
    if not isinstance(backend, str) or not isinstance(digest, str):
        raise TypeError("model_identity_invalid")
    return f"{backend}:{digest}"


def derive_transcript_id(
    *,
    language: str,
    words: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    engine_id: str,
    model_id: str,
    metadata: Mapping[str, object],
) -> str:
    """Derive the same content identity as Core's Transcript domain.

    The Worker intentionally hashes only the canonical transcript projection;
    local audio/model paths are not part of the identity.
    """
    payload = {
        "language": language,
        "engine_id": engine_id,
        "model_id": model_id,
        "words": [dict(word) for word in words],
        "segments": [dict(segment) for segment in segments],
        "metadata": dict(metadata),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"transcript-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def write_result(workspace: Path, transcript: Mapping[str, object]) -> dict[str, object]:
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = workspace / "result.json.tmp"
    result_path = workspace / "result.json"
    data = (
        json.dumps(
            transcript,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, result_path)
        if os.name != "nt":
            descriptor = os.open(workspace, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "relative_path": "result.json",
        "size_bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
        "schema_id": "captioner.transcript",
        "schema_version": 1,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
