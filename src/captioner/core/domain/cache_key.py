"""Canonical deterministic Stage cache-key derivation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.result import FrozenJsonValue, thaw_json_value

CACHE_KEY_SCHEMA_VERSION = 1


def derive_stage_cache_key(
    *,
    stage_name: str,
    stage_version: str,
    input_artifacts: Sequence[ArtifactRef],
    config: Mapping[str, FrozenJsonValue],
) -> str:
    payload = {
        "schema_version": CACHE_KEY_SCHEMA_VERSION,
        "stage_name": stage_name,
        "stage_version": stage_version,
        "input_artifact_sha256": [artifact.sha256 for artifact in input_artifacts],
        "config": thaw_json_value(config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
