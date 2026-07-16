"""Atomic cached Manifest projection derived exclusively from Journal replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.ports.manifest import ManifestStatus

MANIFEST_SCHEMA_VERSION = 1


def projected_data(projection: BatchProjection) -> dict[str, JsonValue]:
    value: object = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "batch_id": projection.batch_id,
        "last_event_seq": projection.last_event_seq,
        "state": projection.state.value,
        "jobs": [
            {
                "job_id": job.job_id,
                "input_path": job.input_path,
                "state": job.state.value,
                "config": job.config.to_dict(),
                "stages": {
                    stage.name.value: {
                        "state": stage.state.value,
                        "attempt": stage.attempt,
                        "cache_key": stage.cache_key,
                        "artifacts": [artifact.to_dict() for artifact in stage.artifacts],
                    }
                    for stage in job.stages
                },
            }
            for job in projection.jobs
        ],
    }
    return cast(dict[str, JsonValue], value)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def projection_hash(data: dict[str, JsonValue]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(data).rstrip(b'\n')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class JsonManifestStore:
    path: Path

    def read(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        try:
            value = cast(object, json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppError("manifest.inconsistent", {"reason": "invalid_json"}) from exc
        if not isinstance(value, dict):
            raise AppError("manifest.inconsistent", {"reason": "invalid_root"})
        return cast(dict[str, object], value)

    def write(self, projection: BatchProjection) -> None:
        data = projected_data(projection)
        document: dict[str, JsonValue] = {**data, "projection_hash": projection_hash(data)}
        encoded = canonical_json_bytes(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise AppError("manifest.projection_failed", {"path": str(self.path)}) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def inspect(self, projection: BatchProjection) -> ManifestStatus:
        try:
            manifest = self.read()
        except AppError:
            return "invalid"
        if manifest is None:
            return "missing"
        seq = manifest.get("last_event_seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            return "invalid"
        if seq > projection.last_event_seq:
            return "ahead"
        if seq < projection.last_event_seq:
            return "stale"
        expected_data = projected_data(projection)
        expected_hash = projection_hash(expected_data)
        actual_hash = manifest.get("projection_hash")
        actual_data = {key: value for key, value in manifest.items() if key != "projection_hash"}
        if actual_hash != expected_hash or actual_data != expected_data:
            return "projection_mismatch"
        return "current"

    def reconcile(self, projection: BatchProjection) -> ManifestStatus:
        status = self.inspect(projection)
        if status in {"missing", "stale"}:
            self.write(projection)
            return "current"
        if status != "current":
            reason = {
                "ahead": "ahead_of_journal",
                "projection_mismatch": "projection_mismatch",
                "invalid": "invalid_json",
            }[status]
            raise AppError("manifest.inconsistent", {"reason": reason})
        return status


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
