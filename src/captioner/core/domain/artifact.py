"""Durable artifact identities independent of physical storage paths."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A verified content-addressed artifact reference."""

    sha256: str
    size_bytes: int
    kind: str
    media_type: str
    logical_name: str

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise AppError("artifact.invalid", {"field": "sha256"})
        if self.size_bytes < 0:
            raise AppError("artifact.invalid", {"field": "size_bytes"})
        if not self.kind.strip():
            raise AppError("artifact.invalid", {"field": "kind"})
        if not self.media_type.strip() or "/" not in self.media_type:
            raise AppError("artifact.invalid", {"field": "media_type"})
        if _SAFE_NAME_RE.fullmatch(self.logical_name) is None:
            raise AppError("artifact.invalid", {"field": "logical_name"})

    def to_dict(self) -> dict[str, int | str]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "media_type": self.media_type,
            "logical_name": self.logical_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactRef:
        if not isinstance(value, Mapping):
            raise AppError("artifact.invalid", {"field": "object"})
        raw = cast(Mapping[object, object], value)
        if set(raw) != {
            "sha256",
            "size_bytes",
            "kind",
            "media_type",
            "logical_name",
        }:
            raise AppError("artifact.invalid", {"field": "object"})
        sha256 = raw["sha256"]
        size_bytes = raw["size_bytes"]
        kind = raw["kind"]
        media_type = raw["media_type"]
        logical_name = raw["logical_name"]
        if (
            not isinstance(sha256, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or not isinstance(kind, str)
            or not isinstance(media_type, str)
            or not isinstance(logical_name, str)
        ):
            raise AppError("artifact.invalid", {"field": "types"})
        return cls(sha256, size_bytes, kind, media_type, logical_name)
