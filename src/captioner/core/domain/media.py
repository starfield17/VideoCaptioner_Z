"""Immutable media and normalized-audio domain values."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import FrozenJsonValue, JsonValue, freeze_json_value

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: str, field: str) -> None:
    if not value.strip():
        raise AppError("media.invalid", {"field": field, "reason": "empty"})


def _require_integer(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("media.invalid", {"field": field, "reason": "integer"})


def _require_sha256(value: str, field: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise AppError("media.invalid", {"field": field, "reason": "sha256"})


def _freeze_metadata(metadata: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    try:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(metadata))
    except (TypeError, ValueError) as exc:
        raise AppError("media.invalid", {"field": "metadata", "reason": str(exc)}) from exc
    return cast(Mapping[str, JsonValue], frozen)


@dataclass(frozen=True, slots=True)
class MediaAsset:
    """A validated source media asset discovered by a media inspector."""

    id: str
    source_path: Path
    content_hash: str
    duration_ms: int
    audio_stream_index: int
    container: str
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        if not self.source_path.is_absolute():
            raise AppError("media.invalid", {"field": "source_path", "reason": "absolute"})
        _require_sha256(self.content_hash, "content_hash")
        _require_integer(self.duration_ms, "duration_ms")
        if self.duration_ms <= 0:
            raise AppError("media.duration_invalid", {"duration_ms": self.duration_ms})
        _require_integer(self.audio_stream_index, "audio_stream_index")
        if self.audio_stream_index < 0:
            raise AppError("media.invalid", {"field": "audio_stream_index", "reason": "negative"})
        _require_text(self.container, "container")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """The fixed Phase 1 audio representation consumed by ASR."""

    artifact_id: str
    path: Path
    sha256: str
    sample_rate: int
    channels: int
    duration_ms: int
    codec: str

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        if not self.path.is_absolute():
            raise AppError("media.invalid", {"field": "path", "reason": "absolute"})
        _require_sha256(self.sha256, "sha256")
        _require_integer(self.sample_rate, "sample_rate")
        _require_integer(self.channels, "channels")
        if self.sample_rate != 16_000 or self.channels != 1 or self.codec != "pcm_s16le":
            raise AppError("media.normalized_audio_invalid", {"reason": "format"})
        _require_integer(self.duration_ms, "duration_ms")
        if self.duration_ms <= 0:
            raise AppError("media.normalized_audio_invalid", {"reason": "duration"})
