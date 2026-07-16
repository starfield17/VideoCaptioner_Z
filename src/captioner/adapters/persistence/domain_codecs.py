"""Strict deterministic JSON codecs for Phase 2/3 Stage artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

from captioner.adapters.exporters.transcript_json import transcript_to_dict
from captioner.core.domain.errors import AppError
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget
from captioner.core.domain.result import JsonValue, thaw_json_value
from captioner.core.domain.subtitle import (
    LEGACY_POLICY_SIGNATURE,
    SubtitleCue,
    SubtitleTrack,
)
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken

SCHEMA_VERSION = 1
TRACK_SCHEMA_VERSION = 2
_POLICY_SIGNATURE = re.compile(r"^policy-[0-9a-f]{64}$")


def encode_json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def decode_json(data: bytes) -> dict[str, object]:
    try:
        value = cast(
            object,
            json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AppError("artifact.codec_invalid", {"reason": "json"}) from exc
    if not isinstance(value, dict):
        raise AppError("artifact.codec_invalid", {"reason": "root"})
    return cast(dict[str, object], value)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_value:{value}")


def encode_media(asset: MediaAsset) -> bytes:
    return encode_json(
        {
            "schema_version": SCHEMA_VERSION,
            "media": {
                "id": asset.id,
                "source_path": str(asset.source_path),
                "content_hash": asset.content_hash,
                "duration_ms": asset.duration_ms,
                "audio_stream_index": asset.audio_stream_index,
                "container": asset.container,
                "metadata": thaw_json_value(asset.metadata),  # type: ignore[arg-type]  # domain metadata is frozen JSON
            },
        }
    )


def decode_media(data: bytes) -> MediaAsset:
    root = decode_json(data)
    raw = _object(root, "media", {"schema_version", "media"})
    _fields(
        raw,
        {
            "id",
            "source_path",
            "content_hash",
            "duration_ms",
            "audio_stream_index",
            "container",
            "metadata",
        },
    )
    from pathlib import Path

    return MediaAsset(
        _str(raw, "id"),
        Path(_str(raw, "source_path")),
        _str(raw, "content_hash"),
        _int(raw, "duration_ms"),
        _int(raw, "audio_stream_index"),
        _str(raw, "container"),
        cast(Mapping[str, JsonValue], _mapping(raw, "metadata")),
    )


def encode_audio(audio: AudioArtifact) -> bytes:
    return encode_json(
        {
            "schema_version": 1,
            "audio": {
                "artifact_id": audio.artifact_id,
                "sha256": audio.sha256,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "duration_ms": audio.duration_ms,
                "codec": audio.codec,
            },
        }
    )


def decode_audio(data: bytes, *, path: str) -> AudioArtifact:
    from pathlib import Path

    root = decode_json(data)
    raw = _object(root, "audio", {"schema_version", "audio"})
    _fields(raw, {"artifact_id", "sha256", "sample_rate", "channels", "duration_ms", "codec"})
    return AudioArtifact(
        _str(raw, "artifact_id"),
        Path(path),
        _str(raw, "sha256"),
        _int(raw, "sample_rate"),
        _int(raw, "channels"),
        _int(raw, "duration_ms"),
        _str(raw, "codec"),
    )


def encode_transcript(transcript: Transcript) -> bytes:
    return encode_json(transcript_to_dict(transcript))


def decode_transcript(data: bytes) -> Transcript:
    root = decode_json(data)
    raw = _object(root, "transcript", {"schema_version", "transcript"})
    _fields(raw, {"id", "language", "engine_id", "model_id", "words", "segments", "metadata"})
    words = tuple(
        WordToken(
            _str(item, "id"),
            _str(item, "text"),
            _int(item, "start_ms"),
            _int(item, "end_ms"),
            _float_or_none(item, "confidence"),
            _str_or_none(item, "speaker_id"),
        )
        for item in _objects(raw, "words")
    )
    segments = tuple(
        TranscriptSegment(
            _str(item, "id"),
            tuple(_strings(item, "word_ids")),
            _str(item, "raw_text"),
            _int(item, "start_ms"),
            _int(item, "end_ms"),
            _float_or_none(item, "confidence"),
        )
        for item in _objects(raw, "segments")
    )
    return Transcript(
        _str(raw, "id"),
        _str(raw, "language"),
        words,
        segments,
        _str(raw, "engine_id"),
        _str(raw, "model_id"),
        cast(Mapping[str, JsonValue], _mapping(raw, "metadata")),
    )


def encode_track(track: SubtitleTrack) -> bytes:
    if _POLICY_SIGNATURE.fullmatch(track.policy_signature) is None:
        raise AppError("artifact.codec_invalid", {"reason": "policy_signature"})
    return encode_json(
        {
            "schema_version": TRACK_SCHEMA_VERSION,
            "subtitle_track": {
                "id": track.id,
                "source_transcript_id": track.source_transcript_id,
                "language": track.language,
                "revision": track.revision,
                "policy_signature": track.policy_signature,
                "cues": [
                    {
                        "id": cue.id,
                        "start_ms": cue.start_ms,
                        "end_ms": cue.end_ms,
                        "source_word_ids": list(cue.source_word_ids),
                        "source_text": cue.source_text,
                        "translated_text": cue.translated_text,
                        "lines": list(cue.lines),
                    }
                    for cue in track.cues
                ],
            },
        }
    )


def decode_track(data: bytes) -> SubtitleTrack:
    root = decode_json(data)
    _fields(root, {"schema_version", "subtitle_track"})
    schema_version = root.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in {1, TRACK_SCHEMA_VERSION}
        or not isinstance(root.get("subtitle_track"), dict)
    ):
        raise AppError("artifact.codec_invalid", {"reason": "subtitle_track"})
    raw = cast(dict[str, object], root["subtitle_track"])
    legacy_fields = {"id", "source_transcript_id", "language", "revision", "cues"}
    current_fields = {*legacy_fields, "policy_signature"}
    expected_fields = legacy_fields if schema_version == 1 else current_fields
    _fields(raw, expected_fields)
    cue_fields = {
        "id",
        "start_ms",
        "end_ms",
        "source_word_ids",
        "source_text",
        "translated_text",
        "lines",
    }
    raw_cues = _objects(raw, "cues")
    for cue in raw_cues:
        _fields(cue, cue_fields)
    cues = tuple(
        SubtitleCue(
            _str(item, "id"),
            _int(item, "start_ms"),
            _int(item, "end_ms"),
            tuple(_strings(item, "source_word_ids")),
            _str(item, "source_text"),
            _str_or_none(item, "translated_text"),
            tuple(_strings(item, "lines")),
        )
        for item in raw_cues
    )
    return SubtitleTrack(
        _str(raw, "id"),
        _str(raw, "source_transcript_id"),
        _str_or_none(raw, "language"),
        cues,
        _int(raw, "revision"),
        LEGACY_POLICY_SIGNATURE
        if schema_version == 1
        else _policy_signature(raw, "policy_signature"),
    )


def encode_publication_receipt(receipt: PublicationReceipt) -> bytes:
    return encode_json(
        {
            "schema_version": receipt.schema_version,
            "output_generation": receipt.output_generation,
            "targets": [
                {
                    "path": target.path,
                    "sha256": target.sha256,
                    "size_bytes": target.size_bytes,
                    "logical_name": target.logical_name,
                }
                for target in receipt.targets
            ],
        }
    )


def decode_publication_receipt(data: bytes) -> PublicationReceipt:
    root = decode_json(data)
    _fields(root, {"schema_version", "output_generation", "targets"})
    if root.get("schema_version") != SCHEMA_VERSION:
        raise AppError("output.publication_invalid", {"reason": "schema"})
    targets = _objects(root, "targets")
    for target in targets:
        _fields(target, {"path", "sha256", "size_bytes", "logical_name"})
    return PublicationReceipt(
        _str(root, "output_generation"),
        tuple(
            PublishedTarget(
                _str(target, "path"),
                _str(target, "sha256"),
                _int(target, "size_bytes"),
                _str(target, "logical_name"),
            )
            for target in targets
        ),
        _int(root, "schema_version"),
    )


def _object(root: dict[str, object], key: str, expected: set[str]) -> dict[str, object]:
    _fields(root, expected)
    if root.get("schema_version") != SCHEMA_VERSION or not isinstance(root.get(key), dict):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return cast(dict[str, object], root[key])


def _fields(value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise AppError("artifact.codec_invalid", {"reason": "fields"})


def _str(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return item


def _policy_signature(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or _POLICY_SIGNATURE.fullmatch(item) is None:
        raise AppError("artifact.codec_invalid", {"reason": key})
    return item


def _str_or_none(value: Mapping[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return item


def _int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return item


def _float_or_none(value: Mapping[str, object], key: str) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return float(item)


def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return cast(dict[str, object], item)


def _objects(value: Mapping[str, object], key: str) -> tuple[dict[str, object], ...]:
    item = value.get(key)
    if not isinstance(item, list):
        raise AppError("artifact.codec_invalid", {"reason": key})
    entries = cast(list[object], item)
    if any(not isinstance(entry, dict) for entry in entries):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return tuple(cast(dict[str, object], entry) for entry in entries)


def _strings(value: Mapping[str, object], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list):
        raise AppError("artifact.codec_invalid", {"reason": key})
    entries = cast(list[object], item)
    if any(not isinstance(entry, str) for entry in entries):
        raise AppError("artifact.codec_invalid", {"reason": key})
    return tuple(cast(str, entry) for entry in entries)
