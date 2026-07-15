"""FFprobe-backed media inspection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import MediaAsset
from captioner.core.domain.result import JsonValue
from captioner.core.ports.process import ProcessPort, ProcessResult

_MAX_STDERR = 2_000


@dataclass(slots=True)
class FFprobeMediaInspector:
    runner: ProcessPort
    executable: str = "ffprobe"

    async def inspect(self, source_path: Path, context: ExecutionContext) -> MediaAsset:
        source = source_path.expanduser().resolve()
        if not source.exists():
            raise AppError("media.input_missing", {"path": str(source)})
        if not source.is_file():
            raise AppError("media.input_not_file", {"path": str(source)})
        context.raise_if_cancelled()
        arguments = (
            self.executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        )
        try:
            result = await self.runner.run(arguments, context)
        except AppError as exc:
            if exc.code == "operation.cancelled":
                raise
            if exc.code == "process.executable_not_found":
                raise AppError("media.ffprobe_not_found", {"executable": self.executable}) from exc
            raise AppError("media.ffprobe_failed", {"reason": exc.code}) from exc
        if result.returncode != 0:
            raise AppError(
                "media.ffprobe_failed",
                {"returncode": result.returncode, "stderr": _decode_stderr(result.stderr)},
            )
        document = _parse_json(result)
        streams_value = document.get("streams")
        streams = cast(list[object], streams_value) if isinstance(streams_value, list) else None
        if streams is None:
            raise AppError("media.no_audio_stream", {"path": str(source)})
        audio = _first_audio_stream(streams)
        if audio is None:
            raise AppError("media.no_audio_stream", {"path": str(source)})
        format_value = document.get("format")
        format_data = (
            cast(dict[str, object], format_value) if isinstance(format_value, dict) else {}
        )
        duration_ms = _duration_ms(format_data, audio)
        context.raise_if_cancelled()
        content_hash = _sha256_file(source, context)
        container_value = format_data.get("format_name")
        container = container_value if isinstance(container_value, str) else ""
        if not container.strip():
            raise AppError("media.ffprobe_invalid_json", {"reason": "container"})
        metadata: dict[str, JsonValue] = {"format_name": container}
        long_name = format_data.get("format_long_name")
        codec = audio.get("codec_name")
        if isinstance(long_name, str) and long_name.strip():
            metadata["format_long_name"] = long_name
        if isinstance(codec, str) and codec.strip():
            metadata["audio_codec"] = codec
        return MediaAsset(
            id=f"media-{content_hash}",
            source_path=source,
            content_hash=content_hash,
            duration_ms=duration_ms,
            audio_stream_index=_stream_index(audio),
            container=container,
            metadata=metadata,
        )


def _parse_json(result: ProcessResult) -> dict[str, object]:
    try:
        value = json.loads(result.stdout.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppError("media.ffprobe_invalid_json", {"reason": "json"}) from exc
    if not isinstance(value, dict):
        raise AppError("media.ffprobe_invalid_json", {"reason": "root"})
    return cast(dict[str, object], value)


def _first_audio_stream(streams: list[object]) -> dict[str, object] | None:
    candidates: list[tuple[int, dict[str, object]]] = []
    for raw in streams:
        if not isinstance(raw, dict):
            continue
        stream = cast(dict[str, object], raw)
        if stream.get("codec_type") != "audio":
            continue
        index = stream.get("index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            candidates.append((index, stream))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _stream_index(stream: Mapping[str, object]) -> int:
    value = stream.get("index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AppError("media.ffprobe_invalid_json", {"reason": "stream_index"})
    return value


def _duration_ms(format_data: Mapping[str, object], audio: Mapping[str, object]) -> int:
    for candidate in (format_data.get("duration"), audio.get("duration")):
        if isinstance(candidate, (int, float, str)) and not isinstance(candidate, bool):
            try:
                seconds = float(candidate)
            except ValueError:
                continue
            if math.isfinite(seconds) and seconds > 0:
                milliseconds = math.floor(seconds * 1000 + 0.5)
                if milliseconds > 0:
                    return milliseconds
    raise AppError("media.duration_invalid")


def _sha256_file(path: Path, context: ExecutionContext) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                context.raise_if_cancelled()
                digest.update(chunk)
    except AppError:
        raise
    except OSError as exc:
        raise AppError("media.input_read_failed", {"path": str(path)}) from exc
    return digest.hexdigest()


def _decode_stderr(data: bytes) -> str:
    """Decode diagnostic bytes explicitly, bounded for structured errors."""
    return data[:_MAX_STDERR].decode("utf-8", errors="replace")
