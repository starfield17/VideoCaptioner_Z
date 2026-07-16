"""FFmpeg audio normalization adapter for the fixed Phase 1 ASR format."""

from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.ports.process import ProcessPort, ProcessResult

_MAX_STDERR = 2_000


@dataclass(slots=True)
class FFmpegAudioNormalizer:
    runner: ProcessPort
    executable: str = "ffmpeg"

    async def normalize(
        self, asset: MediaAsset, workspace: Path, context: ExecutionContext
    ) -> AudioArtifact:
        context.raise_if_cancelled()
        workspace_path = workspace.expanduser().resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        output = workspace_path / "normalized.wav"
        _remove(output)
        arguments = (
            self.executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(asset.source_path),
            "-map",
            f"0:{asset.audio_stream_index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        )
        try:
            result = await self.runner.run(arguments, context)
        except AppError as exc:
            _remove(output)
            if exc.code == "operation.cancelled":
                raise
            if exc.code == "process.executable_not_found":
                raise AppError("media.ffmpeg_not_found", {"executable": self.executable}) from exc
            raise AppError("media.audio_normalization_failed", {"reason": exc.code}) from exc
        if result.returncode != 0:
            _remove(output)
            raise AppError(
                "media.audio_normalization_failed",
                {"returncode": result.returncode, "stderr": _decode_stderr(result)},
            )
        if not output.is_file():
            raise AppError("media.normalized_audio_missing", {"path": str(output)})
        try:
            sample_rate, channels, sample_width, frames = _read_wav_format(output)
        except (OSError, EOFError, wave.Error) as exc:
            _remove(output)
            raise AppError("media.normalized_audio_invalid", {"reason": "wav"}) from exc
        if sample_rate != 16_000 or channels != 1 or sample_width != 2:
            _remove(output)
            raise AppError("media.normalized_audio_invalid", {"reason": "format"})
        duration_ms = math.floor(frames * 1000 / sample_rate + 0.5)
        if duration_ms <= 0:
            _remove(output)
            raise AppError("media.normalized_audio_invalid", {"reason": "duration"})
        try:
            sha256 = _sha256(output, context)
        except AppError:
            _remove(output)
            raise
        except OSError as exc:
            _remove(output)
            raise AppError("media.normalized_audio_read_failed", {"reason": "read"}) from exc
        return AudioArtifact(
            artifact_id=f"audio-{sha256}",
            path=output,
            sha256=sha256,
            sample_rate=sample_rate,
            channels=channels,
            duration_ms=duration_ms,
            codec="pcm_s16le",
        )


def _read_wav_format(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getframerate(),
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getnframes(),
        )


def _sha256(path: Path, context: ExecutionContext) -> str:
    digest = hashlib.sha256()
    first_chunk = True
    try:
        context.raise_if_cancelled()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                context.raise_if_cancelled()
                digest.update(chunk)
                if first_chunk:
                    first_chunk = False
                    context.checkpoint("mid_execute")
    except AppError:
        raise
    except OSError as exc:
        raise AppError("media.normalized_audio_read_failed", {"reason": "read"}) from exc
    return digest.hexdigest()


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise AppError("media.cleanup_failed", {"path": str(path)}) from exc


def _decode_stderr(result: ProcessResult) -> str:
    return result.stderr[:_MAX_STDERR].decode("utf-8", errors="replace")
