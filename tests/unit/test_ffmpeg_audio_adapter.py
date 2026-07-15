from __future__ import annotations

import asyncio
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.support import make_media

from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.process import ProcessResult


def _empty_calls() -> list[tuple[str, ...]]:
    return []


@dataclass
class FFmpegStub:
    result: ProcessResult | AppError
    write_output: bool = True
    calls: list[tuple[str, ...]] = field(default_factory=_empty_calls)

    async def run(self, arguments: Sequence[str], context: ExecutionContext) -> ProcessResult:
        self.calls.append(tuple(arguments))
        context.raise_if_cancelled()
        if self.write_output:
            with wave.open(arguments[-1], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\x00\x00" * 16_000)
        if isinstance(self.result, AppError):
            raise self.result
        return self.result


def test_ffmpeg_normalizes_selected_stream_and_returns_artifact(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "input file.wav"
        source.write_bytes(b"source")
        runner = FFmpegStub(ProcessResult(b"", b"", 0))
        artifact = await FFmpegAudioNormalizer(runner).normalize(
            make_media(source, audio_stream_index=2), tmp_path / "workspace", ExecutionContext()
        )
        assert artifact.sample_rate == 16_000
        assert artifact.channels == 1
        assert artifact.duration_ms == 1_000
        assert len(artifact.sha256) == 64
        command = runner.calls[0]
        assert command[command.index("-map") + 1] == "0:2"
        assert "-nostdin" in command
        assert "pcm_s16le" in command

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("result", "write_output"),
    [(ProcessResult(b"", b"failure", 1), False), (ProcessResult(b"", b"", 0), False)],
)
def test_ffmpeg_failure_or_missing_output_cleans_workspace(
    tmp_path: Path, result: ProcessResult, write_output: bool
) -> None:
    async def scenario() -> None:
        source = tmp_path / "input.wav"
        source.write_bytes(b"source")
        workspace = tmp_path / "workspace"
        with pytest.raises(AppError):
            await FFmpegAudioNormalizer(FFmpegStub(result, write_output)).normalize(
                make_media(source), workspace, ExecutionContext()
            )
        assert not (workspace / "normalized.wav").exists()

    asyncio.run(scenario())


def test_ffmpeg_missing_binary_and_cancellation_are_structured(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "input.wav"
        source.write_bytes(b"source")
        with pytest.raises(AppError, match="ffmpeg_not_found"):
            await FFmpegAudioNormalizer(
                FFmpegStub(AppError("process.executable_not_found"))
            ).normalize(make_media(source), tmp_path / "workspace", ExecutionContext())
        context = ExecutionContext()
        context.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await FFmpegAudioNormalizer(FFmpegStub(ProcessResult(b"", b"", 0))).normalize(
                make_media(source), tmp_path / "cancelled", context
            )

    asyncio.run(scenario())
