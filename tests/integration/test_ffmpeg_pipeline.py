from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from scripts.validate_subtitle import parse_srt
from tests.support import make_transcript

from captioner.adapters.asr.fake import FakeASRAdapter
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.core.application.run_single import RunSingleRequest, RunSingleService
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parents[1] / "fixtures" / "media"


def _service(
    temp_root: Path, asr: FakeASRAdapter, *, ffmpeg_bin: str = "ffmpeg"
) -> RunSingleService:
    runner = AsyncioSubprocessRunner()
    return RunSingleService(
        inspector=FFprobeMediaInspector(runner),
        normalizer=FFmpegAudioNormalizer(runner, executable=ffmpeg_bin),
        asr_engine=asr,
        artifact_store_factory=LocalArtifactStore,
        transcript_serializer=serialize_transcript,
        subtitle_serializer=serialize_srt,
        temp_root=temp_root,
    )


def test_real_ffmpeg_wav_pipeline_and_spaced_unicode_path(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "媒体 samples" / "english sample.wav"
        source.parent.mkdir()
        shutil.copy2(FIXTURES / "english-short.wav", source)
        output = tmp_path / "output"
        result = await _service(
            tmp_path / "runtime", FakeASRAdapter(transcription_result=make_transcript())
        ).run(RunSingleRequest(source, output, "en", False))
        assert result.subtitle_path.is_file()
        assert parse_srt(result.subtitle_path.read_text(encoding="utf-8"))

    asyncio.run(scenario())


def test_real_ffmpeg_normalized_wav_format(tmp_path: Path) -> None:
    async def scenario() -> AudioArtifact:
        source = FIXTURES / "english-short.wav"
        runner = AsyncioSubprocessRunner()
        inspector = FFprobeMediaInspector(runner)
        normalizer = FFmpegAudioNormalizer(runner)
        context = ExecutionContext()
        asset = await inspector.inspect(source, context)
        return await normalizer.normalize(asset, tmp_path / "workspace", context)

    artifact = asyncio.run(scenario())
    with wave.open(str(artifact.path), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


def test_real_ffmpeg_video_audio_and_no_audio_paths(tmp_path: Path) -> None:
    video = tmp_path / "video with audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=1",
            "-i",
            str(FIXTURES / "english-short.wav"),
            "-t",
            "2",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )

    async def scenario() -> None:
        runner = AsyncioSubprocessRunner()
        asset = await FFprobeMediaInspector(runner).inspect(video, ExecutionContext())
        assert asset.audio_stream_index >= 0
        await FFmpegAudioNormalizer(runner).normalize(
            asset, tmp_path / "video-workspace", ExecutionContext()
        )
        with pytest.raises(AppError, match="no_audio_stream"):
            await FFprobeMediaInspector(runner).inspect(
                FIXTURES / "no-audio.mp4", ExecutionContext()
            )

    asyncio.run(scenario())


def test_real_ffmpeg_failure_leaves_no_final_srt(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = FIXTURES / "english-short.wav"
        output = tmp_path / "output"
        service = _service(
            tmp_path / "runtime",
            FakeASRAdapter(transcription_result=make_transcript()),
            ffmpeg_bin="captioner-missing-ffmpeg",
        )
        with pytest.raises(AppError, match="ffmpeg_not_found"):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not list(output.glob("*.srt"))

    asyncio.run(scenario())
