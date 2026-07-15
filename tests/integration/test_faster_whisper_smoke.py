from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import cast

import pytest
from scripts.validate_subtitle import parse_srt

from captioner.adapters.asr.faster_whisper import (
    FasterWhisperConfig,
    FasterWhisperEngine,
    ModelFactory,
    WhisperModelProtocol,
    default_model_factory,
)
from captioner.adapters.exporters.srt import serialize
from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.core.domain.execution import ExecutionContext
from captioner.core.policies.simple_segmentation import segment_transcript
from captioner.core.ports.asr import TranscriptionRequest

pytestmark = pytest.mark.slow
FIXTURE = Path(__file__).parents[1] / "fixtures" / "media" / "english-short.wav"


def test_faster_whisper_cpu_smoke(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper")

    async def scenario() -> None:
        runner = AsyncioSubprocessRunner()
        context = ExecutionContext()
        asset = await FFprobeMediaInspector(runner).inspect(FIXTURE, context)
        audio = await FFmpegAudioNormalizer(runner).normalize(
            asset, tmp_path / "slow-workspace", context
        )
        cache_value = os.environ.get("CAPTIONER_FASTER_WHISPER_CACHE")
        config = FasterWhisperConfig(
            model_id="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            model_cache_dir=None if cache_value is None else Path(cache_value),
        )
        factory_calls = 0

        def factory(value: FasterWhisperConfig) -> WhisperModelProtocol:
            nonlocal factory_calls
            factory_calls += 1
            return default_model_factory(value)

        engine = FasterWhisperEngine(config, cast(ModelFactory, factory))
        transcript = await engine.transcribe(TranscriptionRequest(audio, "en"), context)
        assert transcript.words
        assert all(word.end_ms > word.start_ms for word in transcript.words)
        assert parse_srt(serialize(segment_transcript(transcript)))
        assert factory_calls == 1

    asyncio.run(scenario())
