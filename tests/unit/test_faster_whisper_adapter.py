from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from tests.support import make_audio

import captioner.adapters.asr.faster_whisper as faster_whisper_module
from captioner.adapters.asr.faster_whisper import (
    FasterWhisperConfig,
    FasterWhisperEngine,
    ModelFactory,
    seconds_to_ms,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.asr import TranscriptionRequest


@dataclass(frozen=True)
class FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.8


@dataclass(frozen=True)
class FakeSegment:
    text: str
    start: float
    end: float
    words: tuple[FakeWord, ...] | None


@dataclass(frozen=True)
class FakeInfo:
    language: str = "en"


class FakeModel:
    def __init__(self, segments: tuple[FakeSegment, ...]) -> None:
        self.segments = segments
        self.calls = 0
        self.options: list[dict[str, object]] = []

    def transcribe(self, audio: str, **options: object) -> tuple[tuple[FakeSegment, ...], FakeInfo]:
        assert audio
        self.calls += 1
        self.options.append(options)
        return self.segments, FakeInfo()


def _request(tmp_path: Path) -> TranscriptionRequest:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"audio")
    return TranscriptionRequest(make_audio(audio_path), "en")


def _factory(model: FakeModel) -> ModelFactory:
    def factory(_config: FasterWhisperConfig) -> FakeModel:
        return model

    return cast(ModelFactory, factory)


def test_seconds_to_ms_uses_round_half_up_and_rejects_invalid_values() -> None:
    assert seconds_to_ms(1.004) == 1_004
    assert seconds_to_ms(1.0045) == 1_005
    for value in (-1, float("nan"), float("inf"), None):
        with pytest.raises(AppError, match="word_timestamp_invalid"):
            seconds_to_ms(value)


def test_model_factory_runs_once_and_model_is_reused(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = FakeModel(
            (
                FakeSegment(
                    " hello world",
                    0.0,
                    1.0,
                    (FakeWord(" hello ", 0, 0.5), FakeWord("world", 0.5, 1.0)),
                ),
            )
        )
        factory_calls = 0

        def factory(_config: FasterWhisperConfig) -> FakeModel:
            nonlocal factory_calls
            factory_calls += 1
            return model

        engine = FasterWhisperEngine(
            FasterWhisperConfig(model_id="tiny"), cast(ModelFactory, factory)
        )
        first = await engine.transcribe(_request(tmp_path), ExecutionContext())
        second = await engine.transcribe(_request(tmp_path), ExecutionContext())
        assert factory_calls == 1
        assert model.calls == 2
        assert first.id == second.id
        assert first.words[0].id == "word-000001"
        assert model.options[0]["word_timestamps"] is True
        assert model.options[0]["temperature"] == 0.0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "segment",
    [
        FakeSegment("text", 0.0, 1.0, None),
        FakeSegment("text", 0.0, 1.0, (FakeWord("text", 0.5, 0.5),)),
        FakeSegment("text", 1.0, 0.0, (FakeWord("text", 0.0, 0.5),)),
    ],
)
def test_invalid_word_timestamp_output_is_structured(tmp_path: Path, segment: FakeSegment) -> None:
    async def scenario() -> None:
        model = FakeModel((segment,))
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(model))
        with pytest.raises(AppError):
            await engine.transcribe(_request(tmp_path), ExecutionContext())

    asyncio.run(scenario())


def test_empty_output_and_cancellation_do_not_load_model(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory_calls = 0

        def factory(_config: FasterWhisperConfig) -> FakeModel:
            nonlocal factory_calls
            factory_calls += 1
            return FakeModel(())

        engine = FasterWhisperEngine(FasterWhisperConfig(), cast(ModelFactory, factory))
        with pytest.raises(AppError, match="empty_transcript"):
            await engine.transcribe(_request(tmp_path), ExecutionContext())
        context = ExecutionContext()
        context.cancel()
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await engine.transcribe(_request(tmp_path), context)
        assert factory_calls == 1

    asyncio.run(scenario())


def test_missing_optional_runtime_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(FakeModel(())))
        engine.model_factory = None
        with pytest.raises(AppError, match=r"asr\.(runtime_missing|model_load_failed)"):
            await engine.transcribe(_request(tmp_path), ExecutionContext())

    def missing_runtime(_name: str) -> object:
        raise ModuleNotFoundError("faster_whisper")

    monkeypatch.setattr(faster_whisper_module.importlib, "import_module", missing_runtime)
    asyncio.run(scenario())
