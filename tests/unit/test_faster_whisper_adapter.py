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
    default_model_factory,
    seconds_to_ms,
)
from captioner.adapters.exporters.transcript_json import serialize
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
    text: object
    start: float
    end: float
    words: object


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


def test_transcribe_mid_execute_occurs_on_first_segment_consumption(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = FakeModel((FakeSegment(" hello", 0.0, 1.0, (FakeWord("hello", 0.0, 1.0),)),))
        checkpoints: list[str] = []
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(model))
        transcript = await engine.transcribe(
            _request(tmp_path),
            ExecutionContext(checkpoint_hook=checkpoints.append),
        )
        assert transcript.words
        assert checkpoints == ["mid_execute"]

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


@pytest.mark.parametrize(
    "segment",
    [
        FakeSegment(42, 0.0, 1.0, (FakeWord("word", 0, 1),)),
        FakeSegment(" ", 0.0, 1.0, (FakeWord("word", 0, 1),)),
        FakeSegment(" ", 0.0, 1.0, 42),
    ],
)
def test_malformed_segment_text_or_words_is_rejected(tmp_path: Path, segment: FakeSegment) -> None:
    async def scenario() -> None:
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(FakeModel((segment,))))
        with pytest.raises(AppError, match=r"asr\.output_invalid"):
            await engine.transcribe(_request(tmp_path), ExecutionContext())

    asyncio.run(scenario())


def test_empty_blank_segment_without_words_is_ignored(tmp_path: Path) -> None:
    async def scenario() -> None:
        segment = FakeSegment(" ", 0.0, 1.0, None)
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(FakeModel((segment,))))
        with pytest.raises(AppError, match=r"asr\.empty_transcript"):
            await engine.transcribe(_request(tmp_path), ExecutionContext())

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed_first", [False, True])
def test_partial_valid_output_cannot_hide_malformed_segment(
    tmp_path: Path, malformed_first: bool
) -> None:
    valid = FakeSegment("valid", 0.0, 1.0, (FakeWord("valid", 0.0, 1.0),))
    malformed = FakeSegment(42, 1.0, 2.0, ())
    segments = (malformed, valid) if malformed_first else (valid, malformed)

    async def scenario() -> None:
        engine = FasterWhisperEngine(FasterWhisperConfig(), _factory(FakeModel(segments)))
        with pytest.raises(AppError, match=r"asr\.output_invalid"):
            await engine.transcribe(_request(tmp_path), ExecutionContext())

    asyncio.run(scenario())


def test_named_model_has_stable_public_identity(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = FakeModel((FakeSegment("hello", 0.0, 1.0, (FakeWord("hello", 0.0, 1.0),)),))
        config = FasterWhisperConfig(model_ref="tiny")
        engine = FasterWhisperEngine(config, _factory(model))
        transcript = await engine.transcribe(_request(tmp_path), ExecutionContext())
        assert config.model_identity == "faster-whisper:tiny"
        assert transcript.model_id == "faster-whisper:tiny"
        assert "tiny" in serialize(transcript)

    asyncio.run(scenario())


def _write_model_identity_files(path: Path, value: bytes = b"model") -> None:
    path.mkdir()
    (path / "config.json").write_text('{"model": "tiny"}', encoding="utf-8")
    (path / "model.bin").write_bytes(value)
    (path / "tokenizer.json").write_text('{"tokenizer": 1}', encoding="utf-8")


def test_local_model_identity_is_content_based_and_not_serialized(tmp_path: Path) -> None:
    first_dir = tmp_path / "first model"
    second_dir = tmp_path / "second model"
    different_dir = tmp_path / "different model"
    _write_model_identity_files(first_dir)
    _write_model_identity_files(second_dir)
    _write_model_identity_files(different_dir, b"different")
    first = FasterWhisperConfig(model_ref=str(first_dir))
    second = FasterWhisperConfig(model_ref=str(second_dir))
    different = FasterWhisperConfig(model_ref=str(different_dir))
    assert first.model_ref == str(first_dir.resolve())
    assert first.model_identity == second.model_identity
    assert first.model_identity != different.model_identity

    async def scenario() -> None:
        segments = (FakeSegment("hello", 0.0, 1.0, (FakeWord("hello", 0.0, 1.0),)),)
        first_transcript = await FasterWhisperEngine(
            first, _factory(FakeModel(segments))
        ).transcribe(_request(tmp_path), ExecutionContext())
        second_transcript = await FasterWhisperEngine(
            second, _factory(FakeModel(segments))
        ).transcribe(_request(tmp_path), ExecutionContext())
        first_serialized = serialize(first_transcript)
        assert first_serialized == serialize(second_transcript)
        assert str(first_dir) not in first_serialized
        assert str(second_dir) not in first_serialized
        assert str(first_dir) not in first_transcript.id

    asyncio.run(scenario())


@pytest.mark.parametrize("model_ref", ["/missing/model", "./missing-model"])
def test_invalid_local_model_reference_fails_structurally(tmp_path: Path, model_ref: str) -> None:
    selected = model_ref if model_ref.startswith("/") else str(tmp_path / model_ref[2:])
    with pytest.raises(AppError, match=r"asr\.model_config_invalid"):
        FasterWhisperConfig(model_ref=selected)


def test_local_model_without_identity_files_fails_structurally(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    with pytest.raises(AppError, match=r"asr\.model_config_invalid"):
        FasterWhisperConfig(model_ref=str(model_dir))


def test_absolute_model_reference_is_passed_to_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "model"
    _write_model_identity_files(model_dir)
    received: list[str] = []

    class SDKModel:
        def __init__(self, model_ref: str, **options: object) -> None:
            del options
            received.append(model_ref)

    class SDKModule:
        WhisperModel = SDKModel

    def import_sdk(_name: str) -> object:
        return SDKModule

    monkeypatch.setattr(faster_whisper_module.importlib, "import_module", import_sdk)
    config = FasterWhisperConfig(model_ref=str(model_dir))
    default_model_factory(config)
    assert received == [str(model_dir.resolve())]
