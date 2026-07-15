"""Lazy, single-model Faster Whisper adapter."""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.result import JsonValue
from captioner.core.domain.transcript import (
    Transcript,
    TranscriptSegment,
    WordToken,
    derive_transcript_id,
)
from captioner.core.ports import CapabilityProbe
from captioner.core.ports.asr import ASRCapabilities, TranscriptionRequest


class _SDKWord(Protocol):
    word: str
    start: float | None
    end: float | None
    probability: float | None


class _SDKSegment(Protocol):
    text: str
    start: float | None
    end: float | None
    words: Iterable[_SDKWord] | None


class _SDKInfo(Protocol):
    language: str | None


class WhisperModelProtocol(Protocol):
    def transcribe(
        self, audio: str, **options: object
    ) -> tuple[Iterable[_SDKSegment], _SDKInfo]: ...


@dataclass(frozen=True, slots=True)
class FasterWhisperConfig:
    model_id: str = "tiny"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = None
    vad_filter: bool = True
    model_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.device.strip() or not self.compute_type.strip():
            raise ValueError


ModelFactory = Callable[[FasterWhisperConfig], WhisperModelProtocol]


@dataclass(slots=True)
class FasterWhisperEngine:
    config: FasterWhisperConfig
    model_factory: ModelFactory | None = None
    _model: WhisperModelProtocol | None = None

    @property
    def engine_id(self) -> str:
        return "faster-whisper"

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            word_timestamps=True,
            segment_timestamps=True,
            language_detection=True,
            native_long_audio=True,
            internal_batching=False,
            supported_languages=None,
            supported_devices=frozenset({"auto", "cpu", "cuda"}),
        )

    async def probe(self) -> CapabilityProbe:
        available = importlib.util.find_spec("faster_whisper") is not None
        return CapabilityProbe(
            available=available,
            details={"engine_id": self.engine_id, "model_id": self.config.model_id},
        )

    async def transcribe(
        self, request: TranscriptionRequest, context: ExecutionContext
    ) -> Transcript:
        context.raise_if_cancelled()
        model = self._get_or_load_model(context)
        context.raise_if_cancelled()
        if not request.audio.path.is_file():
            raise AppError("asr.transcription_failed", {"reason": "audio_missing"})
        language = request.language or self.config.language
        options: dict[str, object] = {
            "word_timestamps": True,
            "temperature": 0.0,
            "vad_filter": self.config.vad_filter,
        }
        if language is not None:
            options["language"] = language
        try:
            raw_segments, info = model.transcribe(str(request.audio.path), **options)
        except AppError:
            raise
        except Exception as exc:
            raise AppError("asr.transcription_failed", {"reason": "sdk_call"}) from exc
        words: list[WordToken] = []
        transcript_segments: list[TranscriptSegment] = []
        try:
            for raw_segment in raw_segments:
                context.raise_if_cancelled()
                segment = raw_segment
                text = cast(object, getattr(segment, "text", None))
                if not isinstance(text, str):
                    continue
                if not text.strip():
                    continue
                segment_start = seconds_to_ms(getattr(segment, "start", None))
                segment_end = seconds_to_ms(getattr(segment, "end", None))
                if segment_end <= segment_start:
                    _raise_invalid_segment_range()
                raw_words = cast(object, getattr(segment, "words", None))
                if raw_words is None:
                    _raise_word_timestamp_missing()
                if not isinstance(raw_words, Iterable):
                    _raise_word_timestamp_missing()
                try:
                    segment_words = tuple(cast(Iterable[_SDKWord], raw_words))
                except TypeError as exc:
                    _raise_word_timestamp_missing_from(exc)
                if not segment_words:
                    _raise_word_timestamp_missing()
                segment_word_ids: list[str] = []
                for raw_word in segment_words:
                    context.raise_if_cancelled()
                    word = raw_word
                    word_text = cast(object, getattr(word, "word", None))
                    if not isinstance(word_text, str):
                        _raise_invalid_word_text()
                    if not word_text.strip():
                        _raise_invalid_word_text()
                    word_start = seconds_to_ms(getattr(word, "start", None))
                    word_end = seconds_to_ms(getattr(word, "end", None))
                    if word_end <= word_start:
                        _raise_invalid_word_range()
                    probability = getattr(word, "probability", None)
                    confidence = _confidence(probability)
                    word_id = f"word-{len(words) + 1:06d}"
                    words.append(
                        WordToken(
                            id=word_id,
                            text=word_text,
                            start_ms=word_start,
                            end_ms=word_end,
                            confidence=confidence,
                        )
                    )
                    segment_word_ids.append(word_id)
                transcript_segments.append(
                    TranscriptSegment(
                        id=f"segment-{len(transcript_segments) + 1:06d}",
                        word_ids=tuple(segment_word_ids),
                        raw_text=text.strip(),
                        start_ms=segment_start,
                        end_ms=segment_end,
                        confidence=None,
                    )
                )
        except AppError as exc:
            if exc.code == "operation.cancelled":
                raise
            if exc.code in {"asr.word_timestamp_missing", "asr.word_timestamp_invalid"}:
                raise
            raise AppError("asr.output_invalid", {"reason": exc.code}) from exc
        except Exception as exc:
            raise AppError("asr.output_invalid", {"reason": "segments"}) from exc
        if not words or not transcript_segments:
            raise AppError("asr.empty_transcript")
        detected_language = _detected_language(info, language)
        metadata: dict[str, JsonValue] = {
            "word_timestamps": True,
            "vad_filter": self.config.vad_filter,
        }
        transcript_id = derive_transcript_id(
            language=detected_language,
            words=words,
            segments=transcript_segments,
            engine_id=self.engine_id,
            model_id=self.config.model_id,
            metadata=metadata,
        )
        try:
            return Transcript(
                id=transcript_id,
                language=detected_language,
                words=tuple(words),
                segments=tuple(transcript_segments),
                engine_id=self.engine_id,
                model_id=self.config.model_id,
                metadata=metadata,
            )
        except AppError as exc:
            raise AppError("asr.output_invalid", {"reason": exc.code}) from exc

    def _get_or_load_model(self, context: ExecutionContext) -> WhisperModelProtocol:
        if self._model is not None:
            return self._model
        context.raise_if_cancelled()
        factory = self.model_factory or default_model_factory
        try:
            self._model = factory(self.config)
        except AppError:
            raise
        except (ImportError, ModuleNotFoundError) as exc:
            raise AppError("asr.runtime_missing") from exc
        except Exception as exc:
            raise AppError("asr.model_load_failed", {"model_id": self.config.model_id}) from exc
        return self._model


def seconds_to_ms(value: object, *, code: str = "asr.word_timestamp_invalid") -> int:
    """Convert non-negative SDK seconds using round-half-up millisecond rounding."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AppError(code, {"reason": "not_number"})
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise AppError(code, {"reason": "non_finite_or_negative"})
    return math.floor(seconds * 1000 + 0.5)


def _confidence(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AppError("asr.output_invalid", {"reason": "confidence"})
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise AppError("asr.output_invalid", {"reason": "confidence"})
    return confidence


def _detected_language(info: _SDKInfo, requested: str | None) -> str:
    value = getattr(info, "language", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if requested is not None and requested.strip():
        return requested.strip()
    return "und"


def default_model_factory(config: FasterWhisperConfig) -> WhisperModelProtocol:
    try:
        module = importlib.import_module("faster_whisper")
    except (ImportError, ModuleNotFoundError) as exc:
        raise AppError("asr.runtime_missing") from exc
    constructor = getattr(module, "WhisperModel", None)
    if not callable(constructor):
        raise AppError("asr.runtime_missing", {"reason": "WhisperModel"})
    options: dict[str, object] = {
        "device": config.device,
        "compute_type": config.compute_type,
    }
    if config.model_cache_dir is not None:
        options["download_root"] = str(config.model_cache_dir)
    try:
        model = constructor(config.model_id, **options)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AppError("asr.runtime_missing") from exc
    except Exception as exc:
        raise AppError("asr.model_load_failed", {"model_id": config.model_id}) from exc
    return cast(WhisperModelProtocol, model)


def _raise_word_timestamp_missing() -> NoReturn:
    raise AppError("asr.word_timestamp_missing")


def _raise_word_timestamp_missing_from(cause: TypeError) -> NoReturn:
    raise AppError("asr.word_timestamp_missing") from cause


def _raise_invalid_word_text() -> NoReturn:
    raise AppError("asr.output_invalid", {"reason": "word_text"})


def _raise_invalid_word_range() -> NoReturn:
    raise AppError("asr.word_timestamp_invalid", {"reason": "range"})


def _raise_invalid_segment_range() -> NoReturn:
    raise AppError("asr.word_timestamp_invalid", {"reason": "segment_range"})
