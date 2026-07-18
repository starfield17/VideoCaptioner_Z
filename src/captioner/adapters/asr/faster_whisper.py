"""Lazy, single-model Faster Whisper adapter."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
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


_IDENTITY_PATTERNS = (
    "config.json",
    "model*.bin",
    "model*.safetensors",
    "tokenizer.json",
    "vocabulary.*",
)


@dataclass(frozen=True, slots=True, init=False)
class FasterWhisperConfig:
    """Adapter configuration separating SDK loading from public identity.

    ``model_ref`` is passed to Faster Whisper and may be an absolute local
    directory.  ``model_identity`` is derived from stable model content and is
    the only model value allowed into Transcript artifacts.  ``model_id`` is a
    compatibility-only constructor/property alias for the former Phase 1 API.
    """

    model_ref: str
    model_identity: str
    device: str
    compute_type: str
    language: str | None
    vad_filter: bool
    model_cache_dir: Path | None

    def __init__(
        self,
        model_ref: str = "tiny",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        vad_filter: bool = True,
        model_cache_dir: Path | None = None,
        *,
        model_identity: str | None = None,
        model_id: str | None = None,
    ) -> None:
        if model_id is not None:
            if model_ref != "tiny" and model_ref != model_id:
                raise ValueError
            model_ref = model_id
        normalized_ref = _normalize_model_ref(model_ref)
        if not normalized_ref or not device.strip() or not compute_type.strip():
            raise ValueError
        identity = (
            derive_model_identity(normalized_ref)
            if model_identity is None
            else _validate_model_identity(model_identity)
        )
        object.__setattr__(self, "model_ref", normalized_ref)
        object.__setattr__(self, "model_identity", identity)
        object.__setattr__(self, "device", device.strip())
        object.__setattr__(self, "compute_type", compute_type.strip())
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "vad_filter", vad_filter)
        object.__setattr__(self, "model_cache_dir", model_cache_dir)

    @property
    def model_id(self) -> str:
        """Return the stable public identity for old adapter callers."""
        return self.model_identity


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
            details={"engine_id": self.engine_id, "model_identity": self.config.model_identity},
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
            for segment_index, raw_segment in enumerate(raw_segments):
                context.raise_if_cancelled()
                segment = raw_segment
                text = cast(object, getattr(segment, "text", None))
                if not isinstance(text, str):
                    _raise_invalid_segment_output("segment_text")
                if segment_index == 0:
                    context.checkpoint("mid_execute")
                raw_words = cast(object, getattr(segment, "words", None))
                if not text.strip():
                    # Faster Whisper can emit an empty no-speech segment.  It
                    # is ignored only when it has no word payload at all; a
                    # blank segment with words is malformed and must fail.
                    if raw_words is None:
                        continue
                    if not isinstance(raw_words, Iterable):
                        _raise_invalid_segment_output("blank_segment_with_words")
                    try:
                        blank_words = tuple(cast(Iterable[_SDKWord], raw_words))
                    except TypeError as exc:
                        _raise_invalid_segment_output_from("blank_segment_with_words", exc)
                    if blank_words:
                        _raise_invalid_segment_output("blank_segment_with_words")
                    continue
                segment_start = seconds_to_ms(getattr(segment, "start", None))
                segment_end = seconds_to_ms(getattr(segment, "end", None))
                if segment_end <= segment_start:
                    _raise_invalid_segment_range()
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
            if exc.code in {
                "operation.cancelled",
                "asr.output_invalid",
                "asr.word_timestamp_missing",
                "asr.word_timestamp_invalid",
            }:
                raise
            raise AppError("asr.output_invalid", {"reason": exc.code}) from exc
        except Exception as exc:
            raise AppError("asr.transcription_failed", {"reason": "segments"}) from exc
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
            model_id=self.config.model_identity,
            metadata=metadata,
        )
        try:
            return Transcript(
                id=transcript_id,
                language=detected_language,
                words=tuple(words),
                segments=tuple(transcript_segments),
                engine_id=self.engine_id,
                model_id=self.config.model_identity,
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
            raise AppError(
                "asr.model_load_failed", {"model_id": self.config.model_identity}
            ) from exc
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


def _normalize_model_ref(model_ref: object) -> str:
    if not isinstance(model_ref, str):
        raise TypeError
    normalized = model_ref.strip()
    if not normalized:
        raise ValueError
    candidate = Path(normalized).expanduser()
    # Keep POSIX-style absolute references recognizable when the CLI is being
    # exercised on Windows.  ``WindowsPath('/missing/model').is_absolute()``
    # is false, but the user still supplied a path rather than a model name.
    local_hint = (
        candidate.is_absolute() or normalized.startswith("/") or normalized.startswith((".", "~"))
    )
    if local_hint or candidate.exists():
        if not candidate.is_dir():
            raise AppError(
                "asr.model_config_invalid",
                {"reason": "local_model_directory"},
            )
        return str(candidate.resolve())
    return normalized


def _validate_model_identity(model_identity: object) -> str:
    if not isinstance(model_identity, str) or not model_identity.strip():
        raise ValueError
    if Path(model_identity).is_absolute():
        raise AppError("asr.model_config_invalid", {"reason": "absolute_model_identity"})
    return model_identity.strip()


def derive_model_identity(model_ref: str) -> str:
    """Derive a stable public identity without including local paths.

    Named models use their provider reference.  Local directories use only a
    bounded, direct-child manifest of recognized model identity files; cache
    files, directory names and modification times are deliberately excluded.
    """
    normalized_ref = _normalize_model_ref(model_ref)
    candidate = Path(normalized_ref)
    if not candidate.is_dir():
        return f"faster-whisper:{normalized_ref}"
    identity_files = _model_identity_files(candidate)
    if not identity_files:
        raise AppError("asr.model_config_invalid", {"reason": "identity_files"})
    manifest: list[dict[str, JsonValue]] = []
    for path in identity_files:
        try:
            digest = _hash_file(path)
            size = path.stat().st_size
        except OSError as exc:
            raise AppError("asr.model_identity_failed", {"reason": "read"}) from exc
        manifest.append({"name": path.name, "size": size, "sha256": digest})
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"faster-whisper-local:{digest}"


def _model_identity_files(root: Path) -> tuple[Path, ...]:
    selected: list[Path] = []
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise AppError("asr.model_identity_failed", {"reason": "directory"}) from exc
    for path in children:
        if path.is_symlink():
            raise AppError("asr.model_config_invalid", {"reason": "symlink"})
        if path.is_file() and any(
            fnmatchcase(path.name, pattern) for pattern in _IDENTITY_PATTERNS
        ):
            selected.append(path)
    return tuple(selected)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    # Named model refs are downloaded into the application-owned model store.
    # An absolute ref is already a user-selected local directory and must be
    # passed through unchanged without redirecting its loading semantics.
    if config.model_cache_dir is not None and not Path(config.model_ref).is_absolute():
        options["download_root"] = str(config.model_cache_dir)
    try:
        model = constructor(config.model_ref, **options)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AppError("asr.runtime_missing") from exc
    except Exception as exc:
        raise AppError("asr.model_load_failed", {"model_id": config.model_identity}) from exc
    return cast(WhisperModelProtocol, model)


def _raise_word_timestamp_missing() -> NoReturn:
    raise AppError("asr.word_timestamp_missing")


def _raise_word_timestamp_missing_from(cause: TypeError) -> NoReturn:
    raise AppError("asr.word_timestamp_missing") from cause


def _raise_invalid_word_text() -> NoReturn:
    raise AppError("asr.output_invalid", {"reason": "word_text"})


def _raise_invalid_segment_output(reason: str) -> NoReturn:
    raise AppError("asr.output_invalid", {"reason": reason})


def _raise_invalid_segment_output_from(reason: str, cause: TypeError) -> NoReturn:
    raise AppError("asr.output_invalid", {"reason": reason}) from cause


def _raise_invalid_word_range() -> NoReturn:
    raise AppError("asr.word_timestamp_invalid", {"reason": "range"})


def _raise_invalid_segment_range() -> NoReturn:
    raise AppError("asr.word_timestamp_invalid", {"reason": "segment_range"})
