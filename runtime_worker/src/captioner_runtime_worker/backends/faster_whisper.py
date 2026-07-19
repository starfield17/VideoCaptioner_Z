"""Faster Whisper CPU backend; model loading is strictly local and offline."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Event
from typing import Protocol, cast

from ..transcript import derive_transcript_id
from .base import Backend, ProgressCallback


class _WhisperModel(Protocol):
    def transcribe(self, audio_path: str, **kwargs: object) -> tuple[Iterable[object], object]: ...


class _WhisperInfo(Protocol):
    language: str | None


class FasterWhisperCPUBackend(Backend):
    def __init__(self, *, backend_version: str) -> None:
        self.backend_version = backend_version
        self._model: _WhisperModel | None = None
        self._model_key: tuple[str, str] | None = None

    def doctor_import(self) -> bool:
        __import__("faster_whisper")
        __import__("ctranslate2")
        return True

    def load_model(
        self,
        *,
        model_directory: Path,
        options: Mapping[str, object],
        model_identity: Mapping[str, object] | None = None,
    ) -> bool:
        _require_local_model(model_directory)
        compute_type = options.get("compute_type", "int8")
        if compute_type not in {"int8", "float32"}:
            raise ValueError("cpu_compute_type_invalid")
        from faster_whisper import WhisperModel

        with contextlib.redirect_stdout(__import__("sys").stderr):
            self._model = cast(
                _WhisperModel,
                WhisperModel(
                    str(model_directory), device="cpu", compute_type=cast(str, compute_type)
                ),
            )
        digest = None if model_identity is None else model_identity.get("manifest_sha256")
        self._model_key = (
            (str(model_directory.resolve()), digest) if isinstance(digest, str) else None
        )
        return True

    def transcribe(
        self,
        *,
        audio_path: Path,
        model_directory: Path,
        language: str | None,
        task: str,
        initial_prompt: str | None,
        options: Mapping[str, object],
        cancelled: Event,
        progress: ProgressCallback,
        model_identity: Mapping[str, object],
        runtime_info: Mapping[str, object],
    ) -> dict[str, object]:
        _require_local_model(model_directory)
        if not audio_path.is_absolute() or not audio_path.is_file():
            raise ValueError("audio_missing")
        compute_type = options.get("compute_type", "int8")
        if compute_type not in {"int8", "float32"}:
            raise ValueError("cpu_compute_type_invalid")
        digest = model_identity.get("manifest_sha256")
        if not isinstance(digest, str):
            raise TypeError("model_identity_invalid")
        if cancelled.is_set():
            raise CancelledError
        key = (str(model_directory.resolve()), digest)
        progress("loading_model")
        if self._model is None or self._model_key != key:
            self.load_model(
                model_directory=model_directory,
                options=options,
                model_identity=model_identity,
            )
            self._model_key = key
        if cancelled.is_set():
            raise CancelledError
        progress("preparing_audio")
        model = self._model
        if model is None:
            raise RuntimeError("whisper_model_missing")
        transcribe = model.transcribe
        kwargs: dict[str, object] = {
            "word_timestamps": True,
            "temperature": 0.0,
            "vad_filter": bool(options.get("vad_filter", True)),
            "task": task,
        }
        if language is not None:
            kwargs["language"] = language
        if initial_prompt is not None:
            kwargs["initial_prompt"] = initial_prompt
        progress("detecting_language")
        with contextlib.redirect_stdout(__import__("sys").stderr):
            raw_segments, info = transcribe(str(audio_path), **kwargs)
        progress("transcribing")
        segments, words = _map_segments(raw_segments, cancelled)
        if not words or not segments:
            raise ValueError("empty_transcript")
        detected_language = cast(_WhisperInfo, info).language or language or "und"
        return _transcript(
            language=detected_language,
            segments=segments,
            words=words,
            model_identity=model_identity,
            runtime_info=runtime_info,
        )


class CancelledError(Exception):
    pass


def _map_segments(
    raw_segments: Iterable[object], cancelled: Event
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segments: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for raw_segment in raw_segments:
        if cancelled.is_set():
            raise CancelledError
        text = getattr(raw_segment, "text", None)
        start = _milliseconds(getattr(raw_segment, "start", None))
        end = _milliseconds(getattr(raw_segment, "end", None))
        raw_words = getattr(raw_segment, "words", None)
        if not isinstance(text, str) or not text.strip() or end <= start:
            raise ValueError("invalid_segment")
        if not isinstance(raw_words, Iterable):
            raise TypeError("word_timestamps_missing")
        segment_word_ids: list[str] = []
        for raw_word in raw_words:
            word_text = getattr(raw_word, "word", None)
            word_start = _milliseconds(getattr(raw_word, "start", None))
            word_end = _milliseconds(getattr(raw_word, "end", None))
            if (
                not isinstance(word_text, str)
                or not word_text.strip()
                or word_end <= word_start
                or word_start < start
                or word_end > end
            ):
                raise ValueError("invalid_word")
            word_id = f"word-{len(words) + 1:06d}"
            words.append(
                {
                    "id": word_id,
                    "text": word_text,
                    "start_ms": word_start,
                    "end_ms": word_end,
                    "confidence": getattr(raw_word, "probability", None),
                    "speaker_id": None,
                }
            )
            segment_word_ids.append(word_id)
        if not segment_word_ids:
            raise ValueError("word_timestamps_missing")
        segments.append(
            {
                "id": f"segment-{len(segments) + 1:06d}",
                "word_ids": segment_word_ids,
                "raw_text": text.strip(),
                "start_ms": start,
                "end_ms": end,
                "confidence": None,
            }
        )
    return segments, words


def _transcript(
    *,
    language: str,
    segments: list[dict[str, object]],
    words: list[dict[str, object]],
    model_identity: Mapping[str, object],
    runtime_info: Mapping[str, object],
) -> dict[str, object]:
    backend = cast(str, runtime_info["backend_id"])
    model_id = f"{backend}:{cast(str, model_identity['manifest_sha256'])}"
    metadata = {
        "runtime_identity": runtime_info["runtime_id"],
        "runtime_version": runtime_info["runtime_version"],
        "backend_version": runtime_info["backend_version"],
        "worker_version": runtime_info["worker_version"],
        "device_kind": runtime_info["device_kind"],
        "model_identity": dict(model_identity),
        "word_timestamps": True,
    }
    transcript = {
        "id": derive_transcript_id(
            language=language,
            words=words,
            segments=segments,
            engine_id=backend,
            model_id=model_id,
            metadata=metadata,
        ),
        "language": language,
        "engine_id": backend,
        "model_id": model_id,
        "words": words,
        "segments": segments,
        "metadata": metadata,
    }
    return {"schema_version": 1, "transcript": transcript}


def _milliseconds(value: object) -> int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("timestamp_invalid")
    return math.floor(float(value) * 1000 + 0.5)


def _require_local_model(path: Path) -> None:
    if path.is_symlink() or not path.is_absolute() or not path.is_dir():
        raise ValueError("local_model_directory_required")
    for name in ("config.json", "model.bin", "tokenizer.json"):
        candidate = path / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"ct2_{name.replace('.', '_')}_missing")
    if (path / "model.bin").stat().st_size <= 0:
        raise ValueError("ct2_model_bin_empty")
    for name in ("config.json", "tokenizer.json"):
        candidate = path / name
        if candidate.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("ct2_json_too_large")
        try:
            json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"ct2_{name.replace('.', '_')}_invalid") from exc


__all__ = ["CancelledError", "FasterWhisperCPUBackend"]
