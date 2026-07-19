"""MLX Whisper Metal backend with local-only model loading."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Event
from typing import cast

from ..transcript import derive_transcript_id
from .base import Backend, ProgressCallback
from .faster_whisper import CancelledError


class MLXWhisperMetalBackend(Backend):
    def __init__(self, *, backend_version: str) -> None:
        self.backend_version = backend_version

    def doctor_import(self) -> bool:
        __import__("mlx")
        __import__("mlx_whisper")
        core = __import__("mlx.core", fromlist=("array", "eval"))
        value = core.array([1])
        core.eval(value)
        return True

    def load_model(
        self,
        *,
        model_directory: Path,
        options: Mapping[str, object],
        model_identity: Mapping[str, object] | None = None,
    ) -> bool:
        del options
        del model_identity
        _validate_model_directory(model_directory)
        __import__("mlx_whisper")
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
        del options
        _validate_model_directory(model_directory)
        if not audio_path.is_absolute() or not audio_path.is_file():
            raise ValueError("audio_missing")
        if cancelled.is_set():
            raise CancelledError
        progress("loading_model")
        import mlx_whisper

        kwargs: dict[str, object] = {
            "path_or_hf_repo": str(model_directory),
            "verbose": None,
            "word_timestamps": True,
            "task": task,
        }
        if language is not None:
            kwargs["language"] = language
        if initial_prompt is not None:
            kwargs["initial_prompt"] = initial_prompt
        progress("preparing_audio")
        with contextlib.redirect_stdout(__import__("sys").stderr):
            raw = mlx_whisper.transcribe(str(audio_path), **kwargs)
        if cancelled.is_set():
            raise CancelledError
        progress("transcribing")
        if not isinstance(raw, Mapping):
            raise TypeError("invalid_backend_result")
        segments, words = _map_segments(raw.get("segments"), cancelled)
        if not segments or not words:
            raise ValueError("empty_transcript")
        language_value = raw.get("language")
        detected_language = language_value if isinstance(language_value, str) else language or "und"
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
        return {
            "schema_version": 1,
            "transcript": {
                "id": derive_transcript_id(
                    language=detected_language,
                    words=words,
                    segments=segments,
                    engine_id=backend,
                    model_id=model_id,
                    metadata=metadata,
                ),
                "language": detected_language,
                "engine_id": backend,
                "model_id": model_id,
                "words": words,
                "segments": segments,
                "metadata": metadata,
            },
        }


def _validate_model_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_absolute() or not path.is_dir():
        raise ValueError("local_model_directory_required")
    config = path / "config.json"
    if config.is_symlink() or not config.is_file():
        raise ValueError("mlx_config_missing")
    weight_paths = [
        path / name for name in ("model.safetensors", "weights.safetensors", "weights.npz")
    ]
    if not any(not item.is_symlink() and item.is_file() for item in weight_paths):
        raise ValueError("mlx_weights_missing")
    tokenizer = path / "tokenizer.json"
    vocab = path / "vocab.json"
    merges = path / "merges.txt"
    if not (
        (not tokenizer.is_symlink() and tokenizer.is_file())
        or (
            not vocab.is_symlink()
            and vocab.is_file()
            and not merges.is_symlink()
            and merges.is_file()
        )
    ):
        raise ValueError("mlx_tokenizer_missing")
    try:
        if config.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("mlx_config_too_large")
        value = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("mlx_config_invalid") from exc
    if not isinstance(value, dict):
        raise TypeError("mlx_config_invalid")


def _map_segments(
    raw_segments: object, cancelled: Event
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(raw_segments, Iterable) or isinstance(raw_segments, (str, bytes, dict)):
        raise TypeError("word_timestamps_missing")
    segments: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for raw_segment in raw_segments:
        if cancelled.is_set():
            raise CancelledError
        if not isinstance(raw_segment, Mapping):
            raise TypeError("invalid_segment")
        text = raw_segment.get("text")
        start = _milliseconds(raw_segment.get("start"))
        end = _milliseconds(raw_segment.get("end"))
        raw_words = raw_segment.get("words")
        if not isinstance(text, str) or not text.strip() or end <= start:
            raise ValueError("invalid_segment")
        if not isinstance(raw_words, Iterable) or isinstance(raw_words, (str, bytes, dict)):
            raise TypeError("word_timestamps_missing")
        segment_word_ids: list[str] = []
        for raw_word in raw_words:
            if not isinstance(raw_word, Mapping):
                raise TypeError("invalid_word")
            word_text = raw_word.get("word")
            word_start = _milliseconds(raw_word.get("start"))
            word_end = _milliseconds(raw_word.get("end"))
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
                    "confidence": raw_word.get("probability"),
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


def _milliseconds(value: object) -> int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("timestamp_invalid")
    return math.floor(float(value) * 1000 + 0.5)


__all__ = ["MLXWhisperMetalBackend"]
