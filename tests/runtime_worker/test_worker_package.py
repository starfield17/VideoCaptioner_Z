from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest
from captioner_runtime_worker.backends.faster_whisper import (
    CancelledError,
    FasterWhisperCPUBackend,
)
from captioner_runtime_worker.backends.mlx_whisper import MLXWhisperMetalBackend
from captioner_runtime_worker.protocol import decode, encode
from captioner_runtime_worker.transcript import write_result
from captioner_runtime_worker.worker import ProtocolWriter


def test_protocol_writer_keeps_jsonl_on_original_stdout() -> None:
    import io

    stream = io.BytesIO()
    ProtocolWriter(stream).send("operation.progress", "request-1", {"phase": "loading_model"})
    value = json.loads(stream.getvalue())
    assert value["message_type"] == "operation.progress"
    assert value["sequence"] == 0


def test_local_result_is_written_atomically(tmp_path: Path) -> None:
    descriptor = write_result(
        tmp_path,
        {"schema_version": 1, "transcript": {"engine_id": "faster-whisper"}},
    )
    assert descriptor["relative_path"] == "result.json"
    assert (tmp_path / "result.json").is_file()
    assert not (tmp_path / "result.json.tmp").exists()


def test_faster_whisper_backend_requires_local_model_and_maps_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    fake_module = ModuleType("faster_whisper")

    class FakeModel:
        def transcribe(self, _audio: str, **_kwargs: object) -> tuple[object, object]:
            segment = SimpleNamespace(
                text="hello",
                start=0.0,
                end=1.0,
                words=(SimpleNamespace(word="hello", start=0.0, end=1.0, probability=0.9),),
            )
            return (segment,), SimpleNamespace(language="en")

    def build_model(*_args: object, **_kwargs: object) -> FakeModel:
        return FakeModel()

    fake_module.__dict__["WhisperModel"] = build_model
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", ModuleType("ctranslate2"))
    progress: list[str] = []
    result = FasterWhisperCPUBackend(backend_version="1.2.1").transcribe(
        audio_path=audio,
        model_directory=model_dir,
        language=None,
        task="transcribe",
        initial_prompt=None,
        options={"compute_type": "int8"},
        cancelled=Event(),
        progress=progress.append,
        model_identity={"manifest_sha256": "a" * 64},
        runtime_info={
            "backend_id": "faster-whisper",
            "runtime_id": "runtime",
            "runtime_version": "1.0.0",
            "backend_version": "1.2.1",
            "worker_version": "1.0.0",
            "device_kind": "cpu",
        },
    )
    assert result["schema_version"] == 1
    assert progress == ["loading_model", "preparing_audio", "detecting_language", "transcribing"]


def test_faster_whisper_backend_rejects_cancelled_inference(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        FasterWhisperCPUBackend(backend_version="1.2.1").transcribe(
            audio_path=audio,
            model_directory=model_dir,
            language=None,
            task="transcribe",
            initial_prompt=None,
            options={},
            cancelled=cancelled,
            progress=lambda _phase: None,
            model_identity={"manifest_sha256": "a" * 64},
            runtime_info={"backend_id": "faster-whisper"},
        )


def test_mlx_backend_requires_config_and_weight_alternative(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    backend = MLXWhisperMetalBackend(backend_version="0.4.3")
    with pytest.raises(ValueError, match="mlx_weights_missing"):
        backend.transcribe(
            audio_path=tmp_path / "audio.wav",
            model_directory=model_dir,
            language=None,
            task="transcribe",
            initial_prompt=None,
            options={},
            cancelled=Event(),
            progress=lambda _phase: None,
            model_identity={"manifest_sha256": "a" * 64},
            runtime_info={"backend_id": "mlx-whisper"},
        )


def test_codec_round_trip_matches_worker_protocol() -> None:
    line = encode("shutdown.request", "request-1", 0, {"reason": "test"})
    value = decode(line)
    assert value["protocol"] == "captioner.worker"
