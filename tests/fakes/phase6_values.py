"""Small builders shared by Phase 6 contract tests."""

from __future__ import annotations

from pathlib import Path

from captioner.core.domain.asr_backend import BackendCapability
from captioner.core.domain.model import (
    ModelFileEntry,
    ModelIdentity,
    ModelInstallation,
    ModelManifest,
    ModelState,
)
from captioner.core.domain.runtime import (
    RuntimeFileEntry,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeManifest,
    RuntimeState,
    RuntimeTarget,
)
from captioner.core.domain.worker_protocol import (
    ResultDescriptor,
    TranscribeRequest,
    WorkerHandshake,
)


def runtime_manifest(
    *,
    backend_id: str = "faster-whisper",
    device_kind: str = "cpu",
    model_format: str = "faster-whisper-ct2",
    platform: str = "macos",
    architecture: str = "arm64",
    runtime_id: str = "faster-whisper-cpu-macos-arm64",
    version: str = "1.0.0",
) -> RuntimeManifest:
    capability = BackendCapability(
        backend_id=backend_id,
        device_kind=device_kind,
        supported_model_formats=(model_format,),
        word_timestamps=True,
        language_detection=True,
        translation_task=True,
    )
    return RuntimeManifest(
        schema_version=1,
        runtime_identity=RuntimeIdentity(runtime_id, version),
        worker_protocol_version="1.0",
        backend_id=backend_id,
        backend_version="1.0.0",
        target=RuntimeTarget(platform, architecture, device_kind, "14.0"),
        capabilities=capability,
        supported_model_formats=(model_format,),
        archive_sha256="a" * 64,
        files=(RuntimeFileEntry("worker", 1, "b" * 64, True),),
    )


def runtime_installation(
    *,
    state: RuntimeState = RuntimeState.AVAILABLE,
    doctor_passed: bool | None = None,
    **kwargs: str,
) -> RuntimeInstallation:
    manifest = runtime_manifest(**kwargs)
    return RuntimeInstallation(
        identity=manifest.runtime_identity,
        manifest=manifest,
        install_path=Path("/captioner/runtime") / manifest.runtime_identity.runtime_id,
        state=state,
        doctor_passed=doctor_passed,
    )


def model_manifest(
    *,
    backend_id: str = "faster-whisper",
    model_format: str = "faster-whisper-ct2",
    source_id: str = "huggingface",
    repository_id: str = "org/model",
    revision: str = "revision-a",
    display_name: str = "large-v3",
) -> ModelManifest:
    files = (
        (
            ModelFileEntry("config.json", 1, "b" * 64),
            ModelFileEntry("model.bin", 1, "c" * 64),
        )
        if model_format != "mlx-whisper"
        else (
            ModelFileEntry("config.json", 1, "b" * 64),
            ModelFileEntry("model.safetensors", 1, "c" * 64),
        )
    )
    identity = ModelIdentity(
        backend_id=backend_id,
        source_id=source_id,
        repository_id=repository_id,
        revision=revision,
        model_format=model_format,
        manifest_sha256="d" * 64,
    )
    return ModelManifest(
        schema_version=1,
        identity=identity,
        display_name=display_name,
        files=files,
        compatible_runtime_backends=(backend_id,),
        model_format=model_format,
    )


def model_installation(
    *,
    state: ModelState = ModelState.INSTALLED,
    managed: bool | None = None,
    load_verified: bool = False,
    **kwargs: str,
) -> ModelInstallation:
    manifest = model_manifest(**kwargs)
    return ModelInstallation(
        identity=manifest.identity,
        manifest=manifest,
        model_directory=Path("/captioner/models")
        / manifest.identity.repository_id.replace("/", "-"),
        state=state,
        managed=managed,
        load_verified=load_verified,
    )


def worker_handshake() -> WorkerHandshake:
    return WorkerHandshake(
        protocol_version="1.0",
        runtime_id="faster-whisper-cpu-macos-arm64",
        runtime_version="1.0.0",
        backend_id="faster-whisper",
        backend_version="1.0.0",
        worker_version="1.0.0",
        platform="macos",
        architecture="arm64",
        capabilities=("word_timestamps", "language_detection", "translation_task"),
        supported_devices=("cpu",),
        supported_model_formats=("faster-whisper-ct2",),
        supported_result_schema_versions=(1,),
    )


def transcribe_request() -> TranscribeRequest:
    runtime = runtime_installation()
    model = model_installation()
    return TranscribeRequest(
        normalized_audio_path=Path("/captioner/work/audio.wav"),
        attempt_workspace=Path("/captioner/work/attempt"),
        model_directory=model.model_directory,
        backend_id=runtime.manifest.backend_id,
        runtime_identity=runtime.identity,
        model_identity=model.identity,
        result_schema_version=1,
        language="en",
        task="transcribe",
        word_timestamps=True,
        request_id="request-1",
        job_id="job-1",
        stage_attempt_id="attempt-1",
    )


def result_descriptor(*, relative_path: str = "result.json") -> ResultDescriptor:
    return ResultDescriptor(relative_path, 3, "a" * 64, "transcript", 1)


__all__ = [
    "model_installation",
    "model_manifest",
    "result_descriptor",
    "runtime_installation",
    "runtime_manifest",
    "transcribe_request",
    "worker_handshake",
]
