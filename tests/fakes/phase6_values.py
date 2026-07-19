"""Small builders shared by Phase 6 contract tests."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path

from captioner.core.domain.asr_backend import BackendCapability
from captioner.core.domain.model import (
    ModelFileEntry,
    ModelIdentity,
    ModelInstallation,
    ModelManifest,
    ModelState,
    compute_model_manifest_sha256,
)
from captioner.core.domain.result import JsonValue
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
    runtime_id: str | None = None,
    version: str = "1.0.0",
    minimum_os_version: str = "14.0",
    additional_capabilities: tuple[str, ...] = (),
) -> RuntimeManifest:
    effective_runtime_id = runtime_id or f"{backend_id}-{device_kind}-{platform}-{architecture}"
    capability = BackendCapability(
        backend_id=backend_id,
        device_kind=device_kind,
        supported_model_formats=(model_format,),
        word_timestamps=True,
        language_detection=True,
        translation_task=True,
        additional_capabilities=additional_capabilities,
    )
    return RuntimeManifest(
        schema_version=1,
        runtime_identity=RuntimeIdentity(effective_runtime_id, version),
        worker_protocol_version="1.1",
        backend_id=backend_id,
        backend_version="1.0.0",
        target=RuntimeTarget(platform, architecture, device_kind, minimum_os_version),
        capabilities=capability,
        supported_model_formats=(model_format,),
        archive_sha256="a" * 64,
        files=(RuntimeFileEntry("worker", 1, "b" * 64, True),),
    )


def runtime_installation(
    *,
    backend_id: str = "faster-whisper",
    device_kind: str = "cpu",
    model_format: str = "faster-whisper-ct2",
    platform: str = "macos",
    architecture: str = "arm64",
    runtime_id: str | None = None,
    version: str = "1.0.0",
    state: RuntimeState = RuntimeState.AVAILABLE,
    doctor_passed: bool | None = None,
    minimum_os_version: str = "14.0",
    additional_capabilities: tuple[str, ...] = (),
) -> RuntimeInstallation:
    manifest = runtime_manifest(
        backend_id=backend_id,
        device_kind=device_kind,
        model_format=model_format,
        platform=platform,
        architecture=architecture,
        runtime_id=runtime_id,
        version=version,
        minimum_os_version=minimum_os_version,
        additional_capabilities=additional_capabilities,
    )
    return RuntimeInstallation(
        identity=manifest.runtime_identity,
        manifest=manifest,
        install_path=Path(tempfile.gettempdir())
        / "captioner-runtime-test"
        / manifest.runtime_identity.runtime_id,
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
    source_metadata: Mapping[str, JsonValue] | None = None,
    description: str = "",
    required_capabilities: tuple[str, ...] = (),
    required_device_kind: str | None = None,
    required_platform: str | None = None,
    files: tuple[ModelFileEntry, ...] | None = None,
    compatible_runtime_backends: tuple[str, ...] | None = None,
) -> ModelManifest:
    default_files = (
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
    effective_files = default_files if files is None else files
    effective_backends = (
        (backend_id,) if compatible_runtime_backends is None else compatible_runtime_backends
    )
    normalized_metadata = {} if source_metadata is None else dict(source_metadata)
    identity_without_digest = ModelIdentity(
        backend_id=backend_id,
        source_id=source_id,
        repository_id=repository_id,
        revision=revision,
        model_format=model_format,
        manifest_sha256="0" * 64,
    )
    manifest_sha256 = compute_model_manifest_sha256(
        schema_version=1,
        identity=identity_without_digest,
        display_name=display_name,
        files=effective_files,
        compatible_runtime_backends=effective_backends,
        model_format=model_format,
        source_metadata=normalized_metadata,
        description=description,
        required_capabilities=required_capabilities,
        required_device_kind=required_device_kind,
        required_platform=required_platform,
    )
    identity = ModelIdentity(
        backend_id=backend_id,
        source_id=source_id,
        repository_id=repository_id,
        revision=revision,
        model_format=model_format,
        manifest_sha256=manifest_sha256,
    )
    return ModelManifest(
        schema_version=1,
        identity=identity,
        display_name=display_name,
        files=effective_files,
        compatible_runtime_backends=effective_backends,
        model_format=model_format,
        source_metadata=normalized_metadata,
        description=description,
        required_capabilities=required_capabilities,
        required_device_kind=required_device_kind,
        required_platform=required_platform,
    )


def model_installation(
    *,
    backend_id: str = "faster-whisper",
    model_format: str = "faster-whisper-ct2",
    source_id: str = "huggingface",
    repository_id: str = "org/model",
    revision: str = "revision-a",
    display_name: str = "large-v3",
    state: ModelState = ModelState.INSTALLED,
    managed: bool | None = None,
    load_verified: bool | None = None,
    source_metadata: Mapping[str, JsonValue] | None = None,
    description: str = "",
    required_capabilities: tuple[str, ...] = (),
    required_device_kind: str | None = None,
    required_platform: str | None = None,
    validation_passed: bool | None = None,
    files: tuple[ModelFileEntry, ...] | None = None,
    compatible_runtime_backends: tuple[str, ...] | None = None,
) -> ModelInstallation:
    manifest = model_manifest(
        backend_id=backend_id,
        model_format=model_format,
        source_id=source_id,
        repository_id=repository_id,
        revision=revision,
        display_name=display_name,
        source_metadata=source_metadata,
        description=description,
        required_capabilities=required_capabilities,
        required_device_kind=required_device_kind,
        required_platform=required_platform,
        files=files,
        compatible_runtime_backends=compatible_runtime_backends,
    )
    return ModelInstallation(
        identity=manifest.identity,
        manifest=manifest,
        model_directory=Path(tempfile.gettempdir())
        / "captioner-model-test"
        / manifest.identity.repository_id.replace("/", "-"),
        state=state,
        managed=managed,
        load_verified=load_verified,
        validation_passed=validation_passed,
    )


def worker_handshake() -> WorkerHandshake:
    return WorkerHandshake(
        protocol_version="1.1",
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
