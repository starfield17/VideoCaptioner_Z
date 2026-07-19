from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from captioner.adapters.runtime.runtime_archive import (
    build_file_manifest,
    create_deterministic_archive,
    safe_extract_archive,
    sha256_file,
    validate_archive,
    verify_runtime_payload,
)
from captioner.core.domain.asr_backend import BackendCapability
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    RuntimeIdentity,
    RuntimeManifest,
    RuntimeTarget,
)


def _manifest(root: Path, *, archive_sha256: str = "a" * 64) -> RuntimeManifest:
    capability = BackendCapability(
        backend_id="faster-whisper",
        device_kind="cpu",
        supported_model_formats=("faster-whisper-ct2",),
        word_timestamps=True,
        language_detection=True,
        translation_task=True,
        additional_capabilities=("runtime_doctor",),
    )
    return RuntimeManifest(
        schema_version=1,
        runtime_identity=RuntimeIdentity("faster-whisper-cpu-test", "1.0.0"),
        worker_protocol_version="1.2",
        backend_id="faster-whisper",
        backend_version="1.2.1",
        target=RuntimeTarget("macos", "arm64", "cpu", "14.0"),
        capabilities=capability,
        supported_model_formats=("faster-whisper-ct2",),
        archive_sha256=archive_sha256,
        files=build_file_manifest(root),
    )


def _payload(root: Path) -> None:
    executable = root / "payload" / "python" / "bin" / "python3"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    executable.chmod(0o755)
    (root / "payload" / "build_info.json").write_text("{}\n", encoding="utf-8")


def _archive_with_members(path: Path, members: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(data) if kind == "file" else 0
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "payload/worker"
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = "payload/worker"
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
            if kind == "file":
                archive.addfile(info, io.BytesIO(data))
            else:
                archive.addfile(info)


def test_deterministic_archive_round_trips_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _payload(source)
    manifest = _manifest(source)
    archive = tmp_path / "runtime.tar.gz"
    create_deterministic_archive(source, archive)

    assert validate_archive(archive, manifest)
    extracted = tmp_path / "extracted"
    safe_extract_archive(archive, extracted, manifest)
    verify_runtime_payload(extracted, manifest)
    assert sha256_file(archive) == sha256_file(archive)


def test_archive_rejects_hash_and_size_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _payload(source)
    manifest = _manifest(source)
    archive = tmp_path / "runtime.tar.gz"
    create_deterministic_archive(source, archive)
    tampered = tmp_path / "tampered.tar.gz"
    tampered.write_bytes(archive.read_bytes() + b"x")

    validate_archive(tampered, manifest)
    assert sha256_file(tampered) != sha256_file(archive)

    bad_manifest = RuntimeManifest(
        schema_version=manifest.schema_version,
        runtime_identity=manifest.runtime_identity,
        worker_protocol_version=manifest.worker_protocol_version,
        backend_id=manifest.backend_id,
        backend_version=manifest.backend_version,
        target=manifest.target,
        capabilities=manifest.capabilities,
        supported_model_formats=manifest.supported_model_formats,
        archive_sha256=manifest.archive_sha256,
        files=tuple(
            type(entry)(entry.relative_path, entry.size_bytes, "b" * 64, entry.executable)
            for entry in manifest.files
        ),
    )
    with pytest.raises(AppError, match=r"runtime\.archive_hash_mismatch"):
        safe_extract_archive(archive, tmp_path / "bad", bad_manifest)


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("file", "../escape"),
        ("file", "/absolute"),
        ("file", "C:/drive-path"),
        ("symlink", "payload/worker"),
        ("hardlink", "payload/worker"),
        ("fifo", "payload/worker"),
    ),
)
def test_archive_rejects_unsafe_members(tmp_path: Path, kind: str, name: str) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _archive_with_members(archive, [(name, b"x", kind)])
    manifest = _manifest_for_one_file("payload/worker")

    with pytest.raises(AppError):
        validate_archive(archive, manifest)


def test_archive_rejects_duplicate_extra_and_missing_files(tmp_path: Path) -> None:
    manifest = _manifest_for_one_file("payload/worker")
    duplicate = tmp_path / "duplicate.tar.gz"
    _archive_with_members(
        duplicate,
        [("payload/worker", b"x", "file"), ("payload/worker", b"x", "file")],
    )
    with pytest.raises(AppError, match=r"runtime\.archive_entry_invalid"):
        validate_archive(duplicate, manifest)

    extra = tmp_path / "extra.tar.gz"
    _archive_with_members(
        extra,
        [("payload/worker", b"x", "file"), ("payload/extra", b"y", "file")],
    )
    with pytest.raises(AppError, match=r"runtime\.archive_extra_file"):
        validate_archive(extra, manifest)

    missing = tmp_path / "missing.tar.gz"
    _archive_with_members(missing, [("payload/other", b"x", "file")])
    with pytest.raises(AppError):
        validate_archive(missing, manifest)


def test_archive_enforces_total_uncompressed_size_limit(tmp_path: Path) -> None:
    data = b"0123456789"
    archive = tmp_path / "large-logical.tar.gz"
    _archive_with_members(archive, [("payload/worker", data, "file")])
    digest = hashlib.sha256(data).hexdigest()
    manifest = _manifest_for_file("payload/worker", len(data), digest)

    with pytest.raises(AppError, match=r"runtime\.archive_too_large"):
        validate_archive(archive, manifest, max_extracted_bytes=9)


def _manifest_for_one_file(relative_path: str) -> RuntimeManifest:
    digest = hashlib.sha256(b"x").hexdigest()
    capability = BackendCapability(
        backend_id="faster-whisper",
        device_kind="cpu",
        supported_model_formats=("faster-whisper-ct2",),
        word_timestamps=True,
        language_detection=True,
        translation_task=True,
        additional_capabilities=("runtime_doctor",),
    )
    from captioner.core.domain.runtime import RuntimeFileEntry

    return RuntimeManifest(
        schema_version=1,
        runtime_identity=RuntimeIdentity("faster-whisper-cpu-test", "1.0.0"),
        worker_protocol_version="1.2",
        backend_id="faster-whisper",
        backend_version="1.2.1",
        target=RuntimeTarget("macos", "arm64", "cpu", "14.0"),
        capabilities=capability,
        supported_model_formats=("faster-whisper-ct2",),
        archive_sha256="a" * 64,
        files=(RuntimeFileEntry(relative_path, 1, digest, False),),
    )


def _manifest_for_file(relative_path: str, size: int, digest: str) -> RuntimeManifest:
    manifest = _manifest_for_one_file(relative_path)
    from captioner.core.domain.runtime import RuntimeFileEntry

    return RuntimeManifest(
        schema_version=manifest.schema_version,
        runtime_identity=manifest.runtime_identity,
        worker_protocol_version=manifest.worker_protocol_version,
        backend_id=manifest.backend_id,
        backend_version=manifest.backend_version,
        target=manifest.target,
        capabilities=manifest.capabilities,
        supported_model_formats=manifest.supported_model_formats,
        archive_sha256=manifest.archive_sha256,
        files=(RuntimeFileEntry(relative_path, size, digest, False),),
    )
