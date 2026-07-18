from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.fakes.in_memory_runtime_repository import InMemoryRuntimeRepository
from tests.fakes.phase6_values import runtime_manifest

from captioner.adapters.runtime.runtime_archive import (
    FilesystemRuntimeArchive,
    build_file_manifest,
    create_deterministic_archive,
    sha256_file,
)
from captioner.core.application.runtime_manager import RuntimeManager
from captioner.core.application.runtime_selection import HostFacts
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import DoctorCheck, DoctorPhase, DoctorReport, RuntimeState
from captioner.core.domain.runtime_package import RuntimePackageDescriptor
from captioner.core.ports.runtime_package_source import RuntimePackageSource


@dataclass(frozen=True, slots=True)
class _Paths:
    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def runtimes_dir(self) -> Path:
        return self.data_dir / "runtimes"

    @property
    def downloads_dir(self) -> Path:
        return self.root / "downloads"

    @property
    def staging_dir(self) -> Path:
        return self.root / "staging"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def log_dir(self) -> Path:
        return self.root / "log"


@dataclass(slots=True)
class _LocalSource(RuntimePackageSource):
    archive: Path
    descriptor: RuntimePackageDescriptor

    def resolve(self, reference: str | Path, destination: Path) -> RuntimePackageDescriptor:
        del reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.archive, destination)
        return self.descriptor


class _Archive(FilesystemRuntimeArchive):
    pass


def _report(ok: bool) -> DoctorReport:
    check = DoctorCheck(
        "test",
        ok,
        None if ok else "runtime.test_doctor_failed",
        None if ok else "runtime.test_doctor_failed",
    )
    return DoctorReport(
        ok=ok,
        phase=DoctorPhase.STATIC.value,
        checks=(check,),
        error_code=None if ok else check.error_code,
        message_code=None if ok else check.message_code,
    )


def _package(tmp_path: Path) -> tuple[_LocalSource, object]:
    root = tmp_path / "package-root"
    worker = root / "payload" / "worker"
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b"worker")
    manifest = runtime_manifest()
    manifest = type(manifest)(
        schema_version=manifest.schema_version,
        runtime_identity=manifest.runtime_identity,
        worker_protocol_version=manifest.worker_protocol_version,
        backend_id=manifest.backend_id,
        backend_version=manifest.backend_version,
        target=manifest.target,
        capabilities=manifest.capabilities,
        supported_model_formats=manifest.supported_model_formats,
        archive_sha256=manifest.archive_sha256,
        files=build_file_manifest(root),
    )
    archive = tmp_path / "runtime.tar.gz"
    create_deterministic_archive(root, archive)
    manifest = type(manifest)(
        schema_version=manifest.schema_version,
        runtime_identity=manifest.runtime_identity,
        worker_protocol_version=manifest.worker_protocol_version,
        backend_id=manifest.backend_id,
        backend_version=manifest.backend_version,
        target=manifest.target,
        capabilities=manifest.capabilities,
        supported_model_formats=manifest.supported_model_formats,
        archive_sha256=sha256_file(archive),
        files=manifest.files,
    )
    descriptor = RuntimePackageDescriptor(
        package_schema_version=1,
        archive_filename=archive.name,
        archive_size_bytes=archive.stat().st_size,
        runtime_manifest=manifest,
    )
    return _LocalSource(archive, descriptor), manifest.runtime_identity


def _manager(
    tmp_path: Path, *, activation_ok: bool = True
) -> tuple[RuntimeManager, _LocalSource, InMemoryRuntimeRepository]:
    source, identity = _package(tmp_path)
    del identity
    paths = _Paths(tmp_path / "app")
    repository = InMemoryRuntimeRepository()
    from tests.fakes.fake_runtime_doctor import FakeRuntimeDoctor

    doctor = FakeRuntimeDoctor(_report(True), _report(activation_ok))
    manager = RuntimeManager(
        paths=paths,
        repository=repository,
        archive=_Archive(),
        package_source=source,
        doctor=doctor,
        host_facts=HostFacts("macos", "arm64", "14.0", True),
    )
    return manager, source, repository


def test_install_is_verified_and_repeat_install_is_idempotent(tmp_path: Path) -> None:
    manager, source, repository = _manager(tmp_path)

    first = manager.install("local", activate=False)
    second = manager.install("local", activate=False)

    assert first.identity == second.identity
    assert first.state is RuntimeState.INSTALLED
    assert repository.list_installations() == (first,)
    assert source.archive.is_file()


def test_recover_cleans_orphaned_transactions_and_partial_archives(tmp_path: Path) -> None:
    manager, _source, _repository = _manager(tmp_path)
    paths = _Paths(tmp_path / "app")
    orphan = paths.staging_dir / "runtimes" / "interrupted"
    orphan.mkdir(parents=True)
    (orphan / "payload").write_bytes(b"partial")
    partial = paths.downloads_dir / "runtimes" / "interrupted.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    assert manager.recover() == ()
    assert not orphan.exists()
    assert not partial.exists()


def test_activation_failure_keeps_no_new_active_pointer(tmp_path: Path) -> None:
    manager, _source, repository = _manager(tmp_path, activation_ok=False)

    with pytest.raises(AppError, match=r"runtime\.test_doctor_failed"):
        manager.install("local", activate=True)

    installed = repository.list_installations()
    assert len(installed) == 1
    assert installed[0].state is RuntimeState.FAILED
    assert (
        repository.get_active_pointer(
            installed[0].manifest.backend_id, installed[0].manifest.target
        )
        is None
    )
