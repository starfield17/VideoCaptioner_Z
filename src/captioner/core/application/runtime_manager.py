"""Application service for transactional Runtime installation and activation."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from captioner.core.application.runtime_selection import HostFacts
from captioner.core.domain.errors import AppError
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.domain.runtime import (
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeManifest,
    RuntimeState,
    RuntimeTarget,
)
from captioner.core.domain.runtime_package import RuntimePackageDescriptor
from captioner.core.ports.runtime_archive import RuntimeArchive
from captioner.core.ports.runtime_doctor import RuntimeDoctor
from captioner.core.ports.runtime_package_source import RuntimePackageSource
from captioner.core.ports.runtime_repository import RuntimeRepository

ProgressCallback = Callable[[OperationProgress], None]


class RuntimeManagerPaths(Protocol):
    @property
    def config_dir(self) -> Path: ...

    @property
    def data_dir(self) -> Path: ...

    @property
    def runtimes_dir(self) -> Path: ...

    @property
    def downloads_dir(self) -> Path: ...

    @property
    def staging_dir(self) -> Path: ...

    @property
    def temp_dir(self) -> Path: ...

    @property
    def log_dir(self) -> Path: ...


class RuntimeManager:
    """Coordinate package source, repository, and static/activation Doctor."""

    def __init__(
        self,
        *,
        paths: RuntimeManagerPaths,
        repository: RuntimeRepository,
        archive: RuntimeArchive,
        package_source: RuntimePackageSource,
        doctor: RuntimeDoctor,
        host_facts: HostFacts,
    ) -> None:
        self._paths = paths
        self._repository = repository
        self._archive = archive
        self._package_source = package_source
        self._doctor = doctor
        self._host = host_facts

    def list_runtimes(self) -> tuple[RuntimeInstallation, ...]:
        return self._repository.list_installations()

    def install(
        self,
        package_source: str | Path,
        *,
        activate: bool = True,
        progress: ProgressCallback | None = None,
    ) -> RuntimeInstallation:
        transaction_id = uuid.uuid4().hex
        staging_root = self._paths.staging_dir / "runtimes" / transaction_id
        archive_part = self._paths.downloads_dir / "runtimes" / f"{transaction_id}.part"
        self._emit(progress, "resolving_release")
        with self._repository.manager_lock():
            try:
                descriptor = self._package_source.resolve(package_source, archive_part)
                manifest = descriptor.runtime_manifest
                self._validate_target(manifest.target)
                self._emit(progress, "downloading")
                self._verify_archive(archive_part, descriptor)
                self._emit(progress, "verifying_archive")
                staging_root.mkdir(parents=True, exist_ok=False)
                self._emit(progress, "extracting")
                self._archive.extract(archive_part, staging_root, manifest)
                self._write_metadata(staging_root, descriptor)
                staged = RuntimeInstallation(
                    identity=manifest.runtime_identity,
                    manifest=manifest,
                    install_path=staging_root,
                    state=RuntimeState.STAGED,
                    managed=True,
                    doctor_passed=False,
                )
                self._emit(progress, "verifying_installation")
                static_report = self._doctor.static_doctor(staged)
                if not static_report.ok:
                    raise AppError(
                        static_report.error_code or "runtime.static_doctor_failed",
                        {"phase": "static"},
                    )
                final_root = (
                    self._paths.runtimes_dir
                    / manifest.runtime_identity.runtime_id
                    / manifest.runtime_identity.version
                )
                existing = self._repository.get_by_identity(manifest.runtime_identity)
                if existing is not None:
                    if existing.manifest != manifest:
                        raise AppError("runtime.identity_manifest_conflict")
                    if activate:
                        self._activate(existing.identity, progress)
                    return existing
                if final_root.exists():
                    raise AppError("runtime.version_directory_conflict")
                installation = RuntimeInstallation(
                    identity=manifest.runtime_identity,
                    manifest=manifest,
                    install_path=final_root,
                    state=RuntimeState.INSTALLED,
                    managed=True,
                    doctor_passed=False,
                )
                # Keep the complete installation record inside the staging
                # root so a final-directory move cannot expose a payload
                # without its registration metadata.
                self._write_installation_metadata(staging_root, installation)
                _fsync_directory(staging_root)
                final_root.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(staging_root, final_root)
                    _fsync_directory(final_root.parent)
                except OSError as exc:
                    if exc.errno == 28:
                        raise AppError("runtime.disk_full") from exc
                    raise AppError("runtime.install_move_failed") from exc
                self._repository.register_installation(installation)
                if activate:
                    self._activate(installation.identity, progress)
                    refreshed = self._repository.get_by_identity(installation.identity)
                    if refreshed is None:
                        raise AppError("runtime.registration_missing")
                    installation = refreshed
                self._emit(progress, "completed")
            except OSError as exc:
                if exc.errno == 28:
                    raise AppError("runtime.disk_full") from exc
                raise AppError("runtime.install_failed") from exc
            else:
                return installation
            finally:
                _remove_tree(staging_root)
                archive_part.unlink(missing_ok=True)
                self._emit(progress, "cleaning_staging")

    def activate(
        self, identity: RuntimeIdentity, *, progress: ProgressCallback | None = None
    ) -> RuntimeInstallation:
        with self._repository.manager_lock():
            self._activate(identity, progress)
            result = self._repository.get_by_identity(identity)
            if result is None:
                raise AppError("runtime.registration_missing")
            return result

    def rollback(
        self,
        backend_id: str,
        target: RuntimeTarget,
        *,
        progress: ProgressCallback | None = None,
    ) -> RuntimeInstallation:
        with self._repository.manager_lock():
            pointer = self._repository.get_active_pointer(backend_id, target)
            if pointer is None or pointer.previous is None:
                raise AppError("runtime.rollback_unavailable")
            self._emit(progress, "rolling_back")
            self._activate(pointer.previous, progress)
            result = self._repository.get_by_identity(pointer.previous)
            if result is None:
                raise AppError("runtime.registration_missing")
            self._emit(progress, "completed")
            return result

    def doctor(self, identity: RuntimeIdentity, *, activation: bool = False):
        runtime = self._repository.get_by_identity(identity)
        if runtime is None:
            raise AppError("runtime.not_registered")
        report = self._doctor.static_doctor(runtime)
        if activation and report.ok:
            report = self._doctor.activation_doctor(
                runtime, self._activation_workspace(runtime.identity, "doctor")
            )
        return report

    def register_external(
        self,
        manifest_path: Path,
        runtime_root: Path,
        *,
        developer_mode: bool,
        progress: ProgressCallback | None = None,
    ) -> RuntimeInstallation:
        if not developer_mode:
            raise AppError("runtime.developer_mode_required")
        manifest = _load_manifest(manifest_path)
        self._validate_target(manifest.target)
        installation = RuntimeInstallation(
            identity=manifest.runtime_identity,
            manifest=manifest,
            install_path=runtime_root.expanduser().resolve(),
            state=RuntimeState.EXTERNAL_UNMANAGED,
            managed=False,
            doctor_passed=False,
        )
        with self._repository.manager_lock():
            report = self._doctor.static_doctor(installation)
            if not report.ok:
                raise AppError(report.error_code or "runtime.static_doctor_failed")
            self._repository.register_installation(installation)
            self._emit(progress, "running_doctor")
            try:
                report = self._doctor.activation_doctor(
                    installation,
                    self._activation_workspace(installation.identity, "external"),
                )
            except AppError:
                self._repository.remove_installation_record(installation.identity)
                raise
            if not report.ok:
                self._repository.remove_installation_record(installation.identity)
                raise AppError(report.error_code or "runtime.activation_doctor_failed")
            available = RuntimeInstallation(
                identity=installation.identity,
                manifest=installation.manifest,
                install_path=installation.install_path,
                state=RuntimeState.EXTERNAL_UNMANAGED,
                managed=False,
                doctor_passed=True,
            )
            self._repository.update_installation(available)
            self._emit(progress, "completed")
            return available

    def remove(self, identity: RuntimeIdentity) -> None:
        with self._repository.manager_lock():
            installation = self._repository.get_by_identity(identity)
            if installation is None:
                raise AppError("runtime.not_registered")
            if installation.managed:
                self._repository.remove_managed_files(identity)
            else:
                self._repository.remove_installation_record(identity)

    def recover(self) -> tuple[RuntimeIdentity, ...]:
        with self._repository.manager_lock():
            recovered = self._repository.recover()
            self._clean_interrupted_transactions()
            return recovered

    def _activate(self, identity: RuntimeIdentity, progress: ProgressCallback | None) -> None:
        runtime = self._repository.get_by_identity(identity)
        if runtime is None:
            raise AppError("runtime.not_registered")
        if runtime.state not in {
            RuntimeState.INSTALLED,
            RuntimeState.AVAILABLE,
            RuntimeState.EXTERNAL_UNMANAGED,
        }:
            raise AppError("runtime.not_installable")
        pointer = self._repository.get_active_pointer(
            runtime.manifest.backend_id, runtime.manifest.target
        )
        if (
            pointer is not None
            and pointer.current == identity
            and pointer.pending_activation is None
            and runtime.is_available
        ):
            # Re-activating a healthy current Runtime is deliberately a pure
            # idempotent read.  Health checks belong to the explicit Doctor
            # command and must not rewrite current/previous pointers.
            return
        self._repository.prepare_activation(identity)
        self._emit(progress, "activating")
        try:
            report = self._doctor.activation_doctor(
                runtime, self._activation_workspace(runtime.identity, "activation")
            )
        except AppError:
            self._repository.restore_pending_activation(identity)
            self._mark_activation_failed(runtime)
            raise
        if not report.ok:
            self._repository.restore_pending_activation(identity)
            self._mark_activation_failed(runtime)
            raise AppError(report.error_code or "runtime.activation_doctor_failed")
        self._repository.update_installation(
            RuntimeInstallation(
                identity=runtime.identity,
                manifest=runtime.manifest,
                install_path=runtime.install_path,
                state=(
                    RuntimeState.EXTERNAL_UNMANAGED
                    if runtime.state is RuntimeState.EXTERNAL_UNMANAGED
                    else RuntimeState.AVAILABLE
                ),
                managed=runtime.managed,
                doctor_passed=True,
            )
        )
        self._repository.complete_activation(identity)

    def _mark_activation_failed(self, runtime: RuntimeInstallation) -> None:
        self._repository.update_installation(
            RuntimeInstallation(
                identity=runtime.identity,
                manifest=runtime.manifest,
                install_path=runtime.install_path,
                state=(
                    RuntimeState.EXTERNAL_UNMANAGED
                    if runtime.state is RuntimeState.EXTERNAL_UNMANAGED
                    else RuntimeState.FAILED
                ),
                managed=runtime.managed,
                doctor_passed=False,
            )
        )

    def _validate_target(self, target: RuntimeTarget) -> None:
        host = self._host
        if target.platform != host.platform or target.architecture != host.architecture:
            raise AppError("runtime.target_unsupported")
        if not _version_at_least(host.os_version, target.minimum_os_version):
            raise AppError("runtime.minimum_os_unsupported")
        if target.device_kind == "metal" and not host.native_architecture:
            raise AppError("runtime.target_unsupported")

    def _activation_workspace(self, identity: RuntimeIdentity, purpose: str) -> Path:
        return self._paths.temp_dir / "runtimes" / identity.runtime_id / identity.version / purpose

    def _clean_interrupted_transactions(self) -> None:
        staging_root = self._paths.staging_dir / "runtimes"
        if staging_root.is_dir():
            for transaction in staging_root.iterdir():
                try:
                    if transaction.is_symlink() or transaction.is_file():
                        transaction.unlink()
                    elif transaction.is_dir():
                        shutil.rmtree(transaction)
                except OSError as exc:
                    raise AppError(
                        "runtime.persistence_failed", {"reason": "recovery_cleanup"}
                    ) from exc

        downloads_root = self._paths.downloads_dir / "runtimes"
        if downloads_root.is_dir():
            for partial in downloads_root.glob("*.part"):
                try:
                    partial.unlink()
                except OSError as exc:
                    raise AppError(
                        "runtime.persistence_failed", {"reason": "recovery_cleanup"}
                    ) from exc

    def _verify_archive(self, archive: Path, descriptor: RuntimePackageDescriptor) -> None:
        try:
            if archive.stat().st_size != descriptor.archive_size_bytes:
                raise AppError("runtime.archive_size_mismatch")
            if self._archive.sha256_file(archive) != descriptor.archive_sha256:
                raise AppError("runtime.archive_hash_mismatch")
        except OSError as exc:
            raise AppError("runtime.archive_read_failed") from exc

    @staticmethod
    def _write_metadata(root: Path, descriptor: RuntimePackageDescriptor) -> None:
        _write_json(root / "runtime-package.json", descriptor.to_dict())
        _write_json(root / "runtime-manifest.json", descriptor.runtime_manifest.to_dict())

    @staticmethod
    def _write_installation_metadata(root: Path, installation: RuntimeInstallation) -> None:
        _write_json(
            root / "installation.json",
            {
                "schema_version": 1,
                "identity": installation.identity.to_dict(),
                "manifest": installation.manifest.to_dict(),
                "install_path": str(installation.install_path),
                "state": installation.state.value,
                "managed": installation.managed,
                "doctor_passed": installation.doctor_passed,
            },
        )

    @staticmethod
    def _emit(progress: ProgressCallback | None, phase: str) -> None:
        if progress is not None:
            progress(OperationProgress("runtime", phase, f"runtime.{phase}", {}))


def _load_manifest(path: Path) -> RuntimeManifest:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise AppError("runtime.manifest_invalid") from exc
    return RuntimeManifest.from_dict(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        if exc.errno == 28:
            raise AppError("runtime.disk_full") from exc
        raise AppError("runtime.persistence_failed") from exc


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise AppError("runtime.persistence_failed", {"reason": "directory_fsync"}) from exc


def _version_at_least(actual: str, required: str) -> bool:
    actual_parts = tuple(int(item) for item in actual.split("."))
    required_parts = tuple(int(item) for item in required.split("."))
    width = max(len(actual_parts), len(required_parts))
    return (actual_parts + (0,) * (width - len(actual_parts))) >= (
        required_parts + (0,) * (width - len(required_parts))
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


__all__ = ["ProgressCallback", "RuntimeManager", "RuntimeManagerPaths"]
