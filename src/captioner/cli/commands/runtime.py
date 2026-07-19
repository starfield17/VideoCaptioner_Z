"""Runtime management command adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from captioner.adapters.runtime.filesystem_runtime_repository import FilesystemRuntimeRepository
from captioner.adapters.runtime.host_probe import probe_host_facts
from captioner.adapters.runtime.runtime_archive import FilesystemRuntimeArchive
from captioner.adapters.runtime.runtime_doctor import FilesystemRuntimeDoctor, WorkerClientFactory
from captioner.adapters.runtime.runtime_package_source import LocalOrHTTPSRuntimePackageSource
from captioner.adapters.runtime.subprocess_worker_client import SubprocessWorkerClient
from captioner.core.application.runtime_manager import RuntimeManager
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import (
    DoctorReport,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeTarget,
)
from captioner.infrastructure.app_paths import AppPaths


@dataclass(frozen=True, slots=True)
class RuntimeCommandContext:
    paths: AppPaths


class RuntimeNamespace(Protocol):
    runtime_command: str
    reference: str
    no_activate: bool
    activation: bool
    runtime_id: str
    runtime_version: str
    backend: str
    platform: str
    architecture: str
    device: str
    manifest: Path
    root: Path
    developer_mode: bool


def build_manager(paths: AppPaths, *, activation_client: bool = False) -> RuntimeManager:
    repository = FilesystemRuntimeRepository(paths.runtimes_dir)
    host = probe_host_facts()
    worker_factory: WorkerClientFactory | None = (
        (
            lambda _runtime: SubprocessWorkerClient(
                log_dir=paths.log_dir,
                runtime_use_lock=repository.use_lock,
            )
        )
        if activation_client
        else None
    )
    doctor = FilesystemRuntimeDoctor(
        host_facts=host,
        worker_client_factory=worker_factory,
    )
    manager = RuntimeManager(
        paths=paths,
        repository=repository,
        archive=FilesystemRuntimeArchive(),
        package_source=LocalOrHTTPSRuntimePackageSource(),
        doctor=doctor,
        host_facts=host,
    )
    manager.recover()
    return manager


def execute(namespace: object, *, paths: AppPaths) -> dict[str, JsonValue]:
    args = cast(RuntimeNamespace, namespace)
    command = args.runtime_command
    manager = build_manager(
        paths,
        activation_client=command
        in {"doctor", "activate", "rollback", "install", "register-external"},
    )
    if command == "list":
        return {"runtimes": [_installation_payload(item) for item in manager.list_runtimes()]}
    if command == "install":
        installation = manager.install(
            args.reference,
            activate=not args.no_activate,
        )
        return {"runtime": _installation_payload(installation)}
    if command == "doctor":
        identity = _identity(namespace)
        report = manager.doctor(identity, activation=args.activation)
        return {"report": _report_payload(report)}
    if command == "activate":
        installation = manager.activate(_identity(namespace))
        return {"runtime": _installation_payload(installation)}
    if command == "rollback":
        target = RuntimeTarget(
            args.platform,
            args.architecture,
            args.device,
            "0.0.0",
        )
        installation = manager.rollback(args.backend, target)
        return {"runtime": _installation_payload(installation)}
    if command == "remove":
        manager.remove(_identity(namespace))
        return {"removed": True}
    if command == "register-external":
        installation = manager.register_external(
            args.manifest,
            args.root,
            developer_mode=args.developer_mode,
        )
        return {"runtime": _installation_payload(installation)}
    raise AppError("cli.unknown_command")


def _identity(namespace: object) -> RuntimeIdentity:
    args = cast(RuntimeNamespace, namespace)
    return RuntimeIdentity(args.runtime_id, args.runtime_version)


def _installation_payload(installation: RuntimeInstallation) -> dict[str, JsonValue]:
    return {
        "identity": installation.identity.to_dict(),
        "state": installation.state.value,
        "managed": installation.managed,
        "available": installation.is_available,
        "install_path": str(installation.install_path),
        "backend_id": installation.manifest.backend_id,
        "target": installation.manifest.target.to_dict(),
        "supported_model_formats": list(installation.manifest.supported_model_formats),
    }


def _report_payload(report: DoctorReport) -> dict[str, JsonValue]:
    checks = report.checks
    return {
        "ok": report.ok,
        "phase": report.phase,
        "error_code": report.error_code,
        "message_code": report.message_code,
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "error_code": check.error_code,
                "message_code": check.message_code,
                "details": dict(check.details),
            }
            for check in checks
        ],
    }


__all__ = ["RuntimeCommandContext", "build_manager", "execute"]
