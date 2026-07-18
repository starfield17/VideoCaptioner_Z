"""Filesystem Static Doctor and real Worker-backed Activation Doctor."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import cast

from captioner.adapters.runtime.runtime_archive import (
    verify_runtime_payload,
)
from captioner.adapters.runtime.subprocess_worker_client import runtime_interpreter
from captioner.core.application.runtime_selection import HostFacts
from captioner.core.application.worker_handshake_validation import validate_worker_handshake
from captioner.core.application.worker_result_validation import validate_worker_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    DoctorCheck,
    DoctorPhase,
    DoctorReport,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeManifest,
)
from captioner.core.domain.runtime_package import RuntimePackageDescriptor
from captioner.core.domain.worker_protocol import DoctorRequest, DoctorResponse, HandshakeRequest
from captioner.core.ports.worker_client import WorkerClient

WorkerClientFactory = Callable[[RuntimeInstallation], WorkerClient]
RuntimeUseLockFactory = Callable[[RuntimeIdentity], AbstractContextManager[None]]


class FilesystemRuntimeDoctor:
    """Perform static checks locally and activation checks through a Worker."""

    def __init__(
        self,
        *,
        host_facts: HostFacts,
        worker_client_factory: WorkerClientFactory | None = None,
        runtime_use_lock: RuntimeUseLockFactory | None = None,
    ) -> None:
        self._host = host_facts
        self._worker_client_factory = worker_client_factory
        self._runtime_use_lock = runtime_use_lock

    def static_doctor(self, runtime: RuntimeInstallation) -> DoctorReport:
        checks: list[DoctorCheck] = []
        checks.append(_check("installation_root", lambda: _check_root(runtime)))
        checks.append(_check("target_platform", lambda: _check_platform(runtime, self._host)))
        checks.append(_check("minimum_os", lambda: _check_minimum_os(runtime, self._host)))
        checks.append(_check("interpreter", lambda: _check_interpreter(runtime)))
        checks.append(
            _check(
                "manifest_files",
                lambda: verify_runtime_payload(
                    runtime.install_path,
                    runtime.manifest,
                    allowed_extra_paths=(
                        "runtime-package.json",
                        "runtime-manifest.json",
                        "installation.json",
                        ".use.lock",
                    ),
                ),
            )
        )
        checks.append(_check("package_metadata", lambda: _check_package_metadata(runtime)))
        checks.append(_check("build_info", lambda: _check_build_info(runtime)))
        checks.append(
            _check(
                "runtime_doctor_capability",
                lambda: _require_runtime_doctor_capability(runtime),
            )
        )
        return _report(DoctorPhase.STATIC.value, checks)

    def activation_doctor(self, runtime: RuntimeInstallation, workspace: Path) -> DoctorReport:
        return asyncio.run(self._activation_doctor(runtime, workspace))

    async def _activation_doctor(
        self, runtime: RuntimeInstallation, workspace: Path
    ) -> DoctorReport:
        lock = (
            nullcontext()
            if self._runtime_use_lock is None
            else self._runtime_use_lock(runtime.identity)
        )
        with lock:
            return await self._activation_doctor_unlocked(runtime, workspace)

    async def _activation_doctor_unlocked(
        self, runtime: RuntimeInstallation, workspace: Path
    ) -> DoctorReport:
        static = self.static_doctor(runtime)
        if not static.ok:
            return DoctorReport(
                ok=False,
                phase=DoctorPhase.ACTIVATION.value,
                checks=static.checks,
                error_code="runtime.static_doctor_failed",
                message_code="runtime.static_doctor_failed",
            )
        if self._worker_client_factory is None:
            return _report(
                DoctorPhase.ACTIVATION.value,
                (
                    DoctorCheck(
                        "worker_client",
                        False,
                        "runtime.worker_client_unavailable",
                        "runtime.worker_client_unavailable",
                    ),
                ),
            )
        workspace.mkdir(parents=True, exist_ok=True)
        client = self._worker_client_factory(runtime)
        handshake_request = HandshakeRequest(
            required_capabilities=tuple(
                sorted(runtime.manifest.capabilities.advertised_capabilities | {"runtime_doctor"})
            ),
            required_backend_id=runtime.manifest.backend_id,
            required_result_schema_versions=(1,),
        )
        checks: list[DoctorCheck] = []
        try:
            handshake = await client.start(runtime, workspace, handshake_request)
            validation = validate_worker_handshake(runtime.manifest, handshake_request, handshake)
            checks.append(
                DoctorCheck(
                    "protocol_handshake",
                    validation.ok,
                    None if validation.ok else validation.error_code,
                    None if validation.ok else validation.message_code,
                    {"reasons": list(validation.reasons)},
                )
            )
            if not validation.ok:
                return _report(DoctorPhase.ACTIVATION.value, checks)
            doctor_request = DoctorRequest(uuid.uuid4().hex, "doctor-probe.json")
            response = await client.doctor(doctor_request)
            probe_ok = _validate_probe(response, workspace)
            checks.extend(
                (
                    DoctorCheck(
                        "backend_import",
                        response.backend_import_ok,
                        "runtime.backend_import_failed" if not response.backend_import_ok else None,
                    ),
                    DoctorCheck(
                        "device_visibility",
                        response.device_kind == runtime.manifest.target.device_kind,
                        "runtime.device_probe_failed"
                        if response.device_kind != runtime.manifest.target.device_kind
                        else None,
                    ),
                    DoctorCheck(
                        "probe_nonce",
                        response.nonce == doctor_request.nonce,
                        "runtime.workspace_probe_failed"
                        if response.nonce != doctor_request.nonce
                        else None,
                    ),
                    DoctorCheck(
                        "workspace_round_trip",
                        probe_ok,
                        "runtime.workspace_probe_failed" if not probe_ok else None,
                    ),
                )
            )
            return _report(DoctorPhase.ACTIVATION.value, checks)
        except AppError as exc:
            checks.append(DoctorCheck("activation", False, exc.code, exc.code))
            return _report(DoctorPhase.ACTIVATION.value, checks)
        finally:
            await client.shutdown()


def _check(name: str, action: Callable[[], None]) -> DoctorCheck:
    try:
        action()
    except AppError as exc:
        return DoctorCheck(name, False, exc.code, exc.code)
    except OSError:
        return DoctorCheck(name, False, "runtime.filesystem_invalid", "runtime.filesystem_invalid")
    return DoctorCheck(name, True)


def _report(phase: str, checks: tuple[DoctorCheck, ...] | list[DoctorCheck]) -> DoctorReport:
    normalized = tuple(checks)
    failed = next((check for check in normalized if not check.ok), None)
    return DoctorReport(
        ok=failed is None,
        phase=phase,
        checks=normalized,
        error_code=None if failed is None else failed.error_code,
        message_code=None if failed is None else failed.message_code,
    )


def _check_root(runtime: RuntimeInstallation) -> None:
    if not runtime.install_path.is_dir():
        raise AppError("runtime.installation_root_missing")
    if not (runtime.install_path / "payload").is_dir():
        raise AppError("runtime.payload_missing")


def _check_platform(runtime: RuntimeInstallation, host: HostFacts) -> None:
    target = runtime.manifest.target
    if target.platform != host.platform or target.architecture != host.architecture:
        raise AppError("runtime.target_unsupported")


def _check_minimum_os(runtime: RuntimeInstallation, host: HostFacts) -> None:
    actual = tuple(int(item) for item in host.os_version.split("."))
    required = tuple(int(item) for item in runtime.manifest.target.minimum_os_version.split("."))
    width = max(len(actual), len(required))
    if (actual + (0,) * (width - len(actual))) < (required + (0,) * (width - len(required))):
        raise AppError("runtime.minimum_os_unsupported")


def _check_interpreter(runtime: RuntimeInstallation) -> None:
    interpreter = runtime_interpreter(runtime)
    if not interpreter.is_file() or (
        runtime.manifest.target.platform != "windows" and not _is_executable(interpreter)
    ):
        raise AppError("runtime.interpreter_invalid")


def _check_build_info(runtime: RuntimeInstallation) -> None:
    path = runtime.install_path / "payload" / "build_info.json"
    if not path.is_file():
        raise AppError("runtime.build_info_missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError("runtime.build_info_invalid") from exc
    if not isinstance(raw, dict):
        raise AppError("runtime.build_info_invalid")
    values = cast(dict[str, object], raw)
    expected = {
        "schema_version": 1,
        "runtime_id": runtime.identity.runtime_id,
        "runtime_version": runtime.identity.version,
        "backend_id": runtime.manifest.backend_id,
        "backend_version": runtime.manifest.backend_version,
        "platform": runtime.manifest.target.platform,
        "architecture": runtime.manifest.target.architecture,
        "device_kind": runtime.manifest.target.device_kind,
        "protocol_version": runtime.manifest.worker_protocol_version,
        "supported_model_formats": list(runtime.manifest.supported_model_formats),
        "capabilities": sorted(runtime.manifest.capabilities.advertised_capabilities),
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise AppError("runtime.build_info_mismatch")
    if not isinstance(values.get("worker_version"), str):
        raise AppError("runtime.build_info_mismatch")


def _check_package_metadata(runtime: RuntimeInstallation) -> None:
    manifest_path = runtime.install_path / "runtime-manifest.json"
    package_path = runtime.install_path / "runtime-package.json"
    if runtime.managed and (not manifest_path.is_file() or not package_path.is_file()):
        raise AppError("runtime.package_metadata_missing")
    if manifest_path.is_file():
        manifest = RuntimeManifest.from_dict(_read_json(manifest_path))
        if manifest != runtime.manifest:
            raise AppError("runtime.manifest_metadata_mismatch")
    if package_path.is_file():
        descriptor = RuntimePackageDescriptor.from_dict(_read_json(package_path))
        if descriptor.runtime_manifest != runtime.manifest:
            raise AppError("runtime.package_metadata_mismatch")


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise AppError("runtime.package_metadata_invalid") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def _require_runtime_doctor_capability(runtime: RuntimeInstallation) -> None:
    if "runtime_doctor" not in runtime.manifest.capabilities.advertised_capabilities:
        raise AppError("runtime.doctor_capability_missing")


def _validate_probe(response: DoctorResponse, workspace: Path) -> bool:
    descriptor = response.probe_result
    try:
        validate_worker_result(
            descriptor,
            workspace,
            supported_schema_versions={"captioner.runtime-doctor": {1}},
        )
    except AppError:
        return False
    return True


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


__all__ = ["FilesystemRuntimeDoctor", "RuntimeUseLockFactory", "WorkerClientFactory"]
