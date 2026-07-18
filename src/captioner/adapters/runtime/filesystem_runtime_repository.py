"""Crash-safe filesystem Runtime installation repository."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    ActiveRuntimePointer,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeManifest,
    RuntimeState,
    RuntimeTarget,
)

_SLOT_RE = re.compile(
    r"^(?P<backend>[^|]+)\|(?P<platform>[^|]+)\|(?P<arch>[^|]+)\|(?P<device>[^|]+)$"
)


class FilesystemRuntimeRepository:
    """Persist managed and external Runtime records below ``runtimes_dir``."""

    def __init__(self, runtimes_dir: Path, *, lock_timeout: float = 30.0) -> None:
        self.root = runtimes_dir.expanduser().resolve()
        self.active_path = self.root / "active.json"
        self.manager_lock_path = self.root / ".manager.lock"
        self.external_root = self.root / "external"
        self._lock_timeout = lock_timeout

    @contextmanager
    def manager_lock(self) -> Generator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.manager_lock_path), timeout=self._lock_timeout)
        try:
            lock.acquire()
        except Timeout as exc:
            raise AppError("runtime.manager_busy") from exc
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def use_lock(self, identity: RuntimeIdentity) -> Generator[None]:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        lock = FileLock(str(installation.install_path / ".use.lock"), timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise AppError("runtime.in_use") from exc
        try:
            yield
        finally:
            lock.release()

    def list_installations(self) -> tuple[RuntimeInstallation, ...]:
        if not self.root.is_dir():
            return ()
        installations: list[RuntimeInstallation] = []
        if self.external_root.is_dir():
            for record in sorted(self.external_root.glob("*.json")):
                installations.append(self._read_installation(record))
        for runtime_dir in sorted(self.root.iterdir()):
            if not runtime_dir.is_dir() or runtime_dir.name == "external":
                continue
            for version_dir in sorted(runtime_dir.iterdir()):
                if version_dir.is_dir() and (version_dir / "installation.json").is_file():
                    installations.append(self._read_installation(version_dir / "installation.json"))
        return tuple(
            sorted(
                installations, key=lambda item: (item.identity.runtime_id, item.identity.version)
            )
        )

    def get_by_identity(self, identity: RuntimeIdentity) -> RuntimeInstallation | None:
        for installation in self.list_installations():
            if installation.identity == identity:
                return installation
        return None

    def get(self, identity: RuntimeIdentity) -> RuntimeInstallation | None:
        return self.get_by_identity(identity)

    def register_installation(self, installation: RuntimeInstallation) -> None:
        record = self._record_path(installation)
        existing = self.get_by_identity(installation.identity)
        if existing is not None and existing.manifest != installation.manifest:
            raise AppError("runtime.identity_manifest_conflict")
        record.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(record, _installation_to_dict(installation))

    def register(self, installation: RuntimeInstallation) -> None:
        self.register_installation(installation)

    def update_installation(self, installation: RuntimeInstallation) -> None:
        if self.get_by_identity(installation.identity) is None:
            raise AppError("runtime.not_registered")
        self.register_installation(installation)

    def get_active_runtime(
        self, backend_id: str, target: RuntimeTarget
    ) -> RuntimeInstallation | None:
        pointer = self.get_active_pointer(backend_id, target)
        if pointer is None or pointer.current is None:
            return None
        installation = self.get_by_identity(pointer.current)
        if installation is None:
            raise AppError("runtime.active_pointer_invalid")
        if not installation.is_available:
            raise AppError("runtime.active_pointer_invalid")
        return installation

    def get_active_pointer(
        self, backend_id: str, target: RuntimeTarget
    ) -> ActiveRuntimePointer | None:
        slots = self._read_active_slots()
        return slots.get(_slot_key(backend_id, target))

    def set_active_runtime(
        self, identity: RuntimeIdentity, backend_id: str, target: RuntimeTarget
    ) -> None:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        if not installation.is_available:
            raise AppError("runtime.not_available")
        if installation.manifest.backend_id != backend_id or installation.manifest.target != target:
            raise AppError("runtime.active_pointer_invalid")
        old = self.get_active_pointer(backend_id, target)
        pointer = ActiveRuntimePointer(
            backend_id,
            target,
            current=identity,
            previous=None if old is None else old.current,
            pending_activation=None,
        )
        self._write_active_pointer(pointer)

    def prepare_activation(self, identity: RuntimeIdentity) -> ActiveRuntimePointer:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        old = self.get_active_pointer(
            installation.manifest.backend_id, installation.manifest.target
        )
        pointer = ActiveRuntimePointer(
            installation.manifest.backend_id,
            installation.manifest.target,
            current=identity,
            previous=None if old is None else old.current,
            pending_activation=identity,
        )
        self._write_active_pointer(pointer)
        return pointer

    def complete_activation(self, identity: RuntimeIdentity) -> None:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        pointer = self.get_active_pointer(
            installation.manifest.backend_id, installation.manifest.target
        )
        if pointer is None or pointer.pending_activation != identity:
            raise AppError("runtime.active_pointer_invalid")
        self._write_active_pointer(
            ActiveRuntimePointer(
                pointer.backend_id,
                pointer.target,
                current=pointer.current,
                previous=pointer.previous,
                pending_activation=None,
            )
        )

    def restore_pending_activation(self, identity: RuntimeIdentity) -> None:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        pointer = self.get_active_pointer(
            installation.manifest.backend_id, installation.manifest.target
        )
        if pointer is None or pointer.pending_activation != identity:
            return
        if pointer.previous is None:
            slots = self._read_active_slots()
            slots.pop(_slot_key(pointer.backend_id, pointer.target), None)
            self._write_active_slots(slots)
            return
        self._write_active_pointer(
            ActiveRuntimePointer(
                pointer.backend_id,
                pointer.target,
                current=pointer.previous,
                previous=None,
                pending_activation=None,
            )
        )

    def clear_active_runtime(self, backend_id: str, target: RuntimeTarget) -> None:
        slots = self._read_active_slots()
        slots.pop(_slot_key(backend_id, target), None)
        self._write_active_slots(slots)

    def remove_installation_record(self, identity: RuntimeIdentity) -> None:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        pointer = self.get_active_pointer(
            installation.manifest.backend_id, installation.manifest.target
        )
        if pointer is not None and identity in {
            pointer.current,
            pointer.previous,
            pointer.pending_activation,
        }:
            raise AppError("runtime.active_or_previous")
        record = self._record_path(installation)
        try:
            record.unlink()
        except OSError as exc:
            raise AppError("runtime.record_remove_failed") from exc

    def remove_managed_files(self, identity: RuntimeIdentity) -> None:
        installation = self.get_by_identity(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        if not installation.can_delete_files:
            raise AppError("runtime.external_files_protected")
        pointer = self.get_active_pointer(
            installation.manifest.backend_id, installation.manifest.target
        )
        if pointer is not None and identity in {
            pointer.current,
            pointer.previous,
            pointer.pending_activation,
        }:
            raise AppError("runtime.active_or_previous")
        record = self._record_path(installation)
        lock_path = installation.install_path / ".use.lock"
        try:
            with self.use_lock(identity):
                for child in installation.install_path.iterdir():
                    if child in {record, lock_path}:
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                record.unlink()
            lock_path.unlink(missing_ok=True)
            installation.install_path.rmdir()
        except Timeout as exc:
            raise AppError("runtime.in_use") from exc
        except OSError as exc:
            raise AppError("runtime.files_remove_failed") from exc

    def recover(self) -> tuple[RuntimeIdentity, ...]:
        recovered: list[RuntimeIdentity] = []
        for pointer in tuple(self._read_active_slots().values()):
            if pointer.pending_activation is None:
                continue
            candidate = self.get_by_identity(pointer.pending_activation)
            if candidate is not None:
                self.update_installation(
                    RuntimeInstallation(
                        identity=candidate.identity,
                        manifest=candidate.manifest,
                        install_path=candidate.install_path,
                        state=(
                            RuntimeState.EXTERNAL_UNMANAGED
                            if candidate.state is RuntimeState.EXTERNAL_UNMANAGED
                            else RuntimeState.FAILED
                        ),
                        managed=candidate.managed,
                        doctor_passed=False,
                    )
                )
                recovered.append(candidate.identity)
                self.restore_pending_activation(candidate.identity)
        return tuple(recovered)

    def _record_path(self, installation: RuntimeInstallation) -> Path:
        if installation.managed:
            return installation.install_path / "installation.json"
        safe = f"{installation.identity.runtime_id}-{installation.identity.version}".replace(
            ".", "_"
        )
        return self.external_root / f"{safe}.json"

    def _read_installation(self, record: Path) -> RuntimeInstallation:
        try:
            value = json.loads(
                record.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise AppError("runtime.installation_invalid") from exc
        if not isinstance(value, Mapping):
            raise AppError("runtime.installation_invalid")
        raw = cast(Mapping[object, object], value)
        if raw.get("schema_version") != 1:
            raise AppError("runtime.installation_invalid", {"field": "schema_version"})
        identity = RuntimeIdentity.from_dict(_required(raw, "identity"))
        manifest = RuntimeManifest.from_dict(_required(raw, "manifest"))
        install_path_value = _required(raw, "install_path")
        if not isinstance(install_path_value, str):
            raise AppError("runtime.installation_invalid")
        state_value = _required(raw, "state")
        managed_value = _required(raw, "managed")
        doctor_value = _required(raw, "doctor_passed")
        if not isinstance(state_value, str) or not isinstance(managed_value, bool):
            raise AppError("runtime.installation_invalid")
        if not isinstance(doctor_value, bool):
            raise AppError("runtime.installation_invalid")
        try:
            state = RuntimeState(state_value)
        except ValueError as exc:
            raise AppError("runtime.installation_invalid") from exc
        return RuntimeInstallation(
            identity=identity,
            manifest=manifest,
            install_path=Path(install_path_value),
            state=state,
            managed=managed_value,
            doctor_passed=doctor_value,
        )

    def _read_active_slots(self) -> dict[str, ActiveRuntimePointer]:
        if not self.active_path.is_file():
            return {}
        try:
            value = json.loads(
                self.active_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise AppError("runtime.active_pointer_invalid") from exc
        if not isinstance(value, Mapping):
            raise AppError("runtime.active_pointer_invalid")
        raw_value = cast(Mapping[object, object], value)
        if raw_value.get("schema_version") != 1:
            raise AppError("runtime.active_pointer_invalid")
        slots_value = raw_value.get("slots")
        if not isinstance(slots_value, Mapping):
            raise AppError("runtime.active_pointer_invalid")
        raw_slots = cast(Mapping[object, object], slots_value)
        slots: dict[str, ActiveRuntimePointer] = {}
        for raw_key, raw_slot in raw_slots.items():
            if not isinstance(raw_key, str):
                raise AppError("runtime.active_pointer_invalid")
            backend, target = _parse_slot_key(raw_key)
            if not isinstance(raw_slot, Mapping):
                raise AppError("runtime.active_pointer_invalid")
            slot = cast(Mapping[object, object], raw_slot)
            target_value = slot.get("target")
            if target_value is not None:
                target = RuntimeTarget.from_dict(target_value)
            pointer = ActiveRuntimePointer(
                backend,
                target,
                current=_optional_identity(slot, "current"),
                previous=_optional_identity(slot, "previous"),
                pending_activation=_optional_identity(slot, "pending_activation"),
            )
            slots[raw_key] = pointer
        return slots

    def _write_active_pointer(self, pointer: ActiveRuntimePointer) -> None:
        slots = self._read_active_slots()
        slots[_slot_key(pointer.backend_id, pointer.target)] = pointer
        self._write_active_slots(slots)

    def _write_active_slots(self, slots: Mapping[str, ActiveRuntimePointer]) -> None:
        payload = {
            "schema_version": 1,
            "slots": {
                key: {
                    "target": pointer.target.to_dict(),
                    "current": None if pointer.current is None else pointer.current.to_dict(),
                    "previous": None if pointer.previous is None else pointer.previous.to_dict(),
                    "pending_activation": (
                        None
                        if pointer.pending_activation is None
                        else pointer.pending_activation.to_dict()
                    ),
                }
                for key, pointer in sorted(slots.items())
            },
        }
        self._write_json(self.active_path, payload)

    def _write_json(self, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if exc.errno == 28:
                raise AppError("runtime.disk_full") from exc
            raise AppError("runtime.persistence_failed") from exc


def _installation_to_dict(installation: RuntimeInstallation) -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": installation.identity.to_dict(),
        "manifest": installation.manifest.to_dict(),
        "install_path": str(installation.install_path),
        "state": installation.state.value,
        "managed": installation.managed,
        "doctor_passed": installation.doctor_passed,
    }


def _slot_key(backend_id: str, target: RuntimeTarget) -> str:
    return f"{backend_id}|{target.platform}|{target.architecture}|{target.device_kind}"


def _parse_slot_key(value: str) -> tuple[str, RuntimeTarget]:
    match = _SLOT_RE.fullmatch(value)
    if match is None:
        raise AppError("runtime.active_pointer_invalid")
    target = RuntimeTarget(
        match.group("platform"),
        match.group("arch"),
        match.group("device"),
        "0.0.0",
    )
    return match.group("backend"), target


def _optional_identity(value: Mapping[object, object], key: str) -> RuntimeIdentity | None:
    if key not in value:
        raise AppError("runtime.active_pointer_invalid", {"field": key})
    raw = value[key]
    return None if raw is None else RuntimeIdentity.from_dict(raw)


def _required(value: Mapping[object, object], key: str) -> object:
    if key not in value:
        raise AppError("runtime.installation_invalid", {"field": key})
    return value[key]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


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


__all__ = ["FilesystemRuntimeRepository"]
