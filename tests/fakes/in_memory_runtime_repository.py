"""Deterministic in-memory Runtime Repository fake."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    ActiveRuntimePointer,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeState,
    RuntimeTarget,
)


def _empty_installations() -> dict[RuntimeIdentity, RuntimeInstallation]:
    return {}


def _empty_active() -> dict[tuple[str, tuple[str, str, str]], RuntimeIdentity]:
    return {}


def _empty_in_use() -> set[RuntimeIdentity]:
    return set()


def _empty_pointers() -> dict[tuple[str, tuple[str, str, str]], ActiveRuntimePointer]:
    return {}


@dataclass(slots=True)
class InMemoryRuntimeRepository:
    initial_installations: Iterable[RuntimeInstallation] = ()
    _installations: dict[RuntimeIdentity, RuntimeInstallation] = field(
        default_factory=_empty_installations, init=False
    )
    _active: dict[tuple[str, tuple[str, str, str]], RuntimeIdentity] = field(
        default_factory=_empty_active, init=False
    )
    _in_use: set[RuntimeIdentity] = field(default_factory=_empty_in_use, init=False)
    _pointers: dict[tuple[str, tuple[str, str, str]], ActiveRuntimePointer] = field(
        default_factory=_empty_pointers, init=False
    )

    def __post_init__(self) -> None:
        for installation in self.initial_installations:
            self.register(installation)

    def list_installations(self) -> tuple[RuntimeInstallation, ...]:
        return tuple(
            self._installations[key]
            for key in sorted(self._installations, key=lambda item: (item.runtime_id, item.version))
        )

    def get_by_identity(self, identity: RuntimeIdentity) -> RuntimeInstallation | None:
        return self._installations.get(identity)

    def get(self, identity: RuntimeIdentity) -> RuntimeInstallation | None:
        return self.get_by_identity(identity)

    def register_installation(self, installation: RuntimeInstallation) -> None:
        self.register(installation)

    def update_installation(self, installation: RuntimeInstallation) -> None:
        if installation.identity not in self._installations:
            raise AppError("runtime.not_registered")
        self._installations[installation.identity] = installation

    @contextmanager
    def manager_lock(self) -> Generator[None]:
        yield

    def register(self, installation: RuntimeInstallation) -> None:
        if installation.identity in self._installations:
            raise AppError("runtime.duplicate_installation")
        self._installations[installation.identity] = installation

    def get_active_runtime(
        self, backend_id: str, target: RuntimeTarget
    ) -> RuntimeInstallation | None:
        identity = self._active.get((backend_id, target.key))
        return None if identity is None else self._installations.get(identity)

    def set_active_runtime(
        self, identity: RuntimeIdentity, backend_id: str, target: RuntimeTarget
    ) -> None:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        if not installation.is_available:
            raise AppError("runtime.not_available")
        if installation.manifest.backend_id != backend_id:
            raise AppError("runtime.active_pointer_invalid")
        if installation.manifest.target != target:
            raise AppError("runtime.active_pointer_invalid")
        self._active[(backend_id, target.key)] = identity
        self._pointers[(backend_id, target.key)] = ActiveRuntimePointer(
            backend_id, target, current=identity
        )

    def clear_active_runtime(self, backend_id: str, target: RuntimeTarget) -> None:
        self._active.pop((backend_id, target.key), None)
        self._pointers.pop((backend_id, target.key), None)

    def get_active_pointer(
        self, backend_id: str, target: RuntimeTarget
    ) -> ActiveRuntimePointer | None:
        return self._pointers.get((backend_id, target.key))

    def prepare_activation(self, identity: RuntimeIdentity) -> ActiveRuntimePointer:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        key = (installation.manifest.backend_id, installation.manifest.target.key)
        old = self._pointers.get(key)
        pointer = ActiveRuntimePointer(
            installation.manifest.backend_id,
            installation.manifest.target,
            current=identity,
            previous=None if old is None else old.current,
            pending_activation=identity,
        )
        self._pointers[key] = pointer
        return pointer

    def complete_activation(self, identity: RuntimeIdentity) -> None:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        key = (installation.manifest.backend_id, installation.manifest.target.key)
        pointer = self._pointers.get(key)
        if pointer is None or pointer.pending_activation != identity:
            raise AppError("runtime.active_pointer_invalid")
        self._pointers[key] = ActiveRuntimePointer(
            pointer.backend_id,
            pointer.target,
            current=pointer.current,
            previous=pointer.previous,
        )
        self._active[key] = identity

    def restore_pending_activation(self, identity: RuntimeIdentity) -> None:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        key = (installation.manifest.backend_id, installation.manifest.target.key)
        pointer = self._pointers.get(key)
        if pointer is None or pointer.pending_activation != identity:
            return
        if pointer.previous is None:
            self._pointers.pop(key, None)
            self._active.pop(key, None)
        else:
            self._pointers[key] = ActiveRuntimePointer(
                pointer.backend_id, pointer.target, current=pointer.previous
            )
            self._active[key] = pointer.previous

    def remove_installation_record(self, identity: RuntimeIdentity) -> None:
        if identity in self._in_use:
            raise AppError("runtime.busy")
        if identity not in self._installations:
            raise AppError("runtime.not_registered")
        if any(
            identity in {pointer.current, pointer.previous, pointer.pending_activation}
            for pointer in self._pointers.values()
        ):
            raise AppError("runtime.active")
        del self._installations[identity]

    def remove_managed_files(self, identity: RuntimeIdentity) -> None:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        if not installation.can_delete_files:
            raise AppError("runtime.external_files_protected")
        self.remove_installation_record(identity)

    def recover(self) -> tuple[RuntimeIdentity, ...]:
        recovered: list[RuntimeIdentity] = []
        for pointer in tuple(self._pointers.values()):
            if pointer.pending_activation is None:
                continue
            candidate = self._installations.get(pointer.pending_activation)
            if candidate is not None:
                self._installations[candidate.identity] = replace(
                    candidate,
                    state=(
                        RuntimeState.EXTERNAL_UNMANAGED
                        if candidate.state is RuntimeState.EXTERNAL_UNMANAGED
                        else RuntimeState.FAILED
                    ),
                    doctor_passed=False,
                )
                recovered.append(candidate.identity)
                self.restore_pending_activation(candidate.identity)
        return tuple(recovered)

    def mark_state(
        self,
        identity: RuntimeIdentity,
        state: RuntimeState,
        *,
        doctor_passed: bool | None = None,
    ) -> RuntimeInstallation:
        installation = self._installations.get(identity)
        if installation is None:
            raise AppError("runtime.not_registered")
        updated = replace(installation, state=state, doctor_passed=doctor_passed)
        self._installations[identity] = updated
        return updated

    def set_in_use(self, identity: RuntimeIdentity, in_use: bool = True) -> None:
        if identity not in self._installations:
            raise AppError("runtime.not_registered")
        if in_use:
            self._in_use.add(identity)
        else:
            self._in_use.discard(identity)


__all__ = ["InMemoryRuntimeRepository"]
