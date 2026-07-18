"""Deterministic in-memory Runtime Repository fake."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeState,
    RuntimeTarget,
)


def _empty_installations() -> dict[RuntimeIdentity, RuntimeInstallation]:
    return {}


def _empty_active() -> dict[tuple[str, tuple[str, str, str, str]], RuntimeIdentity]:
    return {}


def _empty_in_use() -> set[RuntimeIdentity]:
    return set()


@dataclass(slots=True)
class InMemoryRuntimeRepository:
    initial_installations: Iterable[RuntimeInstallation] = ()
    _installations: dict[RuntimeIdentity, RuntimeInstallation] = field(
        default_factory=_empty_installations, init=False
    )
    _active: dict[tuple[str, tuple[str, str, str, str]], RuntimeIdentity] = field(
        default_factory=_empty_active, init=False
    )
    _in_use: set[RuntimeIdentity] = field(default_factory=_empty_in_use, init=False)

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

    def clear_active_runtime(self, backend_id: str, target: RuntimeTarget) -> None:
        self._active.pop((backend_id, target.key), None)

    def remove_installation_record(self, identity: RuntimeIdentity) -> None:
        if identity in self._in_use:
            raise AppError("runtime.busy")
        if identity not in self._installations:
            raise AppError("runtime.not_registered")
        if identity in self._active.values():
            raise AppError("runtime.active")
        del self._installations[identity]

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
