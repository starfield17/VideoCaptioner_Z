"""Runtime installation and active-pointer boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from captioner.core.domain.runtime import (
    ActiveRuntimePointer,
    RuntimeIdentity,
    RuntimeInstallation,
    RuntimeTarget,
)


class RuntimeRepository(Protocol):
    """Store contract for Runtime records; implementations own persistence."""

    def list_installations(self) -> tuple[RuntimeInstallation, ...]: ...

    def get_by_identity(self, identity: RuntimeIdentity) -> RuntimeInstallation | None: ...

    def get(self, identity: RuntimeIdentity) -> RuntimeInstallation | None: ...

    def register_installation(self, installation: RuntimeInstallation) -> None: ...

    def register(self, installation: RuntimeInstallation) -> None: ...

    def update_installation(self, installation: RuntimeInstallation) -> None: ...

    def manager_lock(self) -> AbstractContextManager[None]: ...

    def get_active_runtime(
        self, backend_id: str, target: RuntimeTarget
    ) -> RuntimeInstallation | None: ...

    def set_active_runtime(
        self, identity: RuntimeIdentity, backend_id: str, target: RuntimeTarget
    ) -> None: ...

    def clear_active_runtime(self, backend_id: str, target: RuntimeTarget) -> None: ...

    def get_active_pointer(
        self, backend_id: str, target: RuntimeTarget
    ) -> ActiveRuntimePointer | None: ...

    def prepare_activation(self, identity: RuntimeIdentity) -> ActiveRuntimePointer: ...

    def complete_activation(self, identity: RuntimeIdentity) -> None: ...

    def restore_pending_activation(self, identity: RuntimeIdentity) -> None: ...

    def remove_installation_record(self, identity: RuntimeIdentity) -> None: ...

    def remove_managed_files(self, identity: RuntimeIdentity) -> None: ...

    def recover(self) -> tuple[RuntimeIdentity, ...]: ...


RuntimeRepositoryPort = RuntimeRepository

__all__ = ["RuntimeRepository", "RuntimeRepositoryPort"]
