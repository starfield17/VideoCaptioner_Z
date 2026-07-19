"""Model installation and compatibility-query boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from captioner.core.domain.model import ModelIdentity, ModelInstallation
from captioner.core.domain.runtime import RuntimeInstallation


class ModelRepository(Protocol):
    def list_installed_models(self) -> tuple[ModelInstallation, ...]: ...

    def get_by_identity(self, identity: ModelIdentity) -> ModelInstallation | None: ...

    def get(self, identity: ModelIdentity) -> ModelInstallation | None: ...

    def register_managed_model(self, model: ModelInstallation) -> None: ...

    def register_external_model(self, model: ModelInstallation) -> None: ...

    def update_model(self, model: ModelInstallation) -> None: ...

    def mark_load_verified(self, identity: ModelIdentity) -> ModelInstallation: ...

    def remove_managed_model_record(self, identity: ModelIdentity) -> None: ...

    def remove_model(self, identity: ModelIdentity) -> None: ...

    def manager_lock(self) -> AbstractContextManager[None]: ...

    def use_lock(self, identity: ModelIdentity) -> AbstractContextManager[None]: ...

    def recover(self) -> tuple[ModelIdentity, ...]: ...

    def find_compatible_models(
        self, runtime: RuntimeInstallation
    ) -> tuple[ModelInstallation, ...]: ...


ModelRepositoryPort = ModelRepository

__all__ = ["ModelRepository", "ModelRepositoryPort"]
