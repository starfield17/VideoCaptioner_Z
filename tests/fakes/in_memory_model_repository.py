"""Deterministic in-memory Model Repository fake."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from captioner.core.application.model_compatibility import check_model_compatibility
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelIdentity, ModelInstallation, ModelState
from captioner.core.domain.runtime import RuntimeInstallation


def _empty_models() -> dict[ModelIdentity, ModelInstallation]:
    return {}


@dataclass(slots=True)
class InMemoryModelRepository:
    initial_models: Iterable[ModelInstallation] = ()
    _models: dict[ModelIdentity, ModelInstallation] = field(
        default_factory=_empty_models, init=False
    )

    def __post_init__(self) -> None:
        for model in self.initial_models:
            if model.managed is True:
                self.register_managed_model(model)
            else:
                self.register_external_model(model)

    def list_installed_models(self) -> tuple[ModelInstallation, ...]:
        installed = tuple(
            model
            for model in self._models.values()
            if model.state
            in {ModelState.INSTALLED, ModelState.LOAD_VERIFIED, ModelState.EXTERNAL_UNMANAGED}
        )
        return tuple(
            sorted(
                installed,
                key=lambda item: (
                    item.identity.source_id,
                    item.identity.repository_id,
                    item.identity.revision,
                    item.identity.manifest_sha256,
                ),
            )
        )

    def get_by_identity(self, identity: ModelIdentity) -> ModelInstallation | None:
        return self._models.get(identity)

    def get(self, identity: ModelIdentity) -> ModelInstallation | None:
        return self.get_by_identity(identity)

    def register_managed_model(self, model: ModelInstallation) -> None:
        if model.managed is not True or model.state is ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.managed_registration_invalid")
        self._register(model)

    def register_external_model(self, model: ModelInstallation) -> None:
        if model.managed is not False or model.state is not ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.external_registration_invalid")
        self._register(model)

    def _register(self, model: ModelInstallation) -> None:
        if model.identity in self._models:
            raise AppError("model.duplicate_identity")
        self._models[model.identity] = model

    def mark_load_verified(self, identity: ModelIdentity) -> ModelInstallation:
        model = self._models.get(identity)
        if model is None:
            raise AppError("model.not_registered")
        state = (
            ModelState.EXTERNAL_UNMANAGED if model.managed is False else ModelState.LOAD_VERIFIED
        )
        updated = replace(model, state=state, load_verified=True)
        self._models[identity] = updated
        return updated

    def remove_managed_model_record(self, identity: ModelIdentity) -> None:
        model = self._models.get(identity)
        if model is None:
            raise AppError("model.not_registered")
        if model.managed is not True or not model.can_delete_files:
            raise AppError("model.external_unmanaged")
        del self._models[identity]

    def find_compatible_models(self, runtime: RuntimeInstallation) -> tuple[ModelInstallation, ...]:
        return tuple(
            model
            for model in self.list_installed_models()
            if check_model_compatibility(runtime, model).compatible
        )


__all__ = ["InMemoryModelRepository"]
