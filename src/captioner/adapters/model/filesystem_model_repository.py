"""Crash-safe filesystem repository for managed and external models."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from captioner.core.application.model_compatibility import check_model_compatibility
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelIdentity,
    ModelInstallation,
    ModelManifest,
    ModelState,
    compute_model_identity_sha256,
)
from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.ports.model_repository import ModelRepository


class FilesystemModelRepository(ModelRepository):
    """Persist model metadata below ``<data_dir>/models`` only."""

    def __init__(
        self,
        models_dir: Path,
        *,
        staging_dir: Path | None = None,
        lock_timeout: float = 30.0,
    ) -> None:
        self.root = models_dir.expanduser().resolve()
        self.managed_root = self.root / "managed"
        self.external_root = self.root / "external"
        self.use_root = self.root / ".use"
        self.recovery_root = self.root / ".recovery"
        self.manager_lock_path = self.root / ".manager.lock"
        self.staging_root = (
            ((staging_dir or self.root.parent / "staging") / "models").expanduser().resolve()
        )
        self._lock_timeout = lock_timeout

    @contextmanager
    def manager_lock(self) -> Generator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.manager_lock_path), timeout=self._lock_timeout)
        try:
            lock.acquire()
        except Timeout as exc:
            raise AppError("model.manager_busy") from exc
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def use_lock(self, identity: ModelIdentity) -> Generator[None]:
        if self.get_by_identity(identity) is None:
            raise AppError("model.not_registered")
        self.use_root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.use_root / f"{_identity_digest(identity)}.lock"), timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise AppError("model.in_use") from exc
        try:
            yield
        finally:
            lock.release()

    def list_installed_models(self) -> tuple[ModelInstallation, ...]:
        if not self.root.is_dir():
            return ()
        models: list[ModelInstallation] = []
        if self.external_root.is_dir():
            for record in sorted(self.external_root.glob("*.json")):
                models.append(self._read_installation(record))
        if self.managed_root.is_dir():
            for identity_root in sorted(self.managed_root.iterdir()):
                if identity_root.is_symlink() or not identity_root.is_dir():
                    continue
                record = identity_root / "installation.json"
                if record.is_file():
                    models.append(self._read_installation(record))
        return tuple(
            sorted(
                models,
                key=lambda model: (
                    model.identity.source_id,
                    model.identity.repository_id,
                    model.identity.revision,
                    model.identity.manifest_sha256,
                ),
            )
        )

    def get_by_identity(self, identity: ModelIdentity) -> ModelInstallation | None:
        for model in self.list_installed_models():
            if model.identity == identity:
                return model
        return None

    def get(self, identity: ModelIdentity) -> ModelInstallation | None:
        return self.get_by_identity(identity)

    def register_managed_model(self, model: ModelInstallation) -> None:
        if model.managed is not True or model.state is ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.managed_registration_invalid")
        record = self._managed_record_path(model)
        existing = self.get_by_identity(model.identity)
        if existing is not None and existing.manifest != model.manifest:
            raise AppError("model.identity_manifest_conflict")
        record.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(record, model.to_dict())

    def register_external_model(self, model: ModelInstallation) -> None:
        if model.managed is not False or model.state is not ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.external_registration_invalid")
        existing = self.get_by_identity(model.identity)
        if existing is not None and existing.manifest != model.manifest:
            raise AppError("model.identity_manifest_conflict")
        self.external_root.mkdir(parents=True, exist_ok=True)
        self._write_json(self._external_record_path(model.identity), model.to_dict())

    def update_model(self, model: ModelInstallation) -> None:
        if self.get_by_identity(model.identity) is None:
            raise AppError("model.not_registered")
        if model.managed is True:
            self.register_managed_model(model)
        else:
            self.register_external_model(model)

    def mark_load_verified(self, identity: ModelIdentity) -> ModelInstallation:
        model = self.get_by_identity(identity)
        if model is None:
            raise AppError("model.not_registered")
        updated = ModelInstallation(
            identity=model.identity,
            manifest=model.manifest,
            model_directory=model.model_directory,
            state=(
                ModelState.EXTERNAL_UNMANAGED
                if model.managed is False
                else ModelState.LOAD_VERIFIED
            ),
            managed=model.managed,
            load_verified=True,
            validation_passed=True,
        )
        self.update_model(updated)
        return updated

    def remove_managed_model_record(self, identity: ModelIdentity) -> None:
        self.remove_model(identity)

    def remove_model(self, identity: ModelIdentity) -> None:
        model = self.get_by_identity(identity)
        if model is None:
            raise AppError("model.not_registered")
        record = (
            self._managed_record_path(model)
            if model.managed
            else self._external_record_path(identity)
        )
        try:
            with self.use_lock(identity):
                if model.managed:
                    installation_root = model.model_directory.parent
                    if installation_root.exists():
                        shutil.rmtree(installation_root)
                else:
                    record.unlink(missing_ok=True)
        except Timeout as exc:
            raise AppError("model.in_use") from exc
        except OSError as exc:
            raise AppError("model.remove_failed") from exc

    def find_compatible_models(self, runtime: RuntimeInstallation) -> tuple[ModelInstallation, ...]:
        return tuple(
            model
            for model in self.list_installed_models()
            if check_model_compatibility(runtime, model).compatible
        )

    def recover(self) -> tuple[ModelIdentity, ...]:
        recovered: list[ModelIdentity] = []
        if self.staging_root.is_dir():
            try:
                for child in tuple(self.staging_root.iterdir()):
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
            except OSError as exc:
                raise AppError("model.recovery_cleanup_failed") from exc
        if not self.managed_root.is_dir():
            return ()
        for identity_root in tuple(self.managed_root.iterdir()):
            if identity_root.is_symlink() or not identity_root.is_dir():
                self._quarantine(identity_root)
                continue
            record = identity_root / "installation.json"
            try:
                if record.is_file():
                    model = self._read_installation(record)
                    self._validate_complete_managed_directory(identity_root, model)
                    continue
                model = self._reconstruct(identity_root)
            except (AppError, OSError):
                self._quarantine(identity_root)
                continue
            self._write_json(record, model.to_dict())
            recovered.append(model.identity)
        return tuple(recovered)

    def _reconstruct(self, identity_root: Path) -> ModelInstallation:
        manifest_path = identity_root / "model-manifest.json"
        payload = identity_root / "payload"
        manifest = ModelManifest.from_dict(self._read_json(manifest_path))
        from captioner.adapters.model.filesystem_model_validator import (
            FilesystemModelValidator,
        )

        report = FilesystemModelValidator().validate(manifest, payload)
        if not report.ok:
            raise AppError("model.recovery_incomplete")
        if identity_root.name != manifest.identity.digest:
            raise AppError("model.recovery_incomplete")
        return ModelInstallation(
            identity=manifest.identity,
            manifest=manifest,
            model_directory=payload,
            state=ModelState.INSTALLED,
            managed=True,
            load_verified=False,
            validation_passed=True,
        )

    def _validate_complete_managed_directory(
        self,
        identity_root: Path,
        model: ModelInstallation,
    ) -> None:
        if model.managed is not True or identity_root.name != model.identity.digest:
            raise AppError("model.recovery_incomplete")
        expected_payload = identity_root / "payload"
        if expected_payload.is_symlink() or not expected_payload.is_dir():
            raise AppError("model.recovery_incomplete")
        if model.model_directory.resolve() != expected_payload.resolve():
            raise AppError("model.recovery_incomplete")
        manifest = ModelManifest.from_dict(self._read_json(identity_root / "model-manifest.json"))
        if manifest != model.manifest:
            raise AppError("model.recovery_incomplete")
        from captioner.adapters.model.filesystem_model_validator import (
            FilesystemModelValidator,
        )

        report = FilesystemModelValidator().validate(manifest, expected_payload)
        if not report.ok:
            raise AppError("model.recovery_incomplete")

    def _quarantine(self, identity_root: Path) -> None:
        self.recovery_root.mkdir(parents=True, exist_ok=True)
        target = self.recovery_root / f"{identity_root.name}-{os.urandom(8).hex()}"
        os.replace(identity_root, target)

    def _managed_record_path(self, model: ModelInstallation) -> Path:
        expected = self._expected_managed_payload(model.identity)
        if model.model_directory.resolve() != expected.resolve():
            raise AppError("model.installation_invalid", {"field": "model_directory"})
        return expected.parent / "installation.json"

    def _external_record_path(self, identity: ModelIdentity) -> Path:
        return self.external_root / f"{_identity_digest(identity)}.json"

    def _read_installation(self, path: Path) -> ModelInstallation:
        if path.is_symlink() or not path.is_file():
            raise AppError("model.installation_invalid")
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise AppError("model.installation_invalid") from exc
        try:
            return ModelInstallation.from_dict(raw)
        except AppError:
            raise
        except (TypeError, ValueError) as exc:
            raise AppError("model.installation_invalid") from exc

    def _read_json(self, path: Path) -> object:
        if path.is_symlink() or not path.is_file():
            raise AppError("model.manifest_invalid")
        try:
            return json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise AppError("model.manifest_invalid") from exc

    def _write_json(self, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.urandom(8).hex()}.tmp")
        try:
            data = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(data)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        except (OSError, TypeError, ValueError) as exc:
            raise AppError("model.metadata_write_failed") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _expected_managed_payload(self, identity: ModelIdentity) -> Path:
        return self.managed_root / identity.digest / "payload"


def _identity_digest(identity: ModelIdentity) -> str:
    return compute_model_identity_sha256(identity)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
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
        raise AppError("model.metadata_write_failed", {"reason": "directory_fsync"}) from exc


__all__ = ["FilesystemModelRepository"]
