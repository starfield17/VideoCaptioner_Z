"""Application service for safe Model Source and local model operations."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from captioner.core.application.model_compatibility import ensure_model_compatibility
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    LocalModelInspection,
    ModelIdentity,
    ModelInstallation,
    ModelManifest,
    ModelSourceCandidate,
    ModelSourceReference,
    ModelState,
    ModelValidationReport,
    compute_model_manifest_sha256,
)
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.ports.local_model_inspector import LocalModelInspector
from captioner.core.ports.model_repository import ModelRepository
from captioner.core.ports.model_source import ModelMaterializer, ModelSource
from captioner.core.ports.model_validator import ModelValidator

ProgressCallback = Callable[[OperationProgress], None]
LoadVerifier = Callable[[ModelInstallation, object | None, str], None]

_REMOTE_ALLOWED = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "preprocessor_config.json",
    "vocabulary.json",
    "vocabulary.txt",
    "vocabulary.*",
    "model.safetensors",
    "weights.safetensors",
    "weights.npz",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "README.md",
    "LICENSE*",
)
_FORBIDDEN_SUFFIXES = {
    ".py",
    ".sh",
    ".bat",
    ".cmd",
    ".ps1",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
}


class ModelManager:
    """Own model identity construction and installation commit boundaries."""

    def __init__(
        self,
        *,
        repository: ModelRepository,
        inspector: LocalModelInspector,
        validator: ModelValidator,
        models_dir: Path,
        staging_dir: Path,
        downloads_dir: Path | None = None,
        sources: Mapping[str, ModelSource] | None = None,
        materializers: Mapping[str, ModelMaterializer] | None = None,
        load_verifier: LoadVerifier | None = None,
        max_files: int = 4096,
        max_total_bytes: int = 32 * 1024 * 1024 * 1024,
    ) -> None:
        self.repository = repository
        self.inspector = inspector
        self.validator = validator
        self.models_dir = models_dir.expanduser().resolve()
        self.staging_dir = staging_dir.expanduser().resolve() / "models"
        self.downloads_dir = (
            downloads_dir or staging_dir.expanduser().resolve().parent / "downloads"
        ).expanduser().resolve() / "models"
        self.sources = {} if sources is None else dict(sources)
        self.materializers = {} if materializers is None else dict(materializers)
        self.load_verifier = load_verifier
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes

    def list_models(self) -> tuple[ModelInstallation, ...]:
        return self.repository.list_installed_models()

    def search_huggingface(
        self, query: str, backend_id: str, limit: int = 20
    ) -> tuple[ModelSourceCandidate, ...]:
        source = self._source("huggingface")
        return source.search(query, backend_id, limit)

    def install_remote(
        self,
        source_id: str,
        repository_id: str,
        revision: str | None,
        backend_id: str,
        model_format: str,
        *,
        display_name: str | None = None,
        progress: ProgressCallback | None = None,
        verify_load: bool = False,
        runtime: object | None = None,
        device: str = "auto",
    ) -> ModelInstallation:
        source = self._source(source_id)
        materializer = self._materializer(source_id)
        with self.repository.manager_lock():
            reference = source.resolve_exact(
                repository_id,
                revision,
                backend_id,
                model_format_hint=model_format,
            )
            transaction = self._transaction_root()
            download_transaction = self.downloads_dir / transaction.name
            result: ModelInstallation | None = None
            try:
                download_transaction.mkdir(parents=True, exist_ok=False)
                source_root = download_transaction / "source"
                installation_root = transaction / "installation"
                _emit(progress, "resolving_source")
                materializer.materialize(reference, source_root, progress=progress)
                _emit(progress, "copying")
                payload = installation_root / "payload"
                self._copy_clean_tree(source_root, payload, remote=True)
                _emit(progress, "inspecting")
                inspection = self.inspector.inspect(
                    payload,
                    backend_hint=backend_id,
                    model_format_hint=model_format,
                )
                _require_inspection(inspection)
                _emit(progress, "hashing")
                _emit(progress, "validating")
                model = self._build_managed_model(
                    inspection,
                    reference,
                    payload,
                    display_name=display_name,
                )
                _emit(progress, "installing")
                result = self._commit_staged_model(
                    model,
                    installation_root,
                    progress=progress,
                    verify_load=verify_load,
                    runtime=runtime,
                    device=device,
                )
            except OSError as exc:
                if exc.errno == 28:
                    raise AppError("model.disk_full") from exc
                raise
            finally:
                _cleanup_transaction(transaction, progress)
                _cleanup_transaction(download_transaction, progress)
            _emit(progress, "completed")
            return result

    def import_local(
        self,
        directory: Path,
        backend_hint: str | None = None,
        format_hint: str | None = None,
        display_name: str | None = None,
        *,
        progress: ProgressCallback | None = None,
        verify_load: bool = False,
        runtime: object | None = None,
        device: str = "auto",
    ) -> ModelInstallation:
        original = directory.expanduser().absolute()
        with self.repository.manager_lock():
            inspection = self.inspector.inspect(original, backend_hint, format_hint)
            _require_inspection(inspection)
            transaction = self._transaction_root()
            result: ModelInstallation | None = None
            try:
                payload = transaction / "installation" / "payload"
                _emit(progress, "copying")
                self._copy_clean_tree(original, payload, remote=False)
                copied = self.inspector.inspect(payload, backend_hint, format_hint)
                _require_inspection(copied)
                _emit(progress, "hashing")
                reference = ModelSourceReference(
                    source_id="local-import",
                    repository_id=f"local-import/{uuid.uuid4().hex}",
                    revision="1",
                    backend_id=copied.detected_backend_id or backend_hint or "",
                    model_format_hint=copied.detected_model_format or format_hint,
                )
                model = self._build_managed_model(
                    copied,
                    reference,
                    payload,
                    display_name=display_name,
                )
                _emit(progress, "validating")
                _emit(progress, "installing")
                result = self._commit_staged_model(
                    model,
                    transaction / "installation",
                    progress=progress,
                    verify_load=verify_load,
                    runtime=runtime,
                    device=device,
                )
            finally:
                _cleanup_transaction(transaction, progress)
            _emit(progress, "completed")
            return result

    def register_external(
        self,
        directory: Path,
        backend_hint: str | None = None,
        format_hint: str | None = None,
        display_name: str | None = None,
        *,
        developer_mode: bool = False,
        progress: ProgressCallback | None = None,
        verify_load: bool = False,
        runtime: object | None = None,
        device: str = "auto",
    ) -> ModelInstallation:
        if not developer_mode:
            raise AppError("model.developer_mode_required")
        original = directory.expanduser().absolute()
        with self.repository.manager_lock():
            _emit(progress, "inspecting")
            inspection = self.inspector.inspect(original, backend_hint, format_hint)
            _require_inspection(inspection)
            reference = ModelSourceReference(
                source_id="external-path",
                repository_id=f"external/{uuid.uuid4().hex}",
                revision="1",
                backend_id=inspection.detected_backend_id or backend_hint or "",
                model_format_hint=inspection.detected_model_format or format_hint,
            )
            model = self._build_external_model(
                inspection,
                reference,
                original,
                display_name=display_name,
            )
            _emit(progress, "validating")
            self._require_static_validation(model)
            self.repository.register_external_model(model)
            if verify_load:
                model = self._verify_load_unlocked(
                    model.identity,
                    runtime=runtime,
                    device=device,
                    progress=progress,
                )
            _emit(progress, "completed")
            return self.repository.get_by_identity(model.identity) or model

    def validate(self, identity: ModelIdentity) -> ModelValidationReport:
        with self.repository.manager_lock():
            return self._validate_unlocked(identity)

    def _validate_unlocked(self, identity: ModelIdentity) -> ModelValidationReport:
        model = self._model(identity)
        report = self.validator.validate(model.manifest, model.model_directory)
        if model.state is ModelState.EXTERNAL_UNMANAGED:
            if not report.ok and model.validation_passed:
                self.repository.update_model(_with_validation(model, False))
            elif report.ok and not model.validation_passed:
                self.repository.update_model(_with_validation(model, True))
            return report
        if report.ok and not model.validation_passed:
            self.repository.update_model(_with_validation(model, True))
        elif not report.ok and model.state is not ModelState.FAILED:
            self.repository.update_model(
                ModelInstallation(
                    identity=model.identity,
                    manifest=model.manifest,
                    model_directory=model.model_directory,
                    state=ModelState.FAILED,
                    managed=True,
                    load_verified=False,
                    validation_passed=False,
                )
            )
        return report

    def verify_load(
        self,
        identity: ModelIdentity,
        *,
        runtime: object | None = None,
        device: str = "auto",
        progress: ProgressCallback | None = None,
    ) -> ModelInstallation:
        with self.repository.manager_lock():
            model = self._verify_load_unlocked(
                identity,
                runtime=runtime,
                device=device,
                progress=progress,
            )
        _emit(progress, "completed")
        return model

    def _verify_load_unlocked(
        self,
        identity: ModelIdentity,
        *,
        runtime: object | None,
        device: str,
        progress: ProgressCallback | None,
    ) -> ModelInstallation:
        model = self._model(identity)
        if isinstance(runtime, RuntimeInstallation):
            if not runtime.is_available:
                raise AppError("runtime.not_available")
            ensure_model_compatibility(runtime, model)
        report = self._validate_unlocked(identity)
        if not report.ok:
            raise AppError(report.error_code or "model.validation_failed")
        if self.load_verifier is None:
            raise AppError("model.load_verifier_unavailable")
        with self.repository.use_lock(identity):
            _emit(progress, "load_verifying")
            try:
                self.load_verifier(model, runtime, device)
            except AppError:
                raise
            except Exception as exc:
                raise AppError("model.load_failed") from exc
            updated = self.repository.mark_load_verified(identity)
        return updated

    def remove(self, identity: ModelIdentity) -> None:
        with self.repository.manager_lock():
            self.repository.remove_model(identity)

    def recover(self) -> tuple[ModelIdentity, ...]:
        with self.repository.manager_lock():
            recovered = self.repository.recover()
            self._clean_download_transactions()
            return recovered

    def _source(self, source_id: str) -> ModelSource:
        source = self.sources.get(source_id)
        if source is None:
            raise AppError("model.source_unavailable", {"source_id": source_id})
        return source

    def _materializer(self, source_id: str) -> ModelMaterializer:
        materializer = self.materializers.get(source_id)
        if materializer is None:
            raise AppError("model.source_materialize_unavailable", {"source_id": source_id})
        return materializer

    def _model(self, identity: ModelIdentity) -> ModelInstallation:
        model = self.repository.get_by_identity(identity)
        if model is None:
            raise AppError("model.not_installed")
        return model

    def _transaction_root(self) -> Path:
        root = self.staging_dir / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        return root

    def _clean_download_transactions(self) -> None:
        if not self.downloads_dir.exists():
            return
        try:
            for child in tuple(self.downloads_dir.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
        except OSError as exc:
            raise AppError("model.recovery_cleanup_failed") from exc

    def _build_managed_model(
        self,
        inspection: LocalModelInspection,
        reference: ModelSourceReference,
        payload: Path,
        *,
        display_name: str | None,
    ) -> ModelInstallation:
        if display_name is None and reference.source_id in {"huggingface", "modelscope"}:
            display_name = reference.repository_id.rsplit("/", 1)[-1]
        return self._build_model(
            inspection,
            reference,
            payload,
            display_name=display_name,
            managed=True,
        )

    def _build_external_model(
        self,
        inspection: LocalModelInspection,
        reference: ModelSourceReference,
        directory: Path,
        *,
        display_name: str | None,
    ) -> ModelInstallation:
        return self._build_model(
            inspection,
            reference,
            directory,
            display_name=display_name,
            managed=False,
        )

    def _build_model(
        self,
        inspection: LocalModelInspection,
        reference: ModelSourceReference,
        directory: Path,
        *,
        display_name: str | None,
        managed: bool,
    ) -> ModelInstallation:
        backend_id = inspection.detected_backend_id or reference.backend_id
        model_format = inspection.detected_model_format or reference.model_format_hint
        if not backend_id or not model_format:
            raise AppError("model.format_unknown")
        effective_name = (display_name or inspection.display_name_suggestion or "model").strip()
        if not effective_name:
            raise AppError("model.display_name_invalid")
        source_metadata: dict[str, JsonValue] = {"source_id": reference.source_id}
        identity_base = ModelIdentity(
            backend_id=backend_id,
            source_id=reference.source_id,
            repository_id=reference.repository_id,
            revision=reference.revision,
            model_format=model_format,
            manifest_sha256="0" * 64,
        )
        digest = compute_model_manifest_sha256(
            schema_version=1,
            identity=identity_base,
            display_name=effective_name,
            files=inspection.file_inventory,
            compatible_runtime_backends=(backend_id,),
            model_format=model_format,
            source_metadata=source_metadata,
            description="",
            required_capabilities=(),
            required_device_kind="metal" if model_format == "mlx-whisper" else None,
            required_platform="macos" if model_format == "mlx-whisper" else None,
        )
        identity = ModelIdentity(
            backend_id=backend_id,
            source_id=reference.source_id,
            repository_id=reference.repository_id,
            revision=reference.revision,
            model_format=model_format,
            manifest_sha256=digest,
        )
        manifest = ModelManifest(
            schema_version=1,
            identity=identity,
            display_name=effective_name,
            files=inspection.file_inventory,
            compatible_runtime_backends=(backend_id,),
            model_format=model_format,
            source_metadata=source_metadata,
            required_device_kind="metal" if model_format == "mlx-whisper" else None,
            required_platform="macos" if model_format == "mlx-whisper" else None,
        )
        return ModelInstallation(
            identity=identity,
            manifest=manifest,
            model_directory=(directory.expanduser().resolve() if managed else directory),
            state=(ModelState.INSTALLED if managed else ModelState.EXTERNAL_UNMANAGED),
            managed=managed,
            load_verified=False,
            validation_passed=True,
        )

    def _commit_staged_model(
        self,
        model: ModelInstallation,
        installation_root: Path,
        *,
        progress: ProgressCallback | None,
        verify_load: bool,
        runtime: object | None,
        device: str,
    ) -> ModelInstallation:
        if not model.managed:
            raise AppError("model.installation_invalid")
        self._require_static_validation(model)
        final_root = self.models_dir / "managed" / model.identity.digest
        existing = self.repository.get_by_identity(model.identity)
        if existing is not None:
            if existing.manifest != model.manifest:
                raise AppError("model.identity_manifest_conflict")
            result = existing
        else:
            if final_root.exists():
                self.repository.recover()
                existing = self.repository.get_by_identity(model.identity)
                if existing is None and final_root.exists():
                    raise AppError("model.version_directory_conflict")
            if existing is not None:
                if existing.manifest != model.manifest:
                    raise AppError("model.identity_manifest_conflict")
                result = existing
            else:
                installation_root.mkdir(parents=True, exist_ok=True)
                committed_model = ModelInstallation(
                    identity=model.identity,
                    manifest=model.manifest,
                    model_directory=final_root / "payload",
                    state=model.state,
                    managed=True,
                    load_verified=model.load_verified,
                    validation_passed=model.validation_passed,
                )
                _write_json(
                    installation_root / "model-manifest.json",
                    committed_model.manifest.to_dict(),
                )
                _write_json(installation_root / "installation.json", committed_model.to_dict())
                _fsync_tree_directories(installation_root)
                final_root.parent.mkdir(parents=True, exist_ok=True)
                os.replace(installation_root, final_root)
                _fsync_directory(final_root.parent)
                self.repository.register_managed_model(committed_model)
                result = self.repository.get_by_identity(model.identity) or committed_model
        if verify_load:
            result = self._verify_load_unlocked(
                result.identity,
                runtime=runtime,
                device=device,
                progress=progress,
            )
        return result

    def _require_static_validation(self, model: ModelInstallation) -> None:
        report = self.validator.validate(model.manifest, model.model_directory)
        if not report.ok:
            raise AppError(report.error_code or "model.validation_failed")

    def _copy_clean_tree(
        self,
        source: Path,
        destination: Path,
        *,
        remote: bool,
    ) -> None:
        source = source.expanduser()
        if source.is_symlink():
            raise AppError("model.symlink_rejected")
        if not source.is_dir():
            raise AppError("model.directory_missing")
        source = source.resolve()
        destination.mkdir(parents=True, exist_ok=False)
        count = 0
        total = 0
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise AppError("model.symlink_rejected")
            if path.is_dir():
                continue
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise AppError("model.special_file_rejected")
            relative = path.relative_to(source).as_posix()
            if any(part.startswith(".") for part in PurePosixPath(relative).parts):
                continue
            name = PurePosixPath(relative).name
            if name.casefold().endswith(tuple(_FORBIDDEN_SUFFIXES)):
                continue
            if remote and not any(
                fnmatch.fnmatchcase(relative, pattern) for pattern in _REMOTE_ALLOWED
            ):
                continue
            if path.stat().st_nlink > 1:
                raise AppError("model.hardlink_rejected")
            count += 1
            size = path.stat().st_size
            total += size
            if count > self.max_files or total > self.max_total_bytes:
                raise AppError("model.total_too_large")
            target = destination / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("rb") as input_stream, target.open("xb") as output_stream:
                    for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                        output_stream.write(block)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
            except OSError as exc:
                if exc.errno == 28:
                    raise AppError("model.disk_full") from exc
                raise AppError("model.copy_failed") from exc


def _require_inspection(inspection: LocalModelInspection) -> None:
    if not inspection.validation_passed:
        raise AppError(inspection.validation_report.error_code or "model.validation_failed")


def _with_validation(model: ModelInstallation, passed: bool) -> ModelInstallation:
    if not passed:
        return ModelInstallation(
            identity=model.identity,
            manifest=model.manifest,
            model_directory=model.model_directory,
            state=(
                ModelState.EXTERNAL_UNMANAGED
                if model.state is ModelState.EXTERNAL_UNMANAGED
                else ModelState.FAILED
            ),
            managed=model.managed,
            load_verified=False,
            validation_passed=False,
        )
    state = model.state
    if state in {ModelState.STAGED, ModelState.FAILED}:
        state = ModelState.INSTALLED
    return ModelInstallation(
        identity=model.identity,
        manifest=model.manifest,
        model_directory=model.model_directory,
        state=state,
        managed=model.managed,
        load_verified=model.load_verified if state is ModelState.LOAD_VERIFIED else False,
        validation_passed=passed,
    )


def _cleanup_transaction(transaction: Path, progress: ProgressCallback | None) -> None:
    if not transaction.exists():
        return
    _emit(progress, "cleaning_staging")
    try:
        shutil.rmtree(transaction)
    except OSError as exc:
        raise AppError("model.staging_cleanup_failed") from exc


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, OSError) and exc.errno == 28:
            raise AppError("model.disk_full") from exc
        raise AppError("model.metadata_write_failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _emit(progress: ProgressCallback | None, phase: str) -> None:
    if progress is not None:
        progress(OperationProgress("model", phase, f"model.{phase}", {}))


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


def _fsync_tree_directories(root: Path) -> None:
    """Flush staged directory entries before the transaction is committed."""
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


__all__ = ["LoadVerifier", "ModelManager", "ProgressCallback"]
