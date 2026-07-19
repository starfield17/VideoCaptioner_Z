"""CLI boundary for Model Manager operations.

The command module only assembles ports, parses the namespace, and renders
safe projections.  Source SDKs and filesystem mutation stay in adapters and
the application service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from captioner.adapters.model.filesystem_local_model_inspector import (
    FilesystemLocalModelInspector,
)
from captioner.adapters.model.filesystem_model_repository import (
    FilesystemModelRepository,
)
from captioner.adapters.model.filesystem_model_validator import FilesystemModelValidator
from captioner.adapters.model.huggingface_materializer import HuggingFaceModelMaterializer
from captioner.adapters.model.huggingface_source import HuggingFaceModelSource
from captioner.adapters.model.model_load_verifier import WorkerModelLoadVerifier
from captioner.adapters.model.modelscope_materializer import ModelScopeModelMaterializer
from captioner.adapters.model.modelscope_source import ModelScopeModelSource
from captioner.adapters.runtime.filesystem_runtime_repository import (
    FilesystemRuntimeRepository,
)
from captioner.adapters.runtime.host_probe import probe_host_facts
from captioner.adapters.runtime.subprocess_worker_client import SubprocessWorkerClient
from captioner.core.application.model_manager import ModelManager
from captioner.core.application.model_selector import select_model
from captioner.core.application.runtime_selection import select_runtime
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation, ModelValidationReport
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import RuntimeIdentity, RuntimeInstallation
from captioner.infrastructure.app_paths import AppPaths
from captioner.infrastructure.model_source_config import load_all_model_source_configs


class ModelNamespace(Protocol):
    model_command: str
    query: str
    backend: str
    limit: int
    repository_id: str
    revision: str | None
    model_format: str
    display_name: str | None
    directory: Path
    selector: str
    developer_mode: bool
    verify_load: bool
    runtime_id: str | None
    runtime_version: str | None
    device: str


def build_manager(paths: AppPaths) -> ModelManager:
    """Compose the filesystem repository and lazy remote source adapters."""
    repository = FilesystemModelRepository(
        paths.models_dir,
        staging_dir=paths.staging_dir,
    )
    validator = FilesystemModelValidator()
    inspector = FilesystemLocalModelInspector(validator)
    configs = load_all_model_source_configs(paths.config_dir)
    sources = {
        "huggingface": HuggingFaceModelSource(configs["huggingface"]),
        "modelscope": ModelScopeModelSource(configs["modelscope"]),
    }
    materializers = {
        "huggingface": HuggingFaceModelMaterializer(configs["huggingface"]),
        "modelscope": ModelScopeModelMaterializer(configs["modelscope"]),
    }
    runtime_repository = FilesystemRuntimeRepository(paths.runtimes_dir)
    load_verifier = WorkerModelLoadVerifier(
        workspace_root=paths.workspaces_dir / "model-load",
        worker_factory=lambda _runtime: SubprocessWorkerClient(
            log_dir=paths.log_dir,
            runtime_use_lock=runtime_repository.use_lock,
        ),
    )
    manager = ModelManager(
        repository=repository,
        inspector=inspector,
        validator=validator,
        models_dir=paths.models_dir,
        staging_dir=paths.staging_dir,
        downloads_dir=paths.downloads_dir,
        sources=sources,
        materializers=materializers,
        load_verifier=load_verifier,
    )
    manager.recover()
    return manager


def execute(namespace: object, *, paths: AppPaths) -> dict[str, JsonValue]:
    args = cast(ModelNamespace, namespace)
    manager = build_manager(paths)
    command = args.model_command
    if command == "list":
        return {"models": [_installation_payload(item) for item in manager.list_models()]}
    if command == "search-hf":
        candidates = manager.search_huggingface(args.query, args.backend, args.limit)
        return {"candidates": [candidate.to_dict() for candidate in candidates]}
    if command == "install-hf":
        model = manager.install_remote(
            "huggingface",
            args.repository_id,
            args.revision,
            args.backend,
            args.model_format,
            display_name=args.display_name,
        )
        return _maybe_verify(model, args, manager, paths)
    if command == "install-modelscope":
        model = manager.install_remote(
            "modelscope",
            args.repository_id,
            args.revision,
            args.backend,
            args.model_format,
            display_name=args.display_name,
        )
        return _maybe_verify(model, args, manager, paths)
    if command == "import":
        model = manager.import_local(
            args.directory,
            args.backend,
            args.model_format,
            args.display_name,
        )
        return _maybe_verify(model, args, manager, paths)
    if command == "register-external":
        model = manager.register_external(
            args.directory,
            args.backend,
            args.model_format,
            args.display_name,
            developer_mode=args.developer_mode,
        )
        return _maybe_verify(model, args, manager, paths)
    if command == "validate":
        model = _select(args.selector, manager)
        return {
            "model": _installation_payload(model),
            "report": _report_payload(manager.validate(model.identity)),
        }
    if command == "verify-load":
        model = _select(args.selector, manager)
        runtime = _resolve_runtime(model, args, paths)
        verified = manager.verify_load(
            model.identity,
            runtime=runtime,
            device=args.device,
        )
        return {"model": _installation_payload(verified)}
    if command == "remove":
        model = _select(args.selector, manager)
        manager.remove(model.identity)
        return {"removed": True, "identity": model.identity.to_dict()}
    raise AppError("cli.unknown_command")


def _maybe_verify(
    model: ModelInstallation,
    args: ModelNamespace,
    manager: ModelManager,
    paths: AppPaths,
) -> dict[str, JsonValue]:
    if not args.verify_load:
        return {"model": _installation_payload(model)}
    runtime = _resolve_runtime(model, args, paths)
    verified = manager.verify_load(
        model.identity,
        runtime=runtime,
        device=args.device,
    )
    return {"model": _installation_payload(verified)}


def _select(selector: str, manager: ModelManager) -> ModelInstallation:
    return select_model(selector, manager.list_models())


def _resolve_runtime(
    model: ModelInstallation,
    args: ModelNamespace,
    paths: AppPaths,
) -> RuntimeInstallation:
    repository = FilesystemRuntimeRepository(paths.runtimes_dir)
    if args.runtime_id is not None or args.runtime_version is not None:
        if args.runtime_id is None or args.runtime_version is None:
            raise AppError("runtime.identity_required")
        identity = RuntimeIdentity(args.runtime_id, args.runtime_version)
        runtime = repository.get_by_identity(identity)
        if runtime is None or not runtime.is_available:
            raise AppError("runtime.not_registered")
        if args.device != "auto" and runtime.manifest.target.device_kind != args.device:
            raise AppError("runtime.model_device_mismatch")
        return runtime
    active_runtimes = tuple(
        runtime
        for runtime in repository.list_installations()
        if runtime.is_available
        and (
            pointer := repository.get_active_pointer(
                runtime.manifest.backend_id,
                runtime.manifest.target,
            )
        )
        is not None
        and pointer.current == runtime.identity
    )
    selection = select_runtime(
        requested_backend_id=model.identity.backend_id,
        requested_device=args.device,
        host=probe_host_facts(),
        active_runtimes=active_runtimes,
        model=model,
    )
    runtime = repository.get_by_identity(selection.effective_runtime_identity)
    if runtime is None:
        raise AppError("runtime.not_registered")
    return runtime


def _installation_payload(model: ModelInstallation) -> dict[str, JsonValue]:
    return {
        "identity": model.identity.to_dict(),
        "state": model.state.value,
        "managed": model.managed,
        "validation_passed": model.validation_passed,
        "load_verified": model.load_verified,
        "model_directory": str(model.model_directory),
        "display_name": model.manifest.display_name,
        "model_format": model.manifest.model_format,
    }


def _report_payload(report: ModelValidationReport) -> dict[str, JsonValue]:
    return {
        "ok": report.ok,
        "error_code": report.error_code,
        "message_code": report.message_code,
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "error_code": check.error_code,
                "message_code": check.message_code,
            }
            for check in report.checks
        ],
    }


__all__ = ["ModelNamespace", "build_manager", "execute"]
