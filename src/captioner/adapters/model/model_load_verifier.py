"""Worker-backed model load verification adapter.

The Model Manager is synchronous because model installation and CLI commands
are synchronous application operations.  This adapter is the only boundary
that bridges that API to the asynchronous Worker Client lifecycle.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from captioner.adapters.runtime.subprocess_worker_client import SubprocessWorkerClient
from captioner.core.application.worker_handshake_validation import (
    validate_worker_handshake,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation
from captioner.core.domain.runtime import RuntimeIdentity, RuntimeInstallation
from captioner.core.domain.worker_protocol import HandshakeRequest, ModelLoadRequest
from captioner.core.ports.worker_client import WorkerClient

WorkerClientFactory = Callable[[RuntimeInstallation], WorkerClient]


class WorkerModelLoadVerifier:
    """Verify one installed model through its compatible Runtime Worker."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        worker_factory: WorkerClientFactory,
    ) -> None:
        if not workspace_root.is_absolute():
            raise AppError("worker.workspace_invalid")
        self.workspace_root = workspace_root
        self.worker_factory = worker_factory

    def __call__(
        self,
        model: ModelInstallation,
        runtime: object | None,
        device: str,
    ) -> None:
        if not isinstance(runtime, RuntimeInstallation):
            raise AppError("model.runtime_required")
        if device != "auto" and runtime.manifest.target.device_kind != device:
            raise AppError("runtime.model_device_mismatch")
        workspace = self.workspace_root / f"model-load-{uuid.uuid4().hex}"
        worker = self.worker_factory(runtime)
        try:
            asyncio.run(self._verify(worker, runtime, model, workspace))
        finally:
            try:
                shutil.rmtree(workspace)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise AppError("model.load_workspace_cleanup_failed") from exc

    async def _verify(
        self,
        worker: WorkerClient,
        runtime: RuntimeInstallation,
        model: ModelInstallation,
        workspace: Path,
    ) -> None:
        workspace.mkdir(parents=True, exist_ok=False)
        request = HandshakeRequest(
            required_capabilities=tuple(
                sorted(runtime.manifest.capabilities.advertised_capabilities)
            ),
            required_backend_id=runtime.manifest.backend_id,
            required_result_schema_versions=(1,),
        )
        started = False
        try:
            handshake = await worker.start(runtime, workspace, request)
            started = True
            validation = validate_worker_handshake(runtime.manifest, request, handshake)
            if not validation.ok:
                raise AppError(
                    validation.error_code or "worker.handshake_invalid",
                    {"reasons": list(validation.reasons)},
                )
            response = await worker.load_model(
                ModelLoadRequest(
                    model_directory=model.model_directory,
                    model_identity=model.identity,
                    backend_options={},
                )
            )
            if not response.loaded:
                raise AppError("model.load_failed")
            if response.model_identity != model.identity:
                raise AppError("model.load_identity_mismatch")
            if response.backend_id != runtime.manifest.backend_id:
                raise AppError("model.load_identity_mismatch", {"field": "backend_id"})
            if response.device_kind != runtime.manifest.target.device_kind:
                raise AppError("model.load_identity_mismatch", {"field": "device_kind"})
        finally:
            if started:
                await worker.shutdown()


def subprocess_model_load_verifier(
    *,
    log_dir: Path,
    workspace_root: Path,
    runtime_use_lock: Callable[[RuntimeIdentity], AbstractContextManager[None]] | None = None,
) -> WorkerModelLoadVerifier:
    """Build the production verifier without importing an ASR SDK."""

    def factory(runtime: RuntimeInstallation) -> WorkerClient:
        del runtime
        return SubprocessWorkerClient(
            log_dir=log_dir,
            runtime_use_lock=runtime_use_lock,
        )

    return WorkerModelLoadVerifier(workspace_root=workspace_root, worker_factory=factory)


__all__ = [
    "WorkerClientFactory",
    "WorkerModelLoadVerifier",
    "subprocess_model_load_verifier",
]
