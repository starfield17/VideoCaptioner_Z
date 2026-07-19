from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.phase6_values import model_installation, runtime_installation, worker_handshake
from tests.fakes.scripted_worker_client import ScriptedWorkerClient

from captioner.adapters.model.model_load_verifier import WorkerModelLoadVerifier
from captioner.core.domain.errors import AppError
from captioner.core.domain.worker_protocol import ModelLoadResponse


def test_load_verifier_uses_handshake_and_typed_local_load(tmp_path: Path) -> None:
    runtime = runtime_installation()
    model = model_installation()
    client = ScriptedWorkerClient(
        worker_handshake(),
        load_response=ModelLoadResponse(
            model.identity,
            runtime.manifest.backend_id,
            runtime.manifest.target.device_kind,
            True,
        ),
    )
    verifier = WorkerModelLoadVerifier(
        workspace_root=tmp_path / "workspaces",
        worker_factory=lambda _runtime: client,
    )

    verifier(model, runtime, "auto")

    assert len(client.start_calls) == 1
    assert client.load_calls[0].model_directory == model.model_directory
    assert client.shutdown_calls == [False]


def test_load_verifier_rejects_wrong_response_identity_and_always_shutdowns(
    tmp_path: Path,
) -> None:
    runtime = runtime_installation()
    model = model_installation()
    wrong_model = model_installation(repository_id="org/other")
    client = ScriptedWorkerClient(
        worker_handshake(),
        load_response=ModelLoadResponse(
            wrong_model.identity,
            runtime.manifest.backend_id,
            runtime.manifest.target.device_kind,
            True,
        ),
    )
    verifier = WorkerModelLoadVerifier(
        workspace_root=tmp_path / "workspaces",
        worker_factory=lambda _runtime: client,
    )

    with pytest.raises(AppError, match=r"model\.load_identity_mismatch"):
        verifier(model, runtime, "auto")
    assert client.shutdown_calls == [False]
