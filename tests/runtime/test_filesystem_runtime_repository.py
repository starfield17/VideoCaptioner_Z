from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.fakes.phase6_values import runtime_installation

from captioner.adapters.runtime.filesystem_runtime_repository import FilesystemRuntimeRepository
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeState, RuntimeTarget


def _installation(tmp_path: Path, version: str, *, state: RuntimeState) -> RuntimeInstallation:
    base = runtime_installation(version=version, state=state)
    return replace(
        base,
        install_path=tmp_path / "runtimes" / base.identity.runtime_id / base.identity.version,
    )


def test_active_pointer_is_atomic_and_minimum_os_is_not_a_slot_key(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first = _installation(tmp_path, "1.0.0", state=RuntimeState.AVAILABLE)
    second = _installation(tmp_path, "2.0.0", state=RuntimeState.AVAILABLE)
    repository.register_installation(first)
    repository.register_installation(second)
    target = first.manifest.target
    repository.set_active_runtime(first.identity, first.manifest.backend_id, target)
    repository.prepare_activation(second.identity)
    repository.complete_activation(second.identity)

    pointer = repository.get_active_pointer(first.manifest.backend_id, target)
    assert pointer is not None
    assert pointer.current == second.identity
    assert pointer.previous == first.identity
    assert (tmp_path / "runtimes" / "active.json").is_file()
    assert not list((tmp_path / "runtimes").glob(".active.json.*"))

    other_target = RuntimeTarget("macos", "arm64", "cpu", "15.0")
    assert repository.get_active_pointer(first.manifest.backend_id, other_target) == pointer


def test_pending_activation_recovers_previous_and_marks_candidate_failed(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first = _installation(tmp_path, "1.0.0", state=RuntimeState.AVAILABLE)
    second = _installation(tmp_path, "2.0.0", state=RuntimeState.AVAILABLE)
    repository.register_installation(first)
    repository.register_installation(second)
    repository.set_active_runtime(first.identity, first.manifest.backend_id, first.manifest.target)
    repository.prepare_activation(second.identity)

    assert repository.recover() == (second.identity,)
    recovered = repository.get_by_identity(second.identity)
    assert recovered is not None
    assert recovered.state is RuntimeState.FAILED
    pointer = repository.get_active_pointer(first.manifest.backend_id, first.manifest.target)
    assert pointer is not None
    assert pointer.current == first.identity
    assert pointer.pending_activation is None


def test_active_and_previous_runtimes_cannot_be_removed(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first = _installation(tmp_path, "1.0.0", state=RuntimeState.AVAILABLE)
    second = _installation(tmp_path, "2.0.0", state=RuntimeState.AVAILABLE)
    repository.register_installation(first)
    repository.register_installation(second)
    repository.set_active_runtime(first.identity, first.manifest.backend_id, first.manifest.target)
    repository.prepare_activation(second.identity)
    repository.complete_activation(second.identity)

    with pytest.raises(AppError, match=r"runtime\.active_or_previous"):
        repository.remove_managed_files(first.identity)
    with pytest.raises(AppError, match=r"runtime\.active_or_previous"):
        repository.remove_managed_files(second.identity)


def test_failed_inactive_runtime_can_be_removed(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    failed = _installation(tmp_path, "3.0.0", state=RuntimeState.FAILED)
    failed.install_path.mkdir(parents=True)
    repository.register_installation(failed)

    repository.remove_managed_files(failed.identity)
    assert repository.get_by_identity(failed.identity) is None
    assert not failed.install_path.exists()


def test_external_record_removal_does_not_delete_external_files(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    external_root = tmp_path / "external-runtime"
    external_root.mkdir()
    external = _installation(tmp_path, "4.0.0", state=RuntimeState.EXTERNAL_UNMANAGED)
    external = replace(external, install_path=external_root, managed=False, doctor_passed=True)
    repository.register_installation(external)

    repository.remove_installation_record(external.identity)
    assert external_root.is_dir()
    assert repository.get_by_identity(external.identity) is None
