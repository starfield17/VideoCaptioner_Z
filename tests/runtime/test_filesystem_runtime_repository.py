from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.fakes.phase6_values import runtime_installation

from captioner.adapters.runtime.filesystem_runtime_repository import FilesystemRuntimeRepository
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeInstallation, RuntimeState, RuntimeTarget


def _installation(
    tmp_path: Path,
    version: str,
    *,
    state: RuntimeState,
    runtime_id: str | None = None,
) -> RuntimeInstallation:
    base = runtime_installation(version=version, state=state, runtime_id=runtime_id)
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


def test_prepare_activation_leaves_current_and_previous_unchanged(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first = _installation(tmp_path, "1.0.0", state=RuntimeState.AVAILABLE)
    second = _installation(tmp_path, "2.0.0", state=RuntimeState.AVAILABLE)
    candidate = _installation(tmp_path, "3.0.0", state=RuntimeState.AVAILABLE)
    for installation in (first, second, candidate):
        repository.register_installation(installation)

    repository.set_active_runtime(first.identity, first.manifest.backend_id, first.manifest.target)
    repository.set_active_runtime(
        second.identity, second.manifest.backend_id, second.manifest.target
    )
    prepared = repository.prepare_activation(candidate.identity)

    assert prepared.current == second.identity
    assert prepared.previous == first.identity
    assert prepared.pending_activation == candidate.identity
    active = repository.get_active_runtime(second.manifest.backend_id, second.manifest.target)
    assert active is not None
    assert active.identity == second.identity

    repository.complete_activation(candidate.identity)
    committed = repository.get_active_pointer(
        candidate.manifest.backend_id, candidate.manifest.target
    )
    assert committed is not None
    assert committed.current == candidate.identity
    assert committed.previous == second.identity
    assert committed.pending_activation is None


def test_pending_activation_recovers_previous_and_marks_candidate_failed(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first = _installation(tmp_path, "1.0.0", state=RuntimeState.AVAILABLE)
    second = _installation(tmp_path, "2.0.0", state=RuntimeState.AVAILABLE)
    candidate = _installation(tmp_path, "3.0.0", state=RuntimeState.AVAILABLE)
    repository.register_installation(first)
    repository.register_installation(second)
    repository.register_installation(candidate)
    repository.set_active_runtime(first.identity, first.manifest.backend_id, first.manifest.target)
    repository.prepare_activation(second.identity)
    repository.complete_activation(second.identity)
    repository.prepare_activation(candidate.identity)

    assert repository.recover() == (candidate.identity,)
    recovered = repository.get_by_identity(candidate.identity)
    assert recovered is not None
    assert recovered.state is RuntimeState.FAILED
    pointer = repository.get_active_pointer(first.manifest.backend_id, first.manifest.target)
    assert pointer is not None
    assert pointer.current == second.identity
    assert pointer.previous == first.identity
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


def test_external_record_removal_observes_use_lock_and_preserves_external_files(
    tmp_path: Path,
) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    external_root = tmp_path / "external-runtime"
    external_root.mkdir()
    marker = external_root / "model.bin"
    marker.write_bytes(b"external")
    external = _installation(tmp_path, "4.1.0", state=RuntimeState.EXTERNAL_UNMANAGED)
    external = replace(external, install_path=external_root, managed=False, doctor_passed=True)
    repository.register_installation(external)

    with repository.use_lock(external.identity), pytest.raises(AppError, match=r"runtime\.in_use"):
        repository.remove_installation_record(external.identity)

    repository.remove_installation_record(external.identity)
    assert marker.read_bytes() == b"external"
    assert external_root.is_dir()


def test_external_record_identity_digest_avoids_filename_collision(tmp_path: Path) -> None:
    repository = FilesystemRuntimeRepository(tmp_path / "runtimes")
    first_root = tmp_path / "external-one"
    second_root = tmp_path / "external-two"
    first_root.mkdir()
    second_root.mkdir()
    first = _installation(
        tmp_path,
        "1.0.0",
        state=RuntimeState.EXTERNAL_UNMANAGED,
        runtime_id="foo.bar",
    )
    first = replace(first, install_path=first_root, managed=False, doctor_passed=True)
    second = _installation(
        tmp_path,
        "1.0.0",
        state=RuntimeState.EXTERNAL_UNMANAGED,
        runtime_id="foo_bar",
    )
    second = replace(second, install_path=second_root, managed=False, doctor_passed=True)
    repository.register_installation(first)
    repository.register_installation(second)

    records = tuple((tmp_path / "runtimes" / "external").glob("*.json"))
    assert len(records) == 2
    assert repository.get_by_identity(first.identity) == first
    assert repository.get_by_identity(second.identity) == second

    repository.remove_installation_record(first.identity)
    assert repository.get_by_identity(first.identity) is None
    assert repository.get_by_identity(second.identity) == second
    assert len(tuple((tmp_path / "runtimes" / "external").glob("*.json"))) == 1
