from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.phase6_values import runtime_installation, runtime_manifest

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import (
    DoctorCheck,
    DoctorPhase,
    DoctorReport,
    RuntimeFileEntry,
    RuntimeIdentity,
    RuntimeState,
    RuntimeTarget,
)


def test_runtime_identity_is_stable_and_path_free() -> None:
    identity = RuntimeIdentity("faster-whisper-cpu-macos-arm64", "1.0.0")
    assert identity.runtime_id == "faster-whisper-cpu-macos-arm64"
    with pytest.raises(AppError, match=r"runtime\.identity_invalid"):
        RuntimeIdentity("/private/runtime", "1.0.0")


@pytest.mark.parametrize(
    "relative_path",
    ["/worker", "../worker", "worker/../secret", "C:/worker", "worker\\worker"],
)
def test_runtime_file_manifest_rejects_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(AppError, match=r"runtime\.file_invalid"):
        RuntimeFileEntry(relative_path, 1, "a" * 64, True)


def test_runtime_file_manifest_rejects_bad_hash_and_size() -> None:
    with pytest.raises(AppError, match=r"runtime\.file_invalid"):
        RuntimeFileEntry("worker", 1, "not-a-hash", True)
    with pytest.raises(AppError, match=r"runtime\.file_invalid"):
        RuntimeFileEntry("worker", -1, "a" * 64, True)


def test_installed_and_available_are_distinct() -> None:
    installed = runtime_installation(state=RuntimeState.INSTALLED)
    available = runtime_installation(state=RuntimeState.AVAILABLE)
    external = runtime_installation(
        state=RuntimeState.EXTERNAL_UNMANAGED,
        doctor_passed=True,
    )
    assert not installed.is_available
    assert available.is_available
    assert external.state is RuntimeState.EXTERNAL_UNMANAGED
    assert external.is_available


def test_external_runtime_without_activation_doctor_is_not_available() -> None:
    external = runtime_installation(state=RuntimeState.EXTERNAL_UNMANAGED)
    assert not external.is_available
    assert not external.can_delete_files


def test_doctor_report_is_structured_and_phase_specific() -> None:
    report = DoctorReport(
        ok=True,
        phase=DoctorPhase.STATIC.value,
        checks=(DoctorCheck("manifest", True),),
    )
    assert report.ok
    assert report.phase == "static"
    assert report.checks[0].name == "manifest"
    with pytest.raises(AppError, match=r"runtime\.doctor_invalid"):
        DoctorReport(
            ok=True,
            phase="static",
            checks=(DoctorCheck("manifest", False),),
        )


def test_runtime_manifest_does_not_create_or_reference_install_path(tmp_path: Path) -> None:
    manifest = runtime_manifest()
    assert "install_path" not in manifest.to_dict()
    assert str(tmp_path) not in repr(manifest)


def test_runtime_target_key_ignores_minimum_os_version() -> None:
    first = RuntimeTarget("macos", "arm64", "cpu", "13.0")
    second = RuntimeTarget("macos", "arm64", "cpu", "14.0")
    assert first.key == second.key == ("macos", "arm64", "cpu")


@pytest.mark.parametrize(
    "values",
    [
        ("darwin", "arm64", "cpu"),
        ("windows", "amd64", "cpu"),
        ("linux", "x86_64", "gpu"),
    ],
)
def test_runtime_target_requires_normalized_values(values: tuple[str, str, str]) -> None:
    with pytest.raises(AppError, match=r"runtime\.target_invalid"):
        RuntimeTarget(*values, "1.0")
