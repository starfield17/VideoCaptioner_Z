from __future__ import annotations

from pathlib import Path

import pytest
from scripts.build_nuitka import (
    BuildError,
    build_command,
    clean_owned_paths,
    find_unique_artifact,
    layout_for_platform,
    safe_remove_owned,
    validate_version,
    windows_compiler_options,
    windows_numeric_version,
)


def test_version_validation() -> None:
    assert validate_version("0.0.0") == "0.0.0"
    assert validate_version("1.2.3-beta+build") == "1.2.3-beta+build"
    assert windows_numeric_version("1.2.3-beta+build") == "1.2.3.0"
    with pytest.raises(ValueError):
        validate_version("0.0")


@pytest.mark.parametrize(
    ("platform_name", "final_name", "executable_name"),
    [
        ("linux", "captioner", "captioner"),
        ("windows", "captioner", "captioner.exe"),
        ("macos", "Captioner.app", "captioner"),
    ],
)
def test_platform_output_paths(
    tmp_path: Path, platform_name: str, final_name: str, executable_name: str
) -> None:
    layout = layout_for_platform(platform_name, dist_root=tmp_path / "dist")
    assert layout.final_root.name == final_name
    assert layout.executable_path.name == executable_name
    assert layout.work_root.parent == layout.dist_root
    expected_notice = (
        layout.final_root / "Contents" / "Resources" / "THIRD_PARTY_NOTICES.md"
        if platform_name == "macos"
        else layout.final_root / "THIRD_PARTY_NOTICES.md"
    )
    assert layout.notice_path == expected_notice


def test_build_command_contains_plugin_package_and_resources(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0", layout, python_executable=Path("python"), project_root=tmp_path
    )
    joined = " ".join(command)
    assert "--enable-plugin=pyside6" in command
    assert "--include-package=captioner" in command
    assert f"--include-data-dir={tmp_path / 'resources'}=resources" in command
    assert f"--include-data-files={tmp_path / 'README.md'}=README.md" in command
    assert (
        f"--include-data-files={tmp_path / 'THIRD_PARTY_NOTICES.md'}=THIRD_PARTY_NOTICES.md"
        in command
    )
    assert "--nofollow-import-to=faster_whisper" in command
    assert "--nofollow-import-to=ctranslate2" in command
    assert "--nofollow-import-to=torch" in command
    assert "--version" not in joined


def test_windows_python_313_command_uses_msvc_and_numeric_metadata(tmp_path: Path) -> None:
    layout = layout_for_platform("windows", dist_root=tmp_path / "dist")
    command = build_command(
        "1.2.3-beta",
        layout,
        python_executable=Path("python3.13"),
        project_root=tmp_path,
        architecture="x86_64",
    )
    assert "--msvc=latest" in command
    assert not any("--mingw64" in argument for argument in command)
    assert "--product-version=1.2.3.0" in command
    assert "--file-version=1.2.3.0" in command


def test_windows_arm64_compiler_seam_is_explicit() -> None:
    assert windows_compiler_options("arm64") == ("--clang",)


def test_clean_path_protection(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    layout.dist_root.mkdir()
    layout.work_root.mkdir()
    layout.final_root.mkdir()
    clean_owned_paths(layout)
    assert not layout.work_root.exists()
    assert not layout.final_root.exists()
    with pytest.raises(BuildError, match="uncontrolled"):
        safe_remove_owned(tmp_path / "outside", (layout.final_root,))


def test_clean_does_not_follow_owned_output_symlink(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    layout.dist_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    layout.final_root.symlink_to(outside, target_is_directory=True)
    clean_owned_paths(layout)
    assert marker.is_file()
    assert not layout.final_root.exists()


def test_unique_artifact_detection(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")


def test_macos_build_command_keeps_app_bundle_resource_destination(tmp_path: Path) -> None:
    layout = layout_for_platform("macos", dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0", layout, python_executable=Path("python"), project_root=tmp_path
    )
    assert "--macos-create-app-bundle" in command
    assert (
        f"--include-data-files={tmp_path / 'THIRD_PARTY_NOTICES.md'}=THIRD_PARTY_NOTICES.md"
        in command
    )
    (tmp_path / "one.dist").mkdir()
    assert find_unique_artifact(tmp_path, ".dist").name == "one.dist"
    (tmp_path / "two.dist").mkdir()
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")
