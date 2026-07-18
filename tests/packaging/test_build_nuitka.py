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
    stage_artifact,
    stage_release_documents,
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
    if platform_name == "macos":
        assert layout.readme_path == (layout.final_root / "Contents" / "Resources" / "README.md")
        assert layout.notice_path == (
            layout.final_root / "Contents" / "Resources" / "THIRD_PARTY_NOTICES.md"
        )
        assert layout.resource_root == (layout.final_root / "Contents" / "Resources" / "resources")
    else:
        assert layout.readme_path == layout.final_root / "README.md"
        assert layout.notice_path == layout.final_root / "THIRD_PARTY_NOTICES.md"
        assert layout.resource_root == layout.final_root / "resources"


def test_build_command_contains_plugin_package_and_resources(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0", layout, python_executable=Path("python"), project_root=tmp_path
    )
    joined = " ".join(command)
    assert "--enable-plugin=pyside6" in command
    assert "--include-package=captioner" in command
    assert "--assume-yes-for-downloads" in command
    assert f"--include-data-dir={tmp_path / 'resources'}=resources" in command
    assert not any("--include-data-files=" in argument for argument in command)
    assert "README.md" not in joined or "--include-data-files" not in joined
    assert "--nofollow-import-to=faster_whisper" in command
    assert "--nofollow-import-to=ctranslate2" in command
    assert "--nofollow-import-to=torch" in command
    assert "--nofollow-import-to=transformers" in command
    assert "--version" not in joined


@pytest.mark.parametrize("platform_name", ["linux", "windows", "macos"])
def test_build_command_assumes_yes_for_downloads_on_every_platform(
    tmp_path: Path, platform_name: str
) -> None:
    layout = layout_for_platform(platform_name, dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0",
        layout,
        python_executable=Path("python"),
        project_root=tmp_path,
        architecture="x86_64",
    )
    assert "--assume-yes-for-downloads" in command
    assert f"--include-data-dir={tmp_path / 'resources'}=resources" in command
    assert not any(
        argument.startswith("--include-data-files=") and "README.md" in argument
        for argument in command
    )
    assert not any(
        argument.startswith("--include-data-files=") and "THIRD_PARTY_NOTICES.md" in argument
        for argument in command
    )


def test_packaged_gui_smoke_command_shapes() -> None:
    """Document the packaged bilingual GUI smoke invocations used by Release Full Gate."""
    en = ["captioner", "--gui", "--lang", "en", "--smoke-test"]
    zh = ["captioner", "--gui", "--lang", "zh-CN", "--smoke-test"]
    assert en[1:4] == ["--gui", "--lang", "en"]
    assert zh[1:4] == ["--gui", "--lang", "zh-CN"]
    assert en[-1] == "--smoke-test"
    assert zh[-1] == "--smoke-test"


def test_diagnostics_modules_are_package_discoverable() -> None:
    import captioner.adapters.diagnostics.local_diagnostics as local_diagnostics
    import captioner.core.application.diagnostics as diagnostics
    import captioner.gui.diagnostics_controller as diagnostics_controller
    import captioner.gui.pages.diagnostics_page as diagnostics_page

    assert diagnostics.DIAGNOSTICS_SCHEMA_VERSION == 1
    assert local_diagnostics.LocalDiagnosticsAdapter is not None
    assert diagnostics_controller.DiagnosticsController is not None
    assert diagnostics_page.DiagnosticsPage is not None


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
    assert "--assume-yes-for-downloads" in command
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
    assert "--assume-yes-for-downloads" in command
    assert not any(
        argument.startswith("--include-data-files=") and "THIRD_PARTY_NOTICES.md" in argument
        for argument in command
    )
    (tmp_path / "one.dist").mkdir()
    assert find_unique_artifact(tmp_path, ".dist").name == "one.dist"
    (tmp_path / "two.dist").mkdir()
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")


def _write_resource_tree(resource_root: Path) -> None:
    for directory in ("i18n", "prompts", "runtime", "tokenizers"):
        (resource_root / directory).mkdir(parents=True, exist_ok=True)
    (resource_root / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    for filename in (
        "tokenizer-manifest.json",
        "cl100k_base.tiktoken",
        "o200k_base.tiktoken",
    ):
        (resource_root / "tokenizers" / filename).write_bytes(b"resource")


def _write_project_docs(project_root: Path) -> None:
    (project_root / "README.md").write_text("readme-source", encoding="utf-8")
    (project_root / "THIRD_PARTY_NOTICES.md").write_text("notice-source", encoding="utf-8")


def test_stage_artifact_linux_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_docs(project_root)
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    (artifact / "captioner").write_text("binary", encoding="utf-8")
    _write_resource_tree(artifact / "resources")

    stage_artifact(layout, artifact, project_root=project_root)

    assert layout.executable_path.is_file()
    assert layout.readme_path.read_text(encoding="utf-8") == "readme-source"
    assert layout.notice_path.read_text(encoding="utf-8") == "notice-source"
    assert (layout.resource_root / "i18n" / "en.json").is_file()
    assert not artifact.exists()
    assert (project_root / "README.md").read_text(encoding="utf-8") == "readme-source"
    assert (project_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == "notice-source"


def test_stage_artifact_windows_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_docs(project_root)
    layout = layout_for_platform("windows", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    (artifact / "captioner.exe").write_text("binary", encoding="utf-8")
    _write_resource_tree(artifact / "resources")

    stage_artifact(layout, artifact, project_root=project_root)

    assert layout.executable_path.name == "captioner.exe"
    assert layout.executable_path.is_file()
    assert layout.readme_path == layout.final_root / "README.md"
    assert layout.notice_path == layout.final_root / "THIRD_PARTY_NOTICES.md"
    assert layout.readme_path.is_file()
    assert layout.notice_path.is_file()
    assert (layout.resource_root / "tokenizers" / "tokenizer-manifest.json").is_file()


def test_stage_artifact_macos_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_docs(project_root)
    layout = layout_for_platform("macos", dist_root=tmp_path / "dist")
    fake_app = layout.work_root / "main.app"
    resource_root = fake_app / "Contents" / "Resources" / "resources"
    executable = fake_app / "Contents" / "MacOS" / "captioner"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")
    _write_resource_tree(resource_root)

    stage_artifact(layout, fake_app, project_root=project_root)

    assert layout.executable_path == (layout.final_root / "Contents" / "MacOS" / "captioner")
    assert layout.executable_path.is_file()
    assert layout.readme_path == (layout.final_root / "Contents" / "Resources" / "README.md")
    assert layout.notice_path == (
        layout.final_root / "Contents" / "Resources" / "THIRD_PARTY_NOTICES.md"
    )
    assert layout.readme_path.read_text(encoding="utf-8") == "readme-source"
    assert layout.notice_path.read_text(encoding="utf-8") == "notice-source"
    assert (layout.resource_root / "i18n" / "en.json").is_file()
    assert (layout.resource_root / "tokenizers" / "tokenizer-manifest.json").is_file()
    assert not fake_app.exists()


def test_stage_release_documents_missing_readme(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "THIRD_PARTY_NOTICES.md").write_text("notice", encoding="utf-8")
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="required release source file is missing"):
        stage_release_documents(layout, project_root=project_root)


def test_stage_release_documents_missing_notice(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "README.md").write_text("readme", encoding="utf-8")
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="required release source file is missing"):
        stage_release_documents(layout, project_root=project_root)


def test_stage_artifact_missing_executable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_docs(project_root)
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    _write_resource_tree(artifact / "resources")
    with pytest.raises(BuildError, match="packaged layout is missing"):
        stage_artifact(layout, artifact, project_root=project_root)


def test_stage_artifact_missing_resource_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_project_docs(project_root)
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    (artifact / "captioner").write_text("binary", encoding="utf-8")
    for directory in ("i18n", "prompts", "runtime", "tokenizers"):
        (artifact / "resources" / directory).mkdir(parents=True)
    (artifact / "resources" / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    # Intentionally omit tokenizer files required by validate_resource_root.
    with pytest.raises(BuildError, match="packaged layout is missing"):
        stage_artifact(layout, artifact, project_root=project_root)


def test_find_unique_artifact_rejects_multiple(tmp_path: Path) -> None:
    (tmp_path / "one.dist").mkdir()
    (tmp_path / "two.dist").mkdir()
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")


def test_find_unique_artifact_rejects_none(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")
