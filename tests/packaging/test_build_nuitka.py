from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.build_nuitka import (
    BuildError,
    BuildLayout,
    BuildTarget,
    artifact_suffix,
    build_command,
    clean_owned_paths,
    find_unique_artifact,
    layout_for_platform,
    safe_remove_owned,
    stage_artifact,
    stage_release_documents,
    stage_resource_tree,
    validate_version,
    windows_compiler_options,
    windows_numeric_version,
)

ROOT = Path(__file__).resolve().parents[2]
CLI_ENTRY = ROOT / "scripts" / "nuitka_cli_entry.py"
WORKFLOW = ROOT / ".github" / "workflows" / "release-full.yml"


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
def test_desktop_platform_output_paths(
    tmp_path: Path, platform_name: str, final_name: str, executable_name: str
) -> None:
    layout = layout_for_platform(platform_name, target="desktop", dist_root=tmp_path / "dist")
    assert layout.target == "desktop"
    assert layout.final_root.name == final_name
    assert layout.executable_path.name == executable_name
    assert layout.work_root == layout.dist_root / ".nuitka-build" / "desktop"
    if platform_name == "macos":
        assert layout.readme_path == (layout.final_root / "Contents" / "Resources" / "README.md")
        assert layout.notice_path == (
            layout.final_root / "Contents" / "Resources" / "THIRD_PARTY_NOTICES.md"
        )
        assert layout.resource_root == (layout.final_root / "Contents" / "Resources" / "resources")
        assert artifact_suffix(layout) == ".app"
    else:
        assert layout.readme_path == layout.final_root / "README.md"
        assert layout.notice_path == layout.final_root / "THIRD_PARTY_NOTICES.md"
        assert layout.resource_root == layout.final_root / "resources"
        assert artifact_suffix(layout) == ".dist"


@pytest.mark.parametrize(
    ("platform_name", "executable_name"),
    [
        ("linux", "captioner"),
        ("windows", "captioner.exe"),
        ("macos", "captioner"),
    ],
)
def test_cli_platform_output_paths(
    tmp_path: Path, platform_name: str, executable_name: str
) -> None:
    layout = layout_for_platform(platform_name, target="cli", dist_root=tmp_path / "dist")
    assert layout.target == "cli"
    assert layout.final_root == layout.dist_root / "captioner"
    assert layout.executable_path == layout.final_root / executable_name
    assert layout.resource_root == layout.final_root / "resources"
    assert layout.readme_path == layout.final_root / "README.md"
    assert layout.notice_path == layout.final_root / "THIRD_PARTY_NOTICES.md"
    assert layout.work_root == layout.dist_root / ".nuitka-build" / "cli"
    assert artifact_suffix(layout) == ".dist"


@pytest.mark.parametrize("platform_name", ["linux", "windows", "macos"])
def test_cli_build_command_excludes_gui_and_resources(tmp_path: Path, platform_name: str) -> None:
    layout = layout_for_platform(platform_name, target="cli", dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0",
        layout,
        python_executable=Path("python"),
        project_root=tmp_path,
        architecture="x86_64",
    )
    joined = " ".join(command)
    assert str(tmp_path / "scripts" / "nuitka_cli_entry.py") in command
    assert "--assume-yes-for-downloads" in command
    assert "--enable-plugin=pyside6" not in command
    assert "--include-package=captioner" not in command
    assert not any("--include-data-dir=" in argument for argument in command)
    assert "--nofollow-import-to=captioner.gui" in command
    assert "--nofollow-import-to=captioner.gui.*" in command
    assert "--nofollow-import-to=PySide6" in command
    assert "--nofollow-import-to=PySide6.*" in command
    assert "--macos-create-app-bundle" not in command
    assert "--macos-app-name=Captioner" not in command
    assert str(tmp_path / "main.py") not in command
    assert "--nofollow-import-to=faster_whisper" in command
    assert "--version" not in joined


@pytest.mark.parametrize("platform_name", ["linux", "windows", "macos"])
def test_desktop_build_command_keeps_gui_plugin_without_data_dir(
    tmp_path: Path, platform_name: str
) -> None:
    layout = layout_for_platform(platform_name, target="desktop", dist_root=tmp_path / "dist")
    command = build_command(
        "0.0.0",
        layout,
        python_executable=Path("python"),
        project_root=tmp_path,
        architecture="x86_64",
    )
    assert str(tmp_path / "main.py") in command
    assert "--enable-plugin=pyside6" in command
    assert "--include-package=captioner" in command
    assert "--assume-yes-for-downloads" in command
    assert not any("--include-data-dir=" in argument for argument in command)
    assert not any(
        argument.startswith("--include-data-files=") and "README.md" in argument
        for argument in command
    )
    if platform_name == "macos":
        assert "--macos-create-app-bundle" in command
        assert "--macos-app-name=Captioner" in command
    else:
        assert "--macos-create-app-bundle" not in command


def test_windows_python_313_command_uses_msvc_and_numeric_metadata(tmp_path: Path) -> None:
    layout = layout_for_platform("windows", target="desktop", dist_root=tmp_path / "dist")
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
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.dist_root.mkdir()
    layout.work_root.mkdir(parents=True)
    layout.final_root.mkdir()
    clean_owned_paths(layout)
    assert not layout.work_root.exists()
    assert not layout.final_root.exists()
    with pytest.raises(BuildError, match="uncontrolled"):
        safe_remove_owned(tmp_path / "outside", (layout.final_root,))


def test_clean_does_not_follow_owned_output_symlink(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
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


def test_find_unique_artifact_rejects_multiple(tmp_path: Path) -> None:
    (tmp_path / "one.dist").mkdir()
    (tmp_path / "two.dist").mkdir()
    with pytest.raises(BuildError, match="exactly one"):
        find_unique_artifact(tmp_path, ".dist")


def test_find_unique_artifact_rejects_none(tmp_path: Path) -> None:
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


def _prepare_project(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_project_docs(project_root)
    _write_resource_tree(project_root / "resources")


def _make_executable_only_artifact(
    layout: BuildLayout, *, executable_name: str, nested_macos_app: bool = False
) -> Path:
    if nested_macos_app:
        artifact = layout.work_root / "main.app"
        executable = artifact / "Contents" / "MacOS" / "captioner"
        executable.parent.mkdir(parents=True)
        executable.write_text("binary", encoding="utf-8")
        return artifact
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    (artifact / executable_name).write_text("binary", encoding="utf-8")
    return artifact


@pytest.mark.parametrize(
    ("platform_name", "target", "executable_name", "nested_macos_app"),
    [
        ("linux", "cli", "captioner", False),
        ("windows", "cli", "captioner.exe", False),
        ("macos", "cli", "captioner", False),
        ("linux", "desktop", "captioner", False),
        ("windows", "desktop", "captioner.exe", False),
        ("macos", "desktop", "captioner", True),
    ],
)
def test_stage_artifact_copies_resources_from_project(
    tmp_path: Path,
    platform_name: str,
    target: BuildTarget,
    executable_name: str,
    nested_macos_app: bool,
) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    layout = layout_for_platform(platform_name, target=target, dist_root=tmp_path / "dist")
    artifact = _make_executable_only_artifact(
        layout, executable_name=executable_name, nested_macos_app=nested_macos_app
    )
    assert not (artifact / "resources").exists()
    if nested_macos_app:
        assert not (artifact / "Contents" / "Resources" / "resources").exists()

    stage_artifact(layout, artifact, project_root=project_root)

    assert layout.executable_path.is_file()
    assert layout.readme_path.read_text(encoding="utf-8") == "readme-source"
    assert layout.notice_path.read_text(encoding="utf-8") == "notice-source"
    assert (layout.resource_root / "i18n" / "en.json").is_file()
    assert (layout.resource_root / "tokenizers" / "tokenizer-manifest.json").is_file()
    assert not artifact.exists()
    assert (project_root / "resources" / "i18n" / "en.json").is_file()


def test_stage_resource_tree_missing_source(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="resource source tree is invalid"):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_resource_tree_incomplete_source(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "resources" / "i18n").mkdir(parents=True)
    (project_root / "resources" / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="resource source tree is invalid"):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_resource_tree_rejects_symlink_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    target = project_root / "outside.txt"
    target.write_text("x", encoding="utf-8")
    link = project_root / "resources" / "i18n" / "link.json"
    link.symlink_to(target)
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="resource source tree is invalid"):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_resource_tree_rejects_symlink_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (project_root / "resources" / "extra").symlink_to(outside, target_is_directory=True)
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="resource source tree is invalid"):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_resource_tree_rejects_existing_destination(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    layout.resource_root.mkdir()
    with pytest.raises(BuildError, match="resource destination already exists"):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_resource_tree_copy_failure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with (
        patch("scripts.build_nuitka.shutil.copytree", side_effect=OSError("disk full")),
        pytest.raises(BuildError, match="failed to stage resources"),
    ):
        stage_resource_tree(layout, project_root=project_root)


def test_stage_release_documents_missing_readme(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "THIRD_PARTY_NOTICES.md").write_text("notice", encoding="utf-8")
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="required release source file is missing"):
        stage_release_documents(layout, project_root=project_root)


def test_stage_release_documents_missing_notice(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "README.md").write_text("readme", encoding="utf-8")
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    layout.final_root.mkdir(parents=True)
    with pytest.raises(BuildError, match="required release source file is missing"):
        stage_release_documents(layout, project_root=project_root)


def test_stage_artifact_missing_executable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _prepare_project(project_root)
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    with pytest.raises(BuildError, match="packaged layout is missing"):
        stage_artifact(layout, artifact, project_root=project_root)


def test_nuitka_cli_entry_imports_only_cli_boundary() -> None:
    source = CLI_ENTRY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "captioner.cli.cli_entry" in imported
    assert "captioner.entrypoint" not in imported
    assert not any(name == "PySide6" or name.startswith("PySide6.") for name in imported)
    assert not any(
        name == "captioner.gui" or name.startswith("captioner.gui.") for name in imported
    )

    probe = (
        "import ast, pathlib, sys\n"
        f"source = pathlib.Path({str(CLI_ENTRY)!r}).read_text(encoding='utf-8')\n"
        "tree = ast.parse(source)\n"
        "mods = set()\n"
        "for node in ast.walk(tree):\n"
        "    if isinstance(node, ast.Import):\n"
        "        mods.update(alias.name for alias in node.names)\n"
        "    elif isinstance(node, ast.ImportFrom) and node.module:\n"
        "        mods.add(node.module)\n"
        "assert 'captioner.cli.cli_entry' in mods\n"
        "assert 'captioner.entrypoint' not in mods\n"
        "assert not any(m == 'PySide6' or m.startswith('PySide6.') for m in mods)\n"
        "assert not any(m == 'captioner.gui' or m.startswith('captioner.gui.') for m in mods)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_release_full_gate_is_cli_only_packaging() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("--target cli") == 3
    assert "--gui" not in text
    assert "Captioner.app" not in text
    assert "captioner-cli-ubuntu-" in text
    assert "captioner-cli-windows-" in text
    assert "captioner-cli-macos-" in text
    assert "captioner-cli-linux.tar.gz" in text
    assert "captioner-cli-windows.zip" in text
    assert "captioner-cli-macos.zip" in text
    assert " --cli " not in text
    assert '"--cli"' not in text
    assert "ubuntu-24.04-cli-package" in text
    assert "windows-2022-cli-package" in text
    assert "macos-15-cli-package" in text


def test_diagnostics_modules_are_package_discoverable() -> None:
    import captioner.adapters.diagnostics.local_diagnostics as local_diagnostics
    import captioner.core.application.diagnostics as diagnostics
    import captioner.gui.diagnostics_controller as diagnostics_controller
    import captioner.gui.pages.diagnostics_page as diagnostics_page

    assert diagnostics.DIAGNOSTICS_SCHEMA_VERSION == 1
    assert local_diagnostics.LocalDiagnosticsAdapter is not None
    assert diagnostics_controller.DiagnosticsController is not None
    assert diagnostics_page.DiagnosticsPage is not None
