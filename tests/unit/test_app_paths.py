from __future__ import annotations

from pathlib import Path

import pytest

from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import (
    CompiledRuntime,
    ensure_runtime_layout,
    resolve_app_paths,
)


def _make_resources(root: Path) -> None:
    for directory in ("i18n", "prompts", "runtime", "tokenizers"):
        (root / directory).mkdir(parents=True)
    (root / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    (root / "tokenizers" / "tokenizer-manifest.json").write_text("{}", encoding="utf-8")
    (root / "tokenizers" / "cl100k_base.tiktoken").write_bytes(b"cl100k")
    (root / "tokenizers" / "o200k_base.tiktoken").write_bytes(b"o200k")


def test_source_override_and_writable_layout_are_separate(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    _make_resources(resources)
    paths = resolve_app_paths(
        platform_name="linux",
        base_dir=tmp_path / "user",
        executable_path=tmp_path / "source" / "main.py",
        compiled=False,
        resource_root_override=resources,
    )
    ensure_runtime_layout(paths)
    assert paths.resource_root == resources.resolve()
    assert paths.i18n_resource_dir.is_dir()
    assert all(directory.is_dir() for directory in paths.writable_directories)
    assert paths.batches_dir.is_dir()
    assert (paths.artifacts_dir / ".incoming").is_dir()
    assert (paths.artifacts_dir / "sha256").is_dir()
    assert all(resources not in directory.parents for directory in paths.writable_directories)


def test_compiled_linux_resource_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    resources = bundle / "resources"
    _make_resources(resources)
    paths = resolve_app_paths(
        platform_name="linux",
        base_dir=tmp_path / "user",
        executable_path=bundle / "captioner",
        compiled=True,
    )
    assert paths.resource_root == resources.resolve()


def test_compiled_macos_app_resource_path(tmp_path: Path) -> None:
    app = tmp_path / "Captioner.app"
    resources = app / "Contents" / "Resources" / "resources"
    _make_resources(resources)
    executable = app / "Contents" / "MacOS" / "captioner"
    paths = resolve_app_paths(
        platform_name="darwin",
        base_dir=tmp_path / "user",
        executable_path=executable,
        compiled=True,
    )
    assert paths.resource_root == resources.resolve()


def test_compiled_windows_runtime_descriptor_resolves_distribution_resources(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "captioner"
    resources = bundle / "resources"
    _make_resources(resources)
    paths = resolve_app_paths(
        platform_name="win32",
        base_dir=tmp_path / "user",
        executable_path=bundle / "captioner.exe",
        compiled_runtime=CompiledRuntime(True, bundle),
    )
    assert paths.resource_root == resources.resolve()


def test_compiled_runtime_never_falls_back_to_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    _make_resources(cwd / "resources")
    monkeypatch.chdir(cwd)
    with pytest.raises(AppError, match=r"runtime\.resource_root_invalid"):
        resolve_app_paths(
            platform_name="linux",
            base_dir=tmp_path / "user",
            executable_path=tmp_path / "missing" / "captioner",
            compiled_runtime=CompiledRuntime(True, tmp_path / "missing"),
        )


def test_incomplete_resource_roots_fail_closed(tmp_path: Path) -> None:
    incomplete = tmp_path / "resources"
    (incomplete / "i18n").mkdir(parents=True)
    (incomplete / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AppError, match=r"runtime\.resource_root_invalid"):
        resolve_app_paths(
            base_dir=tmp_path / "user",
            resource_root_override=incomplete,
            compiled_runtime=CompiledRuntime(False, None),
        )


def test_injected_home_uses_platform_standard_shapes(tmp_path: Path) -> None:
    linux = resolve_app_paths(platform_name="linux", home_dir=tmp_path / "home")
    windows = resolve_app_paths(platform_name="win32", home_dir=tmp_path / "home")
    macos = resolve_app_paths(platform_name="darwin", home_dir=tmp_path / "home")
    assert linux.config_dir == tmp_path / "home" / ".config" / "Captioner"
    assert windows.config_dir == tmp_path / "home" / "AppData" / "Roaming" / "Captioner"
    assert macos.cache_dir == tmp_path / "home" / "Library" / "Caches" / "Captioner"


def test_default_platformdirs_paths_are_not_bundle_relative() -> None:
    paths = resolve_app_paths()
    assert paths.config_dir != paths.resource_root
    assert paths.data_dir != paths.resource_root
