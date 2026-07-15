from __future__ import annotations

from pathlib import Path

from captioner.infrastructure.app_paths import ensure_runtime_layout, resolve_app_paths


def _make_resources(root: Path) -> None:
    (root / "i18n").mkdir(parents=True)
    (root / "i18n" / "en.json").write_text("{}", encoding="utf-8")


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
