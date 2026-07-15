"""Initial standalone Nuitka build wrapper for Captioner."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = PROJECT_ROOT / "dist"
VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class BuildError(RuntimeError):
    """A deterministic build or artifact-layout failure."""

    @classmethod
    def unsupported_platform(cls, value: str) -> BuildError:
        return cls(f"unsupported build platform: {value}")

    @classmethod
    def uncontrolled_path(cls, path: Path) -> BuildError:
        return cls(f"refusing to clean uncontrolled path: {path}")

    @classmethod
    def artifact_count(cls, suffix: str, count: int, names: str) -> BuildError:
        return cls(f"expected exactly one {suffix} artifact, found {count}: {names}")

    @classmethod
    def missing_files(cls, paths: str) -> BuildError:
        return cls(f"packaged layout is missing: {paths}")


@dataclass(frozen=True, slots=True)
class BuildLayout:
    platform_name: str
    dist_root: Path
    work_root: Path
    final_root: Path
    executable_path: Path
    resource_root: Path


def validate_version(value: str) -> str:
    """Validate a simple SemVer-like build version."""
    if VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError
    return value


def windows_numeric_version(value: str) -> str:
    """Convert a release version into Windows' four-part numeric metadata."""
    display_version = validate_version(value)
    core = display_version.split("-", 1)[0].split("+", 1)[0]
    components = tuple(int(part) for part in core.split("."))
    if any(component > 65535 for component in components):
        raise ValueError
    return ".".join((*map(str, components), "0"))


def windows_compiler_options(architecture: str | None = None) -> tuple[str, ...]:
    """Select the Windows compiler, with an explicit ARM64 extension seam."""
    normalized = (platform.machine() if architecture is None else architecture).lower()
    if normalized in {"x86_64", "amd64", "x64"}:
        return ("--msvc=latest",)
    if normalized in {"arm64", "aarch64"}:
        # ARM64 support stays isolated so its future Clang policy is easy to revise.
        return ("--clang",)
    raise ValueError


def normalize_platform(value: str | None = None) -> str:
    """Map platform names to the three supported build branches."""
    value = platform.system() if value is None else value
    lowered = value.lower()
    if lowered.startswith("win"):
        return "windows"
    if lowered in {"darwin", "mac", "macos"}:
        return "macos"
    if lowered in {"linux", "linux2"}:
        return "linux"
    raise BuildError.unsupported_platform(value)


def layout_for_platform(
    platform_name: str | None = None, *, dist_root: Path = DIST_ROOT
) -> BuildLayout:
    """Return the standardized output and executable paths."""
    normalized = normalize_platform(platform_name)
    root = dist_root.expanduser().resolve()
    work_root = root / ".nuitka-build"
    if normalized == "macos":
        final_root = root / "Captioner.app"
        executable = final_root / "Contents" / "MacOS" / "captioner"
        resource_root = final_root / "Contents" / "Resources" / "resources"
    else:
        final_root = root / "captioner"
        executable_name = "captioner.exe" if normalized == "windows" else "captioner"
        executable = final_root / executable_name
        resource_root = final_root / "resources"
    return BuildLayout(
        platform_name=normalized,
        dist_root=root,
        work_root=work_root,
        final_root=final_root,
        executable_path=executable,
        resource_root=resource_root,
    )


def safe_remove_owned(path: Path, owned_paths: tuple[Path, ...]) -> None:
    """Delete one exact path only when it is explicitly owned by this wrapper."""
    normalized = Path(os.path.abspath(path.expanduser()))
    owned = {Path(os.path.abspath(candidate.expanduser())) for candidate in owned_paths}
    if normalized not in owned:
        raise BuildError.uncontrolled_path(path)
    if not normalized.exists() and not normalized.is_symlink():
        return
    if normalized.is_dir() and not normalized.is_symlink():
        shutil.rmtree(normalized)
    else:
        normalized.unlink()


def clean_owned_paths(layout: BuildLayout) -> None:
    """Clean only the build staging and final output paths owned by the wrapper."""
    owned_paths = (layout.work_root, layout.final_root)
    for path in owned_paths:
        safe_remove_owned(path, owned_paths)


def build_command(
    version: str,
    layout: BuildLayout,
    *,
    python_executable: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    architecture: str | None = None,
) -> list[str]:
    """Build the platform-specific Nuitka command without executing it."""
    display_version = validate_version(version)
    python = str(sys.executable if python_executable is None else python_executable)
    resources = project_root / "resources"
    readme = project_root / "README.md"
    command = [
        python,
        "-m",
        "nuitka",
        "--standalone",
        "--static-libpython=no",
        "--enable-plugin=pyside6",
        "--include-package=captioner",
        f"--include-data-dir={resources}=resources",
        f"--include-data-files={readme}=README.md",
        "--nofollow-import-to=tests",
        f"--output-dir={layout.work_root}",
        "--output-filename=captioner",
    ]
    if layout.platform_name == "windows":
        numeric_version = windows_numeric_version(display_version)
        command.extend(windows_compiler_options(architecture))
        command.extend(
            (
                f"--product-version={numeric_version}",
                f"--file-version={numeric_version}",
            )
        )
    else:
        command.append(f"--product-version={display_version}")
    if layout.platform_name == "macos":
        command.extend(("--macos-create-app-bundle", "--macos-app-name=Captioner"))
    command.append(str(project_root / "main.py"))
    return command


def find_unique_artifact(work_root: Path, suffix: str) -> Path:
    """Find exactly one generated ``.dist`` or ``.app`` directory."""
    artifacts = sorted(path for path in work_root.glob(f"*{suffix}") if path.is_dir())
    if len(artifacts) != 1:
        names = ", ".join(str(path) for path in artifacts) or "none"
        raise BuildError.artifact_count(suffix, len(artifacts), names)
    return artifacts[0]


def stage_artifact(layout: BuildLayout, artifact: Path) -> None:
    """Copy one Nuitka artifact into the standardized final location."""
    safe_remove_owned(layout.final_root, (layout.work_root, layout.final_root))
    if layout.platform_name == "macos":
        shutil.copytree(artifact, layout.final_root)
    else:
        shutil.copytree(artifact, layout.final_root)
    verify_layout(layout)


def verify_layout(layout: BuildLayout) -> None:
    """Verify the executable and staged resources needed by smoke tests."""
    required = (
        layout.executable_path,
        layout.resource_root / "i18n" / "en.json",
        layout.final_root / "README.md",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError.missing_files(", ".join(missing))


def build(version: str, *, clean: bool = False, platform_name: str | None = None) -> BuildLayout:
    """Compile, stage, and verify the local platform artifact."""
    validate_version(version)
    layout = layout_for_platform(platform_name)
    if clean:
        clean_owned_paths(layout)
    layout.work_root.mkdir(parents=True, exist_ok=True)
    command = build_command(version, layout)
    print("==> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    suffix = ".app" if layout.platform_name == "macos" else ".dist"
    artifact = find_unique_artifact(layout.work_root, suffix)
    stage_artifact(layout, artifact)
    print(f"Nuitka output: {layout.final_root}")
    return layout


def main(argv: list[str] | None = None) -> int:
    """Parse build options and run the wrapper."""
    parser = argparse.ArgumentParser(description="Build Captioner standalone with Nuitka")
    parser.add_argument(
        "--clean", action="store_true", help="Clean owned staging/output paths first"
    )
    parser.add_argument("--version", default="0.0.0", help="Package version")
    namespace = parser.parse_args(argv)
    try:
        build(namespace.version, clean=namespace.clean)
    except (BuildError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Nuitka build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
