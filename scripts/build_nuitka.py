"""Initial standalone Nuitka build wrapper for Captioner."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

source_root_text = str(SOURCE_ROOT)

if source_root_text not in sys.path:
    sys.path.insert(0, source_root_text)

# SOURCE_ROOT must be on sys.path before project imports when not installed.
from captioner.core.domain.errors import AppError  # noqa: E402
from captioner.infrastructure.app_paths import validate_resource_root  # noqa: E402

DIST_ROOT = PROJECT_ROOT / "dist"
VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
BuildTarget = Literal["cli", "desktop"]
REQUIRED_PYTHON = (3, 13)
MSVC_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
VSWHERE_RELATIVE = Path("Microsoft Visual Studio") / "Installer" / "vswhere.exe"

NUITKA_MISSING_MESSAGE = """\
Nuitka is not installed in the selected Python environment.

Recommended:
  uv sync --frozen

Optional pip setup:
  python -m pip install -e .
  python -m pip install "nuitka==2.8.10"
"""

PYSIDE6_MISSING_MESSAGE = """\
PySide6 is not installed; it is required for the desktop target.

Recommended:
  uv sync --frozen

Optional:
  python -m pip install -e .
"""

PYTHON_VERSION_MESSAGE = """\
Unsupported Python version: {version}.

Captioner builds require Python 3.13.
"""

MSVC_MISSING_MESSAGE = """\
No supported Windows C compiler was found.

Install Visual Studio 2022 Build Tools with:
- Desktop development with C++
- MSVC v143 x64/x86 build tools
- Windows 10 or Windows 11 SDK

uv installs Python packages only; it does not install the native C compiler.
"""


class BuildError(RuntimeError):
    """A deterministic build or artifact-layout failure."""

    @classmethod
    def unsupported_platform(cls, value: str) -> BuildError:
        return cls(f"unsupported build platform: {value}")

    @classmethod
    def unsupported_target(cls, value: str) -> BuildError:
        return cls(f"unsupported build target: {value}")

    @classmethod
    def uncontrolled_path(cls, path: Path) -> BuildError:
        return cls(f"refusing to clean uncontrolled path: {path}")

    @classmethod
    def artifact_count(cls, suffix: str, count: int, names: str) -> BuildError:
        return cls(f"expected exactly one {suffix} artifact, found {count}: {names}")

    @classmethod
    def missing_files(cls, paths: str) -> BuildError:
        return cls(f"packaged layout is missing: {paths}")

    @classmethod
    def source_file_missing(cls, path: Path) -> BuildError:
        return cls(f"required release source file is missing: {path}")

    @classmethod
    def source_file_invalid(cls, path: Path) -> BuildError:
        return cls(f"required release source file is not a regular file: {path}")

    @classmethod
    def staging_destination_escape(cls, path: Path) -> BuildError:
        return cls(f"release document destination escapes final root: {path}")

    @classmethod
    def staging_copy_failed(cls, source: Path, target: Path, detail: str) -> BuildError:
        return cls(f"failed to stage {source} -> {target}: {detail}")

    @classmethod
    def resource_source_invalid(cls, path: Path) -> BuildError:
        return cls(f"resource source tree is invalid: {path}")

    @classmethod
    def resource_destination_exists(cls, path: Path) -> BuildError:
        return cls(f"resource destination already exists: {path}")

    @classmethod
    def resource_copy_failed(cls, source: Path, destination: Path, detail: str) -> BuildError:
        return cls(f"failed to stage resources {source} -> {destination}: {detail}")

    @classmethod
    def nuitka_missing(cls) -> BuildError:
        return cls(NUITKA_MISSING_MESSAGE.strip())

    @classmethod
    def pyside6_missing(cls) -> BuildError:
        return cls(PYSIDE6_MISSING_MESSAGE.strip())

    @classmethod
    def unsupported_python(cls, version_info: tuple[int, ...]) -> BuildError:
        version = ".".join(str(part) for part in version_info[:3])
        return cls(PYTHON_VERSION_MESSAGE.format(version=version).strip())

    @classmethod
    def msvc_missing(cls) -> BuildError:
        return cls(MSVC_MISSING_MESSAGE.strip())


@dataclass(frozen=True, slots=True)
class BuildLayout:
    platform_name: str
    target: BuildTarget
    dist_root: Path
    work_root: Path
    final_root: Path
    executable_path: Path
    resource_root: Path
    readme_path: Path
    notice_path: Path


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
    platform_name: str | None = None,
    *,
    target: BuildTarget = "desktop",
    dist_root: Path = DIST_ROOT,
) -> BuildLayout:
    """Return the standardized output and executable paths for a build target."""
    if target not in {"cli", "desktop"}:
        raise BuildError.unsupported_target(str(target))
    normalized = normalize_platform(platform_name)
    root = dist_root.expanduser().resolve()
    work_root = root / ".nuitka-build" / target
    if target == "desktop" and normalized == "macos":
        final_root = root / "Captioner.app"
        executable = final_root / "Contents" / "MacOS" / "captioner"
        resource_root = final_root / "Contents" / "Resources" / "resources"
        readme_path = final_root / "Contents" / "Resources" / "README.md"
        notice_path = final_root / "Contents" / "Resources" / "THIRD_PARTY_NOTICES.md"
    else:
        final_root = root / "captioner"
        executable_name = "captioner.exe" if normalized == "windows" else "captioner"
        executable = final_root / executable_name
        resource_root = final_root / "resources"
        readme_path = final_root / "README.md"
        notice_path = final_root / "THIRD_PARTY_NOTICES.md"
    return BuildLayout(
        platform_name=normalized,
        target=target,
        dist_root=root,
        work_root=work_root,
        final_root=final_root,
        executable_path=executable,
        resource_root=resource_root,
        readme_path=readme_path,
        notice_path=notice_path,
    )


def artifact_suffix(layout: BuildLayout) -> str:
    """Return the Nuitka artifact suffix expected for the layout."""
    if layout.target == "desktop" and layout.platform_name == "macos":
        return ".app"
    return ".dist"


def build_environment(
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Copy the process environment with the source tree prepended to PYTHONPATH."""
    environment = os.environ.copy()
    source_root = str((project_root / "src").resolve())
    current = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = (
        source_root if not current else os.pathsep.join((source_root, current))
    )
    return environment


def module_available(name: str) -> bool:
    """Return whether an importable module exists in the current environment."""
    return importlib.util.find_spec(name) is not None


def preflight_python_version(
    *,
    version_info: tuple[int, ...] | None = None,
) -> None:
    """Reject interpreters other than Python 3.13."""
    if version_info is None:
        current = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    else:
        current = version_info
    if current[:2] != REQUIRED_PYTHON:
        raise BuildError.unsupported_python(current)


def preflight_dependencies(
    target: BuildTarget,
    *,
    find_module: Callable[[str], bool] = module_available,
) -> None:
    """Reject missing Nuitka (all targets) or PySide6 (desktop) before compilation."""
    if not find_module("nuitka"):
        raise BuildError.nuitka_missing()
    if target == "desktop" and not find_module("PySide6"):
        raise BuildError.pyside6_missing()


def detect_msvc(
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environ: dict[str, str] | None = None,
) -> bool:
    """Return True when MSVC x64 tools are available for Windows Python 3.13 builds."""
    if which("cl") is not None:
        return True

    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    which_vswhere = which("vswhere")
    if which_vswhere is not None:
        candidates.append(Path(which_vswhere))
    program_files_x86 = env.get("ProgramFiles(x86)")
    if program_files_x86:
        candidates.append(Path(program_files_x86) / VSWHERE_RELATIVE)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            continue
        try:
            completed = run(
                [
                    str(resolved),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    MSVC_COMPONENT,
                    "-property",
                    "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if completed.returncode != 0:
            continue
        if completed.stdout.strip():
            return True
    return False


def preflight_windows_compiler(
    platform_name: str,
    *,
    architecture: str | None = None,
    detect: Callable[[], bool] = detect_msvc,
) -> None:
    """Fail early when Windows x64 builds lack MSVC; leave ARM64 policy unchanged."""
    if platform_name != "windows":
        return
    normalized = (platform.machine() if architecture is None else architecture).lower()
    if normalized not in {"x86_64", "amd64", "x64"}:
        return
    if not detect():
        raise BuildError.msvc_missing()


def preflight_build(
    target: BuildTarget,
    *,
    platform_name: str | None = None,
    architecture: str | None = None,
    version_info: tuple[int, ...] | None = None,
    find_module: Callable[[str], bool] = module_available,
    detect_compiler: Callable[[], bool] = detect_msvc,
) -> None:
    """Run deterministic startup checks before invoking Nuitka."""
    preflight_python_version(version_info=version_info)
    preflight_dependencies(target, find_module=find_module)
    normalized = normalize_platform(platform_name)
    preflight_windows_compiler(
        normalized,
        architecture=architecture,
        detect=detect_compiler,
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
    command = [
        python,
        "-m",
        "nuitka",
        "--standalone",
        "--static-libpython=no",
        "--assume-yes-for-downloads",
        "--nofollow-import-to=tests",
        "--nofollow-import-to=faster_whisper",
        "--nofollow-import-to=ctranslate2",
        "--nofollow-import-to=torch",
        "--nofollow-import-to=transformers",
        f"--output-dir={layout.work_root}",
        "--output-filename=captioner",
    ]
    if layout.target == "cli":
        command.extend(
            (
                "--nofollow-import-to=captioner.gui",
                "--nofollow-import-to=captioner.gui.*",
                "--nofollow-import-to=PySide6",
                "--nofollow-import-to=PySide6.*",
            )
        )
        entry = project_root / "scripts" / "nuitka_cli_entry.py"
    else:
        command.extend(("--enable-plugin=pyside6", "--include-package=captioner"))
        entry = project_root / "main.py"
        if layout.platform_name == "macos":
            command.extend(("--macos-create-app-bundle", "--macos-app-name=Captioner"))
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
    command.append(str(entry))
    return command


def find_unique_artifact(work_root: Path, suffix: str) -> Path:
    """Find exactly one generated ``.dist`` or ``.app`` directory."""
    artifacts = sorted(path for path in work_root.glob(f"*{suffix}") if path.is_dir())
    if len(artifacts) != 1:
        names = ", ".join(str(path) for path in artifacts) or "none"
        raise BuildError.artifact_count(suffix, len(artifacts), names)
    return artifacts[0]


def _require_regular_source_file(path: Path) -> Path:
    """Require a non-symlink regular file for release document sources."""
    if path.is_symlink() or not path.is_file():
        if not path.exists() and not path.is_symlink():
            raise BuildError.source_file_missing(path)
        raise BuildError.source_file_invalid(path)
    return path


def _destination_within_final_root(layout: BuildLayout, destination: Path) -> Path:
    """Reject document destinations that escape the staged final root."""
    final_root = layout.final_root.resolve()
    resolved = destination.expanduser()
    # Destination may not exist yet; resolve parents and rejoin the name.
    parent = resolved.parent.resolve()
    candidate = parent / resolved.name
    try:
        candidate.relative_to(final_root)
    except ValueError as exc:
        raise BuildError.staging_destination_escape(destination) from exc
    return candidate


def _assert_resource_tree_safe(source: Path) -> None:
    """Reject symlinks and non-file/non-directory entries in the resource tree."""
    if source.is_symlink() or not source.is_dir():
        raise BuildError.resource_source_invalid(source)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise BuildError.resource_source_invalid(path)
        if not path.is_file() and not path.is_dir():
            raise BuildError.resource_source_invalid(path)


def stage_resource_tree(
    layout: BuildLayout,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Copy the project resource tree into the staged final layout."""
    source = project_root / "resources"
    try:
        validate_resource_root(source)
    except AppError as exc:
        raise BuildError.resource_source_invalid(source) from exc
    _assert_resource_tree_safe(source)
    destination = _destination_within_final_root(layout, layout.resource_root)
    if destination.exists() or destination.is_symlink():
        raise BuildError.resource_destination_exists(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, destination)
    except OSError as exc:
        raise BuildError.resource_copy_failed(source, destination, str(exc)) from exc
    try:
        validate_resource_root(destination)
    except AppError as exc:
        raise BuildError.resource_source_invalid(destination) from exc
    _assert_resource_tree_safe(destination)


def stage_release_documents(
    layout: BuildLayout,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Copy release documentation into the deterministic platform layout."""
    sources = (
        (project_root / "README.md", layout.readme_path),
        (project_root / "THIRD_PARTY_NOTICES.md", layout.notice_path),
    )
    for source, destination in sources:
        _require_regular_source_file(source)
        target = _destination_within_final_root(layout, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise BuildError.staging_copy_failed(source, target, str(exc)) from exc


def stage_artifact(
    layout: BuildLayout,
    artifact: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Stage one Nuitka artifact into the standardized final location.

    Prefer a same-filesystem move over a full tree copy so standalone builds do
    not pay a second multi-hundred-megabyte directory walk. Cross-device failures
    fall back to ``shutil.move``'s copy-and-remove path. Application resources
    and release documentation are staged explicitly after the move so platform
    layout does not depend on Nuitka data-file placement.
    """
    safe_remove_owned(layout.final_root, (layout.work_root, layout.final_root))
    layout.final_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(artifact), str(layout.final_root))
    stage_resource_tree(layout, project_root=project_root)
    stage_release_documents(layout, project_root=project_root)
    verify_layout(layout)


def verify_layout(layout: BuildLayout) -> None:
    """Verify the executable and staged resources needed by smoke tests."""
    required = (
        layout.executable_path,
        layout.readme_path,
        layout.notice_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError.missing_files(", ".join(missing))
    try:
        validate_resource_root(layout.resource_root)
    except AppError as exc:
        raise BuildError.missing_files(str(exc)) from exc


def build(
    version: str,
    *,
    clean: bool = False,
    platform_name: str | None = None,
    target: BuildTarget = "desktop",
) -> BuildLayout:
    """Compile, stage, and verify the local platform artifact."""
    validate_version(version)
    layout = layout_for_platform(platform_name, target=target)
    preflight_build(target, platform_name=layout.platform_name)
    if clean:
        clean_owned_paths(layout)
    layout.work_root.mkdir(parents=True, exist_ok=True)
    command = build_command(version, layout)
    print("==> " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        env=build_environment(),
    )
    artifact = find_unique_artifact(layout.work_root, artifact_suffix(layout))
    stage_artifact(layout, artifact)
    print(f"Nuitka output: {layout.final_root}")
    return layout


def main(argv: list[str] | None = None) -> int:
    """Parse build options and run the wrapper."""
    parser = argparse.ArgumentParser(description="Build Captioner standalone with Nuitka")
    parser.add_argument(
        "--clean", action="store_true", help="Clean owned staging/output paths first"
    )
    parser.add_argument(
        "--target",
        choices=("cli", "desktop"),
        default="desktop",
        help="Build target: cli or desktop (default: desktop)",
    )
    parser.add_argument("--version", default="0.0.0", help="Package version")
    namespace = parser.parse_args(argv)
    try:
        build(namespace.version, clean=namespace.clean, target=namespace.target)
    except (BuildError, ValueError, subprocess.CalledProcessError, OSError) as exc:
        print(f"Nuitka build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
