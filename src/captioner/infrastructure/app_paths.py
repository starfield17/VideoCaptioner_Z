"""Read-only resource and OS-standard writable path resolution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from platformdirs import user_cache_dir, user_config_dir, user_data_dir, user_log_dir

from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier
from captioner.core.domain.result import JsonValue

APP_NAME = "Captioner"

# Nuitka exposes this module attribute when this module is compiled.
try:
    _nuitka_compiled = cast(object, __compiled__)  # type: ignore[name-defined]
except NameError:
    _nuitka_compiled: object | None = None


@dataclass(frozen=True, slots=True)
class CompiledRuntime:
    """The runtime identity needed to resolve read-only bundled resources."""

    compiled: bool
    containing_dir: Path | None


_REQUIRED_RESOURCE_DIRECTORIES = ("i18n", "prompts", "runtime", "tokenizers")
_REQUIRED_RESOURCE_FILES = (
    Path("i18n") / "en.json",
    Path("tokenizers") / "tokenizer-manifest.json",
    Path("tokenizers") / "cl100k_base.tiktoken",
    Path("tokenizers") / "o200k_base.tiktoken",
)


@dataclass(frozen=True, slots=True)
class AppPaths:
    """All paths used by the application, separated by mutability."""

    app_name: str
    resource_root: Path
    i18n_resource_dir: Path
    prompt_resource_dir: Path
    runtime_manifest_resource_dir: Path
    config_dir: Path
    data_dir: Path
    cache_dir: Path
    log_dir: Path
    temp_dir: Path

    @property
    def batches_dir(self) -> Path:
        return self.data_dir / "batches"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def runtimes_dir(self) -> Path:
        return self.data_dir / "runtimes"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def tokenizer_resource_dir(self) -> Path:
        return self.resource_root / "tokenizers"

    @property
    def tokenizer_manifest_path(self) -> Path:
        return self.tokenizer_resource_dir / "tokenizer-manifest.json"

    @property
    def writable_directories(self) -> tuple[Path, ...]:
        return (
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.log_dir,
            self.temp_dir,
            self.batches_dir,
            self.artifacts_dir,
            self.artifacts_dir / ".incoming",
            self.artifacts_dir / "sha256",
            self.models_dir,
            self.runtimes_dir,
            self.workspaces_dir,
            self.downloads_dir,
            self.staging_dir,
        )


def resolve_app_paths(
    *,
    app_name: str = APP_NAME,
    platform_name: str | None = None,
    home_dir: Path | None = None,
    base_dir: Path | None = None,
    executable_path: Path | None = None,
    compiled: bool | None = None,
    compiled_runtime: CompiledRuntime | None = None,
    resource_root_override: Path | None = None,
) -> AppPaths:
    """Resolve bundle resources and writable directories.

    ``base_dir`` and ``resource_root_override`` are deterministic test seams.
    They never change the production default, which uses ``platformdirs``.
    """
    normalized_platform = _normalize_platform(platform_name or sys.platform)
    executable = (executable_path or Path(sys.executable)).expanduser().resolve()
    runtime = _compiled_runtime() if compiled_runtime is None else compiled_runtime
    if compiled is not None:
        runtime = CompiledRuntime(compiled, executable.parent if compiled else None)
    resource_root = _resolve_resource_root(
        normalized_platform,
        executable,
        runtime,
        resource_root_override,
    )
    writable = _resolve_writable_dirs(
        app_name,
        normalized_platform,
        home_dir=home_dir,
        base_dir=base_dir,
    )
    return AppPaths(
        app_name=app_name,
        resource_root=resource_root,
        i18n_resource_dir=resource_root / "i18n",
        prompt_resource_dir=resource_root / "prompts",
        runtime_manifest_resource_dir=resource_root / "runtime",
        **writable,
    )


def compiled_runtime_from_descriptor(value: object) -> CompiledRuntime:
    """Convert Nuitka's injected descriptor at the compiled entry boundary."""
    containing_dir = getattr(value, "containing_dir", None)
    if not isinstance(containing_dir, str) or not containing_dir:
        raise AppError("runtime.resource_root_invalid", {"reason": "compiled_containing_dir"})
    return CompiledRuntime(True, Path(containing_dir).expanduser().resolve())


def ensure_runtime_layout(paths: AppPaths) -> None:
    """Create only writable directories; bundled resources remain untouched."""
    for directory in paths.writable_directories:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_safe_child(root: Path, identifier: str, *, field: str) -> Path:
    """Resolve one validated identifier directly below ``root``."""
    validated = validate_identifier(identifier, field=field)
    resolved_root = root.expanduser().resolve()
    child = (resolved_root / validated).resolve()
    if child.parent != resolved_root:
        raise AppError("path.outside_runtime_root", {"field": field})
    return child


def _normalize_platform(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("win"):
        return "win32"
    if lowered in {"darwin", "mac", "macos"}:
        return "darwin"
    return "linux"


def _compiled_runtime() -> CompiledRuntime:
    """Read Nuitka's compile-time descriptor without importing Nuitka."""
    compiled = _nuitka_compiled
    if compiled is None:
        main_module = sys.modules.get("__main__")
        compiled = None if main_module is None else getattr(main_module, "__compiled__", None)
    if compiled is None:
        return CompiledRuntime(False, None)
    return compiled_runtime_from_descriptor(compiled)


def validate_resource_root(root: Path) -> Path:
    """Require the complete immutable resource layout used by the application."""
    resolved = root.expanduser().resolve()
    missing: list[JsonValue] = []
    for directory in _REQUIRED_RESOURCE_DIRECTORIES:
        if not (resolved / directory).is_dir():
            missing.append(directory)
    for relative_path in _REQUIRED_RESOURCE_FILES:
        if not (resolved / relative_path).is_file():
            missing.append(str(relative_path))
    if missing:
        raise AppError(
            "runtime.resource_root_invalid",
            {"missing": missing},
        )
    return resolved


def _resolve_resource_root(
    platform_name: str,
    executable: Path,
    runtime: CompiledRuntime,
    override: Path | None,
) -> Path:
    if override is not None:
        return validate_resource_root(override)

    candidates: list[Path] = []
    if runtime.compiled:
        containing_dir = runtime.containing_dir
        if containing_dir is None:
            raise AppError("runtime.resource_root_invalid", {"reason": "compiled_containing_dir"})
        if platform_name != "darwin" and containing_dir == executable.parent:
            candidates.append(containing_dir / "resources")
        if platform_name == "darwin":
            candidates.extend(
                (
                    containing_dir / "Resources" / "resources",
                    containing_dir.parent / "Resources" / "resources",
                    containing_dir / ".." / "Resources" / "resources",
                    containing_dir.parent / "Contents" / "Resources" / "resources",
                )
            )
        candidates.append(executable.parent / "resources")
        candidates.append(executable.parent.parent / "resources")
        if platform_name == "darwin":
            candidates.extend(
                (
                    executable.parent / "Resources" / "resources",
                    executable.parent.parent / "Resources" / "resources",
                    executable.parent / ".." / "Resources" / "resources",
                    executable.parent.parent / "Contents" / "Resources" / "resources",
                )
            )
    else:
        module_path = Path(__file__).resolve()
        candidates.extend(parent / "resources" for parent in module_path.parents)
        candidates.append(Path.cwd() / "resources")

    unique_candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in unique_candidates:
        try:
            return validate_resource_root(candidate)
        except AppError:
            # Candidate probing is explicit; only a complete root is accepted.
            continue
    raise AppError(
        "runtime.resource_root_invalid",
        {"reason": "no_complete_candidate", "compiled": runtime.compiled},
    )


def _resolve_writable_dirs(
    app_name: str,
    platform_name: str,
    *,
    home_dir: Path | None,
    base_dir: Path | None,
) -> dict[str, Path]:
    if base_dir is not None:
        root = base_dir.expanduser().resolve()
        return {
            "config_dir": root / "config",
            "data_dir": root / "data",
            "cache_dir": root / "cache",
            "log_dir": root / "log",
            "temp_dir": root / "temp",
        }
    if home_dir is None:
        config_root = Path(user_config_dir(app_name, appauthor=False))
        data_root = Path(user_data_dir(app_name, appauthor=False))
        if platform_name == "darwin":
            # platformdirs intentionally maps both macOS roots to the same
            # Application Support directory. Keep the application's config
            # and durable data namespaces explicit, matching the injected-home
            # seam and avoiding one mixed-purpose writable root.
            config_root /= "config"
            data_root /= "data"
        cache_root = Path(user_cache_dir(app_name, appauthor=False))
        return {
            "config_dir": config_root,
            "data_dir": data_root,
            "cache_dir": cache_root,
            "log_dir": Path(user_log_dir(app_name, appauthor=False)),
            "temp_dir": cache_root / "tmp",
        }

    home = home_dir.expanduser().resolve()
    if platform_name == "win32":
        roaming = home / "AppData" / "Roaming" / app_name
        local = home / "AppData" / "Local" / app_name
        return {
            "config_dir": roaming,
            "data_dir": local / "data",
            "cache_dir": local / "cache",
            "log_dir": local / "log",
            "temp_dir": local / "temp",
        }
    if platform_name == "darwin":
        application_support = home / "Library" / "Application Support" / app_name
        cache = home / "Library" / "Caches" / app_name
        return {
            "config_dir": application_support / "config",
            "data_dir": application_support / "data",
            "cache_dir": cache,
            "log_dir": home / "Library" / "Logs" / app_name,
            "temp_dir": cache / "tmp",
        }
    cache = home / ".cache" / app_name
    return {
        "config_dir": home / ".config" / app_name,
        "data_dir": home / ".local" / "share" / app_name,
        "cache_dir": cache,
        "log_dir": home / ".local" / "state" / app_name / "log",
        "temp_dir": cache / "tmp",
    }
