"""Read-only resource and OS-standard writable path resolution."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir, user_log_dir

APP_NAME = "Captioner"


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
    def writable_directories(self) -> tuple[Path, ...]:
        return (self.config_dir, self.data_dir, self.cache_dir, self.log_dir, self.temp_dir)


def resolve_app_paths(
    *,
    app_name: str = APP_NAME,
    platform_name: str | None = None,
    home_dir: Path | None = None,
    base_dir: Path | None = None,
    executable_path: Path | None = None,
    compiled: bool | None = None,
    resource_root_override: Path | None = None,
) -> AppPaths:
    """Resolve bundle resources and writable directories.

    ``base_dir`` and ``resource_root_override`` are deterministic test seams.
    They never change the production default, which uses ``platformdirs``.
    """
    normalized_platform = _normalize_platform(platform_name or sys.platform)
    executable = (executable_path or Path(sys.executable)).expanduser().resolve()
    is_compiled = _is_compiled() if compiled is None else compiled
    resource_root = _resolve_resource_root(
        normalized_platform,
        executable,
        is_compiled,
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


def ensure_runtime_layout(paths: AppPaths) -> None:
    """Create only writable directories; bundled resources remain untouched."""
    for directory in paths.writable_directories:
        directory.mkdir(parents=True, exist_ok=True)
    paths.batches_dir.mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / ".incoming").mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "sha256").mkdir(parents=True, exist_ok=True)


def _normalize_platform(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("win"):
        return "win32"
    if lowered in {"darwin", "mac", "macos"}:
        return "darwin"
    return "linux"


def _is_compiled() -> bool:
    return bool(getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None))


def _resolve_resource_root(
    platform_name: str,
    executable: Path,
    compiled: bool,
    override: Path | None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()

    candidates: list[Path] = []
    if compiled:
        executable_dir = executable.parent
        candidates.append(executable_dir / "resources")
        candidates.append(executable_dir / "Resources" / "resources")
        if platform_name == "darwin":
            contents_dir = executable_dir.parent
            candidates.append(contents_dir / "Resources" / "resources")
            candidates.append(contents_dir / "Resources")
            candidates.append(contents_dir.parent / "Resources" / "resources")
    else:
        module_path = Path(__file__).resolve()
        candidates.extend(parent / "resources" for parent in module_path.parents)
        candidates.append(Path.cwd() / "resources")

    unique_candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in unique_candidates:
        if (candidate / "i18n").is_dir():
            return candidate
    return unique_candidates[0]


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
        return {
            "config_dir": Path(user_config_dir(app_name, appauthor=False)),
            "data_dir": Path(user_data_dir(app_name, appauthor=False)),
            "cache_dir": Path(user_cache_dir(app_name, appauthor=False)),
            "log_dir": Path(user_log_dir(app_name, appauthor=False)),
            "temp_dir": Path(user_cache_dir(app_name, appauthor=False)) / "tmp",
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
