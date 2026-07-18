"""Host fact normalization at the adapter boundary."""

from __future__ import annotations

import os
import platform as platform_module
import subprocess
import sys
from collections.abc import Callable

from captioner.core.application.runtime_selection import HostFacts
from captioner.core.domain.errors import AppError


def normalize_platform(value: str) -> str:
    normalized = value.strip().casefold()
    mapping = {
        "darwin": "macos",
        "macos": "macos",
        "win32": "windows",
        "windows": "windows",
        "linux": "linux",
    }
    result = mapping.get(normalized)
    if result is None:
        raise AppError("runtime.host_unsupported", {"field": "platform"})
    return result


def normalize_architecture(value: str) -> str:
    normalized = value.strip().casefold()
    mapping = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "arm64e": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "x86-64": "x86_64",
    }
    result = mapping.get(normalized)
    if result is None:
        raise AppError("runtime.host_unsupported", {"field": "architecture"})
    return result


class HostProbe:
    """Probe and normalize the current process host without Domain guessing."""

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform_module.system,
        machine: Callable[[], str] = platform_module.machine,
        mac_version: Callable[[], str] | None = None,
        environment: dict[str, str] | None = None,
        translated_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._system = system
        self._machine = machine
        self._mac_version = mac_version or (lambda: platform_module.mac_ver()[0])
        self._environment = os.environ if environment is None else environment
        self._translated_probe = translated_probe or _is_rosetta_translated

    def probe(self) -> HostFacts:
        normalized_platform = normalize_platform(self._system())
        normalized_architecture = normalize_architecture(
            self._environment.get("PROCESSOR_ARCHITEW6432", self._machine())
        )
        os_version = _os_version(normalized_platform, self._mac_version)
        native = not (
            normalized_platform == "macos"
            and normalized_architecture == "arm64"
            and self._translated_probe()
        )
        return HostFacts(
            platform=normalized_platform,
            architecture=normalized_architecture,
            os_version=os_version,
            native_architecture=native,
        )


def probe_host_facts() -> HostFacts:
    return HostProbe().probe()


def _os_version(normalized_platform: str, mac_version: Callable[[], str]) -> str:
    if normalized_platform == "macos":
        raw = mac_version()
    elif normalized_platform == "windows":
        raw = platform_module.win32_ver()[0]
    else:
        raw = platform_module.release()
    parts = [part for part in raw.split(".") if part.isdigit()]
    return ".".join(parts[:3]) if parts else "0.0.0"


def _is_rosetta_translated() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-in", "sysctl.proc_translated"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "1"


__all__ = [
    "HostProbe",
    "normalize_architecture",
    "normalize_platform",
    "probe_host_facts",
]
