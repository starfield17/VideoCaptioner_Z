"""Immutable Runtime archive package descriptors.

The descriptor is intentionally kept outside the archive.  The archive hash
therefore describes exactly the bytes that are installed and cannot become a
self-referential manifest value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.domain.runtime import RuntimeManifest


@dataclass(frozen=True, slots=True)
class RuntimePackageDescriptor:
    """Sidecar metadata for one Runtime archive."""

    package_schema_version: int
    archive_filename: str
    archive_size_bytes: int
    runtime_manifest: RuntimeManifest

    def __post_init__(self) -> None:
        if type(self.package_schema_version) is not int or self.package_schema_version <= 0:
            raise AppError("runtime.package_invalid", {"field": "package_schema_version"})
        if (
            not self.archive_filename
            or self.archive_filename != self.archive_filename.strip()
            or "/" in self.archive_filename
            or "\\" in self.archive_filename
            or PureWindowsPath(self.archive_filename).drive
            or PureWindowsPath(self.archive_filename).is_absolute()
            or self.archive_filename in {".", ".."}
            or any(
                ord(character) < 32 or ord(character) == 127 for character in self.archive_filename
            )
            or not self.archive_filename.endswith(".tar.gz")
        ):
            raise AppError("runtime.package_invalid", {"field": "archive_filename"})
        if type(self.archive_size_bytes) is not int or self.archive_size_bytes <= 0:
            raise AppError("runtime.package_invalid", {"field": "archive_size_bytes"})

    @property
    def archive_sha256(self) -> str:
        """Return the archive digest declared by the nested manifest."""
        return self.runtime_manifest.archive_sha256

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "package_schema_version": self.package_schema_version,
            "archive_filename": self.archive_filename,
            "archive_size_bytes": self.archive_size_bytes,
            "runtime_manifest": self.runtime_manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimePackageDescriptor:
        if not isinstance(value, Mapping):
            raise AppError("runtime.package_invalid", {"field": "root"})
        if _contains_sensitive_key(cast(Mapping[object, object], value)):
            raise AppError("runtime.package_invalid", {"reason": "sensitive"})
        raw = cast(Mapping[object, object], value)
        return cls(
            package_schema_version=_required_int(raw, "package_schema_version"),
            archive_filename=_required_string(raw, "archive_filename"),
            archive_size_bytes=_required_int(raw, "archive_size_bytes"),
            runtime_manifest=RuntimeManifest.from_dict(_required_value(raw, "runtime_manifest")),
        )


def _required_value(value: Mapping[object, object], key: str) -> object:
    if key not in value:
        raise AppError("runtime.package_invalid", {"field": key})
    return value[key]


def _required_string(value: Mapping[object, object], key: str) -> str:
    item = _required_value(value, key)
    if not isinstance(item, str):
        raise AppError("runtime.package_invalid", {"field": key})
    return item


def _required_int(value: Mapping[object, object], key: str) -> int:
    item = _required_value(value, key)
    if type(item) is not int:
        raise AppError("runtime.package_invalid", {"field": key})
    return item


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, nested in mapping.items():
            if isinstance(key, str) and any(
                marker in key.casefold()
                for marker in (
                    "token",
                    "secret",
                    "password",
                    "credential",
                    "authorization",
                    "api_key",
                    "apikey",
                )
            ):
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_sensitive_key(item) for item in sequence)
    return False


__all__ = ["RuntimePackageDescriptor"]
