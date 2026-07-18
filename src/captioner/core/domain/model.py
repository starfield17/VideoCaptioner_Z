"""Model source, identity, manifest, and installation contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


def _empty_json_mapping() -> dict[str, JsonValue]:
    return {}


class ModelSourceKind(StrEnum):
    """Sources supported by the Phase 6 source boundary."""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    LOCAL_IMPORT = "local-import"
    EXTERNAL_PATH = "external-path"


class ModelState(StrEnum):
    """Lifecycle and ownership states for a model record."""

    STAGED = "staged"
    INSTALLED = "installed"
    LOAD_VERIFIED = "load_verified"
    FAILED = "failed"
    EXTERNAL_UNMANAGED = "external_unmanaged"


ModelStatus = ModelState


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Stable model identity without a machine-local filesystem path."""

    backend_id: str
    source_id: str
    repository_id: str
    revision: str
    model_format: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.backend_id, "backend_id", "model.identity_invalid")
        _require_text(self.source_id, "source_id", "model.identity_invalid")
        _require_repository_id(self.repository_id)
        _require_text(self.revision, "revision", "model.identity_invalid")
        _require_text(self.model_format, "model_format", "model.identity_invalid")
        _require_sha256(self.manifest_sha256, "manifest_sha256", "model.identity_invalid")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "source_id": self.source_id,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "model_format": self.model_format,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelFileEntry:
    """One model file relative to the installed model directory."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_posix_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise AppError("model.file_invalid", {"field": "size_bytes"})
        _require_sha256(self.sha256, "sha256", "model.file_invalid")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


ModelManifestFile = ModelFileEntry
ModelFile = ModelFileEntry


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Static model metadata and its content manifest."""

    schema_version: int
    identity: ModelIdentity
    display_name: str
    files: tuple[ModelFileEntry, ...]
    compatible_runtime_backends: tuple[str, ...]
    model_format: str
    source_metadata: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)
    description: str = ""
    required_capabilities: tuple[str, ...] = ()
    required_device_kind: str | None = None
    required_platform: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise AppError("model.manifest_invalid", {"field": "schema_version"})
        _require_text(self.display_name, "display_name", "model.manifest_invalid")
        description = cast(object, self.description)
        if not isinstance(description, str) or (description and description != description.strip()):
            raise AppError("model.manifest_invalid", {"field": "description"})
        _require_text(self.model_format, "model_format", "model.manifest_invalid")
        if self.identity.model_format != self.model_format:
            raise AppError("model.manifest_invalid", {"field": "model_format"})
        backends = tuple(self.compatible_runtime_backends)
        if not backends or any(
            not isinstance(cast(object, value), str) or not value.strip() or value != value.strip()
            for value in backends
        ):
            raise AppError("model.manifest_invalid", {"field": "compatible_runtime_backends"})
        if self.identity.backend_id not in backends:
            raise AppError("model.manifest_invalid", {"field": "compatible_runtime_backends"})
        if len(set(backends)) != len(backends):
            raise AppError(
                "model.manifest_invalid",
                {"field": "compatible_runtime_backends", "reason": "duplicate"},
            )
        files = tuple(self.files)
        if not files:
            raise AppError("model.manifest_invalid", {"field": "files"})
        paths = tuple(entry.relative_path for entry in files)
        if len(set(paths)) != len(paths):
            raise AppError("model.manifest_invalid", {"field": "files", "reason": "duplicate"})
        capabilities = tuple(self.required_capabilities)
        if any(
            not isinstance(cast(object, value), str) or not value.strip() or value != value.strip()
            for value in capabilities
        ):
            raise AppError("model.manifest_invalid", {"field": "required_capabilities"})
        _optional_text(self.required_device_kind, "required_device_kind")
        _optional_text(self.required_platform, "required_platform")
        if self.required_device_kind == "auto":
            raise AppError("model.manifest_invalid", {"field": "required_device_kind"})
        frozen_metadata = _freeze_metadata(self.source_metadata)
        if self.model_format == "mlx-whisper":
            _validate_mlx_required_files(paths)
        object.__setattr__(self, "compatible_runtime_backends", backends)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "source_metadata", frozen_metadata)

    def has_file(self, relative_path: str) -> bool:
        return relative_path in {entry.relative_path for entry in self.files}

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "display_name": self.display_name,
            "description": self.description,
            "files": [entry.to_dict() for entry in self.files],
            "compatible_runtime_backends": list(self.compatible_runtime_backends),
            "model_format": self.model_format,
            "source_metadata": _thaw_metadata(self.source_metadata),
            "required_capabilities": list(self.required_capabilities),
            "required_device_kind": self.required_device_kind,
            "required_platform": self.required_platform,
        }


@dataclass(frozen=True, slots=True)
class ModelInstallation:
    """A managed or external local model directory projection."""

    identity: ModelIdentity
    manifest: ModelManifest
    model_directory: Path
    state: ModelState = ModelState.STAGED
    managed: bool | None = None
    load_verified: bool = False

    def __post_init__(self) -> None:
        if self.manifest.identity != self.identity:
            raise AppError("model.installation_invalid", {"field": "identity"})
        if not self.model_directory.is_absolute():
            raise AppError("model.installation_invalid", {"field": "model_directory"})
        managed = (
            self.state is not ModelState.EXTERNAL_UNMANAGED
            if self.managed is None
            else self.managed
        )
        if type(managed) is not bool or type(self.load_verified) is not bool:
            raise AppError("model.installation_invalid", {"field": "ownership"})
        if managed and self.state is ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.installation_invalid", {"field": "managed"})
        if not managed and self.state is not ModelState.EXTERNAL_UNMANAGED:
            raise AppError("model.installation_invalid", {"field": "state"})
        verified = self.state is ModelState.LOAD_VERIFIED or self.load_verified
        object.__setattr__(self, "managed", managed)
        object.__setattr__(self, "load_verified", verified)

    @property
    def is_load_verified(self) -> bool:
        return self.load_verified

    @property
    def can_delete_files(self) -> bool:
        return self.managed is True and self.state is not ModelState.EXTERNAL_UNMANAGED


@dataclass(frozen=True, slots=True)
class ModelSourceCapabilities:
    """Operations a Model Source can perform without implying downloads."""

    search: bool
    exact_repository: bool
    local_directory: bool = False
    unmanaged_local_directory: bool = False


@dataclass(frozen=True, slots=True)
class ModelSourceCandidate:
    """Safe source metadata returned by search or exact resolution."""

    identity: ModelIdentity
    display_name: str
    description: str = ""

    def __post_init__(self) -> None:
        _require_text(self.display_name, "display_name", "model.source_result_invalid")
        if self.description and self.description != self.description.strip():
            raise AppError("model.source_result_invalid", {"field": "description"})


@dataclass(frozen=True, slots=True)
class ModelValidationCheck:
    """One safe static model validation check."""

    name: str
    ok: bool
    error_code: str | None = None
    message_code: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name", "model.validation_invalid")
        if type(self.ok) is not bool:
            raise AppError("model.validation_invalid", {"field": "ok"})
        _optional_text(self.error_code, "error_code")
        _optional_text(self.message_code, "message_code")


@dataclass(frozen=True, slots=True)
class ModelValidationReport:
    """Result projection returned by a Model Validator Port."""

    ok: bool
    checks: tuple[ModelValidationCheck, ...]
    error_code: str | None = None
    message_code: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise AppError("model.validation_invalid", {"field": "ok"})
        checks = tuple(self.checks)
        if not checks:
            raise AppError("model.validation_invalid", {"field": "checks"})
        if self.ok and any(not check.ok for check in checks):
            raise AppError("model.validation_invalid", {"field": "ok", "reason": "checks"})
        _optional_text(self.error_code, "error_code")
        _optional_text(self.message_code, "message_code")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "details", _freeze_metadata(self.details))


def required_files_for_format(model_format: str) -> tuple[frozenset[str], ...]:
    """Return alternative required-file groups for one model format."""
    if model_format == "mlx-whisper":
        return (
            frozenset({"config.json"}),
            frozenset({"model.safetensors", "weights.safetensors", "weights.npz"}),
        )
    return ()


def _validate_mlx_required_files(paths: tuple[str, ...]) -> None:
    path_set = set(paths)
    groups = required_files_for_format("mlx-whisper")
    if any(not group <= path_set for group in groups[:1]) or not groups[1] & path_set:
        raise AppError("model.manifest_invalid", {"field": "files", "reason": "mlx_required"})


def _require_text(value: object, field: str, code: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AppError(code, {"field": field})


def _require_repository_id(value: object) -> None:
    _require_text(value, "repository_id", "model.identity_invalid")
    assert isinstance(value, str)
    if (
        "\\" in value
        or ".." in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AppError("model.identity_invalid", {"field": "repository_id"})


def _require_sha256(value: object, field: str, code: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AppError(code, {"field": field})


def _validate_relative_posix_path(value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise AppError("model.file_invalid", {"field": "relative_path"})
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AppError("model.file_invalid", {"field": "relative_path"})


def _optional_text(value: object, field: str) -> None:
    if value is not None:
        _require_text(value, field, "model.manifest_invalid")


def _freeze_metadata(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AppError("model.metadata_invalid", {"reason": "object"})
    mapping = cast(Mapping[str, JsonValue], value)
    if _contains_sensitive_key(mapping):
        raise AppError("model.metadata_invalid", {"reason": "sensitive_key"})
    try:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(mapping))
    except (TypeError, ValueError) as exc:
        raise AppError("model.metadata_invalid", {"reason": "json"}) from exc
    return cast(Mapping[str, JsonValue], frozen)


def _thaw_metadata(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    frozen = cast(Mapping[str, FrozenJsonValue], value)
    thawed = thaw_json_value(frozen)
    if not isinstance(thawed, dict):
        raise AppError("model.metadata_invalid", {"reason": "object"})
    return cast(dict[str, JsonValue], thawed)


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        return any(
            key.lower().replace("-", "_") in _SENSITIVE_KEYS or _contains_sensitive_key(item)
            for key, item in raw.items()
            if isinstance(key, str)
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_sensitive_key(item) for item in sequence)
    return False


__all__ = [
    "ModelFile",
    "ModelFileEntry",
    "ModelIdentity",
    "ModelInstallation",
    "ModelManifest",
    "ModelManifestFile",
    "ModelSourceCandidate",
    "ModelSourceCapabilities",
    "ModelSourceKind",
    "ModelState",
    "ModelStatus",
    "ModelValidationCheck",
    "ModelValidationReport",
    "required_files_for_format",
]
