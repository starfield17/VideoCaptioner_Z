"""Immutable Runtime identities, manifests, installation states, and reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from captioner.core.domain.asr_backend import BackendCapability
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_RUNTIME_PLATFORMS = frozenset({"macos", "windows", "linux"})
SUPPORTED_RUNTIME_ARCHITECTURES = frozenset({"arm64", "x86_64"})
SUPPORTED_RUNTIME_DEVICES = frozenset({"cpu", "cuda", "metal"})


def _empty_json_mapping() -> dict[str, JsonValue]:
    return {}


class RuntimeState(StrEnum):
    """Lifecycle and ownership states for a Runtime installation record."""

    NOT_INSTALLED = "not_installed"
    STAGED = "staged"
    INSTALLED = "installed"
    AVAILABLE = "available"
    FAILED = "failed"
    EXTERNAL_UNMANAGED = "external_unmanaged"


RuntimeStatus = RuntimeState


class DoctorPhase(StrEnum):
    """The two validation layers defined for a Runtime."""

    STATIC = "static"
    ACTIVATION = "activation"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Stable Runtime identity, deliberately independent of install paths."""

    runtime_id: str
    version: str

    def __post_init__(self) -> None:
        _require_identifier(self.runtime_id, "runtime_id", "runtime.identity_invalid")
        _require_version(self.version, "version", "runtime.identity_invalid")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"runtime_id": self.runtime_id, "version": self.version}

    @classmethod
    def from_dict(cls, value: object) -> RuntimeIdentity:
        if not isinstance(value, Mapping):
            raise AppError("runtime.identity_invalid", {"field": "identity"})
        raw = cast(Mapping[object, object], value)
        runtime_id = raw.get("runtime_id")
        version = raw.get("version")
        if not isinstance(runtime_id, str):
            raise AppError("runtime.identity_invalid", {"field": "runtime_id"})
        if not isinstance(version, str):
            raise AppError("runtime.identity_invalid", {"field": "version"})
        return cls(runtime_id, version)


@dataclass(frozen=True, slots=True)
class RuntimeTarget:
    """Normalized host target encoded in a Runtime manifest."""

    platform: str
    architecture: str
    device_kind: str
    minimum_os_version: str

    def __post_init__(self) -> None:
        _require_text(self.platform, "platform", "runtime.target_invalid")
        _require_text(self.architecture, "architecture", "runtime.target_invalid")
        _require_text(self.device_kind, "device_kind", "runtime.target_invalid")
        if self.platform not in SUPPORTED_RUNTIME_PLATFORMS:
            raise AppError("runtime.target_invalid", {"field": "platform"})
        if self.architecture not in SUPPORTED_RUNTIME_ARCHITECTURES:
            raise AppError("runtime.target_invalid", {"field": "architecture"})
        if self.device_kind not in SUPPORTED_RUNTIME_DEVICES:
            raise AppError("runtime.target_invalid", {"field": "device_kind"})
        _require_version(self.minimum_os_version, "minimum_os_version", "runtime.target_invalid")

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the stable key used by an active-runtime pointer."""
        return (
            self.platform,
            self.architecture,
            self.device_kind,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "platform": self.platform,
            "architecture": self.architecture,
            "device_kind": self.device_kind,
            "minimum_os_version": self.minimum_os_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeTarget:
        if not isinstance(value, Mapping):
            raise AppError("runtime.target_invalid", {"field": "target"})
        raw = cast(Mapping[object, object], value)
        return cls(
            platform=_required_string(raw, "platform", "runtime.target_invalid"),
            architecture=_required_string(raw, "architecture", "runtime.target_invalid"),
            device_kind=_required_string(raw, "device_kind", "runtime.target_invalid"),
            minimum_os_version=_required_string(
                raw, "minimum_os_version", "runtime.target_invalid"
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeFileEntry:
    """One verified file relative to the Runtime root."""

    relative_path: str
    size_bytes: int
    sha256: str
    executable: bool

    def __post_init__(self) -> None:
        _validate_relative_posix_path(self.relative_path, "runtime.file_invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise AppError("runtime.file_invalid", {"field": "size_bytes"})
        _require_sha256(self.sha256, "sha256", "runtime.file_invalid")
        if type(self.executable) is not bool:
            raise AppError("runtime.file_invalid", {"field": "executable"})

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "executable": self.executable,
        }


# The longer name is useful at adapter boundaries while the shorter name is
# convenient in manifests and tests.
RuntimeManifestFile = RuntimeFileEntry
RuntimeFile = RuntimeFileEntry


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """Static, content-addressed description of one Runtime archive."""

    schema_version: int
    runtime_identity: RuntimeIdentity
    worker_protocol_version: str
    backend_id: str
    backend_version: str
    target: RuntimeTarget
    capabilities: BackendCapability
    supported_model_formats: tuple[str, ...]
    archive_sha256: str
    files: tuple[RuntimeFileEntry, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise AppError("runtime.manifest_invalid", {"field": "schema_version"})
        _require_version(
            self.worker_protocol_version,
            "worker_protocol_version",
            "runtime.manifest_invalid",
        )
        _require_text(self.backend_id, "backend_id", "runtime.manifest_invalid")
        _require_version(self.backend_version, "backend_version", "runtime.manifest_invalid")
        _require_sha256(self.archive_sha256, "archive_sha256", "runtime.manifest_invalid")
        if self.capabilities.backend_id != self.backend_id:
            raise AppError("runtime.manifest_invalid", {"field": "capabilities"})
        if self.capabilities.device_kind != self.target.device_kind:
            raise AppError("runtime.manifest_invalid", {"field": "target"})
        formats = tuple(self.supported_model_formats)
        if not formats or any(not value.strip() for value in formats):
            raise AppError("runtime.manifest_invalid", {"field": "supported_model_formats"})
        if len(set(formats)) != len(formats):
            raise AppError(
                "runtime.manifest_invalid",
                {"field": "supported_model_formats", "reason": "duplicate"},
            )
        if tuple(self.capabilities.supported_model_formats) != formats:
            raise AppError("runtime.manifest_invalid", {"field": "supported_model_formats"})
        files = tuple(self.files)
        if not files:
            raise AppError("runtime.manifest_invalid", {"field": "files"})
        paths = tuple(entry.relative_path for entry in files)
        if len(set(paths)) != len(paths):
            raise AppError("runtime.manifest_invalid", {"field": "files", "reason": "duplicate"})
        object.__setattr__(self, "supported_model_formats", formats)
        object.__setattr__(self, "files", files)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "runtime_identity": self.runtime_identity.to_dict(),
            "worker_protocol_version": self.worker_protocol_version,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "target": self.target.to_dict(),
            "capabilities": {
                "backend_id": self.capabilities.backend_id,
                "device_kind": self.capabilities.device_kind,
                "supported_model_formats": list(self.capabilities.supported_model_formats),
                "word_timestamps": self.capabilities.word_timestamps,
                "language_detection": self.capabilities.language_detection,
                "translation_task": self.capabilities.translation_task,
                "additional_capabilities": list(self.capabilities.additional_capabilities),
            },
            "supported_model_formats": list(self.supported_model_formats),
            "archive_sha256": self.archive_sha256,
            "files": [entry.to_dict() for entry in self.files],
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeManifest:
        """Decode a sidecar manifest without filling missing required fields."""
        if not isinstance(value, Mapping):
            raise AppError("runtime.manifest_invalid", {"field": "root"})
        raw = cast(Mapping[object, object], value)
        if _contains_sensitive_key(raw):
            raise AppError("runtime.manifest_invalid", {"reason": "sensitive"})
        capabilities_value = _required_mapping(raw, "capabilities", "runtime.manifest_invalid")
        capabilities = BackendCapability(
            backend_id=_required_string(
                capabilities_value, "backend_id", "runtime.manifest_invalid"
            ),
            device_kind=_required_string(
                capabilities_value, "device_kind", "runtime.manifest_invalid"
            ),
            supported_model_formats=_required_string_tuple(
                capabilities_value, "supported_model_formats", "runtime.manifest_invalid"
            ),
            word_timestamps=_required_bool(
                capabilities_value, "word_timestamps", "runtime.manifest_invalid"
            ),
            language_detection=_required_bool(
                capabilities_value, "language_detection", "runtime.manifest_invalid"
            ),
            translation_task=_required_bool(
                capabilities_value, "translation_task", "runtime.manifest_invalid"
            ),
            additional_capabilities=_required_string_tuple(
                capabilities_value, "additional_capabilities", "runtime.manifest_invalid"
            ),
        )
        files_value = _required_sequence(raw, "files", "runtime.manifest_invalid")
        files = tuple(
            RuntimeFileEntry(
                relative_path=_required_string(
                    _mapping_item(item, "file", "runtime.manifest_invalid"),
                    "relative_path",
                    "runtime.manifest_invalid",
                ),
                size_bytes=_required_int(
                    _mapping_item(item, "file", "runtime.manifest_invalid"),
                    "size_bytes",
                    "runtime.manifest_invalid",
                ),
                sha256=_required_string(
                    _mapping_item(item, "file", "runtime.manifest_invalid"),
                    "sha256",
                    "runtime.manifest_invalid",
                ),
                executable=_required_bool(
                    _mapping_item(item, "file", "runtime.manifest_invalid"),
                    "executable",
                    "runtime.manifest_invalid",
                ),
            )
            for item in files_value
        )
        return cls(
            schema_version=_required_int(raw, "schema_version", "runtime.manifest_invalid"),
            runtime_identity=RuntimeIdentity.from_dict(
                _required_value(raw, "runtime_identity", "runtime.manifest_invalid")
            ),
            worker_protocol_version=_required_string(
                raw, "worker_protocol_version", "runtime.manifest_invalid"
            ),
            backend_id=_required_string(raw, "backend_id", "runtime.manifest_invalid"),
            backend_version=_required_string(raw, "backend_version", "runtime.manifest_invalid"),
            target=RuntimeTarget.from_dict(
                _required_value(raw, "target", "runtime.manifest_invalid")
            ),
            capabilities=capabilities,
            supported_model_formats=_required_string_tuple(
                raw, "supported_model_formats", "runtime.manifest_invalid"
            ),
            archive_sha256=_required_string(raw, "archive_sha256", "runtime.manifest_invalid"),
            files=files,
        )


@dataclass(frozen=True, slots=True)
class RuntimeInstallation:
    """A registered Runtime and its current lifecycle/health projection."""

    identity: RuntimeIdentity
    manifest: RuntimeManifest
    install_path: Path
    state: RuntimeState = RuntimeState.NOT_INSTALLED
    managed: bool | None = None
    doctor_passed: bool | None = None

    def __post_init__(self) -> None:
        if self.manifest.runtime_identity != self.identity:
            raise AppError("runtime.installation_invalid", {"field": "identity"})
        if not self.install_path.is_absolute():
            raise AppError("runtime.installation_invalid", {"field": "install_path"})
        managed = (
            self.state is not RuntimeState.EXTERNAL_UNMANAGED
            if self.managed is None
            else self.managed
        )
        if managed and self.state is RuntimeState.EXTERNAL_UNMANAGED:
            raise AppError("runtime.installation_invalid", {"field": "managed"})
        if not managed and self.state is not RuntimeState.EXTERNAL_UNMANAGED:
            raise AppError("runtime.installation_invalid", {"field": "state"})
        doctor_passed = (
            self.state is RuntimeState.AVAILABLE
            if self.doctor_passed is None
            else self.doctor_passed
        )
        if type(managed) is not bool or type(doctor_passed) is not bool:
            raise AppError("runtime.installation_invalid", {"field": "health"})
        if self.state is RuntimeState.EXTERNAL_UNMANAGED:
            if managed:
                raise AppError("runtime.installation_invalid", {"field": "managed"})
        elif self.state is RuntimeState.AVAILABLE:
            if not managed or not doctor_passed:
                raise AppError("runtime.installation_invalid", {"field": "health"})
        elif doctor_passed:
            raise AppError("runtime.installation_invalid", {"field": "health"})
        object.__setattr__(self, "managed", managed)
        object.__setattr__(self, "doctor_passed", doctor_passed)

    @property
    def is_available(self) -> bool:
        """Return whether this record passed activation checks."""
        return (self.state is RuntimeState.AVAILABLE and self.doctor_passed is True) or (
            self.state is RuntimeState.EXTERNAL_UNMANAGED and self.doctor_passed is True
        )

    @property
    def can_delete_files(self) -> bool:
        return self.managed is True and self.state is not RuntimeState.EXTERNAL_UNMANAGED

    @property
    def path(self) -> Path:
        """Compatibility alias for adapters that call the installation root a path."""
        return self.install_path


@dataclass(frozen=True, slots=True)
class ActiveRuntimePointer:
    """One atomically persisted backend/target active slot."""

    backend_id: str
    target: RuntimeTarget
    current: RuntimeIdentity | None = None
    previous: RuntimeIdentity | None = None
    pending_activation: RuntimeIdentity | None = None

    def __post_init__(self) -> None:
        _require_text(self.backend_id, "backend_id", "runtime.active_pointer_invalid")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "target": self.target.to_dict(),
            "current": None if self.current is None else self.current.to_dict(),
            "previous": None if self.previous is None else self.previous.to_dict(),
            "pending_activation": (
                None if self.pending_activation is None else self.pending_activation.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> ActiveRuntimePointer:
        if not isinstance(value, Mapping):
            raise AppError("runtime.active_pointer_invalid", {"field": "slot"})
        raw = cast(Mapping[object, object], value)
        return cls(
            backend_id=_required_string(raw, "backend_id", "runtime.active_pointer_invalid"),
            target=RuntimeTarget.from_dict(
                _required_value(raw, "target", "runtime.active_pointer_invalid")
            ),
            current=_optional_identity(raw, "current"),
            previous=_optional_identity(raw, "previous"),
            pending_activation=_optional_identity(raw, "pending_activation"),
        )


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One safe, structured Runtime Doctor check."""

    name: str
    ok: bool
    error_code: str | None = None
    message_code: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        _require_text(self.name, "name", "runtime.doctor_invalid")
        if type(self.ok) is not bool:
            raise AppError("runtime.doctor_invalid", {"field": "ok"})
        _optional_text(self.error_code, "error_code", "runtime.doctor_invalid")
        _optional_text(self.message_code, "message_code", "runtime.doctor_invalid")
        object.__setattr__(self, "details", _freeze_details(self.details, "runtime.doctor_invalid"))


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Result of either static or activation Runtime Doctor."""

    ok: bool
    phase: str
    checks: tuple[DoctorCheck, ...]
    error_code: str | None = None
    message_code: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise AppError("runtime.doctor_invalid", {"field": "ok"})
        if self.phase not in {phase.value for phase in DoctorPhase}:
            raise AppError("runtime.doctor_invalid", {"field": "phase"})
        checks = tuple(self.checks)
        if not checks:
            raise AppError("runtime.doctor_invalid", {"field": "checks"})
        if self.ok and any(not check.ok for check in checks):
            raise AppError("runtime.doctor_invalid", {"field": "ok", "reason": "checks"})
        _optional_text(self.error_code, "error_code", "runtime.doctor_invalid")
        _optional_text(self.message_code, "message_code", "runtime.doctor_invalid")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "details", _freeze_details(self.details, "runtime.doctor_invalid"))


def _require_text(value: object, field: str, code: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AppError(code, {"field": field})


def _require_identifier(value: object, field: str, code: str) -> None:
    _require_text(value, field, code)
    assert isinstance(value, str)
    if (
        "/" in value
        or "\\" in value
        or ".." in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AppError(code, {"field": field})


def _require_version(value: object, field: str, code: str) -> None:
    _require_text(value, field, code)
    assert isinstance(value, str)
    if re.fullmatch(r"\d+(?:\.\d+)+", value) is None:
        raise AppError(code, {"field": field})


def _require_sha256(value: object, field: str, code: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AppError(code, {"field": field})


def _validate_relative_posix_path(value: object, code: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise AppError(code, {"field": "relative_path"})
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AppError(code, {"field": "relative_path"})


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        for key, nested in raw.items():
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


def _optional_text(value: object, field: str, code: str) -> None:
    if value is not None:
        _require_text(value, field, code)


def _freeze_details(value: object, code: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AppError(code, {"field": "details"})
    mapping = cast(Mapping[str, JsonValue], value)
    try:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(mapping))
    except (TypeError, ValueError) as exc:
        raise AppError(code, {"field": "details"}) from exc
    return cast(Mapping[str, JsonValue], frozen)


def _required_value(value: Mapping[object, object], key: str, code: str) -> object:
    if key not in value:
        raise AppError(code, {"field": key})
    return value[key]


def _required_string(value: Mapping[object, object], key: str, code: str) -> str:
    item = _required_value(value, key, code)
    if not isinstance(item, str):
        raise AppError(code, {"field": key})
    return item


def _required_int(value: Mapping[object, object], key: str, code: str) -> int:
    item = _required_value(value, key, code)
    if type(item) is not int:
        raise AppError(code, {"field": key})
    return item


def _required_bool(value: Mapping[object, object], key: str, code: str) -> bool:
    item = _required_value(value, key, code)
    if type(item) is not bool:
        raise AppError(code, {"field": key})
    return item


def _required_mapping(
    value: Mapping[object, object], key: str, code: str
) -> Mapping[object, object]:
    item = _required_value(value, key, code)
    if not isinstance(item, Mapping):
        raise AppError(code, {"field": key})
    return cast(Mapping[object, object], item)


def _required_sequence(value: Mapping[object, object], key: str, code: str) -> tuple[object, ...]:
    item = _required_value(value, key, code)
    if not isinstance(item, (list, tuple)):
        raise AppError(code, {"field": key})
    return tuple(cast(list[object] | tuple[object, ...], item))


def _required_string_tuple(value: Mapping[object, object], key: str, code: str) -> tuple[str, ...]:
    entries = _required_sequence(value, key, code)
    if any(not isinstance(item, str) for item in entries):
        raise AppError(code, {"field": key})
    return tuple(cast(str, item) for item in entries)


def _mapping_item(value: object, field: str, code: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise AppError(code, {"field": field})
    return cast(Mapping[object, object], value)


def _optional_identity(value: Mapping[object, object], key: str) -> RuntimeIdentity | None:
    item = _required_value(value, key, "runtime.active_pointer_invalid")
    return None if item is None else RuntimeIdentity.from_dict(item)


__all__ = [
    "SUPPORTED_RUNTIME_ARCHITECTURES",
    "SUPPORTED_RUNTIME_DEVICES",
    "SUPPORTED_RUNTIME_PLATFORMS",
    "ActiveRuntimePointer",
    "DoctorCheck",
    "DoctorPhase",
    "DoctorReport",
    "RuntimeFile",
    "RuntimeFileEntry",
    "RuntimeIdentity",
    "RuntimeInstallation",
    "RuntimeManifest",
    "RuntimeManifestFile",
    "RuntimeState",
    "RuntimeStatus",
    "RuntimeTarget",
]
