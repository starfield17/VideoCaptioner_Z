"""Worker JSONL Protocol v1 domain messages and pure codec helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelIdentity
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)
from captioner.core.domain.runtime import RuntimeIdentity

WORKER_PROTOCOL_NAME = "captioner.worker"
WORKER_PROTOCOL_VERSION = "1.0"
_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")
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
        "token_value",
    }
)
_JOB_MESSAGE_TYPES = frozenset(
    {
        "transcribe.request",
        "operation.progress",
        "cancel.request",
        "cancel.acknowledged",
        "operation.cancelled",
        "operation.result",
        "operation.error",
    }
)
_FORBIDDEN_PROGRESS_FIELDS = frozenset(
    {
        "percent",
        "percentage",
        "progress_value",
        "completed_units",
        "total_units",
        "percent_milli",
        "eta",
    }
)


def _empty_json_mapping() -> dict[str, JsonValue]:
    return {}


class WorkerMessageType(StrEnum):
    """Messages in the first Worker transport protocol."""

    HANDSHAKE_REQUEST = "handshake.request"
    HANDSHAKE_RESPONSE = "handshake.response"
    TRANSCRIBE_REQUEST = "transcribe.request"
    OPERATION_PROGRESS = "operation.progress"
    CANCEL_REQUEST = "cancel.request"
    CANCEL_ACKNOWLEDGED = "cancel.acknowledged"
    OPERATION_CANCELLED = "operation.cancelled"
    OPERATION_RESULT = "operation.result"
    OPERATION_ERROR = "operation.error"
    SHUTDOWN_REQUEST = "shutdown.request"
    SHUTDOWN_ACKNOWLEDGED = "shutdown.acknowledged"


@dataclass(frozen=True, slots=True)
class CancelRequest:
    """Wire payload for cancellation of one active request."""

    target_request_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.target_request_id, "target_request_id", "worker.cancel_invalid")

    def to_payload(self) -> dict[str, JsonValue]:
        return {"target_request_id": self.target_request_id}

    @classmethod
    def from_payload(cls, value: object) -> CancelRequest:
        raw = _object(value, "cancel_request")
        return cls(_required_str(raw, "target_request_id"))


@dataclass(frozen=True, slots=True)
class CancelAcknowledged:
    """Minimal typed payload confirming a cancellation request was accepted."""

    target_request_id: str

    def __post_init__(self) -> None:
        _require_nonempty(
            self.target_request_id,
            "target_request_id",
            "worker.cancel_invalid",
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {"target_request_id": self.target_request_id}

    @classmethod
    def from_payload(cls, value: object) -> CancelAcknowledged:
        raw = _object(value, "cancel_acknowledged")
        return cls(_required_str(raw, "target_request_id"))


@dataclass(frozen=True, slots=True)
class OperationCancelled:
    """Minimal typed terminal payload for a cancelled operation."""

    target_request_id: str

    def __post_init__(self) -> None:
        _require_nonempty(
            self.target_request_id,
            "target_request_id",
            "worker.cancel_invalid",
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {"target_request_id": self.target_request_id}

    @classmethod
    def from_payload(cls, value: object) -> OperationCancelled:
        raw = _object(value, "operation_cancelled")
        return cls(_required_str(raw, "target_request_id"))


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    """Wire payload for an orderly Worker shutdown."""

    reason: str = "core_shutdown"

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "reason", "worker.shutdown_invalid")

    def to_payload(self) -> dict[str, JsonValue]:
        return {"reason": self.reason}

    @classmethod
    def from_payload(cls, value: object) -> ShutdownRequest:
        raw = _object(value, "shutdown_request")
        return cls(_required_str(raw, "reason"))


@dataclass(frozen=True, slots=True)
class ShutdownAcknowledged:
    """Minimal typed payload confirming orderly shutdown."""

    acknowledged: bool

    def __post_init__(self) -> None:
        if type(self.acknowledged) is not bool:
            raise AppError("worker.shutdown_invalid", {"field": "acknowledged"})

    def to_payload(self) -> dict[str, JsonValue]:
        return {"acknowledged": self.acknowledged}

    @classmethod
    def from_payload(cls, value: object) -> ShutdownAcknowledged:
        raw = _object(value, "shutdown_acknowledged")
        return cls(_required_bool(raw, "acknowledged"))


@dataclass(frozen=True, slots=True)
class WorkerEnvelope:
    """Common wire envelope shared by every protocol message."""

    protocol: str
    version: str
    message_type: str
    request_id: str
    sequence: int
    payload: Mapping[str, JsonValue]
    job_id: str | None = None
    stage_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if self.protocol != WORKER_PROTOCOL_NAME:
            raise AppError("worker.protocol_invalid", {"field": "protocol"})
        if not check_protocol_compatibility(WORKER_PROTOCOL_VERSION, self.version):
            raise AppError("worker.protocol_version_incompatible", {"field": "version"})
        message_type = (
            self.message_type.value
            if isinstance(self.message_type, WorkerMessageType)
            else self.message_type
        )
        if message_type not in {item.value for item in WorkerMessageType}:
            raise AppError("worker.message_type_unknown", {"field": "message_type"})
        _require_nonempty(self.request_id, "request_id", "worker.envelope_invalid")
        if type(self.sequence) is not int or self.sequence < 0:
            raise AppError("worker.envelope_invalid", {"field": "sequence"})
        if message_type in _JOB_MESSAGE_TYPES:
            _require_nonempty(self.job_id, "job_id", "worker.envelope_invalid")
            _require_nonempty(self.stage_attempt_id, "stage_attempt_id", "worker.envelope_invalid")
        payload = _freeze_public_mapping(self.payload, "worker.envelope_invalid")
        if message_type == WorkerMessageType.OPERATION_PROGRESS.value and (
            _contains_forbidden_progress_key(payload)
        ):
            raise AppError("worker.progress_invalid", {"field": "payload"})
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "payload", payload)
        _decode_typed_payload(
            message_type,
            payload,
            request_id=self.request_id,
            job_id=self.job_id,
            stage_attempt_id=self.stage_attempt_id,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-compatible envelope."""
        result: dict[str, JsonValue] = {
            "protocol": self.protocol,
            "version": self.version,
            "message_type": self.message_type,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "payload": _thaw_public_mapping(self.payload),
        }
        if self.job_id is not None:
            result["job_id"] = self.job_id
        if self.stage_attempt_id is not None:
            result["stage_attempt_id"] = self.stage_attempt_id
        return result

    @classmethod
    def from_dict(cls, value: object) -> WorkerEnvelope:
        if not isinstance(value, Mapping):
            raise AppError("worker.protocol_invalid", {"field": "root"})
        raw = cast(Mapping[object, object], value)
        if _contains_sensitive_key(raw):
            raise AppError("worker.protocol_invalid", {"reason": "sensitive"})
        protocol = _required_str(raw, "protocol")
        version = _required_str(raw, "version")
        message_type = _required_str(raw, "message_type")
        request_id = _required_str(raw, "request_id")
        sequence = _required_int(raw, "sequence")
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            raise AppError("worker.envelope_invalid", {"field": "payload"})
        job_id = _optional_str(raw, "job_id")
        stage_attempt_id = _optional_str(raw, "stage_attempt_id")
        return cls(
            protocol,
            version,
            message_type,
            request_id,
            sequence,
            cast(Mapping[str, JsonValue], payload),
            job_id,
            stage_attempt_id,
        )


@dataclass(frozen=True, slots=True)
class HandshakeRequest:
    """Core requirements sent to a Worker during activation."""

    required_capabilities: tuple[str, ...] = ()
    required_backend_id: str | None = None
    required_result_schema_versions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_string_tuple(self.required_capabilities, "required_capabilities")
        _optional_nonempty(self.required_backend_id, "required_backend_id")
        _validate_positive_int_tuple(
            self.required_result_schema_versions,
            "required_result_schema_versions",
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "required_capabilities": list(self.required_capabilities),
            "required_backend_id": self.required_backend_id,
            "required_result_schema_versions": list(self.required_result_schema_versions),
        }

    @classmethod
    def from_payload(cls, value: object) -> HandshakeRequest:
        raw = _object(value, "handshake_request")
        return cls(
            required_capabilities=_string_tuple(raw, "required_capabilities"),
            required_backend_id=_optional_str(raw, "required_backend_id"),
            required_result_schema_versions=_int_tuple(raw, "required_result_schema_versions"),
        )


@dataclass(frozen=True, slots=True)
class WorkerHandshake:
    """Worker capability response returned by a successful handshake."""

    protocol_version: str
    runtime_id: str
    runtime_version: str
    backend_id: str
    backend_version: str
    worker_version: str
    platform: str
    architecture: str
    capabilities: tuple[str, ...]
    supported_devices: tuple[str, ...]
    supported_model_formats: tuple[str, ...]
    supported_result_schema_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_protocol_version(self.protocol_version)
        for field_name, value in (
            ("runtime_id", self.runtime_id),
            ("runtime_version", self.runtime_version),
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
            ("worker_version", self.worker_version),
            ("platform", self.platform),
            ("architecture", self.architecture),
        ):
            _require_nonempty(value, field_name, "worker.handshake_invalid")
        _validate_string_tuple(self.capabilities, "capabilities")
        _validate_string_tuple(self.supported_devices, "supported_devices")
        _validate_string_tuple(self.supported_model_formats, "supported_model_formats")
        _validate_positive_int_tuple(
            self.supported_result_schema_versions,
            "supported_result_schema_versions",
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "protocol_version": self.protocol_version,
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "worker_version": self.worker_version,
            "platform": self.platform,
            "architecture": self.architecture,
            "capabilities": list(self.capabilities),
            "supported_devices": list(self.supported_devices),
            "supported_model_formats": list(self.supported_model_formats),
            "supported_result_schema_versions": list(self.supported_result_schema_versions),
        }

    @classmethod
    def from_payload(cls, value: object) -> WorkerHandshake:
        raw = _object(value, "handshake")
        return cls(
            _required_str(raw, "protocol_version"),
            _required_str(raw, "runtime_id"),
            _required_str(raw, "runtime_version"),
            _required_str(raw, "backend_id"),
            _required_str(raw, "backend_version"),
            _required_str(raw, "worker_version"),
            _required_str(raw, "platform"),
            _required_str(raw, "architecture"),
            _string_tuple(raw, "capabilities"),
            _string_tuple(raw, "supported_devices"),
            _string_tuple(raw, "supported_model_formats"),
            _int_tuple(raw, "supported_result_schema_versions"),
        )


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    """Normalized local inputs sent to a Worker for one ASR attempt."""

    normalized_audio_path: Path
    attempt_workspace: Path
    model_directory: Path
    backend_id: str
    runtime_identity: RuntimeIdentity
    model_identity: ModelIdentity
    result_schema_version: int
    language: str | None
    task: str
    word_timestamps: bool
    initial_prompt: str | None = None
    backend_options: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)
    request_id: str = "request-1"
    job_id: str = "job-1"
    stage_attempt_id: str = "attempt-1"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("normalized_audio_path", self.normalized_audio_path),
            ("attempt_workspace", self.attempt_workspace),
            ("model_directory", self.model_directory),
        ):
            if not value.is_absolute():
                raise AppError("worker.transcribe_request_invalid", {"field": field_name})
        _require_nonempty(self.backend_id, "backend_id", "worker.transcribe_request_invalid")
        if self.backend_id != self.model_identity.backend_id:
            raise AppError("worker.transcribe_request_invalid", {"field": "backend_id"})
        if self.runtime_identity.runtime_id == "":
            raise AppError("worker.transcribe_request_invalid", {"field": "runtime_identity"})
        if type(self.result_schema_version) is not int or self.result_schema_version <= 0:
            raise AppError("worker.transcribe_request_invalid", {"field": "result_schema_version"})
        _optional_nonempty(self.language, "language")
        _require_nonempty(self.task, "task", "worker.transcribe_request_invalid")
        if type(self.word_timestamps) is not bool:
            raise AppError("worker.transcribe_request_invalid", {"field": "word_timestamps"})
        _optional_nonempty(self.initial_prompt, "initial_prompt")
        _require_nonempty(self.request_id, "request_id", "worker.transcribe_request_invalid")
        _require_nonempty(self.job_id, "job_id", "worker.transcribe_request_invalid")
        _require_nonempty(
            self.stage_attempt_id,
            "stage_attempt_id",
            "worker.transcribe_request_invalid",
        )
        object.__setattr__(
            self,
            "backend_options",
            _freeze_public_mapping(self.backend_options, "worker.transcribe_request_invalid"),
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "normalized_audio_path": str(self.normalized_audio_path),
            "attempt_workspace": str(self.attempt_workspace),
            "model_directory": str(self.model_directory),
            "backend_id": self.backend_id,
            "runtime_identity": self.runtime_identity.to_dict(),
            "model_identity": self.model_identity.to_dict(),
            "result_schema_version": self.result_schema_version,
            "language": self.language,
            "task": self.task,
            "word_timestamps": self.word_timestamps,
            "initial_prompt": self.initial_prompt,
            "backend_options": _thaw_public_mapping(self.backend_options),
        }

    @classmethod
    def from_payload(
        cls,
        value: object,
        *,
        request_id: str = "request-1",
        job_id: str = "job-1",
        stage_attempt_id: str = "attempt-1",
    ) -> TranscribeRequest:
        raw = _object(value, "transcribe_request")
        runtime_identity = RuntimeIdentity.from_dict(raw.get("runtime_identity"))
        model_identity = ModelIdentity.from_dict(raw.get("model_identity"))
        backend_options = raw.get("backend_options", {})
        if not isinstance(backend_options, Mapping):
            raise AppError("worker.transcribe_request_invalid", {"field": "backend_options"})
        return cls(
            normalized_audio_path=Path(_required_str(raw, "normalized_audio_path")),
            attempt_workspace=Path(_required_str(raw, "attempt_workspace")),
            model_directory=Path(_required_str(raw, "model_directory")),
            backend_id=_required_str(raw, "backend_id"),
            runtime_identity=runtime_identity,
            model_identity=model_identity,
            result_schema_version=_required_int(raw, "result_schema_version"),
            language=_optional_str(raw, "language"),
            task=_required_str(raw, "task"),
            word_timestamps=_required_bool(raw, "word_timestamps"),
            initial_prompt=_optional_str(raw, "initial_prompt"),
            backend_options=cast(Mapping[str, JsonValue], backend_options),
            request_id=request_id,
            job_id=job_id,
            stage_attempt_id=stage_attempt_id,
        )


@dataclass(frozen=True, slots=True)
class ResultDescriptor:
    """Small Worker result reference; the complete Transcript stays on disk."""

    relative_path: str
    size_bytes: int
    sha256: str
    schema_id: str
    schema_version: int

    def __post_init__(self) -> None:
        _validate_relative_posix_path(self.relative_path, "worker.result_descriptor_invalid")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise AppError("worker.result_descriptor_invalid", {"field": "size_bytes"})
        sha256 = cast(object, self.sha256)
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise AppError("worker.result_descriptor_invalid", {"field": "sha256"})
        _require_nonempty(self.schema_id, "schema_id", "worker.result_descriptor_invalid")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise AppError("worker.result_descriptor_invalid", {"field": "schema_version"})

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_payload(cls, value: object) -> ResultDescriptor:
        raw = _object(value, "result_descriptor")
        return cls(
            _required_str(raw, "relative_path"),
            _required_int(raw, "size_bytes"),
            _required_str(raw, "sha256"),
            _required_str(raw, "schema_id"),
            _required_int(raw, "schema_version"),
        )


WorkerResultDescriptor = ResultDescriptor


@dataclass(frozen=True, slots=True)
class WorkerError:
    """Safe public Worker failure without traceback or credential material."""

    code: str
    message_code: str
    retryable: bool
    details: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        _require_nonempty(self.code, "code", "worker.error_invalid")
        _require_nonempty(self.message_code, "message_code", "worker.error_invalid")
        if type(self.retryable) is not bool:
            raise AppError("worker.error_invalid", {"field": "retryable"})
        object.__setattr__(
            self, "details", _freeze_public_mapping(self.details, "worker.error_invalid")
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "message_code": self.message_code,
            "retryable": self.retryable,
            "details": _thaw_public_mapping(self.details),
        }

    @classmethod
    def from_payload(cls, value: object) -> WorkerError:
        raw = _object(value, "worker_error")
        details = raw.get("details", {})
        if not isinstance(details, Mapping):
            raise AppError("worker.error_invalid", {"field": "details"})
        return cls(
            _required_str(raw, "code"),
            _required_str(raw, "message_code"),
            _required_bool(raw, "retryable"),
            cast(Mapping[str, JsonValue], details),
        )


@dataclass(frozen=True, slots=True)
class CancelResult:
    """Typed cancellation outcome returned by a Worker Client."""

    request_id: str
    acknowledged: bool
    cancelled: bool = False
    timed_out: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.request_id, "request_id", "worker.cancel_invalid")
        if any(
            type(value) is not bool for value in (self.acknowledged, self.cancelled, self.timed_out)
        ):
            raise AppError("worker.cancel_invalid", {"field": "result"})


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    """Idempotent shutdown outcome."""

    acknowledged: bool

    def __post_init__(self) -> None:
        if type(self.acknowledged) is not bool:
            raise AppError("worker.shutdown_invalid", {"field": "acknowledged"})


@dataclass(frozen=True, slots=True)
class WorkerProgressEvent:
    request_id: str
    job_id: str
    stage_attempt_id: str
    sequence: int
    progress: OperationProgress


@dataclass(frozen=True, slots=True)
class WorkerResultEvent:
    request_id: str
    job_id: str
    stage_attempt_id: str
    sequence: int
    result: ResultDescriptor


@dataclass(frozen=True, slots=True)
class WorkerErrorEvent:
    request_id: str
    job_id: str
    stage_attempt_id: str
    sequence: int
    error: WorkerError


@dataclass(frozen=True, slots=True)
class WorkerCancelledEvent:
    request_id: str
    job_id: str
    stage_attempt_id: str
    sequence: int


type WorkerEvent = WorkerProgressEvent | WorkerResultEvent | WorkerErrorEvent | WorkerCancelledEvent
type WorkerProtocolMessage = (
    HandshakeRequest
    | WorkerHandshake
    | TranscribeRequest
    | OperationProgress
    | CancelRequest
    | CancelAcknowledged
    | OperationCancelled
    | ResultDescriptor
    | WorkerError
    | ShutdownRequest
    | ShutdownAcknowledged
)


def decode_typed_message(envelope: WorkerEnvelope) -> WorkerProtocolMessage:
    """Validate and decode an envelope payload into its typed domain message."""
    return _decode_typed_payload(
        envelope.message_type,
        envelope.payload,
        request_id=envelope.request_id,
        job_id=envelope.job_id,
        stage_attempt_id=envelope.stage_attempt_id,
    )


def _decode_typed_payload(
    message_type: str,
    payload: Mapping[str, JsonValue],
    *,
    request_id: str,
    job_id: str | None,
    stage_attempt_id: str | None,
) -> WorkerProtocolMessage:
    if message_type == WorkerMessageType.HANDSHAKE_REQUEST.value:
        return HandshakeRequest.from_payload(payload)
    if message_type == WorkerMessageType.HANDSHAKE_RESPONSE.value:
        return WorkerHandshake.from_payload(payload)
    if message_type == WorkerMessageType.TRANSCRIBE_REQUEST.value:
        if job_id is None or stage_attempt_id is None:
            raise AppError("worker.envelope_invalid", {"field": "job_id"})
        return TranscribeRequest.from_payload(
            payload,
            request_id=request_id,
            job_id=job_id,
            stage_attempt_id=stage_attempt_id,
        )
    if message_type == WorkerMessageType.OPERATION_PROGRESS.value:
        return OperationProgress.from_payload(payload)
    if message_type == WorkerMessageType.CANCEL_REQUEST.value:
        return CancelRequest.from_payload(payload)
    if message_type == WorkerMessageType.CANCEL_ACKNOWLEDGED.value:
        return CancelAcknowledged.from_payload(payload)
    if message_type == WorkerMessageType.OPERATION_CANCELLED.value:
        return OperationCancelled.from_payload(payload)
    if message_type == WorkerMessageType.OPERATION_RESULT.value:
        return ResultDescriptor.from_payload(payload)
    if message_type == WorkerMessageType.OPERATION_ERROR.value:
        return WorkerError.from_payload(payload)
    if message_type == WorkerMessageType.SHUTDOWN_REQUEST.value:
        return ShutdownRequest.from_payload(payload)
    if message_type == WorkerMessageType.SHUTDOWN_ACKNOWLEDGED.value:
        return ShutdownAcknowledged.from_payload(payload)
    raise AppError("worker.message_type_unknown", {"field": "message_type"})


def check_protocol_compatibility(core_version: str, worker_version: str) -> bool:
    """Allow optional minor additions while rejecting a major mismatch."""
    core_major, _ = _version_parts(core_version)
    worker_major, _ = _version_parts(worker_version)
    return core_major == worker_major


def require_protocol_compatibility(core_version: str, worker_version: str) -> None:
    if not check_protocol_compatibility(core_version, worker_version):
        raise AppError("worker.protocol_version_incompatible", {"reason": "major"})


def validate_sequence(previous_sequence: int | None, sequence: int) -> int:
    """Validate one session's non-decreasing JSONL sequence and return it."""
    if type(sequence) is not int or sequence < 0:
        raise AppError("worker.sequence_invalid")
    if previous_sequence is not None and sequence <= previous_sequence:
        raise AppError("worker.sequence_invalid")
    return sequence


def encode_jsonl(envelope: WorkerEnvelope) -> bytes:
    """Encode one protocol envelope as exactly one UTF-8 JSONL line."""
    try:
        serialized = json.dumps(
            envelope.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AppError("worker.protocol_encode_failed") from exc
    return (serialized + "\n").encode("utf-8")


def decode_jsonl(line: str | bytes) -> WorkerEnvelope:
    """Decode one JSONL line and reject invalid protocol objects."""
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
    except UnicodeDecodeError as exc:
        raise AppError("worker.protocol_invalid_json", {"reason": "utf8"}) from exc
    if not text.strip():
        raise AppError("worker.protocol_invalid_json", {"reason": "empty"})
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise AppError("worker.protocol_invalid_json", {"reason": "json"}) from exc
    return WorkerEnvelope.from_dict(value)


def decode_stdout_line(line: str | bytes) -> WorkerEnvelope:
    """Decode Worker stdout and classify every non-protocol line as contamination."""
    try:
        return decode_jsonl(line)
    except AppError as exc:
        raise AppError("worker.protocol_contaminated", {"reason": "stdout"}) from exc


decode_message = decode_jsonl
encode_message = encode_jsonl
decode_protocol_line = decode_stdout_line


class JsonlProtocolCodec:
    """Stateless facade for callers that prefer an object-shaped codec."""

    def encode(self, envelope: WorkerEnvelope) -> bytes:
        return encode_jsonl(envelope)

    def decode(self, line: str | bytes) -> WorkerEnvelope:
        return decode_jsonl(line)

    def decode_typed(self, line: str | bytes) -> WorkerProtocolMessage:
        return decode_typed_message(decode_jsonl(line))

    def decode_stdout(self, line: str | bytes) -> WorkerEnvelope:
        return decode_stdout_line(line)


def _version_parts(value: str) -> tuple[int, int]:
    match = _VERSION_RE.fullmatch(value)
    if match is None:
        raise AppError("worker.protocol_version_invalid")
    return int(match.group("major")), int(match.group("minor"))


def _require_protocol_version(value: str) -> None:
    _version_parts(value)


def _require_nonempty(value: object, field: str, code: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AppError(code, {"field": field})


def _optional_nonempty(value: object, field: str) -> None:
    if value is not None:
        _require_nonempty(value, field, "worker.message_invalid")


def _freeze_public_mapping(value: object, code: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AppError(code, {"field": "payload", "reason": "object"})
    mapping = cast(Mapping[str, JsonValue], value)
    if _contains_sensitive_key(mapping):
        raise AppError(code, {"field": "payload", "reason": "sensitive_or_invalid"})
    try:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(mapping))
    except (TypeError, ValueError) as exc:
        raise AppError(code, {"field": "payload", "reason": "json"}) from exc
    return cast(Mapping[str, JsonValue], frozen)


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        for key, item in raw.items():
            if isinstance(key, str) and key.lower().replace("-", "_") in _SENSITIVE_KEYS:
                return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_sensitive_key(item) for item in sequence)
    return False


def _contains_forbidden_progress_key(value: object) -> bool:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        for key, item in raw.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_PROGRESS_FIELDS:
                return True
            if _contains_forbidden_progress_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return any(_contains_forbidden_progress_key(item) for item in sequence)
    return False


def _validate_string_tuple(value: object, field: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise AppError("worker.message_invalid", {"field": field})
    entries = tuple(cast(tuple[object, ...] | list[object], value))
    if any(
        not isinstance(item, str) or not item.strip() or item != item.strip() for item in entries
    ):
        raise AppError("worker.message_invalid", {"field": field})
    if len(set(entries)) != len(entries):
        raise AppError("worker.message_invalid", {"field": field, "reason": "duplicate"})


def _validate_positive_int_tuple(value: tuple[int, ...], field: str) -> None:
    entries = tuple(value)
    if any(type(item) is not int or item <= 0 for item in entries):
        raise AppError("worker.message_invalid", {"field": field})
    if len(set(entries)) != len(entries):
        raise AppError("worker.message_invalid", {"field": field, "reason": "duplicate"})


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
    ):
        raise AppError(code, {"field": "relative_path"})


def _object(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise AppError("worker.message_invalid", {"field": field})
    return cast(Mapping[object, object], dict(cast(Mapping[object, object], value)))


def _required_str(value: Mapping[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise AppError("worker.message_invalid", {"field": key})
    return item


def _optional_str(value: Mapping[object, object], key: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise AppError("worker.message_invalid", {"field": key})
    return item


def _required_int(value: Mapping[object, object], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise AppError("worker.message_invalid", {"field": key})
    return item


def _required_bool(value: Mapping[object, object], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise AppError("worker.message_invalid", {"field": key})
    return item


def _string_tuple(value: Mapping[object, object], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, (list, tuple)):
        raise AppError("worker.message_invalid", {"field": key})
    entries = cast(list[object] | tuple[object, ...], item)
    result: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise AppError("worker.message_invalid", {"field": key})
        result.append(entry)
    return tuple(result)


def _int_tuple(value: Mapping[object, object], key: str) -> tuple[int, ...]:
    item = value.get(key)
    if not isinstance(item, (list, tuple)):
        raise AppError("worker.message_invalid", {"field": key})
    entries = cast(list[object] | tuple[object, ...], item)
    result: list[int] = []
    for entry in entries:
        if type(entry) is not int:
            raise AppError("worker.message_invalid", {"field": key})
        result.append(entry)
    return tuple(result)


def _thaw_public_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    frozen = cast(Mapping[str, FrozenJsonValue], value)
    thawed = thaw_json_value(frozen)
    if not isinstance(thawed, dict):
        raise AppError("worker.protocol_encode_failed")
    return cast(dict[str, JsonValue], thawed)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_value:{value}")


__all__ = [
    "WORKER_PROTOCOL_NAME",
    "WORKER_PROTOCOL_VERSION",
    "CancelAcknowledged",
    "CancelRequest",
    "CancelResult",
    "HandshakeRequest",
    "JsonlProtocolCodec",
    "OperationCancelled",
    "OperationProgress",
    "ResultDescriptor",
    "ShutdownAcknowledged",
    "ShutdownRequest",
    "ShutdownResult",
    "TranscribeRequest",
    "WorkerCancelledEvent",
    "WorkerEnvelope",
    "WorkerError",
    "WorkerErrorEvent",
    "WorkerEvent",
    "WorkerHandshake",
    "WorkerMessageType",
    "WorkerProgressEvent",
    "WorkerProtocolMessage",
    "WorkerResultDescriptor",
    "WorkerResultEvent",
    "check_protocol_compatibility",
    "decode_jsonl",
    "decode_message",
    "decode_protocol_line",
    "decode_stdout_line",
    "decode_typed_message",
    "encode_jsonl",
    "encode_message",
    "require_protocol_compatibility",
    "validate_sequence",
]
