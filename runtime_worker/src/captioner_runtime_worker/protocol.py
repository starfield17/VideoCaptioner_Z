"""Stdlib-only Worker JSONL protocol writer and validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

PROTOCOL = "captioner.worker"
VERSION = "1.1"

_INBOUND_MESSAGE_TYPES = frozenset(
    {
        "handshake.request",
        "doctor.request",
        "transcribe.request",
        "cancel.request",
        "shutdown.request",
    }
)
_JOB_MESSAGE_TYPES = frozenset({"transcribe.request", "cancel.request"})


def encode(
    message_type: str,
    request_id: str,
    sequence: int,
    payload: Mapping[str, object],
    *,
    job_id: str | None = None,
    stage_attempt_id: str | None = None,
) -> bytes:
    if _contains_sensitive_key(payload):
        raise ValueError("sensitive_field")
    envelope: dict[str, object] = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "message_type": message_type,
        "request_id": request_id,
        "sequence": sequence,
        "payload": dict(payload),
    }
    if job_id is not None:
        envelope["job_id"] = job_id
    if stage_attempt_id is not None:
        envelope["stage_attempt_id"] = stage_attempt_id
    return (
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def decode(line: bytes) -> dict[str, object]:
    value = json.loads(
        line.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("protocol_object_required")
    raw = cast(dict[str, object], value)
    if _contains_sensitive_key(raw):
        raise ValueError("sensitive_field")
    version = raw.get("version")
    if (
        raw.get("protocol") != PROTOCOL
        or not isinstance(version, str)
        or _protocol_major(version) != _protocol_major(VERSION)
    ):
        raise ValueError("protocol_version_invalid")
    for key in ("message_type", "request_id", "sequence", "payload"):
        if key not in raw:
            raise ValueError(f"missing_{key}")
    if not isinstance(raw["message_type"], str) or not raw["message_type"].strip():
        raise TypeError("message_type_invalid")
    if not isinstance(raw["request_id"], str) or not raw["request_id"].strip():
        raise TypeError("envelope_type_invalid")
    if type(raw.get("sequence")) is not int or cast(int, raw["sequence"]) < 0:
        raise ValueError("sequence_invalid")
    if not isinstance(raw["payload"], dict):
        raise TypeError("payload_object_required")
    message_type = cast(str, raw["message_type"])
    if message_type not in _INBOUND_MESSAGE_TYPES:
        raise ValueError("unknown_message_type")
    if message_type in _JOB_MESSAGE_TYPES:
        _required_string(raw, "job_id")
        _required_string(raw, "stage_attempt_id")
    _validate_payload(message_type, cast(dict[str, object], raw["payload"]))
    return raw


def write(stream: BinaryIO, message: bytes) -> None:
    stream.write(message)
    stream.flush()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _protocol_major(value: str) -> int:
    major, separator, _minor = value.partition(".")
    if not separator or not major.isdigit():
        raise ValueError("protocol_version_invalid")
    return int(major)


def _validate_payload(message_type: str, payload: dict[str, object]) -> None:
    if message_type == "handshake.request":
        _string_list(payload, "required_capabilities")
        _optional_string(payload, "required_backend_id")
        values = _int_list(payload, "required_result_schema_versions")
        if any(value <= 0 for value in values):
            raise ValueError("required_result_schema_versions_invalid")
        return
    if message_type == "doctor.request":
        _required_string(payload, "nonce")
        _required_string(payload, "probe_filename")
        _optional_string(payload, "workspace")
        return
    if message_type == "transcribe.request":
        for key in ("backend_id", "task"):
            _required_string(payload, key)
        for key in ("normalized_audio_path", "attempt_workspace", "model_directory"):
            _absolute_path(payload, key)
        _required_object(payload, "runtime_identity")
        _required_object(payload, "model_identity")
        result_schema_version = _required_int(payload, "result_schema_version")
        if result_schema_version <= 0:
            raise ValueError("result_schema_version_invalid")
        _optional_string(payload, "language")
        if type(payload.get("word_timestamps")) is not bool:
            raise TypeError("word_timestamps_invalid")
        _optional_string(payload, "initial_prompt")
        _required_object(payload, "backend_options")
        return
    if message_type == "cancel.request":
        _required_string(payload, "target_request_id")
        return
    if message_type == "shutdown.request":
        _required_string(payload, "reason")
        return
    raise ValueError("unknown_message_type")


def _required_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key}_invalid")
    return item


def _optional_string(value: dict[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is not None and (not isinstance(item, str) or not item.strip()):
        raise ValueError(f"{key}_invalid")
    return item


def _required_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise TypeError(f"{key}_invalid")
    return cast(int, item)


def _required_object(value: dict[str, object], key: str) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise TypeError(f"{key}_invalid")
    return cast(dict[str, object], item)


def _absolute_path(value: dict[str, object], key: str) -> str:
    item = _required_string(value, key)
    if not Path(item).is_absolute():
        raise ValueError(f"{key}_absolute_required")
    return item


def _string_list(value: dict[str, object], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(entry, str) for entry in item):
        raise TypeError(f"{key}_invalid")
    return cast(list[str], item)


def _int_list(value: dict[str, object], key: str) -> list[int]:
    item = value.get(key)
    if not isinstance(item, list) or any(type(entry) is not int for entry in item):
        raise TypeError(f"{key}_invalid")
    return cast(list[int], item)


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
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
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False
