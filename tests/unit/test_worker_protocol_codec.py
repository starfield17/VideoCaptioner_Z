from __future__ import annotations

import json

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.domain.worker_protocol import (
    WORKER_PROTOCOL_NAME,
    WORKER_PROTOCOL_VERSION,
    JsonlProtocolCodec,
    WorkerEnvelope,
    WorkerMessageType,
    check_protocol_compatibility,
    decode_jsonl,
    decode_stdout_line,
    encode_jsonl,
)

_JOB_TYPES = {
    WorkerMessageType.TRANSCRIBE_REQUEST.value,
    WorkerMessageType.OPERATION_PROGRESS.value,
    WorkerMessageType.CANCEL_REQUEST.value,
    WorkerMessageType.CANCEL_ACKNOWLEDGED.value,
    WorkerMessageType.OPERATION_CANCELLED.value,
    WorkerMessageType.OPERATION_RESULT.value,
    WorkerMessageType.OPERATION_ERROR.value,
}


def _envelope(message_type: str) -> WorkerEnvelope:
    payload = (
        OperationProgress("asr", "transcribing", "worker.transcribing", {}).to_payload()
        if message_type == WorkerMessageType.OPERATION_PROGRESS.value
        else {"value": "safe"}
    )
    kwargs: dict[str, str] = {}
    if message_type in _JOB_TYPES:
        kwargs = {"job_id": "job-1", "stage_attempt_id": "attempt-1"}
    return WorkerEnvelope(
        protocol=WORKER_PROTOCOL_NAME,
        version=WORKER_PROTOCOL_VERSION,
        message_type=message_type,
        request_id="request-1",
        sequence=0,
        payload=payload,
        **kwargs,
    )


@pytest.mark.parametrize("message_type", [message.value for message in WorkerMessageType])
def test_all_protocol_message_types_round_trip(message_type: str) -> None:
    envelope = _envelope(message_type)
    decoded = decode_jsonl(encode_jsonl(envelope))
    assert decoded == envelope


def test_codec_object_facade_round_trips() -> None:
    codec = JsonlProtocolCodec()
    envelope = _envelope(WorkerMessageType.HANDSHAKE_REQUEST.value)
    assert codec.decode(codec.encode(envelope)) == envelope


@pytest.mark.parametrize("line", [b"not-json", "", "[]", "null"])
def test_invalid_json_or_non_object_is_rejected(line: str | bytes) -> None:
    with pytest.raises(AppError, match=r"worker\.protocol"):
        decode_jsonl(line)


def test_unknown_message_type_is_rejected() -> None:
    with pytest.raises(AppError, match=r"worker\.message_type_unknown"):
        decode_jsonl(
            json.dumps(
                {
                    "protocol": WORKER_PROTOCOL_NAME,
                    "version": WORKER_PROTOCOL_VERSION,
                    "message_type": "future.message",
                    "request_id": "request-1",
                    "sequence": 0,
                    "payload": {},
                }
            )
        )


def test_major_mismatch_is_rejected_but_minor_additions_are_compatible() -> None:
    assert check_protocol_compatibility("1.0", "1.9")
    assert not check_protocol_compatibility("1.0", "2.0")
    with pytest.raises(AppError, match=r"worker\.protocol_version_incompatible"):
        decode_jsonl(
            json.dumps(
                {
                    "protocol": WORKER_PROTOCOL_NAME,
                    "version": "2.0",
                    "message_type": WorkerMessageType.HANDSHAKE_REQUEST.value,
                    "request_id": "request-1",
                    "sequence": 0,
                    "payload": {},
                }
            )
        )


def test_unknown_optional_fields_are_ignored_with_same_major() -> None:
    wire = json.loads(encode_jsonl(_envelope(WorkerMessageType.HANDSHAKE_REQUEST.value)))
    wire["future_optional_field"] = {"future": True}
    decoded = decode_jsonl(json.dumps(wire))
    assert decoded == _envelope(WorkerMessageType.HANDSHAKE_REQUEST.value)


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": ""},
        {"sequence": -1},
        {"job_id": None},
    ],
)
def test_required_envelope_fields_are_validated(change: dict[str, object]) -> None:
    wire = json.loads(encode_jsonl(_envelope(WorkerMessageType.TRANSCRIBE_REQUEST.value)))
    wire.update(change)
    with pytest.raises(AppError, match=r"worker\.envelope_invalid"):
        decode_jsonl(json.dumps(wire))


def test_non_protocol_stdout_is_marked_as_contaminated() -> None:
    with pytest.raises(AppError, match=r"worker\.protocol_contaminated"):
        decode_stdout_line("worker log accidentally printed to stdout")


def test_credentials_are_rejected_before_serialization_and_repr() -> None:
    with pytest.raises(AppError, match=r"worker\.envelope_invalid"):
        WorkerEnvelope(
            WORKER_PROTOCOL_NAME,
            WORKER_PROTOCOL_VERSION,
            WorkerMessageType.HANDSHAKE_REQUEST.value,
            "request-1",
            0,
            {"api_key": "secret"},
        )
    with pytest.raises(AppError, match=r"worker\.protocol_invalid"):
        decode_jsonl(
            json.dumps(
                {
                    "protocol": WORKER_PROTOCOL_NAME,
                    "version": WORKER_PROTOCOL_VERSION,
                    "message_type": WorkerMessageType.HANDSHAKE_REQUEST.value,
                    "request_id": "request-1",
                    "sequence": 0,
                    "payload": {},
                    "api_key": "secret",
                }
            )
        )


def test_progress_payload_cannot_contain_precise_percentage() -> None:
    with pytest.raises(AppError, match=r"worker\.progress_invalid"):
        WorkerEnvelope(
            WORKER_PROTOCOL_NAME,
            WORKER_PROTOCOL_VERSION,
            WorkerMessageType.OPERATION_PROGRESS.value,
            "request-1",
            0,
            {"operation": "asr", "phase": "transcribing", "percentage": 50},
            "job-1",
            "attempt-1",
        )


def test_progress_domain_value_is_immutable_and_has_no_percentage_field() -> None:
    progress = OperationProgress("runtime", "activating", "runtime.activating", {"backend": "mlx"})
    assert "percentage" not in progress.to_payload()
    with pytest.raises(TypeError):
        progress.detail_parameters["backend"] = "cpu"  # type: ignore[index]  # immutable contract test
