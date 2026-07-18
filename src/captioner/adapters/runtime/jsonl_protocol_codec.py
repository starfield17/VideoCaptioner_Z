"""Adapter-facing facade for the pure Worker JSONL Protocol v1 codec."""

from captioner.core.domain.worker_protocol import (
    JsonlProtocolCodec,
    WorkerEnvelope,
    decode_jsonl,
    decode_stdout_line,
    encode_jsonl,
)

encode_line = encode_jsonl
decode_line = decode_jsonl
decode_stdout = decode_stdout_line

__all__ = [
    "JsonlProtocolCodec",
    "WorkerEnvelope",
    "decode_jsonl",
    "decode_line",
    "decode_stdout",
    "decode_stdout_line",
    "encode_jsonl",
    "encode_line",
]
