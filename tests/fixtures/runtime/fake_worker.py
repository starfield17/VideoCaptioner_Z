"""Small subprocess fixture for Core-side Worker lifecycle tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import cast


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
    sequence = 0
    handshake_request = _read()
    _write(
        _envelope(
            "handshake.response",
            cast(str, handshake_request["request_id"]),
            sequence,
            {
                "protocol_version": "1.1",
                "runtime_id": "faster-whisper-cpu-macos-arm64",
                "runtime_version": "1.0.0",
                "backend_id": "faster-whisper",
                "backend_version": "1.2.1",
                "worker_version": "1.0.0",
                "platform": "macos",
                "architecture": "arm64",
                "capabilities": [
                    "language_detection",
                    "runtime_doctor",
                    "translation_task",
                    "word_timestamps",
                ],
                "supported_devices": ["cpu"],
                "supported_model_formats": ["faster-whisper-ct2"],
                "supported_result_schema_versions": [1],
            },
        )
    )
    sequence += 1
    if mode == "crash-after-handshake":
        return 7
    if mode == "contamination":
        sys.stdout.write("human output\n")
        sys.stdout.flush()
        return 0
    if mode == "partial":
        sys.stdout.buffer.write(b'{"partial":true')
        sys.stdout.buffer.flush()
        return 0
    if mode == "stderr":
        print("worker diagnostic", file=sys.stderr, flush=True)

    while True:
        message = _read()
        message_type = cast(str, message["message_type"])
        request_id = cast(str, message["request_id"])
        if message_type == "shutdown.request":
            _write(
                _envelope(
                    "shutdown.acknowledged",
                    request_id,
                    sequence,
                    {"acknowledged": True},
                )
            )
            if mode == "ack-shutdown-but-hang":
                while True:
                    time.sleep(1)
            return 0
        if message_type != "transcribe.request":
            continue
        job_id = cast(str, message["job_id"])
        attempt_id = cast(str, message["stage_attempt_id"])
        if mode == "crash-during-transcribe":
            return 8
        if mode == "wait-cancel":
            while True:
                cancellation = _read()
                if cancellation["message_type"] != "cancel.request":
                    continue
                cancel_request_id = cast(str, cancellation["request_id"])
                _write(
                    _envelope(
                        "cancel.acknowledged",
                        cancel_request_id,
                        sequence,
                        {"target_request_id": request_id},
                        job_id=job_id,
                        stage_attempt_id=attempt_id,
                    )
                )
                sequence += 1
                _write(
                    _envelope(
                        "operation.cancelled",
                        request_id,
                        sequence,
                        {"target_request_id": request_id},
                        job_id=job_id,
                        stage_attempt_id=attempt_id,
                    )
                )
                sequence += 1
                break
            continue
        if mode == "spawn-child":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            print(f"child_pid={child.pid}", file=sys.stderr, flush=True)
            while True:
                incoming = _read()
                if incoming["message_type"] == "shutdown.request":
                    _write(
                        _envelope(
                            "shutdown.acknowledged",
                            cast(str, incoming["request_id"]),
                            sequence,
                            {"acknowledged": True},
                        )
                    )
                    while True:
                        time.sleep(1)
        if mode == "wrong-correlation":
            request_id = "wrong-request"
        result_sequence = 0 if mode == "wrong-sequence" else sequence
        _write(
            _envelope(
                "operation.result",
                request_id,
                result_sequence,
                {
                    "relative_path": "result.json",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "schema_id": "captioner.transcript",
                    "schema_version": 1,
                },
                job_id=job_id,
                stage_attempt_id=attempt_id,
            )
        )
        sequence += 1
        if mode == "ignore-cancel":
            continue
        # The normal fixture completes one request and waits for shutdown.


def _read() -> dict[str, object]:
    line = sys.stdin.buffer.readline()
    if not line:
        raise SystemExit(0)
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError
    return cast(dict[str, object], value)


def _write(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _envelope(
    message_type: str,
    request_id: str,
    sequence: int,
    payload: dict[str, object],
    *,
    job_id: str | None = None,
    stage_attempt_id: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol": "captioner.worker",
        "version": "1.1",
        "message_type": message_type,
        "request_id": request_id,
        "sequence": sequence,
        "payload": payload,
    }
    if job_id is not None:
        result["job_id"] = job_id
    if stage_attempt_id is not None:
        result["stage_attempt_id"] = stage_attempt_id
    return result


if __name__ == "__main__":
    raise SystemExit(main())
