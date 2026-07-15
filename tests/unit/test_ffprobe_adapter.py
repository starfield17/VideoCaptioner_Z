from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.process import ProcessResult


def _empty_calls() -> list[tuple[str, ...]]:
    return []


@dataclass
class StubProcess:
    result: ProcessResult | AppError
    calls: list[tuple[str, ...]] = field(default_factory=_empty_calls)

    async def run(self, arguments: Sequence[str], context: ExecutionContext) -> ProcessResult:
        self.calls.append(tuple(arguments))
        context.raise_if_cancelled()
        if isinstance(self.result, AppError):
            raise self.result
        return self.result


def _probe_document() -> bytes:
    return json.dumps(
        {
            "streams": [
                {"index": 0, "codec_type": "video"},
                {"index": 2, "codec_type": "audio", "codec_name": "aac", "duration": "1.2"},
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": "1.234",
            },
        }
    ).encode()


def test_ffprobe_selects_actual_audio_stream_and_hashes_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "媒体 sample.mp4"
        source.write_bytes(b"input")
        runner = StubProcess(ProcessResult(_probe_document(), b"", 0))
        asset = await FFprobeMediaInspector(runner).inspect(source, ExecutionContext())
        assert asset.audio_stream_index == 2
        assert asset.duration_ms == 1_234
        assert len(asset.content_hash) == 64
        assert runner.calls[0][-1] == str(source.resolve())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "result",
    [
        ProcessResult(b"not json", b"", 0),
        ProcessResult(
            json.dumps({"streams": [{"index": 0, "codec_type": "video"}], "format": {}}).encode(),
            b"",
            0,
        ),
        ProcessResult(b"", b"bad input", 1),
    ],
)
def test_ffprobe_failure_paths(tmp_path: Path, result: ProcessResult) -> None:
    async def scenario() -> None:
        source = tmp_path / "input.wav"
        source.write_bytes(b"input")
        with pytest.raises(AppError):
            await FFprobeMediaInspector(StubProcess(result)).inspect(source, ExecutionContext())

    asyncio.run(scenario())


def test_ffprobe_missing_executable_and_input_are_structured(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(AppError, match="input_missing"):
            await FFprobeMediaInspector(StubProcess(ProcessResult(b"", b"", 0))).inspect(
                tmp_path / "missing.wav", ExecutionContext()
            )
        source = tmp_path / "input.wav"
        source.write_bytes(b"input")
        runner = StubProcess(AppError("process.executable_not_found"))
        with pytest.raises(AppError, match="ffprobe_not_found"):
            await FFprobeMediaInspector(runner).inspect(source, ExecutionContext())

    asyncio.run(scenario())
