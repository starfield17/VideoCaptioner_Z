from __future__ import annotations

import json

from tests.support import make_transcript

from captioner.adapters.exporters.transcript_json import serialize, serialize_bytes


def test_transcript_json_is_deterministic_utf8_and_does_not_mutate() -> None:
    transcript = make_transcript(("你好 ", "世界"), language="zh-CN", metadata={"note": "字幕"})
    first = serialize(transcript)
    second = serialize(transcript)
    assert first == second
    assert first.endswith("\n")
    assert "你好" in first
    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["transcript"]["words"][0]["start_ms"] == 0
    assert serialize_bytes(transcript) == first.encode("utf-8")
