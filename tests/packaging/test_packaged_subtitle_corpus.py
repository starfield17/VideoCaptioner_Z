from __future__ import annotations

import json
import subprocess

import pytest
from scripts.build_nuitka import layout_for_platform


@pytest.mark.skipif(
    not layout_for_platform().executable_path.is_file(),
    reason="run the Nuitka build before the packaged corpus smoke test",
)
def test_packaged_subtitle_corpus_passes_all_fixtures() -> None:
    executable = layout_for_platform().executable_path
    result = subprocess.run(
        [
            str(executable),
            "--cli",
            "subtitle-corpus",
            "tests/fixtures/transcripts",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["fixture_count"] == 14
    assert report["passed"] == 14
    assert report["failed"] == 0
    assert report["errors"] == []
    assert all(not fixture["errors"] for fixture in report["fixtures"])
