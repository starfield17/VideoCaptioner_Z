from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_golden_update_requires_explicit_acknowledgement() -> None:
    golden_root = ROOT / "tests" / "golden" / "data"
    before = {path: path.read_bytes() for path in golden_root.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/update_subtitle_goldens.py",
            "--fixtures",
            "tests/fixtures/transcripts",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    after = {path: path.read_bytes() for path in golden_root.iterdir()}
    assert result.returncode == 2
    assert "PHASE3_GOLDENS_REVIEWED" in result.stderr
    assert after == before
