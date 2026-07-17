from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from captioner.adapters.llm.token_counter import ModelTokenCounter
from captioner.core.domain.errors import AppError

_RESOURCE_DIR = Path("resources/tokenizers").resolve()


def test_packaged_tokenizers_initialize_offline_in_a_clean_subprocess(tmp_path: Path) -> None:
    script = """
from pathlib import Path
import urllib.request

from captioner.adapters.llm.token_counter import ModelTokenCounter

def fail(*args, **kwargs):
    raise AssertionError("network access")

urllib.request.urlopen = fail
resource_dir = Path(r"RESOURCE_DIR")
fixtures = ("ASCII 123", "你好世界", "🙂🚀", "مرحبا")
for tokenizer_id in ("cl100k_base", "o200k_base"):
    counter = ModelTokenCounter(tokenizer_id, resource_dir=resource_dir)
    assert all(counter.count(value) > 0 for value in fixtures)
""".replace("RESOURCE_DIR", str(_RESOURCE_DIR))
    env = dict(os.environ)
    env["TIKTOKEN_CACHE_DIR"] = str(tmp_path / "empty-cache")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "network access" not in result.stderr


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_missing_or_corrupt_packaged_tokenizer_fails_closed(tmp_path: Path, mode: str) -> None:
    resource_dir = tmp_path / "tokenizers"
    shutil.copytree(_RESOURCE_DIR, resource_dir)
    target = resource_dir / "cl100k_base.tiktoken"
    if mode == "missing":
        target.unlink()
    else:
        target.write_bytes(b"corrupt")
    with pytest.raises(AppError, match=r"llm\.tokenizer_unknown"):
        ModelTokenCounter("cl100k_base", resource_dir=resource_dir)
