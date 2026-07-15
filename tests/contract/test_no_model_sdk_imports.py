from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
FORBIDDEN = {"openai", "faster_whisper", "torch", "transformers", "qwen_asr", "nemo"}


def test_source_has_no_forbidden_sdk_imports() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module.split(".", 1)[0]]
            else:
                continue
            assert not FORBIDDEN.intersection(modules), path
