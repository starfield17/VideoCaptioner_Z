"""Runtime-local immutable build metadata loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast


def load_build_info(runtime_root: Path | None = None) -> dict[str, object]:
    root = Path.cwd() if runtime_root is None else runtime_root
    path = root / "payload" / "build_info.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("build_info_object_required")
    return cast(dict[str, object], value)
