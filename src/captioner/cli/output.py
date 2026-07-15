"""CLI output formatting."""

from __future__ import annotations

import json
from collections.abc import Mapping

from captioner.core.domain.result import JsonValue


def render(payload: Mapping[str, JsonValue], *, as_json: bool) -> str:
    """Render a JSON object or a stable human-readable key/value view."""
    if as_json:
        return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2)
    lines = [f"{key}: {payload[key]}" for key in payload]
    return "\n".join(lines)
