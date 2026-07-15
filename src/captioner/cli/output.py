"""CLI output formatting."""

from __future__ import annotations

import json
from collections.abc import Mapping

from captioner.core.domain.result import JsonValue
from captioner.i18n.service import I18nService

DOCTOR_LABEL_KEYS = {
    "version": "cli.doctor.version",
    "python_version": "cli.doctor.python",
    "platform": "cli.doctor.platform",
    "resource_root": "cli.doctor.resource_root",
    "config_dir": "cli.doctor.config_dir",
    "data_dir": "cli.doctor.data_dir",
    "cache_dir": "cli.doctor.cache_dir",
    "log_dir": "cli.doctor.log_dir",
    "temp_dir": "cli.doctor.temp_dir",
    "locale": "cli.doctor.locale",
    "catalog_valid": "cli.doctor.catalog_valid",
}


def doctor_labels(service: I18nService) -> dict[str, str]:
    """Resolve human-readable doctor labels at the CLI output boundary."""
    return {field: service.translate(key) for field, key in DOCTOR_LABEL_KEYS.items()}


def render(
    payload: Mapping[str, JsonValue], *, as_json: bool, labels: Mapping[str, str] | None = None
) -> str:
    """Render a JSON object or a stable human-readable key/value view."""
    if as_json:
        return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2)
    lines = [
        f"{labels.get(key, key) if labels is not None else key}: {payload[key]}" for key in payload
    ]
    return "\n".join(lines)
