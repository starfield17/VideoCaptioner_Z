"""Validate built-in i18n catalogs and GUI literal translation keys."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.i18n.catalog import validate_catalog_directory

ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "src" / "captioner" / "gui"
I18N_DIR = ROOT / "resources" / "i18n"

# Dynamic key families used by GUI code (finite prefixes, not free-form user text).
DYNAMIC_KEY_PREFIXES = (
    "gui.queue.state.",
    "gui.queue.profile.",
    "gui.queue.stage.",
    "gui.queue.column.",
    "gui.profile.",
    "gui.device.",
    "gui.create.collision.",
    "gui.create.input.rejection.",
    "gui.settings.credential.",
    "gui.activity.event.",
    "gui.recovery.state.",
    "gui.job.action.",
    "gui.job.confirm.",
    "gui.diagnostics.",
    "gui.value.",
)

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_messages(path: Path) -> dict[str, str]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"invalid catalog structure: {path}", file=sys.stderr)
        raise SystemExit(1)
    root = cast(dict[str, object], data)
    messages_obj = root.get("messages")
    if not isinstance(messages_obj, dict):
        print(f"invalid catalog structure: {path}", file=sys.stderr)
        raise SystemExit(1)
    messages = cast(dict[object, object], messages_obj)
    result: dict[str, str] = {}
    for raw_key, raw_value in messages.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            print(f"invalid catalog message types: {path}", file=sys.stderr)
            raise SystemExit(1)
        result[raw_key] = raw_value
    return result


def _placeholders(text: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(text)


def _extract_literal_keys(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name: str | None = None
        if (isinstance(func, ast.Name) and func.id == "translate") or (
            isinstance(func, ast.Attribute) and func.attr == "translate"
        ):
            name = "translate"
        if name is None or not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            keys.add(arg0.value)
    return keys


def _check_catalog_parity(en: dict[str, str], zh: dict[str, str]) -> list[str]:
    errors: list[str] = []
    en_keys = set(en)
    zh_keys = set(zh)
    missing_zh = sorted(en_keys - zh_keys)
    missing_en = sorted(zh_keys - en_keys)
    if missing_zh:
        errors.append(f"zh-CN missing keys: {', '.join(missing_zh[:20])}")
    if missing_en:
        errors.append(f"en missing keys: {', '.join(missing_en[:20])}")
    for key in sorted(en_keys & zh_keys):
        en_ph = _placeholders(en[key])
        zh_ph = _placeholders(zh[key])
        if sorted(en_ph) != sorted(zh_ph):
            errors.append(f"placeholder name mismatch for {key}: en={en_ph} zh={zh_ph}")
        if len(en_ph) != len(zh_ph):
            errors.append(f"placeholder count mismatch for {key}")
        for catalog_name, text in (("en", en[key]), ("zh-CN", zh[key])):
            if key.startswith("gui.") and text.strip() != text:
                errors.append(f"{catalog_name} GUI value has outer whitespace: {key}")
            if key.startswith("gui.") and not text:
                errors.append(f"{catalog_name} GUI value is empty: {key}")
            # Malformed braces: odd number of unescaped braces.
            if text.count("{") != text.count("}"):
                errors.append(f"{catalog_name} malformed braces: {key}")
    return errors


def _check_gui_literal_keys(en: dict[str, str], zh: dict[str, str]) -> list[str]:
    errors: list[str] = []
    keys: set[str] = set()
    for path in GUI_ROOT.rglob("*.py"):
        keys |= _extract_literal_keys(path)
    for key in sorted(keys):
        if key not in en:
            errors.append(f"GUI literal key missing in en: {key}")
        if key not in zh:
            errors.append(f"GUI literal key missing in zh-CN: {key}")
    return errors


def main() -> int:
    """Run the built-in catalog validator plus GUI parity checks."""
    try:
        catalogs = validate_catalog_directory(I18N_DIR, strict=True)
    except AppError as exc:
        print(f"i18n validation failed: {exc}", file=sys.stderr)
        return 1

    en = _load_messages(I18N_DIR / "en.json")
    zh = _load_messages(I18N_DIR / "zh-CN.json")
    errors = _check_catalog_parity(en, zh)
    errors.extend(_check_gui_literal_keys(en, zh))
    if errors:
        for error in errors:
            print(f"i18n check failed: {error}", file=sys.stderr)
        return 1
    print(
        f"i18n validation passed: {len(catalogs)} catalog(s), "
        f"{len(en)} keys, GUI literal keys present"
    )
    # Keep dynamic prefixes referenced so audits remain intentional.
    _ = DYNAMIC_KEY_PREFIXES
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
