"""Validate all built-in i18n catalogs."""

from __future__ import annotations

import sys
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.i18n.catalog import validate_catalog_directory

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the built-in catalog validator."""
    resource_dir = ROOT / "resources" / "i18n"
    try:
        catalogs = validate_catalog_directory(resource_dir, strict=True)
    except AppError as exc:
        print(f"i18n validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"i18n validation passed: {len(catalogs)} catalog(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
