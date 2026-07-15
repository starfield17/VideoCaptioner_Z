from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.core.domain.errors import AppError
from captioner.i18n.catalog import (
    Catalog,
    catalog_to_json_value,
    load_catalog,
    validate_catalog_directory,
    validate_catalog_pair,
)


def _catalog_payload(
    locale: str, messages: dict[str, str], fallback: str | None = None
) -> dict[str, object]:
    return {
        "_meta": {"locale": locale, "name": locale, "fallback": fallback, "schema_version": 1},
        "messages": messages,
    }


def _write_catalog(directory: Path, locale: str, messages: dict[str, str]) -> Path:
    path = directory / f"{locale}.json"
    path.write_text(json.dumps(_catalog_payload(locale, messages)), encoding="utf-8")
    return path


def test_builtin_catalogs_validate() -> None:
    resource_dir = Path(__file__).parents[2] / "resources" / "i18n"
    catalogs = validate_catalog_directory(resource_dir)
    assert [catalog.locale for catalog in catalogs] == ["en", "zh-CN"]


def test_missing_and_malformed_catalogs_fail(tmp_path: Path) -> None:
    with pytest.raises(AppError, match="catalog_missing"):
        load_catalog(tmp_path / "en.json", expected_locale="en")
    malformed = tmp_path / "en.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(AppError, match="invalid_json"):
        load_catalog(malformed, expected_locale="en")


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "en.json"
    path.write_text(
        '{"_meta":{"locale":"en","name":"English","fallback":null,"schema_version":1},'
        '"messages":{"same":"one","same":"two"}}',
        encoding="utf-8",
    )
    with pytest.raises(AppError, match="duplicate_key"):
        load_catalog(path, expected_locale="en")


def test_locale_metadata_and_filename_mismatch_fail(tmp_path: Path) -> None:
    path = tmp_path / "zh-CN.json"
    path.write_text(json.dumps(_catalog_payload("en", {"hello": "Hello"})), encoding="utf-8")
    with pytest.raises(AppError, match="locale_mismatch"):
        load_catalog(path, expected_locale="zh-CN")

    bad_filename = tmp_path / "fr.json"
    bad_filename.write_text(
        json.dumps(_catalog_payload("fr", {"hello": "Hello"})), encoding="utf-8"
    )
    _write_catalog(tmp_path, "en", {"hello": "Hello"})
    with pytest.raises(AppError, match="locale_filename_invalid"):
        validate_catalog_directory(tmp_path)


def test_unknown_key_and_placeholder_mismatch_fail() -> None:
    english = Catalog("en", "English", None, 1, {"message": "{current}/{total}"})
    unknown = Catalog("zh-CN", "中文", "en", 1, {"other": "其他"})
    with pytest.raises(AppError, match="unknown_key"):
        validate_catalog_pair(english, unknown)
    mismatch = Catalog("zh-CN", "中文", "en", 1, {"message": "{current}"})
    with pytest.raises(AppError, match="placeholder_mismatch"):
        validate_catalog_pair(english, mismatch)


def test_empty_translation_and_json_representation(tmp_path: Path) -> None:
    path = tmp_path / "en.json"
    path.write_text(json.dumps(_catalog_payload("en", {"hello": "  "})), encoding="utf-8")
    with pytest.raises(AppError, match="empty_translation"):
        load_catalog(path, expected_locale="en")
    catalog = Catalog("en", "English", None, 1, {"hello": "Hello"})
    assert catalog_to_json_value(catalog)["messages"] == {"hello": "Hello"}
