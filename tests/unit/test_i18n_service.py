from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.core.domain.errors import AppError
from captioner.i18n.service import I18nService


def _write(directory: Path, locale: str, messages: dict[str, str]) -> None:
    payload = {
        "_meta": {
            "locale": locale,
            "name": locale,
            "fallback": None if locale == "en" else "en",
            "schema_version": 1,
        },
        "messages": messages,
    }
    (directory / f"{locale}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_per_key_fallback_and_formatting(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"hello": "Hello {name}", "bye": "Goodbye"})
    _write(tmp_path, "zh-CN", {"hello": "你好 {name}"})
    service = I18nService("zh_cn", resource_dir=tmp_path)
    assert service.translate("hello", {"name": "Ada"}) == "你好 Ada"
    assert service.t("bye") == "Goodbye"


def test_override_catalog_has_highest_priority(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"hello": "Hello", "bye": "Bye"})
    _write(tmp_path, "zh-CN", {"hello": "你好", "bye": "再见"})
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps(
            {
                "_meta": {
                    "locale": "zh-CN",
                    "name": "Override",
                    "fallback": "en",
                    "schema_version": 1,
                },
                "messages": {"hello": "嗨"},
            }
        ),
        encoding="utf-8",
    )
    service = I18nService("zh-CN", resource_dir=tmp_path, override_path=override)
    assert service.translate("hello") == "嗨"
    assert service.translate("bye") == "再见"


def test_missing_current_catalog_falls_back_only_in_non_strict_mode(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"hello": "Hello"})
    with pytest.raises(AppError, match="catalog_missing"):
        I18nService("zh-CN", resource_dir=tmp_path, strict=True)
    service = I18nService("zh-CN", resource_dir=tmp_path, strict=False)
    assert service.translate("hello") == "Hello"


def test_unknown_key_and_missing_format_parameter_are_structured(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"hello": "Hello {name}"})
    service = I18nService(resource_dir=tmp_path)
    with pytest.raises(AppError, match="unknown_key"):
        service.translate("unknown")
    with pytest.raises(AppError, match="format_failed"):
        service.translate("hello")


def test_malformed_catalog_is_not_silently_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "en", {"hello": "Hello"})
    (tmp_path / "zh-CN.json").write_text("{", encoding="utf-8")
    with pytest.raises(AppError, match="invalid_json"):
        I18nService("zh-CN", resource_dir=tmp_path)
