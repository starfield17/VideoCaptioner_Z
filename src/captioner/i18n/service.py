"""Locale-aware message lookup with per-key fallback."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.i18n.catalog import Catalog, load_catalog, validate_catalog_pair
from captioner.i18n.locale import normalize_locale
from captioner.infrastructure.app_paths import resolve_app_paths

LOGGER = logging.getLogger("captioner.i18n")


class I18nService:
    """Resolve messages from override, current locale, then English."""

    def __init__(
        self,
        locale: str = "en",
        *,
        resource_dir: Path | None = None,
        override_path: Path | None = None,
        strict: bool = True,
    ) -> None:
        self.strict = strict
        self.locale = normalize_locale(locale, strict=strict)
        self.resource_dir = (
            resolve_app_paths().i18n_resource_dir if resource_dir is None else resource_dir
        )
        self._english = self._load_optional_catalog("en", required=strict)
        self._current = self._load_current_catalog()
        validate_catalog_pair(self._english, self._current, strict=strict)
        self._override = self._load_override(override_path)
        if self._override is not None:
            validate_catalog_pair(self._english, self._override, strict=strict)

    @property
    def english_catalog(self) -> Catalog:
        return self._english

    @property
    def current_catalog(self) -> Catalog:
        return self._current

    @property
    def override_catalog(self) -> Catalog | None:
        return self._override

    def translate(self, key: str, params: Mapping[str, JsonValue] | None = None) -> str:
        """Translate and format one key without producing a localized error."""
        if key not in self._english.messages:
            if self.strict:
                raise AppError("i18n.unknown_key", {"key": key})
            LOGGER.warning("Missing English message: %s", key)
            return key

        value = self._message_for_key(key)
        if value is None:
            LOGGER.warning("Missing English message: %s", key)
            return key
        if params is None:
            params = {}
        try:
            return value.format(**dict(params))
        except (IndexError, KeyError, ValueError) as exc:
            missing = str(exc).strip("'\"")
            raise AppError("i18n.format_failed", {"key": key, "param": missing}) from exc

    def t(self, key: str, params: Mapping[str, JsonValue] | None = None) -> str:
        """Short alias for ``translate``."""
        return self.translate(key, params)

    def _message_for_key(self, key: str) -> str | None:
        if self._override is not None and key in self._override.messages:
            return self._override.messages[key]
        if key in self._current.messages:
            return self._current.messages[key]
        return self._english.messages.get(key)

    def _load_current_catalog(self) -> Catalog:
        if self.locale == "en":
            return self._english
        path = self.resource_dir / f"{self.locale}.json"
        try:
            return load_catalog(path, expected_locale=self.locale)
        except AppError:
            if self.strict:
                raise
            LOGGER.warning("Falling back to English catalog for locale %s", self.locale)
            return _empty_catalog(self.locale, "Fallback")

    def _load_optional_catalog(self, locale: str, *, required: bool) -> Catalog:
        path = self.resource_dir / f"{locale}.json"
        try:
            return load_catalog(path, expected_locale=locale)
        except AppError:
            if required:
                raise
            LOGGER.warning("English catalog is unavailable")
            return _empty_catalog(locale, "English")

    def _load_override(self, path: Path | None) -> Catalog | None:
        if path is None:
            return None
        catalog = load_catalog(path)
        if catalog.locale not in {self.locale, "en"}:
            raise AppError("i18n.override_locale_mismatch", {"locale": catalog.locale})
        return catalog


def _empty_catalog(locale: str, name: str) -> Catalog:
    return Catalog(locale=locale, name=name, fallback="en", schema_version=1, messages={})
