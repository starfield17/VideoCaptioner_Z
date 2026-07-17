"""Phase 0 diagnostics command."""

from __future__ import annotations

import platform
from dataclasses import dataclass

from captioner import __version__
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.i18n.catalog import validate_catalog_directory
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import AppPaths, resolve_app_paths


@dataclass(frozen=True, slots=True)
class DoctorOptions:
    locale: str
    as_json: bool
    paths: AppPaths | None = None
    tokenizer_smoke: bool = False


def run(options: DoctorOptions, *, service: I18nService | None = None) -> dict[str, JsonValue]:
    """Collect read-only Phase 0 diagnostics."""
    paths = resolve_app_paths() if options.paths is None else options.paths
    message_service = (
        I18nService(
            locale=options.locale,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )
        if service is None
        else service
    )
    catalog_valid = True
    try:
        validate_catalog_directory(paths.i18n_resource_dir, strict=True)
    except AppError:
        catalog_valid = False

    payload: dict[str, JsonValue] = {
        "version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "resource_root": str(paths.resource_root),
        "config_dir": str(paths.config_dir),
        "data_dir": str(paths.data_dir),
        "cache_dir": str(paths.cache_dir),
        "log_dir": str(paths.log_dir),
        "temp_dir": str(paths.temp_dir),
        "locale": message_service.locale,
        "catalog_valid": catalog_valid,
    }
    if options.tokenizer_smoke:
        from captioner.adapters.llm.token_counter import ModelTokenCounter

        fixture = "ASCII 中文 日本語 👩🏽‍💻 123"
        payload["tokenizers"] = {
            tokenizer_id: ModelTokenCounter(
                tokenizer_id,
                resource_dir=paths.tokenizer_resource_dir,
            ).count(fixture)
            for tokenizer_id in ("cl100k_base", "o200k_base")
        }
    return payload
