"""Offline static validators for installed model payloads."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelFileEntry,
    ModelManifest,
    ModelValidationCheck,
    ModelValidationReport,
    required_files_for_format,
)

DEFAULT_MAX_MODEL_FILE_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_MAX_MODEL_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_MAX_MODEL_FILES = 4096
DEFAULT_MAX_JSON_BYTES = 8 * 1024 * 1024


class FilesystemModelValidator:
    """Validate only local regular files; never imports a model SDK."""

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_MODEL_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_MODEL_TOTAL_BYTES,
        max_files: int = DEFAULT_MAX_MODEL_FILES,
        max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    ) -> None:
        if min(max_file_bytes, max_total_bytes, max_files, max_json_bytes) <= 0:
            raise ValueError
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_files = max_files
        self.max_json_bytes = max_json_bytes

    def validate(self, manifest: ModelManifest, model_directory: Path) -> ModelValidationReport:
        checks: list[ModelValidationCheck] = []
        try:
            inventory = self.inventory(model_directory)
        except AppError as exc:
            return _failure_report(exc.code)
        actual = {entry.relative_path: entry for entry in inventory}
        expected = {entry.relative_path: entry for entry in manifest.files}
        checks.append(_check("regular_files", True))
        if set(actual) != set(expected):
            checks.append(_check("manifest_files", False, "model.extra_or_missing_file"))
        else:
            checks.append(_check("manifest_files", True))
        hash_ok = all(actual.get(path) == entry for path, entry in expected.items())
        checks.append(_check("file_hashes", hash_ok, "model.file_hash_mismatch"))
        required_ok, required_code = self._required_files_ok(manifest.model_format, set(actual))
        checks.append(_check("required_files", required_ok, required_code))
        if manifest.model_format == "faster-whisper-ct2":
            model_bin = actual.get("model.bin")
            checks.append(
                _check(
                    "model_bin_nonempty",
                    model_bin is not None and model_bin.size_bytes > 0,
                    "model.model_bin_empty",
                )
            )
        elif manifest.model_format == "mlx-whisper":
            weight_entries = [
                actual[name]
                for name in ("model.safetensors", "weights.safetensors", "weights.npz")
                if name in actual
            ]
            checks.append(
                _check(
                    "weights_nonempty",
                    any(entry.size_bytes > 0 for entry in weight_entries),
                    "model.weights_empty",
                )
            )
        checks.extend(self.json_checks(manifest.model_format, model_directory, set(actual)))
        checks.append(
            _check(
                "backend_format",
                manifest.identity.backend_id in manifest.compatible_runtime_backends,
                "model.format_backend_mismatch",
            )
        )
        ok = all(check.ok for check in checks)
        return ModelValidationReport(
            ok=ok,
            checks=tuple(checks),
            error_code=None
            if ok
            else next(
                (check.error_code for check in checks if not check.ok),
                "model.validation_failed",
            ),
            message_code=None
            if ok
            else next(
                (check.message_code for check in checks if not check.ok),
                "model.validation_failed",
            ),
        )

    def inventory(self, model_directory: Path) -> tuple[ModelFileEntry, ...]:
        root = model_directory.expanduser()
        if root.is_symlink():
            raise AppError("model.symlink_rejected")
        root = root.resolve()
        if not root.is_dir():
            raise AppError("model.directory_missing")
        entries: list[ModelFileEntry] = []
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise AppError("model.symlink_rejected")
            try:
                mode = path.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise AppError("model.inventory_failed") from exc
            if path.is_dir():
                continue
            if not stat.S_ISREG(mode):
                raise AppError("model.special_file_rejected")
            relative = path.relative_to(root).as_posix()
            if any(part.startswith(".") for part in relative.split("/")):
                raise AppError("model.hidden_file_rejected")
            size = path.stat().st_size
            if size > self.max_file_bytes:
                raise AppError("model.file_too_large")
            total += size
            if total > self.max_total_bytes:
                raise AppError("model.total_too_large")
            if len(entries) >= self.max_files:
                raise AppError("model.too_many_files")
            entries.append(ModelFileEntry(relative, size, _sha256(path)))
        if not entries:
            raise AppError("model.empty_directory")
        return tuple(entries)

    def _required_files_ok(self, model_format: str, paths: set[str]) -> tuple[bool, str]:
        groups = required_files_for_format(model_format)
        if model_format == "mlx-whisper":
            tokenizer_ok = "tokenizer.json" in paths or {"vocab.json", "merges.txt"} <= paths
            if not tokenizer_ok:
                return False, "model.mlx_tokenizer_missing"
        if not groups or any(not group <= paths for group in groups[:1]):
            return False, "model.required_files_missing"
        if len(groups) > 1 and not groups[1] & paths:
            return False, "model.required_files_missing"
        return True, ""

    def json_checks(
        self, model_format: str, root: Path, paths: set[str]
    ) -> tuple[ModelValidationCheck, ...]:
        names = ["config.json"]
        if model_format == "faster-whisper-ct2":
            names.append("tokenizer.json")
        elif model_format == "mlx-whisper":
            if "tokenizer.json" in paths:
                names.append("tokenizer.json")
            else:
                names.extend(("vocab.json",))
        checks: list[ModelValidationCheck] = []
        for name in names:
            path = root / name
            try:
                size = path.stat().st_size
            except OSError:
                checks.append(_check(f"json:{name}", False, "model.json_invalid"))
                continue
            if size > self.max_json_bytes:
                checks.append(_check(f"json:{name}", False, "model.json_invalid"))
                continue
            try:
                value = json.loads(
                    path.read_text(encoding="utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                checks.append(_check(f"json:{name}", False, "model.json_invalid"))
                continue
            if name == "config.json" and not isinstance(value, dict):
                checks.append(_check(f"json:{name}", False, "model.json_invalid"))
            else:
                checks.append(_check(f"json:{name}", True))
        return tuple(checks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AppError("model.inventory_failed") from exc
    return digest.hexdigest()


def _check(name: str, ok: bool, code: str | None = None) -> ModelValidationCheck:
    return ModelValidationCheck(
        name=name,
        ok=ok,
        error_code=None if ok else code,
        message_code=None if ok else code,
    )


def _failure_report(code: str) -> ModelValidationReport:
    return ModelValidationReport(
        ok=False,
        checks=(_check("inventory", False, code),),
        error_code=code,
        message_code=code,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


__all__ = [
    "DEFAULT_MAX_JSON_BYTES",
    "DEFAULT_MAX_MODEL_FILES",
    "DEFAULT_MAX_MODEL_FILE_BYTES",
    "DEFAULT_MAX_MODEL_TOTAL_BYTES",
    "FilesystemModelValidator",
]
