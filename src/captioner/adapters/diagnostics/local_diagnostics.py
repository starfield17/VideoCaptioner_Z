"""Local diagnostics probes and atomic redacted ZIP bundle writer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

from captioner.core.application.diagnostics import (
    DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    DiagnosticExportRequest,
    DiagnosticExportResult,
    DiagnosticsSnapshot,
    DiagnosticsStorageLocations,
    RuntimeAvailability,
)
from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import AppPaths, resolve_app_paths

_MAX_MEMBER_BYTES = 512 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
_MAX_ZIP_BYTES = 2 * 1024 * 1024

_BUNDLE_MEMBER_ORDER = (
    "application.json",
    "configuration.json",
    "queue.json",
    "recovery.json",
    "capabilities.json",
)


def _canonical_json_bytes(value: object) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_packaged() -> bool:
    if getattr(sys, "frozen", False):
        return True
    module = sys.modules.get("__main__")
    return module is not None and getattr(module, "__compiled__", None) is not None


def _app_version() -> str:
    try:
        return importlib.metadata.version("captioner")
    except Exception:
        return "0.0.0"


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class LocalDiagnosticsAdapter:
    """Implements environment probes and whitelist-only ZIP export."""

    def __init__(self, *, paths: AppPaths | None = None) -> None:
        self._paths = paths

    def collect_runtime_availability(
        self,
        *,
        provider_configured: bool,
        credential_source: Literal["config", "environment", "missing"],
    ) -> RuntimeAvailability:
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        asr_spec = importlib.util.find_spec("faster_whisper")
        return RuntimeAvailability(
            packaged=_is_packaged(),
            operating_system=platform.system() or "unknown",
            architecture=platform.machine() or "unknown",
            python_version=platform.python_version(),
            app_version=_app_version(),
            ffmpeg_available=ffmpeg_path is not None,
            ffprobe_available=ffprobe_path is not None,
            asr_runtime_available=asr_spec is not None,
            provider_configured=provider_configured,
            credential_source=credential_source,
        )

    def collect_storage_locations(self) -> DiagnosticsStorageLocations:
        """Project resolved writable locations without creating or exporting them."""
        paths = self._paths if self._paths is not None else resolve_app_paths()
        return DiagnosticsStorageLocations(
            config_dir=str(paths.config_dir),
            data_dir=str(paths.data_dir),
            models_dir=str(paths.models_dir),
            runtimes_dir=str(paths.runtimes_dir),
            workspaces_dir=str(paths.workspaces_dir),
            cache_dir=str(paths.cache_dir),
            log_dir=str(paths.log_dir),
            downloads_dir=str(paths.downloads_dir),
            artifacts_dir=str(paths.artifacts_dir),
            staging_dir=str(paths.staging_dir),
        )

    def write_bundle(
        self,
        request: DiagnosticExportRequest,
        *,
        snapshot: DiagnosticsSnapshot,
    ) -> DiagnosticExportResult:
        destination = Path(request.destination)
        self._validate_destination(destination, overwrite=request.overwrite)
        members = self._build_members(snapshot)
        total_uncompressed = sum(len(payload) for payload in members.values())
        if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise AppError("diagnostics.bundle_invalid", {"reason": "payload_too_large"})

        parent = destination.parent
        temp_name = f".captioner-diagnostics-{uuid4().hex}.tmp.zip"
        temp_path = parent / temp_name
        try:
            size_bytes, digest = self._write_temp_zip(temp_path, members)
            os.replace(temp_path, destination)
            if os.name == "posix":
                os.chmod(destination, 0o600)
            _fsync_directory(parent)
        except AppError:
            self._cleanup_temp(temp_path)
            raise
        except OSError as exc:
            self._cleanup_temp(temp_path)
            raise AppError("diagnostics.write_failed") from exc
        except Exception as exc:
            self._cleanup_temp(temp_path)
            raise AppError("diagnostics.write_failed") from exc
        finally:
            self._cleanup_temp(temp_path)

        return DiagnosticExportResult(
            request_id=request.request_id,
            destination=str(destination),
            size_bytes=size_bytes,
            sha256=digest,
        )

    def _write_temp_zip(
        self,
        temp_path: Path,
        members: dict[str, bytes],
    ) -> tuple[int, str]:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=False,
        ) as archive:
            # Fixed member order: manifest first, then payload members.
            for name in ("manifest.json", *_BUNDLE_MEMBER_ORDER):
                archive.writestr(name, members[name])
        with temp_path.open("rb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        size_bytes = temp_path.stat().st_size
        if size_bytes > _MAX_ZIP_BYTES:
            raise AppError("diagnostics.bundle_invalid", {"reason": "zip_too_large"})
        digest = _sha256_hex(temp_path.read_bytes())
        return size_bytes, digest

    def _validate_destination(self, destination: Path, *, overwrite: bool) -> None:
        if destination.suffix.lower() != ".zip":
            raise AppError("diagnostics.destination_invalid", {"reason": "suffix"})
        parent = destination.parent
        if not parent.exists() or not parent.is_dir():
            raise AppError("diagnostics.destination_invalid", {"reason": "parent"})
        if destination.exists():
            if destination.is_symlink():
                raise AppError("diagnostics.destination_invalid", {"reason": "symlink"})
            if not overwrite:
                raise AppError("diagnostics.destination_exists")

    def _build_members(self, snapshot: DiagnosticsSnapshot) -> dict[str, bytes]:
        runtime = snapshot.runtime
        application = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "generated_at_utc": snapshot.generated_at_utc,
            "app_version": runtime.app_version,
            "packaged": runtime.packaged,
            "operating_system": runtime.operating_system,
            "architecture": runtime.architecture,
            "python_version": runtime.python_version,
        }
        configuration = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "locale": snapshot.configuration.locale,
            "built_in_preset_count": snapshot.configuration.built_in_preset_count,
            "user_preset_count": snapshot.configuration.user_preset_count,
            "provider_configured": snapshot.configuration.provider_configured,
            "credential_source": snapshot.configuration.credential_source,
            "issue_codes": list(snapshot.configuration.issue_codes),
        }
        queue = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "revision": snapshot.queue.revision,
            "active_jobs": snapshot.queue.active_jobs,
            "terminal_jobs": snapshot.queue.terminal_jobs,
            "omitted_terminal_jobs": snapshot.queue.omitted_terminal_jobs,
            "state_counts": [list(pair) for pair in snapshot.queue.state_counts],
            "profile_counts": [list(pair) for pair in snapshot.queue.profile_counts],
            "stage_counts": [list(pair) for pair in snapshot.queue.stage_counts],
            "issue_codes": [list(pair) for pair in snapshot.queue.issue_codes],
        }
        recovery = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "recoverable_batches": snapshot.recovery.recoverable_batches,
            "blocked_batches": snapshot.recovery.blocked_batches,
            "paused_batches": snapshot.recovery.paused_batches,
            "issue_codes": [list(pair) for pair in snapshot.recovery.issue_codes],
        }
        capabilities = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "ffmpeg_available": runtime.ffmpeg_available,
            "ffprobe_available": runtime.ffprobe_available,
            "asr_runtime_available": runtime.asr_runtime_available,
            "provider_configured": runtime.provider_configured,
            "credential_source": runtime.credential_source,
        }

        payload: dict[str, bytes] = {
            "application.json": _canonical_json_bytes(application),
            "configuration.json": _canonical_json_bytes(configuration),
            "queue.json": _canonical_json_bytes(queue),
            "recovery.json": _canonical_json_bytes(recovery),
            "capabilities.json": _canonical_json_bytes(capabilities),
        }
        for name, body in payload.items():
            if len(body) > _MAX_MEMBER_BYTES:
                raise AppError(
                    "diagnostics.bundle_invalid",
                    {"reason": "member_too_large", "name": name},
                )

        files = [
            {
                "name": name,
                "size_bytes": len(payload[name]),
                "sha256": _sha256_hex(payload[name]),
            }
            for name in _BUNDLE_MEMBER_ORDER
        ]
        manifest = {
            "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            "generated_at_utc": snapshot.generated_at_utc,
            "files": files,
        }
        payload["manifest.json"] = _canonical_json_bytes(manifest)
        if len(payload["manifest.json"]) > _MAX_MEMBER_BYTES:
            raise AppError("diagnostics.bundle_invalid", {"reason": "member_too_large"})
        return payload

    @staticmethod
    def _cleanup_temp(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            return


__all__ = ["LocalDiagnosticsAdapter"]
