"""Safe Runtime archive inspection, extraction, and manifest helpers."""

from __future__ import annotations

import gzip
import hashlib
import os
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NoReturn

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime import RuntimeFileEntry, RuntimeManifest
from captioner.core.ports.runtime_archive import RuntimeArchive

_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024


class FilesystemRuntimeArchive(RuntimeArchive):
    """Filesystem implementation of the Core Runtime archive port."""

    def sha256_file(self, path: Path) -> str:
        return sha256_file(path)

    def extract(self, archive_path: Path, destination: Path, manifest: RuntimeManifest) -> None:
        safe_extract_archive(archive_path, destination, manifest)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AppError("runtime.archive_read_failed", {"reason": "filesystem"}) from exc
    return digest.hexdigest()


def validate_archive(
    archive_path: Path,
    manifest: RuntimeManifest,
    *,
    max_archive_bytes: int = _MAX_ARCHIVE_BYTES,
) -> tuple[tarfile.TarInfo, ...]:
    """Validate every tar entry before any extraction occurs."""
    try:
        size = archive_path.stat().st_size
    except OSError as exc:
        raise AppError("runtime.archive_read_failed", {"reason": "stat"}) from exc
    if size > max_archive_bytes:
        raise AppError("runtime.archive_too_large")
    expected = {entry.relative_path: entry for entry in manifest.files}
    seen: set[str] = set()
    members: list[tarfile.TarInfo] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                name = _validate_member_name(member.name)
                if name in seen:
                    _fail("runtime.archive_entry_invalid", {"reason": "duplicate"})
                seen.add(name)
                if not member.isdir() and not member.isreg():
                    _fail("runtime.archive_entry_invalid", {"reason": "special"})
                if member.isreg():
                    entry = expected.get(name)
                    if entry is None:
                        _fail("runtime.archive_extra_file", {"path": name})
                    if member.size != entry.size_bytes:
                        _fail("runtime.archive_size_mismatch", {"path": name})
                members.append(member)
    except AppError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise AppError("runtime.archive_invalid", {"reason": "tar"}) from exc

    for entry in manifest.files:
        if entry.relative_path not in seen:
            _fail("runtime.archive_missing_file", {"path": entry.relative_path})
    return tuple(members)


def safe_extract_archive(
    archive_path: Path,
    destination: Path,
    manifest: RuntimeManifest,
    *,
    max_archive_bytes: int = _MAX_ARCHIVE_BYTES,
) -> None:
    """Extract only regular files/directories after complete preflight."""
    members = validate_archive(archive_path, manifest, max_archive_bytes=max_archive_bytes)
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in members:
                relative = _validate_member_name(member.name)
                target = (root / PurePosixPath(relative)).resolve()
                _ensure_inside(target, root)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    _fail("runtime.archive_entry_invalid", {"reason": "duplicate"})
                source = archive.extractfile(member)
                if source is None:
                    _fail("runtime.archive_invalid", {"reason": "file_stream"})
                try:
                    with target.open("xb") as output:
                        for block in iter(source.read, b""):
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    source.close()
    except AppError:
        raise
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, OSError) and exc.errno == 28:
            raise AppError("runtime.disk_full") from exc
        raise AppError("runtime.archive_extract_failed", {"reason": "filesystem"}) from exc
    verify_manifest_files(root, manifest)


def verify_manifest_files(root: Path, manifest: RuntimeManifest) -> None:
    """Verify content and reject regular files not declared by the manifest."""
    _verify_manifest_files(root, manifest, allowed_extra_paths=())


def verify_runtime_payload(
    root: Path,
    manifest: RuntimeManifest,
    *,
    allowed_extra_paths: Iterable[str] = (),
) -> None:
    """Verify a managed Runtime while permitting its sidecar records."""
    _verify_manifest_files(root, manifest, allowed_extra_paths=allowed_extra_paths)


def _verify_manifest_files(
    root: Path,
    manifest: RuntimeManifest,
    *,
    allowed_extra_paths: Iterable[str],
) -> None:
    resolved_root = root.resolve()
    expected = {entry.relative_path: entry for entry in manifest.files}
    allowed = set(allowed_extra_paths)
    for path in resolved_root.rglob("*"):
        if path.is_symlink():
            _fail("runtime.archive_entry_invalid", {"reason": "symlink"})
        if not path.is_file():
            continue
        relative = path.relative_to(resolved_root).as_posix()
        if relative not in expected and relative not in allowed:
            _fail("runtime.extra_runtime_file", {"path": relative})
    for relative, entry in expected.items():
        path = (resolved_root / PurePosixPath(relative)).resolve()
        _ensure_inside(path, resolved_root)
        if not path.is_file():
            _fail("runtime.archive_missing_file", {"path": relative})
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise AppError("runtime.archive_read_failed", {"reason": "stat"}) from exc
        if actual_size != entry.size_bytes:
            _fail("runtime.archive_size_mismatch", {"path": relative})
        if sha256_file(path) != entry.sha256:
            _fail("runtime.archive_hash_mismatch", {"path": relative})
        if entry.executable and os.name != "nt":
            try:
                path.chmod(path.stat().st_mode | 0o111)
            except OSError as exc:
                raise AppError("runtime.archive_permission_failed", {"path": relative}) from exc


def build_file_manifest(root: Path) -> tuple[RuntimeFileEntry, ...]:
    """Inventory a payload using stable POSIX paths and content hashes."""
    resolved = root.resolve()
    entries: list[RuntimeFileEntry] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            _fail("runtime.archive_entry_invalid", {"reason": "symlink"})
        if path.is_file():
            relative = path.relative_to(resolved).as_posix()
            mode = path.stat().st_mode
            entries.append(
                RuntimeFileEntry(
                    relative_path=relative,
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    executable=bool(mode & 0o111),
                )
            )
    if not entries:
        raise AppError("runtime.manifest_invalid", {"field": "files"})
    return tuple(entries)


def create_deterministic_archive(payload_root: Path, output_path: Path) -> None:
    """Create a reproducible gzip tar without links or build-machine metadata."""
    root = payload_root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            output_path.open("wb") as raw_output,
            gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    _fail("runtime.archive_entry_invalid", {"reason": "symlink"})
                relative = path.relative_to(root).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if info.isreg():
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                elif info.isdir():
                    archive.addfile(info)
                else:
                    _fail("runtime.archive_entry_invalid", {"reason": "special"})
    except AppError:
        raise
    except OSError as exc:
        raise AppError("runtime.archive_write_failed", {"reason": "filesystem"}) from exc
    _fsync_file(output_path)


def _validate_member_name(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AppError("runtime.archive_entry_invalid", {"reason": "path"})
    if "\\" in value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AppError("runtime.archive_entry_invalid", {"reason": "path"})
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AppError("runtime.archive_entry_invalid", {"reason": "path"})
    return path.as_posix()


def _ensure_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AppError("runtime.archive_entry_invalid", {"reason": "path_escape"}) from exc


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError as exc:
        raise AppError("runtime.archive_write_failed", {"reason": "fsync"}) from exc


def _fail(code: str, params: dict[str, str] | None = None) -> NoReturn:
    raise AppError(code, params)


__all__ = [
    "FilesystemRuntimeArchive",
    "build_file_manifest",
    "create_deterministic_archive",
    "safe_extract_archive",
    "sha256_file",
    "validate_archive",
    "verify_manifest_files",
    "verify_runtime_payload",
]
