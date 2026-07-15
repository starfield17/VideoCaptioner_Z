"""Atomic local artifact storage with relative-key protection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from captioner.core.domain.errors import AppError


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        target = self._target(key)
        if not self.root.is_dir():
            raise AppError("output.not_directory", {"path": str(self.root)})
        if (target.exists() or target.is_symlink()) and not overwrite:
            raise AppError("output.exists", {"path": str(target)})
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            except OSError as exc:
                _remove_temporary(temporary)
                raise AppError("output.write_failed", {"path": str(target)}) from exc
        except AppError:
            raise
        except OSError as exc:
            raise AppError("output.write_failed", {"path": str(target)}) from exc
        return target

    def read_bytes(self, key: str) -> bytes:
        target = self._target(key)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise AppError("output.read_failed", {"path": str(target)}) from exc

    def exists(self, key: str) -> bool:
        return self._target(key).exists()

    def delete(self, key: str) -> None:
        target = self._target(key)
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            raise AppError("output.delete_failed", {"path": str(target)}) from exc

    def _target(self, key: str) -> Path:
        if not key.strip():
            raise AppError("output.path_invalid", {"key": key, "reason": "empty"})
        posix = PurePosixPath(key)
        windows = PureWindowsPath(key)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or ".." in posix.parts
            or ".." in windows.parts
        ):
            raise AppError("output.path_invalid", {"key": key, "reason": "traversal"})
        target = (self.root / Path(key)).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise AppError("output.path_invalid", {"key": key, "reason": "outside_root"}) from exc
        return target


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        raise AppError("output.cleanup_failed", {"path": str(path)}) from cleanup_error
