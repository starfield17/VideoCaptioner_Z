"""Staged atomic local artifact storage with relative-key protection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from captioner.core.domain.errors import AppError
from captioner.core.ports.artifact_store import StagedArtifact


@dataclass(slots=True)
class LocalStagedArtifact:
    """A single-use staged file owned by a :class:`LocalArtifactStore`."""

    _store: LocalArtifactStore
    _key: str
    _target: Path
    _temporary: Path | None
    _state: str = "staged"

    @property
    def key(self) -> str:
        return self._key

    @property
    def target_path(self) -> Path:
        return self._target

    @property
    def committed(self) -> bool:
        return self._state == "committed"

    def commit(self, *, overwrite: bool) -> Path:
        if self._state == "committed":
            raise AppError("output.stage_invalid", {"key": self._key, "reason": "committed"})
        if self._state == "discarded":
            raise AppError("output.stage_invalid", {"key": self._key, "reason": "discarded"})
        if self._temporary is None:
            raise AppError("output.stage_invalid", {"key": self._key, "reason": "missing"})
        if (self._target.exists() or self._target.is_symlink()) and not overwrite:
            raise AppError("output.exists", {"path": str(self._target)})
        try:
            os.replace(self._temporary, self._target)
        except OSError as exc:
            raise AppError("output.write_failed", {"path": str(self._target)}) from exc
        self._temporary = None
        self._state = "committed"
        return self._target

    def discard(self) -> None:
        if self._state != "staged":
            return
        temporary = self._temporary
        self._temporary = None
        self._state = "discarded"
        if temporary is not None:
            _remove_temporary(temporary)


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def stage_bytes(self, key: str, data: bytes) -> StagedArtifact:
        target = self._target(key)
        if not self.root.is_dir():
            raise AppError("output.not_directory", {"path": str(self.root)})
        temporary: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            staged = LocalStagedArtifact(self, key, target, temporary)
            temporary = None
        except OSError as exc:
            raise AppError("output.write_failed", {"path": str(target)}) from exc
        else:
            return staged
        finally:
            if temporary is not None and temporary.exists():
                _remove_temporary(temporary)

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        staged = self.stage_bytes(key, data)
        try:
            return staged.commit(overwrite=overwrite)
        finally:
            staged.discard()

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
        target = self.root / Path(key)
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise AppError("output.path_invalid", {"key": key, "reason": "outside_root"}) from exc
        if target.is_symlink():
            raise AppError("output.path_invalid", {"key": key, "reason": "symlink"})
        return target


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        raise AppError("output.cleanup_failed", {"path": str(path)}) from cleanup_error
