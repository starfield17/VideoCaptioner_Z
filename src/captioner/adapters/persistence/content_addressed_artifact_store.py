"""Local content-addressed durable Artifact Store."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ContentAddressedArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        (root / ".incoming").mkdir(parents=True, exist_ok=True)
        (root / "sha256").mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self, data: bytes, *, kind: str, media_type: str, logical_name: str
    ) -> ArtifactRef:
        incoming = self._new_incoming()
        try:
            with incoming.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return self._commit_incoming(
                incoming,
                hashlib.sha256(data).hexdigest(),
                len(data),
                kind,
                media_type,
                logical_name,
            )
        finally:
            incoming.unlink(missing_ok=True)

    def put_file(
        self, source: Path, *, kind: str, media_type: str, logical_name: str
    ) -> ArtifactRef:
        incoming = self._new_incoming()
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as input_handle, incoming.open("wb") as output_handle:
                while chunk := input_handle.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            return self._commit_incoming(
                incoming, digest.hexdigest(), size, kind, media_type, logical_name
            )
        except OSError as exc:
            raise AppError("artifact.write_failed", {"logical_name": logical_name}) from exc
        finally:
            incoming.unlink(missing_ok=True)

    def verify(self, ref: ArtifactRef) -> None:
        path = self.resolve(ref)
        try:
            if not path.exists():
                raise AppError("artifact.missing", {"sha256": ref.sha256})
            if not path.is_file() or path.is_symlink() or path.stat().st_size != ref.size_bytes:
                raise AppError("artifact.corrupt", {"sha256": ref.sha256, "reason": "size"})
            digest, size = _hash_file(path)
        except OSError as exc:
            raise AppError("artifact.missing", {"sha256": ref.sha256}) from exc
        if size != ref.size_bytes or digest != ref.sha256:
            raise AppError("artifact.corrupt", {"sha256": ref.sha256, "reason": "hash"})

    def resolve(self, ref: ArtifactRef) -> Path:
        return self.root / "sha256" / ref.sha256[:2] / ref.sha256

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        self.verify(ref)
        try:
            return self.resolve(ref).read_bytes()
        except OSError as exc:
            raise AppError("artifact.read_failed", {"sha256": ref.sha256}) from exc

    def materialize(self, ref: ArtifactRef, target: Path, *, overwrite: bool) -> Path:
        self.verify(ref)
        target = target.expanduser().resolve()
        if target.exists() and not overwrite:
            raise AppError("output.exists", {"path": str(target)})
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(name)
            with (
                os.fdopen(descriptor, "wb") as output_handle,
                self.resolve(ref).open("rb") as input_handle,
            ):
                shutil.copyfileobj(input_handle, output_handle)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary, target)
            temporary = None
            _fsync_directory(target.parent)
        except OSError as exc:
            raise AppError("output.write_failed", {"path": str(target)}) from exc
        else:
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _new_incoming(self) -> Path:
        descriptor, name = tempfile.mkstemp(prefix="artifact-", dir=self.root / ".incoming")
        os.close(descriptor)
        return Path(name)

    def _commit_incoming(
        self,
        incoming: Path,
        digest: str,
        size: int,
        kind: str,
        media_type: str,
        logical_name: str,
    ) -> ArtifactRef:
        ref = ArtifactRef(digest, size, kind, media_type, logical_name)
        final = self.resolve(ref)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            self.verify(ref)
            return ref
        try:
            os.replace(incoming, final)
            _fsync_directory(final.parent)
        except OSError as exc:
            raise AppError("artifact.write_failed", {"logical_name": logical_name}) from exc
        self.verify(ref)
        return ref


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
