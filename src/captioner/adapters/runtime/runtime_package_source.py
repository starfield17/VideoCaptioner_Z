"""Local and HTTPS Runtime package descriptor source."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import NoReturn, Protocol, cast
from urllib.parse import urljoin, urlparse

import httpx

from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime_package import RuntimePackageDescriptor

_MAX_DESCRIPTOR_BYTES = 2 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024


class _HTTPURL(Protocol):
    scheme: str


class _HTTPResponse(Protocol):
    url: _HTTPURL

    def raise_for_status(self) -> object: ...

    def iter_bytes(self, chunk_size: int) -> Iterator[bytes]: ...

    def __enter__(self) -> _HTTPResponse: ...

    def __exit__(self, *args: object) -> None: ...


class _HTTPClient(Protocol):
    def stream(self, method: str, url: str) -> object: ...

    def close(self) -> None: ...


class LocalOrHTTPSRuntimePackageSource:
    """Resolve a sidecar and materialize its archive into a caller-owned path."""

    def __init__(
        self,
        *,
        client: _HTTPClient | None = None,
        max_download_bytes: int = _MAX_DOWNLOAD_BYTES,
    ) -> None:
        if max_download_bytes <= 0:
            raise ValueError
        self._client = client
        self._max_download_bytes = max_download_bytes

    def resolve(self, reference: str | Path, destination: Path) -> RuntimePackageDescriptor:
        if isinstance(reference, Path):
            return self._resolve_local(reference, destination)
        if reference.casefold().startswith("https://"):
            parsed = urlparse(reference)
            if parsed.username or parsed.password:
                raise AppError("runtime.source_url_invalid")
            return self._resolve_remote(reference, destination)
        if _looks_like_windows_path(reference):
            return self._resolve_local(Path(reference), destination)
        parsed = urlparse(reference)
        if parsed.scheme:
            raise AppError("runtime.source_url_invalid")
        return self._resolve_local(Path(reference), destination)

    def _resolve_local(self, descriptor_path: Path, destination: Path) -> RuntimePackageDescriptor:
        path = descriptor_path.expanduser().resolve()
        if not path.is_file():
            raise AppError("runtime.package_descriptor_missing")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise AppError("runtime.package_descriptor_missing") from exc
        if size > _MAX_DESCRIPTOR_BYTES:
            raise AppError("runtime.source_descriptor_too_large")
        descriptor = _load_descriptor(_read_bounded_file(path, _MAX_DESCRIPTOR_BYTES))
        if descriptor.archive_size_bytes > self._max_download_bytes:
            raise AppError("runtime.archive_too_large")
        archive = path.parent / descriptor.archive_filename
        if not archive.is_file():
            raise AppError("runtime.archive_missing")
        _copy_archive(archive, destination, descriptor.archive_size_bytes)
        return descriptor

    def _resolve_remote(self, descriptor_url: str, destination: Path) -> RuntimePackageDescriptor:
        descriptor_bytes = self._get_bytes(descriptor_url, _MAX_DESCRIPTOR_BYTES)
        descriptor = _load_descriptor(descriptor_bytes)
        if descriptor.archive_size_bytes > self._max_download_bytes:
            raise AppError("runtime.archive_too_large")
        archive_url = urljoin(descriptor_url, descriptor.archive_filename)
        parsed = urlparse(archive_url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise AppError("runtime.source_url_invalid")
        self._download(archive_url, destination, descriptor.archive_size_bytes)
        return descriptor

    def _get_bytes(self, url: str, limit: int) -> bytes:
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.Client(follow_redirects=True, timeout=30.0)
        try:
            try:
                stream = cast(AbstractContextManager[object], client.stream("GET", url))
                with stream as raw_response:
                    response = cast(_HTTPResponse, raw_response)
                    response.raise_for_status()
                    _ensure_https_response(response)
                    data = bytearray()
                    for block in response.iter_bytes(64 * 1024):
                        if len(data) + len(block) > limit:
                            _fail("runtime.source_descriptor_too_large")
                        data.extend(block)
            except (httpx.HTTPError, OSError) as exc:
                raise AppError("runtime.source_fetch_failed") from exc
            except AppError:
                raise
        finally:
            if owns_client:
                client.close()
        return bytes(data)

    def _download(self, url: str, destination: Path, expected_size: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.Client(follow_redirects=True, timeout=60.0)
        written = 0
        try:
            try:
                stream = cast(AbstractContextManager[object], client.stream("GET", url))
                with stream as raw_response:
                    response = cast(_HTTPResponse, raw_response)
                    response.raise_for_status()
                    _ensure_https_response(response)
                    with destination.open("wb") as stream:
                        for block in response.iter_bytes(1024 * 1024):
                            written += len(block)
                            if written > min(expected_size, self._max_download_bytes):
                                _fail("runtime.archive_too_large")
                            stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
            except AppError:
                raise
            except (httpx.HTTPError, OSError) as exc:
                if isinstance(exc, OSError) and exc.errno == 28:
                    raise AppError("runtime.disk_full") from exc
                raise AppError("runtime.source_fetch_failed") from exc
        finally:
            if owns_client:
                client.close()
        if written != expected_size:
            raise AppError("runtime.archive_size_mismatch")


def _load_descriptor(data: bytes) -> RuntimePackageDescriptor:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AppError("runtime.package_invalid", {"field": "json"}) from exc
    return RuntimePackageDescriptor.from_dict(value)


def _copy_archive(source: Path, destination: Path, expected_size: int) -> None:
    try:
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            _fail("runtime.archive_size_mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, destination.open("wb") as output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except AppError:
        raise
    except OSError as exc:
        if exc.errno == 28:
            raise AppError("runtime.disk_full") from exc
        raise AppError("runtime.source_copy_failed") from exc


def _read_bounded_file(path: Path, limit: int) -> bytes:
    data = bytearray()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(64 * 1024), b""):
                if len(data) + len(block) > limit:
                    _fail("runtime.source_descriptor_too_large")
                data.extend(block)
    except AppError:
        raise
    except OSError as exc:
        raise AppError("runtime.package_descriptor_missing") from exc
    return bytes(data)


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:", value) or value.startswith(("\\\\", "//")))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def _ensure_https_response(response: _HTTPResponse) -> None:
    if response.url.scheme != "https":
        raise AppError("runtime.source_url_invalid")


def _fail(code: str) -> NoReturn:
    raise AppError(code)


__all__ = ["LocalOrHTTPSRuntimePackageSource"]
