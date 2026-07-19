from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from captioner.adapters.runtime.runtime_package_source import LocalOrHTTPSRuntimePackageSource
from captioner.core.domain.errors import AppError
from captioner.core.domain.runtime_package import RuntimePackageDescriptor


@dataclass(frozen=True, slots=True)
class _URL:
    scheme: str
    netloc: str
    path: str

    def __str__(self) -> str:
        return f"{self.scheme}://{self.netloc}{self.path}"


@dataclass(frozen=True, slots=True)
class _Request:
    url: _URL


class _Response:
    def __init__(self, url: _URL, content: bytes) -> None:
        self.url = url
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
        return (
            self.content[offset : offset + chunk_size]
            for offset in range(0, len(self.content), chunk_size)
        )

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _Client:
    def __init__(self, handler: Callable[[_Request], _Response]) -> None:
        self._handler = handler

    def get(self, url: str) -> _Response:
        return self._handler(_Request(_parse_url(url)))

    def stream(self, method: str, url: str) -> _Response:
        assert method == "GET"
        return self._handler(_Request(_parse_url(url)))

    def close(self) -> None:
        return None


def _parse_url(value: str) -> _URL:
    prefix, remainder = value.split("://", 1)
    netloc, _, path = remainder.partition("/")
    return _URL(prefix, netloc, "/" + path)


def _descriptor() -> RuntimePackageDescriptor:
    from tests.fakes.phase6_values import runtime_manifest

    return RuntimePackageDescriptor(
        package_schema_version=1,
        archive_filename="runtime.tar.gz",
        archive_size_bytes=4,
        runtime_manifest=runtime_manifest(),
    )


def test_https_source_streams_archive_to_destination(tmp_path: Path) -> None:
    descriptor = _descriptor()
    descriptor_bytes = json.dumps(descriptor.to_dict()).encode("utf-8")
    calls: list[str] = []

    def handler(request: _Request) -> _Response:
        calls.append(str(request.url))
        if request.url.path.endswith("runtime.runtime.json"):
            return _Response(request.url, descriptor_bytes)
        return _Response(request.url, b"data")

    client = _Client(handler)
    destination = tmp_path / "downloads" / "runtime.part"
    result = LocalOrHTTPSRuntimePackageSource(client=client).resolve(
        "https://example.test/releases/runtime.runtime.json", destination
    )

    assert result == descriptor
    assert destination.read_bytes() == b"data"
    assert calls == [
        "https://example.test/releases/runtime.runtime.json",
        "https://example.test/releases/runtime.tar.gz",
    ]
    client.close()


@pytest.mark.parametrize(
    "reference",
    ("http://example.test/runtime.runtime.json", "https://user:secret@example.test/runtime.json"),
)
def test_source_rejects_non_https_and_url_credentials(reference: str, tmp_path: Path) -> None:
    source = LocalOrHTTPSRuntimePackageSource()
    with pytest.raises(AppError, match=r"runtime\.source_url_invalid"):
        source.resolve(reference, tmp_path / "archive.part")


def test_source_rejects_download_size_mismatch(tmp_path: Path) -> None:
    descriptor = _descriptor()
    descriptor_bytes = json.dumps(descriptor.to_dict()).encode("utf-8")

    def handler(request: _Request) -> _Response:
        if request.url.path.endswith("runtime.runtime.json"):
            return _Response(request.url, descriptor_bytes)
        return _Response(request.url, b"bad")

    client = _Client(handler)
    with pytest.raises(AppError, match=r"runtime\.archive_size_mismatch"):
        LocalOrHTTPSRuntimePackageSource(client=client).resolve(
            "https://example.test/runtime.runtime.json", tmp_path / "archive.part"
        )
    client.close()


@pytest.mark.parametrize(
    "reference",
    (
        r"C:\Downloads\runtime.runtime.json",
        "C:/Downloads/runtime.runtime.json",
        r"\\server\share\runtime.runtime.json",
        "relative/path/runtime.runtime.json",
    ),
)
def test_windows_and_relative_strings_are_local_references(
    reference: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _descriptor()
    calls: list[Path] = []
    source = LocalOrHTTPSRuntimePackageSource()

    def resolve_local(path: Path, destination: Path) -> RuntimePackageDescriptor:
        calls.append(path)
        assert destination == tmp_path / "archive.part"
        return descriptor

    monkeypatch.setattr(source, "_resolve_local", resolve_local)
    assert source.resolve(reference, tmp_path / "archive.part") == descriptor
    assert calls == [Path(reference)]


def test_remote_descriptor_is_streamed_and_stops_at_limit(tmp_path: Path) -> None:
    calls = 0
    oversized = b"x" * (2 * 1024 * 1024 + 1)

    class StreamingResponse(_Response):
        def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
            nonlocal calls
            assert chunk_size == 64 * 1024
            for block in super().iter_bytes(chunk_size):
                calls += 1
                yield block

    class StreamingClient(_Client):
        def __init__(self) -> None:
            super().__init__(lambda request: StreamingResponse(request.url, oversized))

    with pytest.raises(AppError, match=r"runtime\.source_descriptor_too_large"):
        LocalOrHTTPSRuntimePackageSource(client=StreamingClient()).resolve(
            "https://example.test/runtime.runtime.json", tmp_path / "archive.part"
        )
    assert calls < (len(oversized) // (64 * 1024)) + 2


def test_local_descriptor_rejects_size_before_reading(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "runtime.runtime.json"
    descriptor_path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))

    with pytest.raises(AppError, match=r"runtime\.source_descriptor_too_large"):
        LocalOrHTTPSRuntimePackageSource().resolve(descriptor_path, tmp_path / "archive.part")
