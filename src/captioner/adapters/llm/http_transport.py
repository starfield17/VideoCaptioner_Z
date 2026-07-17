"""HTTP transport boundary for the OpenAI-compatible adapter."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol, cast

import httpx


@dataclass(frozen=True, slots=True)
class HTTPTimeout:
    connect_sec: float
    read_sec: float
    write_sec: float
    pool_sec: float

    def __post_init__(self) -> None:
        values = (self.connect_sec, self.read_sec, self.write_sec, self.pool_sec)
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0
            for value in values
        ):
            raise ValueError

    @classmethod
    def all(cls, seconds: float) -> HTTPTimeout:
        return cls(seconds, seconds, seconds, seconds)

    def as_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_sec,
            read=self.read_sec,
            write=self.write_sec,
            pool=self.pool_sec,
        )


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class HTTPTransportError(RuntimeError):
    """A provider-neutral transport failure with no request secret data."""

    kind: str

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)


class HTTPStreamResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def aiter_bytes(self) -> AsyncIterator[bytes]: ...


class AsyncHTTPClient(Protocol):
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        content: bytes,
        timeout: httpx.Timeout,
    ) -> AbstractAsyncContextManager[HTTPStreamResponse]: ...

    async def aclose(self) -> None: ...


class HTTPTransport(Protocol):
    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: HTTPTimeout,
        max_response_bytes: int,
    ) -> HTTPResponse: ...

    async def close(self) -> None: ...


class HttpxTransport:
    """A closeable httpx transport with bounded response reads."""

    def __init__(
        self,
        *,
        timeout: HTTPTimeout,
        max_response_bytes: int = 2 * 1024 * 1024,
        client: AsyncHTTPClient | None = None,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._client: AsyncHTTPClient = (
            cast(
                AsyncHTTPClient,
                httpx.AsyncClient(timeout=timeout.as_httpx(), follow_redirects=False),
            )
            if client is None
            else client
        )
        self._owns_client = client is None

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: HTTPTimeout,
        max_response_bytes: int,
    ) -> HTTPResponse:
        limit = min(self._max_response_bytes, max_response_bytes)
        if limit < 1:
            raise ValueError
        try:
            async with self._client.stream(
                method,
                url,
                headers=dict(headers),
                content=content,
                timeout=timeout.as_httpx(),
            ) as response:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    _append_bounded(body, chunk, limit)
                return HTTPResponse(response.status_code, dict(response.headers), bytes(body))
        except HTTPTransportError:
            raise
        except httpx.TimeoutException as exc:
            raise HTTPTransportError("timeout") from exc
        except httpx.RequestError as exc:
            raise HTTPTransportError("network") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> HttpxTransport:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def _append_bounded(body: bytearray, chunk: bytes, limit: int) -> None:
    if len(body) + len(chunk) > limit:
        raise HTTPTransportError("response_too_large")
    body.extend(chunk)
