"""httpx-compatible adapter around :class:`curl_cffi.requests.AsyncSession`.

Cloudflare's bot rules on ``/i/api/graphql/*`` fingerprint the TLS
ClientHello (JA3/JA4). httpx (which speaks TLS via Python's ``ssl`` module
backed by OpenSSL) does not produce a Safari/Chrome ClientHello, no matter
what ``User-Agent`` or ``Sec-Fetch-*`` headers we set. curl_cffi runs over
libcurl-impersonate, which replays a real browser's ClientHello byte for
byte, so the TLS layer is indistinguishable from Safari/Chrome.

This module exposes :class:`CffiAsyncClient`, a thin wrapper that mimics
just enough of :class:`httpx.AsyncClient` for twikit's
:class:`twikit.client.client.Client` to work without further changes.

Usage in user code::

    from twikit import Client
    from twikit._cffi_http import CffiAsyncClient

    client = Client('en-US')
    client.http = CffiAsyncClient(
        proxy='socks5://user:pass@1.2.3.4:1080',
        impersonate='safari17_0',
    )

Requires ``pip install curl_cffi``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

try:
    from curl_cffi.requests import AsyncSession as _CurlSession
except ImportError as exc:  # pragma: no cover - import-time check
    raise ImportError(
        "twikit._cffi_http requires `curl_cffi`. Install it with: "
        "`pip install curl_cffi`"
    ) from exc


# httpx and curl_cffi use mostly the same kwarg names, with a few exceptions.
# These are normalised at request time.
_KW_RENAMES = {
    'follow_redirects': 'allow_redirects',
}

# Kwargs that httpx accepts but curl_cffi doesn't / shouldn't see.
_DROP_KWARGS = {'http2', 'http1'}


class _CookiesProxy:
    """Subset of :class:`httpx.Cookies` that twikit actually uses."""

    def __init__(self, session: _CurlSession) -> None:
        self._session = session

    @property
    def jar(self):
        return self._session.cookies.jar

    def clear(self) -> None:
        self._session.cookies.clear()

    def update(self, cookies) -> None:
        if cookies is None:
            return
        if isinstance(cookies, dict):
            for name, value in cookies.items():
                self._session.cookies.set(name, value)
            return
        for item in cookies:
            try:
                name, value = item
            except (TypeError, ValueError):
                continue
            self._session.cookies.set(name, value)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self._session.cookies.get(name, default=default)
        except TypeError:
            try:
                return self._session.cookies[name]
            except KeyError:
                return default

    def __iter__(self):
        try:
            return iter(self._session.cookies)
        except TypeError:
            return iter(list(self._session.cookies.jar))

    def __contains__(self, name: str) -> bool:
        try:
            return name in self._session.cookies
        except Exception:
            return any(c.name == name for c in self._session.cookies.jar)

    def __len__(self) -> int:
        return sum(1 for _ in self._session.cookies.jar)


class _StreamContext:
    """Async context manager wrapping ``AsyncSession.request(stream=True)``."""

    def __init__(self, session: _CurlSession, method: str, url: str, kwargs: dict) -> None:
        self._session = session
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._response = None

    async def __aenter__(self):
        kwargs = dict(self._kwargs)
        kwargs['stream'] = True
        self._response = await self._session.request(self._method, self._url, **kwargs)
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._response is not None:
            try:
                await self._response.aclose()
            except Exception:
                pass


class CffiAsyncClient:
    """A tiny httpx-compatible facade backed by curl_cffi.

    Only the surface twikit's :class:`Client` touches is implemented.
    """

    def __init__(
        self,
        proxy: str | None = None,
        *,
        impersonate: str = 'safari17_0',
        **kwargs: Any,
    ) -> None:
        for key in _DROP_KWARGS:
            kwargs.pop(key, None)

        session_kwargs: dict[str, Any] = {'impersonate': impersonate}
        if proxy is not None:
            session_kwargs['proxy'] = proxy
        for key, value in kwargs.items():
            session_kwargs.setdefault(key, value)

        self._session = _CurlSession(**session_kwargs)
        self._cookies = _CookiesProxy(self._session)
        self._proxy = proxy
        self._impersonate = impersonate
        # Twikit reads ``self.http._mounts`` from the ``proxy`` getter. We
        # don't need real httpx mounts; an empty dict makes the getter
        # short-circuit to ``None`` without crashing.
        self._mounts: dict = {}

    @property
    def cookies(self) -> _CookiesProxy:
        return self._cookies

    @cookies.setter
    def cookies(self, value) -> None:
        # Twikit's ``_remove_duplicate_ct0_cookie`` does
        # ``self.http.cookies = list(cookies.items())``. Re-seed the jar
        # from the given (name, value) pairs (or any mapping).
        self._session.cookies.clear()
        if value is None:
            return
        if isinstance(value, dict):
            pairs = value.items()
        else:
            pairs = value
        for item in pairs:
            try:
                name, val = item
            except (TypeError, ValueError):
                continue
            self._session.cookies.set(name, val)

    async def request(self, method: str, url: str, **kwargs: Any):
        for src, dst in _KW_RENAMES.items():
            if src in kwargs:
                kwargs[dst] = kwargs.pop(src)
        for key in _DROP_KWARGS:
            kwargs.pop(key, None)
        return await self._session.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs: Any):
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs: Any):
        return await self.request('POST', url, **kwargs)

    async def put(self, url: str, **kwargs: Any):
        return await self.request('PUT', url, **kwargs)

    async def delete(self, url: str, **kwargs: Any):
        return await self.request('DELETE', url, **kwargs)

    def stream(self, method: str, url: str, **kwargs: Any) -> _StreamContext:
        for src, dst in _KW_RENAMES.items():
            if src in kwargs:
                kwargs[dst] = kwargs.pop(src)
        for key in _DROP_KWARGS:
            kwargs.pop(key, None)
        return _StreamContext(self._session, method, url, kwargs)

    async def aclose(self) -> None:
        try:
            await self._session.close()
        except Exception:
            pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ['CffiAsyncClient']
