"""Keep twikit's GraphQL query IDs in sync with x.com's live web bundle.

X.com embeds the current GraphQL operation IDs (the 22-char hash that
appears before each operation name in ``/i/api/graphql/<hash>/<Operation>``)
inside its main JS bundle. X rotates these every few weeks, which silently
breaks twikit calls -- a stale hash returns a *404 with an empty body*
from the GraphQL gateway because the persisted query no longer exists.

This module scrapes the live hashes once per bundle version, applies them
to :class:`twikit.client.gql.Endpoint`, and caches the result on disk so
subsequent runs are free until x.com ships a new bundle.

Typical usage in user code::

    from twikit import Client
    from twikit.gql_refresh import refresh_query_ids

    client = Client(...)
    client.set_cookies(my_cookies, clear_cookies=True)
    patched = await refresh_query_ids(client.http)
    print(f'patched {patched} GraphQL endpoint(s)')
    tweets = await client.search_tweet('hello', 'Latest')
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from httpx import AsyncClient


from .client.gql import Endpoint as _GQLEndpoint


# Matches `https://abs.twimg.com/responsive-web/client-web[-foo]/main.<hash>.js`
# and the various chunk URLs (api, ondemand, vendor, ...).
_BUNDLE_URL_RE = re.compile(
    r"https://abs\.twimg\.com/responsive-web/client-web(?:-[a-z]+)?/"
    r"(?:main|api|ondemand|i18n|vendors?)\.[a-zA-Z0-9._-]+a?\.js"
)

# X embeds operations as small objects in the JS bundle. Both field orders
# occur in the wild, so we try both.
_OP_FIRST_RE = re.compile(
    r'operationName\s*:\s*"([A-Za-z][A-Za-z0-9_]*)"'
    r'[^}]{0,200}'
    r'queryId\s*:\s*"([A-Za-z0-9_-]{22})"'
)
_QID_FIRST_RE = re.compile(
    r'queryId\s*:\s*"([A-Za-z0-9_-]{22})"'
    r'[^}]{0,200}'
    r'operationName\s*:\s*"([A-Za-z][A-Za-z0-9_]*)"'
)

_DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) '
        'AppleWebKit/605.1.15 (KHTML, like Gecko) '
        'Version/17.5 Safari/605.1.15'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}


def _operation_to_attr(name: str) -> str:
    """``"SearchTimeline" -> "SEARCH_TIMELINE"``."""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).upper()


def _known_operations() -> dict[str, str]:
    """Map ``OperationName -> current Endpoint url`` for every constant on
    :class:`twikit.client.gql.Endpoint` whose value is a graphql URL."""
    ops: dict[str, str] = {}
    for attr, value in vars(_GQLEndpoint).items():
        if not isinstance(value, str) or '/graphql/' not in value:
            continue
        op = value.rsplit('/', 1)[-1]
        if op:
            ops[op] = value
    return ops


def _cache_path() -> Path:
    override = os.environ.get('TWIKIT_GQL_CACHE')
    if override:
        return Path(override).expanduser()
    base = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return Path(base) / 'twikit' / 'gql_query_ids.json'


def _read_cache() -> dict:
    try:
        with _cache_path().open('r', encoding='utf-8') as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def _write_cache(payload: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
    except OSError as exc:
        print(
            f'[twikit.gql_refresh] could not write cache to {path}: {exc}',
            file=sys.stderr,
        )


async def _fetch_text(http: AsyncClient, url: str) -> str | None:
    try:
        response = await http.get(url, headers=_DEFAULT_HEADERS, timeout=20)
    except Exception as exc:
        print(
            f'[twikit.gql_refresh] GET {url} failed: '
            f'{type(exc).__name__}: {exc}',
            file=sys.stderr,
        )
        return None
    if response.status_code >= 400:
        print(
            f'[twikit.gql_refresh] GET {url} -> {response.status_code}',
            file=sys.stderr,
        )
        return None
    return response.text


def _extract_query_ids(js: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _OP_FIRST_RE.finditer(js):
        op, qid = match.group(1), match.group(2)
        found.setdefault(op, qid)
    for match in _QID_FIRST_RE.finditer(js):
        qid, op = match.group(1), match.group(2)
        found.setdefault(op, qid)
    return found


class _CookieJarSnapshot:
    """Snapshot and restore an httpx client's cookie jar.

    Discovery hits ``https://x.com`` and the JS bundle URLs, which causes the
    server to send ``Set-Cookie`` headers (e.g. ``guest_id_ads`` on .x.com).
    Those land in the same jar that already holds the caller's hand-loaded
    cookies, which httpx then sees as duplicates and crashes on
    (``CookieConflict``). We snapshot + restore so discovery is invisible
    to the rest of the client.
    """

    def __init__(self, http: AsyncClient) -> None:
        self._http = http
        self._snapshot: list = []

    def __enter__(self) -> '_CookieJarSnapshot':
        try:
            self._snapshot = list(self._http.cookies.jar)
        except Exception:
            self._snapshot = []
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._http.cookies.clear()
            for cookie in self._snapshot:
                try:
                    self._http.cookies.jar.set_cookie(cookie)
                except Exception:
                    pass
        except Exception:
            pass


async def discover_query_ids(
    http: AsyncClient,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Return ``{OperationName: queryId}`` for every GraphQL operation x.com
    currently advertises.

    Results are cached on disk and re-used as long as x.com keeps serving
    the same set of JS bundle URLs. Pass ``force=True`` to bypass the cache.

    The caller's cookie jar is restored exactly as it was before discovery,
    so this is safe to call after :meth:`Client.set_cookies`.
    """
    with _CookieJarSnapshot(http):
        home = await _fetch_text(http, 'https://x.com')
        if home is None:
            return {}

        bundle_urls = sorted(set(_BUNDLE_URL_RE.findall(home)))
        if not bundle_urls:
            print(
                '[twikit.gql_refresh] could not find any client-web bundles '
                'on x.com (maybe Cloudflare returned an interstitial?).',
                file=sys.stderr,
            )
            return {}

        cache = _read_cache()
        cache_key = '|'.join(bundle_urls)
        if not force and cache.get('bundle_key') == cache_key:
            cached_ids = cache.get('query_ids')
            if isinstance(cached_ids, dict):
                return {str(k): str(v) for k, v in cached_ids.items()}

        query_ids: dict[str, str] = {}
        for url in bundle_urls:
            js = await _fetch_text(http, url)
            if not js:
                continue
            for op, qid in _extract_query_ids(js).items():
                query_ids.setdefault(op, qid)

        if query_ids:
            _write_cache({'bundle_key': cache_key, 'query_ids': query_ids})

        return query_ids


def apply_query_ids(query_ids: dict[str, str]) -> int:
    """Patch :class:`Endpoint` with the given ``{Operation: queryId}`` map.

    Returns the number of endpoints whose URL was actually changed.
    """
    if not query_ids:
        return 0

    patched = 0
    for op, current in _known_operations().items():
        qid = query_ids.get(op)
        if not qid:
            continue
        new_url = f"{current.rsplit('/', 2)[0]}/{qid}/{op}"
        if new_url == current:
            continue
        setattr(_GQLEndpoint, _operation_to_attr(op), new_url)
        patched += 1
    return patched


async def refresh_query_ids(http: AsyncClient, *, force: bool = False) -> int:
    """Discover live query IDs from x.com and apply them to twikit.

    Convenience wrapper around :func:`discover_query_ids` +
    :func:`apply_query_ids`. Returns the number of endpoints patched.
    """
    query_ids = await discover_query_ids(http, force=force)
    return apply_query_ids(query_ids)


__all__ = [
    'apply_query_ids',
    'discover_query_ids',
    'refresh_query_ids',
]


def _cli_main() -> int:  # pragma: no cover - convenience CLI
    """``python -m twikit.gql_refresh`` -> print the current id map."""
    import httpx

    async def _run() -> int:
        async with httpx.AsyncClient(http2=True) as client:
            ids = await discover_query_ids(client, force=True)
        if not ids:
            print('no query ids discovered', file=sys.stderr)
            return 1
        json.dump(ids, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write('\n')
        return 0

    return asyncio.run(_run())


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(_cli_main())
