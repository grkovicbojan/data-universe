"""
Twitter/X scraper backed by the vendored ``scraping.x.twikit`` library (curl_cffi TLS).

Configuration (environment):

- ``TWIKIT_SESSIONS_FILE`` (recommended): path to JSON defining one or more
  ``{proxy, cookies_file}`` and/or ``{proxy, cookies}`` entries (see below).
- ``TWIKIT_CREDENTIALS_FILE``: alias for ``TWIKIT_SESSIONS_FILE``.
- Shortcut: ``TWIKIT_COOKIES`` path to a single cookies JSON plus optional
  ``TWIKIT_PROXY`` builds one session without a JSON wrapper.
- Multiple shortcuts: ``TWIKIT_COOKIE_FILES`` and ``TWIKIT_PROXIES`` as
  comma-separated lists of equal length → one session per index.

JSON shape::

    {
      "language": "en-US",
      "impersonate": "safari17_0",
      "sessions": [
        {
          "proxy": "socks5://user:pass@host:port",
          "cookies_file": "/path/to/cookies.json"
        },
        {
          "proxy": "http://127.0.0.1:8888",
          "cookies": {"auth_token": "...", "ct0": "..."}
        }
      ]
    }

Each ``cookies_file`` is either a Cookie-Editor JSON export (array of objects
with ``name``, ``value``, ``domain``) or a flat ``{"cookie_name": "value"}`` object.
``auth_token`` and ``ct0`` are required after merge.

Proxies rotate in round-robin order; the next session is used automatically
when building a client for a scrape or validation batch.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import datetime as dt

from common.data import DataEntity
from scraping.scraper import ScrapeConfig, Scraper, ValidationResult
from scraping.x.model import XContent
from scraping.x import utils

_TWIKIT_PATCHED = False
_PATCH_LOCK = threading.Lock()

_DROP_COOKIES = {
    "gt",
    "night_mode",
    "dnt",
    "__gads",
    "__gpi",
    "__eoi",
    "_ga",
    "_gid",
    "_gat",
    "mbox",
}


def _ensure_twikit_runtime_patched() -> None:
    """Patch twikit for current x.com HTML/JS bundle layout (ondemand.s chunks)."""
    global _TWIKIT_PATCHED
    with _PATCH_LOCK:
        if _TWIKIT_PATCHED:
            return
        from scraping.x.twikit.x_client_transaction import transaction as _tx_mod

        _tx_mod.ON_DEMAND_FILE_REGEX = re.compile(
            r""",(\d+):["']ondemand\.s["']""",
            flags=(re.VERBOSE | re.MULTILINE),
        )
        _tx_mod.ON_DEMAND_HASH_PATTERN = r',{}:"([0-9a-f]+)"'

        async def _patched_get_indices(self, home_page_response, session, headers):
            key_byte_indices = []
            response = self.validate_response(home_page_response) or self.home_page_response
            match = _tx_mod.ON_DEMAND_FILE_REGEX.search(str(response))
            if not match:
                raise Exception("Couldn't locate ondemand.s reference in home page")
            on_demand_file_index = match.group(1)
            regex = re.compile(_tx_mod.ON_DEMAND_HASH_PATTERN.format(on_demand_file_index))
            hash_match = regex.search(str(response))
            if not hash_match:
                raise Exception("Couldn't locate ondemand.s hash in home page")
            filename = hash_match.group(1)
            on_demand_file_url = (
                f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{filename}a.js"
            )
            on_demand_file_response = await session.request(
                method="GET", url=on_demand_file_url, headers=headers
            )
            key_byte_indices_match = _tx_mod.INDICES_REGEX.finditer(
                str(on_demand_file_response.text)
            )
            for item in key_byte_indices_match:
                key_byte_indices.append(item.group(2))
            if not key_byte_indices:
                raise Exception("Couldn't get KEY_BYTE indices")
            key_byte_indices = list(map(int, key_byte_indices))
            return key_byte_indices[0], key_byte_indices[1:]

        _tx_mod.ClientTransaction.get_indices = _patched_get_indices
        _TWIKIT_PATCHED = True


def _read_cookies_file(path: str) -> Dict[str, str]:
    """Parse cookies JSON (Cookie-Editor array export or flat name/value dict)."""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    if not isinstance(data, list):
        raise ValueError(
            f"{path!r} is neither a JSON array (extension export) nor an object (flat dict)."
        )

    cookies: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        domain = (entry.get("domain") or "").lstrip(".").lower()
        if domain and domain not in {"x.com", "twitter.com"}:
            continue
        if name in _DROP_COOKIES:
            continue
        cookies[name] = value
    return cookies


def _sessions_file_path() -> Optional[str]:
    return os.environ.get("TWIKIT_SESSIONS_FILE") or os.environ.get("TWIKIT_CREDENTIALS_FILE")


def _sessions_from_env_lists() -> List[Dict[str, Any]]:
    """Build sessions from TWIKIT_COOKIE_FILES + TWIKIT_PROXIES (comma-separated, same length)."""
    files_raw = os.environ.get("TWIKIT_COOKIE_FILES", "").strip()
    proxies_raw = os.environ.get("TWIKIT_PROXIES", "").strip()
    if not files_raw:
        return []
    files = [os.path.expanduser(p.strip()) for p in files_raw.split(",") if p.strip()]
    proxies = [p.strip() or None for p in proxies_raw.split(",")] if proxies_raw else []
    if proxies and len(proxies) != len(files):
        bt.logging.error(
            "TWIKIT_COOKIE_FILES and TWIKIT_PROXIES must have the same number of comma-separated entries."
        )
        return []
    if not proxies:
        proxies = [None] * len(files)
    sessions: List[Dict[str, Any]] = []
    for i, fp in enumerate(files):
        if not os.path.isfile(fp):
            bt.logging.error(f"Twikit TWIKIT_COOKIE_FILES entry not found: {fp!r}. Skipping.")
            continue
        merged = _read_cookies_file(fp)
        missing = [n for n in ("auth_token", "ct0") if n not in merged]
        if missing:
            bt.logging.error(f"Twikit cookies file {fp!r} missing {missing}. Skipping.")
            continue
        sessions.append({"proxy": proxies[i], "cookies": merged})
    return sessions


def _sessions_from_single_cookie_env() -> List[Dict[str, Any]]:
    path = os.environ.get("TWIKIT_COOKIES", "").strip()
    if not path:
        return []
    fp = os.path.expanduser(path)
    if not os.path.isfile(fp):
        bt.logging.error(f"TWIKIT_COOKIES path not found: {fp!r}.")
        return []
    merged = _read_cookies_file(fp)
    missing = [n for n in ("auth_token", "ct0") if n not in merged]
    if missing:
        bt.logging.error(f"TWIKIT_COOKIES file {fp!r} missing required cookies: {missing}.")
        return []
    proxy = os.environ.get("TWIKIT_PROXY")
    if proxy is not None:
        proxy = str(proxy).strip() or None
    return [{"proxy": proxy, "cookies": merged}]


def _load_sessions_config() -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Returns (language, impersonate, sessions) where each session dict has
    keys: proxy (optional str), cookies (dict str->str).
    """
    path = _sessions_file_path()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)

        language = str(cfg.get("language") or "en-US")
        impersonate = str(cfg.get("impersonate") or "safari17_0")
        raw_sessions = cfg.get("sessions") or cfg.get("entries")
        if not isinstance(raw_sessions, list) or not raw_sessions:
            bt.logging.warning(
                f"Twikit config {path!r} has no 'sessions' list; scraper will stay idle."
            )
            return language, impersonate, []

        sessions: List[Dict[str, Any]] = []
        for i, entry in enumerate(raw_sessions):
            if not isinstance(entry, dict):
                continue
            proxy = entry.get("proxy")
            if proxy is not None:
                proxy = str(proxy).strip() or None
            merged: Dict[str, str] = {}
            if "cookies" in entry and isinstance(entry["cookies"], dict):
                merged.update({str(k): str(v) for k, v in entry["cookies"].items()})
            cf = entry.get("cookies_file")
            if cf:
                cfp = os.path.expanduser(str(cf))
                if not os.path.isfile(cfp):
                    bt.logging.error(
                        f"Twikit session #{i}: cookies_file not found: {cfp!r}. Skipping this session."
                    )
                    continue
                merged.update(_read_cookies_file(cfp))
            missing = [n for n in ("auth_token", "ct0") if n not in merged]
            if missing:
                bt.logging.error(
                    f"Twikit session #{i} missing required cookie(s) {missing}. Skipping."
                )
                continue
            sessions.append({"proxy": proxy, "cookies": merged})

        return language, impersonate, sessions

    multi = _sessions_from_env_lists()
    if multi:
        return "en-US", "safari17_0", multi

    single = _sessions_from_single_cookie_env()
    if single:
        return "en-US", "safari17_0", single

    return "en-US", "safari17_0", []


def extract_status_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"/status/(\d+)", url or "")
    return m.group(1) if m else None


def build_twitter_search_query(scrape_config: ScrapeConfig) -> str:
    """Build an X search query string aligned with ApiDojo/Microworlds scrapers."""
    date_format = "%Y-%m-%d_%H:%M:%S_UTC"
    parts = [
        f"since:{scrape_config.date_range.start.astimezone(tz=dt.timezone.utc).strftime(date_format)}",
        f"until:{scrape_config.date_range.end.astimezone(tz=dt.timezone.utc).strftime(date_format)}",
    ]
    if scrape_config.labels:
        username_labels: List[str] = []
        keyword_labels: List[str] = []
        for label in scrape_config.labels:
            if label.value.startswith("@"):
                username_labels.append(f"from:{label.value[1:]}")
            else:
                keyword_labels.append(label.value)
        if username_labels:
            parts.append(f"({' OR '.join(username_labels)})")
        if keyword_labels:
            parts.append(f"({' OR '.join(keyword_labels)})")
    else:
        parts.append("e")
    return " ".join(parts)


def _tweet_hashtag_list(tweet) -> List[str]:
    """Build ``#tag`` list (and cashtags as ``#SYMBOL``) like other X scrapers."""
    tags: List[str] = []
    try:
        for t in tweet.hashtags:
            if not t:
                continue
            tags.append(t if str(t).startswith("#") else f"#{t}")
    except Exception:
        pass
    try:
        entities = (tweet._legacy or {}).get("entities") or {}
        for sym in entities.get("symbols", []) or []:
            txt = sym.get("text")
            if txt:
                tags.append(f"#{txt}")
    except Exception:
        pass
    return list(dict.fromkeys(tags))


def _media_urls_from_tweet(tweet) -> Optional[List[str]]:
    try:
        media = tweet.media or []
    except Exception:
        return None
    urls: List[str] = []
    for m in media:
        u = getattr(m, "media_url", None) or getattr(m, "media_url_https", None)
        if u:
            urls.append(u)
    return urls or None


def tweet_to_xcontent(tweet) -> Optional[Tuple[XContent, bool]]:
    """
    Convert a twikit ``Tweet`` to ``XContent`` plus ``is_retweet``.

    Returns None if the tweet should be dropped (e.g. missing user, retweet card).
    """
    from scraping.x.twikit.tweet import Tweet as TwikitTweet

    if not isinstance(tweet, TwikitTweet):
        return None

    is_retweet = tweet.retweeted_tweet is not None
    if is_retweet:
        return None

    user = tweet.user
    if not user or not user.screen_name:
        return None

    try:
        raw_text = tweet.full_text or tweet.text or ""
        text = utils.sanitize_scraped_tweet(raw_text)
        url = f"https://x.com/{user.screen_name}/status/{tweet.id}"
        ts = tweet.created_at_datetime
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)

        legacy = tweet._legacy or {}
        in_reply_to_id = tweet.in_reply_to
        is_reply = bool(in_reply_to_id) if in_reply_to_id is not None else None
        is_quote = bool(tweet.is_quote_status) if tweet.is_quote_status is not None else None
        quoted_id = None
        try:
            if tweet.quote:
                quoted_id = tweet.quote.id
        except Exception:
            quoted_id = None

        in_reply_to_username = legacy.get("in_reply_to_screen_name")
        conversation_id = legacy.get("conversation_id_str")

        x_content = XContent(
            username=user.screen_name,
            text=text,
            url=url,
            timestamp=ts,
            tweet_hashtags=_tweet_hashtag_list(tweet),
            media=_media_urls_from_tweet(tweet),
            user_id=user.id or None,
            user_display_name=user.name or None,
            user_verified=bool(user.verified) if user.verified is not None else None,
            tweet_id=str(tweet.id),
            is_reply=is_reply,
            is_quote=is_quote,
            conversation_id=conversation_id,
            in_reply_to_user_id=in_reply_to_id,
            language=tweet.lang or None,
            in_reply_to_username=in_reply_to_username,
            quoted_tweet_id=quoted_id,
            like_count=tweet.favorite_count,
            retweet_count=tweet.retweet_count,
            reply_count=tweet.reply_count,
            quote_count=tweet.quote_count,
            view_count=tweet.view_count,
            bookmark_count=tweet.bookmark_count,
            user_blue_verified=user.is_blue_verified,
            user_description=user.description or None,
            user_location=user.location or None,
            profile_image_url=user.profile_image_url or None,
            cover_picture_url=user.profile_banner_url,
            user_followers_count=user.followers_count,
            user_following_count=user.following_count,
            scraped_at=dt.datetime.now(dt.timezone.utc),
        )
        return x_content, is_retweet
    except Exception:
        bt.logging.warning(
            f"Twikit: failed to map tweet to XContent: {traceback.format_exc()}."
        )
        return None


class TwikitTwitterScraper(Scraper):
    """Scrape/validate X posts via twikit + curl_cffi (browser TLS impersonation)."""

    concurrent_validates_semaphore = threading.BoundedSemaphore(20)

    def __init__(self) -> None:
        self._language, self._impersonate, self._sessions = _load_sessions_config()
        self._rr = 0
        self._cycle_lock = threading.Lock()

    def _round_robin_start_index(self) -> int:
        """Return the next starting index for session rotation and advance the pointer."""
        with self._cycle_lock:
            if not self._sessions:
                return 0
            start = self._rr % len(self._sessions)
            self._rr = (self._rr + 1) % len(self._sessions)
            return start

    async def _make_client(self, session: Dict[str, Any]):
        _ensure_twikit_runtime_patched()
        from scraping.x.twikit import Client
        from scraping.x.twikit._cffi_http import CffiAsyncClient
        from scraping.x.twikit.gql_refresh import refresh_query_ids

        client = Client(self._language)
        client.http = CffiAsyncClient(
            proxy=session.get("proxy"),
            impersonate=self._impersonate,
        )
        client.set_cookies(session["cookies"], clear_cookies=True)
        await refresh_query_ids(client.http)
        return client

    async def _make_client_rotating(self, max_tries: Optional[int] = None):
        if not self._sessions:
            return None
        n = len(self._sessions)
        tries = min(max_tries or n, n)
        start = self._round_robin_start_index()
        last_error: Optional[BaseException] = None
        for t in range(tries):
            session = self._sessions[(start + t) % n]
            try:
                return await self._make_client(session)
            except Exception as exc:
                last_error = exc
                bt.logging.warning(
                    f"Twikit: failed to initialize client (attempt {t + 1}/{tries}): "
                    f"{type(exc).__name__}: {exc}"
                )
        if last_error:
            bt.logging.error(
                f"Twikit: exhausted session rotation during client init: {last_error!r}"
            )
        return None

    async def validate(self, entities: List[DataEntity]) -> List[ValidationResult]:
        if not entities:
            return []

        if not self._sessions:
            return [
                ValidationResult(
                    is_valid=False,
                    reason="Twikit scraper not configured (set TWIKIT_SESSIONS_FILE to a valid JSON).",
                    content_size_bytes_validated=e.content_size_bytes,
                )
                for e in entities
            ]

        bt.logging.trace("Acquiring semaphore for concurrent twikit validations.")
        with TwikitTwitterScraper.concurrent_validates_semaphore:
            bt.logging.trace("Acquired semaphore for concurrent twikit validations.")

            async def validate_one(entity: DataEntity) -> ValidationResult:
                if not utils.is_valid_twitter_url(entity.uri):
                    return ValidationResult(
                        is_valid=False,
                        reason="Invalid URI.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                tweet_id = extract_status_id_from_url(entity.uri)
                if not tweet_id:
                    return ValidationResult(
                        is_valid=False,
                        reason="Could not parse tweet id from URI.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )

                current_client = await self._make_client_rotating()
                if current_client is None:
                    return ValidationResult(
                        is_valid=False,
                        reason="Twikit client initialization failed for all configured sessions.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )

                attempt = 0
                max_attempts = max(2, len(self._sessions))
                while attempt < max_attempts:
                    attempt += 1
                    try:
                        tw = await current_client.get_tweet_by_id(tweet_id)
                    except Exception:
                        if attempt >= max_attempts:
                            bt.logging.error(
                                f"Twikit validate failed for {entity.uri}: {traceback.format_exc()}."
                            )
                            return ValidationResult(
                                is_valid=False,
                                reason="Failed to fetch tweet from X via twikit.",
                                content_size_bytes_validated=entity.content_size_bytes,
                            )
                        current_client = await self._make_client_rotating()
                        if current_client is None:
                            continue
                        continue

                    if tw.retweeted_tweet is not None:
                        placeholder = XContent(
                            username="_",
                            text="_",
                            url=entity.uri,
                            timestamp=entity.datetime,
                            tweet_hashtags=[],
                        )
                        return utils.validate_tweet_content(
                            actual_tweet=placeholder,
                            entity=entity,
                            is_retweet=True,
                        )

                    parsed = tweet_to_xcontent(tw)
                    if parsed is None:
                        return ValidationResult(
                            is_valid=False,
                            reason="Tweet not found or is invalid.",
                            content_size_bytes_validated=entity.content_size_bytes,
                        )

                    actual_tweet, is_retweet = parsed
                    if utils.normalize_url(actual_tweet.url) != utils.normalize_url(
                        entity.uri
                    ):
                        return ValidationResult(
                            is_valid=False,
                            reason="Tweet URL mismatch.",
                            content_size_bytes_validated=entity.content_size_bytes,
                        )

                    return utils.validate_tweet_content(
                        actual_tweet=actual_tweet,
                        entity=entity,
                        is_retweet=is_retweet,
                    )

                return ValidationResult(
                    is_valid=False,
                    reason="Twikit validation retries exhausted.",
                    content_size_bytes_validated=entity.content_size_bytes,
                )

            return await asyncio.gather(*[validate_one(e) for e in entities])

    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        if not self._sessions:
            bt.logging.warning(
                "Twikit scraper has no sessions (configure TWIKIT_SESSIONS_FILE); skipping scrape."
            )
            return []

        client = await self._make_client_rotating()
        if client is None:
            return []

        query = build_twitter_search_query(scrape_config)
        limit = int(scrape_config.entity_limit or 150)
        collected: List[DataEntity] = []

        try:
            page = await client.search_tweet(query, "Latest", count=20)
        except Exception:
            bt.logging.error(
                f"Twikit search_tweet failed for {query!r}: {traceback.format_exc()}."
            )
            return []

        while len(collected) < limit:
            for tweet in page:
                if len(collected) >= limit:
                    break
                parsed = tweet_to_xcontent(tweet)
                if not parsed:
                    continue
                x_content, _ = parsed
                if not scrape_config.date_range.contains(x_content.timestamp):
                    continue
                collected.append(XContent.to_data_entity(x_content))

            if len(collected) >= limit:
                break

            try:
                nxt = await page.next()
            except Exception:
                bt.logging.trace(f"Twikit: page.next() ended: {traceback.format_exc()}.")
                break

            if not nxt or len(nxt) == 0:
                break
            page = nxt

        bt.logging.success(
            f"Twikit scrape completed for query={query!r}: {len(collected)} entities."
        )
        return collected
