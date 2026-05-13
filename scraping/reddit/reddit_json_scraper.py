import asyncio
import aiohttp
import json
import os
import threading
import traceback
import datetime as dt
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import bittensor as bt
from aiohttp_socks import ProxyConnector
from common.data import DataEntity, DataLabel, DataSource
from common.date_range import DateRange
from scraping.scraper import ScrapeConfig, Scraper, ValidationResult
from scraping.reddit.model import RedditContent, RedditDataType, DELETED_USER
from scraping.reddit.utils import (
    is_valid_reddit_url,
    validate_reddit_content,
    normalize_label,
    normalize_permalink,
    extract_media_urls,
)
from common.protocol import KeywordMode


def _format_host_for_proxy(host: str) -> str:
    """Bracket IPv6 literals so they parse correctly in proxy URLs."""
    h = host.strip()
    if ":" in h and not (h.startswith("[") and h.endswith("]")):
        return f"[{h}]"
    return h


def _socks5_url(host: str, port: int, username: str, password: str) -> str:
    """Build a socks5:// URL for aiohttp-socks (credentials URL-encoded)."""
    h = _format_host_for_proxy(host)
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    if user or pwd:
        return f"socks5://{user}:{pwd}@{h}:{port}"
    return f"socks5://{h}:{port}"


def _load_socks5_specs_from_env() -> Optional[List[dict]]:
    raw = os.getenv("REDDIT_JSON_SOCKS5_PROXIES", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        bt.logging.warning(f"Invalid REDDIT_JSON_SOCKS5_PROXIES JSON: {e}")
        return None
    if not isinstance(data, list):
        bt.logging.warning("REDDIT_JSON_SOCKS5_PROXIES must be a JSON array; ignoring.")
        return None
    return data


def _proxy_routes_from_specs(specs: Optional[List[dict]]) -> List[Optional[str]]:
    """
    Build rotation list: direct (None) first, then up to four SOCKS5 proxy URLs.
    None means use the machine's default egress (no proxy).
    """
    routes: List[Optional[str]] = [None]
    if not specs:
        return routes
    for i, spec in enumerate(specs[:4]):
        try:
            host = str(spec["host"]).strip()
            port = int(spec["port"])
            username = str(spec.get("username") or "")
            password = str(spec.get("password") or "")
            if not host:
                raise ValueError("missing host")
            routes.append(_socks5_url(host, port, username, password))
        except (KeyError, TypeError, ValueError) as e:
            bt.logging.warning(f"Skipping invalid SOCKS5 proxy entry #{i + 1}: {e}")
    return routes


class RedditJsonScraper(Scraper):
    """
    Scrapes Reddit data using Reddit's public JSON API (no authentication required).
    This scraper accesses publicly available data through Reddit's .json endpoints.
    """

    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    BASE_URL = "https://www.reddit.com"

    # On-demand: Reddit returns at most 100 items per listing page; on-demand jobs allow up to 1000 entities.
    REDDIT_LISTING_MAX_PER_PAGE = 100
    ON_DEMAND_MAX_ENTITIES = 1000
    ON_DEMAND_MAX_PAGES = 150  # safety cap for pagination loops

    # Rate limiting settings
    REQUEST_TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, socks5_proxies: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            socks5_proxies: Up to four dicts with keys host, port, and optional username, password.
                If None, reads JSON from env REDDIT_JSON_SOCKS5_PROXIES (same shape).
                Outbound traffic rotates: direct IP, then each SOCKS5, in round-robin order.
        """
        specs = socks5_proxies if socks5_proxies is not None else _load_socks5_specs_from_env()
        self._proxy_routes = _proxy_routes_from_specs(specs)
        self._proxy_rr_lock = threading.Lock()
        self._proxy_rr_index = 0
        if len(self._proxy_routes) > 1:
            bt.logging.info(
                f"RedditJsonScraper: round-robin across {len(self._proxy_routes)} routes "
                f"(direct + {len(self._proxy_routes) - 1} SOCKS5)."
            )

    def _next_proxy_url(self) -> Optional[str]:
        with self._proxy_rr_lock:
            route = self._proxy_routes[self._proxy_rr_index % len(self._proxy_routes)]
            self._proxy_rr_index += 1
            return route

    @asynccontextmanager
    async def _client_session(self):
        route = self._next_proxy_url()
        connector = (
            aiohttp.TCPConnector()
            if route is None
            else ProxyConnector.from_url(route)
        )
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": self.USER_AGENT},
        ) as session:
            yield session

    async def validate(self, entities: List[DataEntity]) -> List[ValidationResult]:
        """
        Validate a list of DataEntity objects using Reddit's public JSON API.
        """
        if not entities:
            return []

        results: List[ValidationResult] = []

        for entity in entities:
            # 1) Basic URI sanity check
            if not is_valid_reddit_url(entity.uri):
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Invalid URI.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 2) Decode RedditContent blob
            try:
                ent_content = RedditContent.from_data_entity(entity)
            except Exception:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Failed to decode data entity.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 3) Fetch live data from Reddit's JSON API
            try:
                # For comments, we pass the expected comment id so we can
                # locate the exact node in the JSON tree instead of taking
                # the first child only.
                live_content = await self._fetch_content_from_url(
                    ent_content.url,
                    ent_content.data_type,
                    expected_comment_id=ent_content.id
                    if ent_content.data_type == RedditDataType.COMMENT
                    else None,
                )
            except Exception as e:
                bt.logging.error(f"Failed to retrieve content for {entity.uri}: {e}")
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Failed to retrieve submission/comment from Reddit.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 4) Live content object exists?
            if not live_content:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        reason="Reddit content not found or invalid.",
                        content_size_bytes_validated=entity.content_size_bytes,
                    )
                )
                continue

            # 5) Field-by-field validation
            validation_result = validate_reddit_content(
                actual_content=live_content,
                entity_to_validate=entity,
            )

            results.append(validation_result)

        return results

    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        """Scrapes a batch of reddit posts/comments according to the scrape config."""
        bt.logging.trace(
            f"Reddit JSON scraper performing scrape with config: {scrape_config}."
        )

        assert (
            not scrape_config.labels or len(scrape_config.labels) <= 1
        ), "Can only scrape 1 subreddit at a time."

        # Strip the r/ from the config or use 'all' if no label is provided.
        subreddit_name = (
            normalize_label(scrape_config.labels[0]) if scrape_config.labels else "all"
        )

        bt.logging.trace(
            f"Running Reddit JSON scraper with subreddit: {subreddit_name}."
        )

        # Get the search parameters
        limit = min(scrape_config.entity_limit, 100)  # Reddit API max is 100
        sort = self._get_sort_for_date_range(scrape_config.date_range.end)

        contents = []
        try:
            # Fetch posts from the subreddit
            # IMPORTANT: raw_json=1 returns unescaped text (e.g., ">" instead of "&gt;")
            # This matches PRAW output format for consistent validation with miners
            url = f"{self.BASE_URL}/r/{subreddit_name}/{sort}.json?limit={limit}&raw_json=1"
            posts = await self._fetch_posts(url)

            for post_data in posts:
                content = self._parse_post(post_data)
                if content:
                    contents.append(content)

        except Exception as e:
            bt.logging.error(
                f"Failed to scrape reddit using subreddit {subreddit_name}: {traceback.format_exc()}."
            )
            return []

        # Filter out NSFW content with media
        filtered_contents = []
        for content in contents:
            if content.is_nsfw and content.media:
                bt.logging.trace(f"Skipping NSFW content with media: {content.url}")
                continue
            filtered_contents.append(content)

        bt.logging.success(
            f"Completed scrape for subreddit {subreddit_name}. Scraped {len(filtered_contents)} items "
            f"(filtered out {len(contents) - len(filtered_contents)} NSFW+media posts)."
        )

        # Convert to DataEntity objects
        data_entities = []
        for content in filtered_contents:
            data_entities.append(RedditContent.to_data_entity(content=content))

        return data_entities

    async def on_demand_scrape(
        self,
        usernames: List[str] = None,
        subreddit: str = "all",
        keywords: List[str] = None,
        keyword_mode: KeywordMode = "all",
        start_datetime: dt.datetime = None,
        end_datetime: dt.datetime = None,
        limit: int = 100,
        reddit_global_search: bool = False,
    ) -> List[DataEntity]:
        """
        Scrapes Reddit data based on specific search criteria using public JSON API.

        Reddit returns at most ``REDDIT_LISTING_MAX_PER_PAGE`` children per HTTP response.
        This method follows Listing ``after`` tokens until ``limit`` matching items are
        collected or the listing ends.

        Args:
            usernames: If non-empty, gather posts and comments for these authors (OR across names).
            subreddit: Target subreddit without ``r/`` when not using global search.
            keywords: Optional substring filters (also passed through Reddit ``q=`` when searching).
            keyword_mode: ``any`` or ``all`` for miner-side keyword matching.
            start_datetime / end_datetime: UTC bounds applied in ``_matches_criteria``.
            limit: Target number of entities (capped at ``ON_DEMAND_MAX_ENTITIES``).
            reddit_global_search: Sitewide ``/search.json`` (requires keywords).

        Returns:
            Up to ``limit`` ``DataEntity`` objects.
        """
        usernames = list(usernames or [])
        keywords = list(keywords or [])

        if (
            not usernames
            and not keywords
            and start_datetime is None
            and end_datetime is None
            and (not subreddit or subreddit == "all")
        ):
            bt.logging.trace("No search parameters, returning empty list")
            return []

        bt.logging.trace(
            f"On-demand scrape with usernames={usernames}, subreddit={subreddit}, "
            f"keywords={keywords}, keyword_mode={keyword_mode}, reddit_global_search={reddit_global_search}, "
            f"start={start_datetime}, end={end_datetime}"
        )

        if reddit_global_search and not keywords:
            bt.logging.trace("reddit_global_search requires keywords, returning empty list")
            return []

        target_limit = min(max(1, int(limit)), self.ON_DEMAND_MAX_ENTITIES)
        page_lim = self.REDDIT_LISTING_MAX_PER_PAGE

        contents: List[RedditContent] = []

        seen_ids: Set[str] = set()

        def try_add(
            c: Optional[RedditContent],
            kw_match: Optional[List[str]],
            posts_from_reddit_search: bool,
        ) -> None:
            if not c or len(contents) >= target_limit:
                return
            dedup = c.id or c.url or ""
            if dedup in seen_ids:
                return
            mkw = None if posts_from_reddit_search else kw_match
            if not self._matches_criteria(
                c, mkw, keyword_mode, start_datetime, end_datetime
            ):
                return
            seen_ids.add(dedup)
            contents.append(c)

        try:
            # Case 1: user profiles — paginate submitted + comments with ``after``
            if usernames:
                for username in usernames:
                    if len(contents) >= target_limit:
                        break
                    try:
                        for path_suffix in ("submitted", "comments"):
                            if len(contents) >= target_limit:
                                break
                            after_cursor: Optional[str] = None
                            pages = 0
                            while (
                                len(contents) < target_limit
                                and pages < self.ON_DEMAND_MAX_PAGES
                            ):
                                base = (
                                    f"{self.BASE_URL}/user/{username}/{path_suffix}.json"
                                    f"?limit={page_lim}&raw_json=1"
                                )
                                url = (
                                    f"{base}&after={quote(after_cursor, safe='')}"
                                    if after_cursor
                                    else base
                                )
                                children, after_cursor = await self._fetch_listing_page(
                                    url
                                )
                                pages += 1
                                for child in children:
                                    if len(contents) >= target_limit:
                                        break
                                    c = self._listing_child_to_content(child)
                                    try_add(c, keywords, False)
                                if not after_cursor:
                                    break
                    except Exception as e:
                        bt.logging.warning(f"Failed to scrape user '{username}': {e}")
                        continue

            # Case 2: sitewide search — paginate ``/search.json``
            elif reddit_global_search:
                if keyword_mode == "all":
                    search_query = ' AND '.join(f'"{kw}"' for kw in keywords)
                else:
                    search_query = ' OR '.join(f'"{kw}"' for kw in keywords)
                q_enc = quote(search_query, safe="")
                after_cursor = None
                pages = 0
                while (
                    len(contents) < target_limit and pages < self.ON_DEMAND_MAX_PAGES
                ):
                    base = (
                        f"{self.BASE_URL}/search.json?q={q_enc}&restrict_sr=0"
                        f"&limit={page_lim}&sort=new&raw_json=1"
                    )
                    url = (
                        f"{base}&after={quote(after_cursor, safe='')}"
                        if after_cursor
                        else base
                    )
                    bt.logging.debug(f"Reddit sitewide search: {url}")
                    children, after_cursor = await self._fetch_listing_page(url)
                    pages += 1
                    for child in children:
                        if len(contents) >= target_limit:
                            break
                        kind = child.get("kind", "")
                        if kind == "t3":
                            c = self._parse_post(child)
                        elif kind == "t1":
                            c = self._parse_comment(child)
                        else:
                            c = self._parse_post(child)
                        try_add(c, None, True)
                    if not after_cursor:
                        break

            # Case 3: subreddit — ``search.json`` or ``/new.json``, paginated
            else:
                subreddit_name = (
                    subreddit.removeprefix("r/")
                    if subreddit and subreddit.startswith("r/")
                    else subreddit
                )

                if not subreddit_name:
                    bt.logging.warning(
                        "Subreddit-scoped on_demand_scrape requires a subreddit; returning no posts"
                    )
                elif keywords:
                    if keyword_mode == "all":
                        search_query = ' AND '.join(f'"{kw}"' for kw in keywords)
                    else:
                        search_query = ' OR '.join(f'"{kw}"' for kw in keywords)
                    q_enc = quote(search_query, safe="")
                    after_cursor = None
                    pages = 0
                    while (
                        len(contents) < target_limit
                        and pages < self.ON_DEMAND_MAX_PAGES
                    ):
                        base = (
                            f"{self.BASE_URL}/r/{subreddit_name}/search.json?q={q_enc}"
                            f"&restrict_sr=1&limit={page_lim}&sort=new&raw_json=1"
                        )
                        url = (
                            f"{base}&after={quote(after_cursor, safe='')}"
                            if after_cursor
                            else base
                        )
                        children, after_cursor = await self._fetch_listing_page(url)
                        pages += 1
                        for child in children:
                            if len(contents) >= target_limit:
                                break
                            c = self._listing_child_to_content(child)
                            try_add(c, keywords, True)
                        if not after_cursor:
                            break
                else:
                    after_cursor = None
                    pages = 0
                    while (
                        len(contents) < target_limit
                        and pages < self.ON_DEMAND_MAX_PAGES
                    ):
                        base = (
                            f"{self.BASE_URL}/r/{subreddit_name}/new.json"
                            f"?limit={page_lim}&raw_json=1"
                        )
                        url = (
                            f"{base}&after={quote(after_cursor, safe='')}"
                            if after_cursor
                            else base
                        )
                        children, after_cursor = await self._fetch_listing_page(url)
                        pages += 1
                        for child in children:
                            if len(contents) >= target_limit:
                                break
                            c = self._listing_child_to_content(child)
                            try_add(c, keywords, False)
                        if not after_cursor:
                            break

        except Exception as e:
            bt.logging.error(f"Failed to perform on-demand scrape: {e}")
            bt.logging.error(traceback.format_exc())
            return []

        filtered_contents: List[RedditContent] = []
        for content in contents:
            if len(filtered_contents) >= target_limit:
                break
            if content.is_nsfw and content.media:
                bt.logging.trace(f"Skipping NSFW content with media: {content.url}")
                continue
            filtered_contents.append(content)

        bt.logging.success(
            f"On-demand scrape completed. Returning {len(filtered_contents)} items "
            f"(target_limit={target_limit}, pre-filter count={len(contents)})."
        )

        data_entities: List[DataEntity] = []
        for content in filtered_contents[:target_limit]:
            data_entities.append(RedditContent.to_data_entity(content=content))

        return data_entities

    async def _fetch_posts(self, url: str) -> List[dict]:
        """
        Fetch posts from Reddit's JSON API with retry logic.

        Returns:
            List of post/comment data dictionaries
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._client_session() as session:
                    async with session.get(url, timeout=self.REQUEST_TIMEOUT) as response:
                        if response.status == 429:
                            # Rate limited, wait and retry (next attempt uses next rotated route)
                            retry_after = int(
                                response.headers.get("Retry-After", self.RETRY_DELAY)
                            )
                            bt.logging.warning(
                                f"Rate limited, waiting {retry_after}s before retry..."
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        if response.status != 200:
                            bt.logging.warning(f"Got status {response.status} from {url}")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_DELAY)
                                continue
                            return []

                        data = await response.json()

                        # Reddit JSON API returns data in "data" -> "children" structure
                        if isinstance(data, dict) and "data" in data:
                            children = data["data"].get("children", [])
                            return children
                        elif isinstance(data, list) and len(data) > 0:
                            # Sometimes Reddit returns a list (e.g., for comments)
                            if "data" in data[0]:
                                return data[0]["data"].get("children", [])

                        return []

            except asyncio.TimeoutError:
                bt.logging.warning(
                    f"Timeout fetching {url}, attempt {attempt + 1}/{self.MAX_RETRIES}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue
            except Exception as e:
                bt.logging.error(f"Error fetching {url}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue

        return []

    async def _fetch_listing_page(self, url: str) -> Tuple[List[dict], Optional[str]]:
        """
        Fetch one Reddit Listing page. Returns (children, after_cursor).

        ``after_cursor`` is Reddit's fullname token for the next page (e.g. t3_xxx).
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._client_session() as session:
                    async with session.get(url, timeout=self.REQUEST_TIMEOUT) as response:
                        if response.status == 429:
                            retry_after = int(
                                response.headers.get("Retry-After", self.RETRY_DELAY)
                            )
                            bt.logging.warning(
                                f"Rate limited, waiting {retry_after}s before retry..."
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        if response.status != 200:
                            bt.logging.warning(f"Got status {response.status} from {url}")
                            if attempt < self.MAX_RETRIES - 1:
                                await asyncio.sleep(self.RETRY_DELAY)
                                continue
                            return [], None

                        data = await response.json()

                        if isinstance(data, dict) and "data" in data:
                            listing = data["data"]
                            children = listing.get("children", [])
                            after = listing.get("after")
                            return children, after
                        if isinstance(data, list) and len(data) > 0:
                            if "data" in data[0]:
                                listing = data[0]["data"]
                                children = listing.get("children", [])
                                after = listing.get("after")
                                return children, after

                        return [], None

            except asyncio.TimeoutError:
                bt.logging.warning(
                    f"Timeout fetching {url}, attempt {attempt + 1}/{self.MAX_RETRIES}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue
            except Exception as e:
                bt.logging.error(f"Error fetching {url}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue

        return [], None

    def _listing_child_to_content(self, post_data: dict) -> Optional[RedditContent]:
        """Map a Listing child ``{kind, data}`` to RedditContent."""
        kind = post_data.get("kind", "")
        if kind == "t3":
            return self._parse_post(post_data)
        if kind == "t1":
            return self._parse_comment(post_data)
        return self._parse_post(post_data)

    async def _fetch_content_from_url(
        self,
        url: str,
        data_type: RedditDataType,
        expected_comment_id: Optional[str] = None,
    ) -> Optional[RedditContent]:
        """
        Fetch and parse a specific post or comment from its URL.
        """
        # Normalize URL: remove .json if present, remove existing query params
        clean_url = url
        if clean_url.rstrip('/').endswith('.json'):
            clean_url = clean_url.rstrip('/')[:-5] + '/'
        if '?' in clean_url:
            clean_url = clean_url.split('?')[0]

        # Add .json and raw_json=1 parameter
        # raw_json=1 returns unescaped text (e.g., ">" instead of "&gt;") to match PRAW output
        # Reddit accepts /.json format (e.g., /comments/abc/.json)
        json_url = f"{clean_url}.json?raw_json=1"

        try:
            async with self._client_session() as session:
                async with session.get(json_url, timeout=self.REQUEST_TIMEOUT) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    if data_type == RedditDataType.POST:
                        # For posts, data is a list where [0] contains the post
                        if isinstance(data, list) and len(data) > 0:
                            children = data[0].get("data", {}).get("children", [])
                            if children:
                                return self._parse_post(children[0])
                    elif data_type == RedditDataType.COMMENT:
                        # For comments, we navigate the full comment tree to find
                        # the node matching the expected comment id, rather than
                        # assuming it is the first child.
                        if isinstance(data, list) and len(data) > 1:
                            # Get parent post's NSFW status (comments inherit from parent)
                            parent_post_data = (
                                data[0]
                                .get("data", {})
                                .get("children", [{}])[0]
                                .get("data", {})
                            )
                            parent_nsfw = parent_post_data.get("over_18", False)

                            def _walk_comments(
                                nodes: List[dict],
                            ) -> Optional[RedditContent]:
                                for node in nodes:
                                    if not isinstance(node, dict):
                                        continue
                                    kind = node.get("kind")
                                    node_data = node.get("data", {}) or {}
                                    name = node_data.get("name") or node_data.get("id")

                                    # Match against the full name (e.g. "t1_xxx") or bare id.
                                    if expected_comment_id and name:
                                        if (
                                            name == expected_comment_id
                                            or name.split("_")[-1]
                                            == expected_comment_id.split("_")[-1]
                                        ):
                                            return self._parse_comment(
                                                node, parent_nsfw=parent_nsfw
                                            )

                                    # Recurse into replies if present.
                                    replies = node_data.get("replies")
                                    if (
                                        isinstance(replies, dict)
                                        and "data" in replies
                                    ):
                                        reply_children = (
                                            replies.get("data", {})
                                            .get("children", [])
                                        )
                                        found = _walk_comments(reply_children)
                                        if found is not None:
                                            return found
                                return None

                            children = (
                                data[1]
                                .get("data", {})
                                .get("children", [])
                            )
                            # If we have an expected id, search the tree; otherwise,
                            # fall back to the first child as before.
                            if children:
                                if expected_comment_id:
                                    found = _walk_comments(children)
                                    if found is not None:
                                        return found
                                # Fallback: preserve previous behaviour.
                                return self._parse_comment(
                                    children[0], parent_nsfw=parent_nsfw
                                )

        except Exception as e:
            bt.logging.error(f"Error fetching content from {url}: {e}")

        return None

    def _parse_post(self, post_data: dict) -> Optional[RedditContent]:
        """
        Parse a Reddit post from JSON API response.
        """
        try:
            data = post_data.get("data", {})

            # Extract media URLs
            media_urls = []
            if data.get("url"):
                # Check if it's an image/video URL
                url = data["url"]
                if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webm']):
                    media_urls.append(url)
                elif 'reddit_video' in str(data.get("media", {})):
                    if data.get("media", {}).get("reddit_video", {}).get("fallback_url"):
                        media_urls.append(data["media"]["reddit_video"]["fallback_url"])

            # Check for gallery
            if data.get("gallery_data"):
                for item in data.get("media_metadata", {}).values():
                    if item.get("s", {}).get("u"):
                        media_urls.append(item["s"]["u"].replace("&amp;", "&"))

            username = data.get("author", DELETED_USER)
            if username == "[deleted]":
                username = DELETED_USER

            # ``name`` (fullname e.g. t3_xxx) is preferred; fall back to ``id`` for OD metadata completeness.
            post_id = data.get("id") or ""
            permalink = (data.get("permalink") or "").strip()
            if permalink:
                post_url = f"{self.BASE_URL}{normalize_permalink(permalink)}"
            else:
                link = (data.get("url") or "").split("?")[0]
                post_url = link if "reddit.com" in link else ""

            if not post_id or not post_url:
                bt.logging.trace("Skipping Reddit post: missing id or url")
                return None

            sub = data.get("subreddit_name_prefixed")
            if not sub and data.get("subreddit"):
                sub = f"r/{data['subreddit']}"

            selftext = data.get("selftext")
            if selftext is None:
                selftext = ""

            over_18 = data.get("over_18")
            if over_18 is None:
                over_18 = False

            score_raw = data.get("score")
            score = int(score_raw) if score_raw is not None else 0

            ur_raw = data.get("upvote_ratio")
            upvote_ratio = float(ur_raw) if ur_raw is not None else 0.0

            nc_raw = data.get("num_comments")
            num_comments = int(nc_raw) if nc_raw is not None else 0

            return RedditContent(
                id=post_id,
                url=post_url,
                username=username,
                communityName=sub or "",
                body=selftext,
                createdAt=dt.datetime.utcfromtimestamp(data.get("created_utc", 0)).replace(
                    tzinfo=dt.timezone.utc
                ),
                dataType=RedditDataType.POST,
                title=data.get("title") or "",
                parentId=None,
                media=media_urls if media_urls else None,
                is_nsfw=bool(over_18),
                score=score,
                upvote_ratio=upvote_ratio,
                num_comments=num_comments,
                scrapedAt=dt.datetime.now(dt.timezone.utc),
            )
        except Exception as e:
            bt.logging.trace(f"Failed to parse post: {e}")
            return None

    def _parse_comment(self, comment_data: dict, parent_nsfw: bool = False) -> Optional[RedditContent]:
        """
        Parse a Reddit comment from JSON API response.

        Args:
            comment_data: The comment data from Reddit JSON API
            parent_nsfw: NSFW status inherited from parent post (comments don't have their own over_18 field)
        """
        try:
            data = comment_data.get("data", {})

            username = data.get("author", DELETED_USER)
            if username == "[deleted]":
                username = DELETED_USER

            comment_id = data.get("name") or data.get("id") or ""
            permalink = (data.get("permalink") or "").strip()
            if permalink:
                comment_url = f"{self.BASE_URL}{normalize_permalink(permalink)}"
            else:
                comment_url = ""

            if not comment_id or not comment_url:
                bt.logging.trace("Skipping Reddit comment: missing id or url")
                return None

            sub = data.get("subreddit_name_prefixed")
            if not sub and data.get("subreddit"):
                sub = f"r/{data['subreddit']}"

            body = data.get("body") or ""

            score_raw = data.get("score")
            score = int(score_raw) if score_raw is not None else 0

            # OD metadata completeness requires non-null (comments have no ratio / thread count on Reddit JSON).
            return RedditContent(
                id=comment_id,
                url=comment_url,
                username=username,
                communityName=sub or "",
                body=body,
                createdAt=dt.datetime.utcfromtimestamp(data.get("created_utc", 0)).replace(
                    tzinfo=dt.timezone.utc
                ),
                dataType=RedditDataType.COMMENT,
                title=None,
                parentId=data.get("parent_id"),
                media=None,
                is_nsfw=bool(parent_nsfw),
                score=score,
                upvote_ratio=0.0,
                num_comments=0,
                scrapedAt=dt.datetime.now(dt.timezone.utc),
            )
        except Exception as e:
            bt.logging.trace(f"Failed to parse comment: {e}")
            return None

    @staticmethod
    def _normalize_inclusive_end_datetime(end: dt.datetime) -> dt.datetime:
        """Treat API end_date at UTC midnight as inclusive end of that calendar day."""
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        if (
            end.hour == 0
            and end.minute == 0
            and end.second == 0
            and end.microsecond == 0
        ):
            return end.replace(hour=23, minute=59, second=59, microsecond=999999)
        return end

    def _matches_criteria(
        self,
        content: RedditContent,
        keywords: List[str] = None,
        keyword_mode: KeywordMode = "all",
        start_datetime: dt.datetime = None,
        end_datetime: dt.datetime = None
    ) -> bool:
        """
        Check if content matches the specified criteria.
        """
        if start_datetime:
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=dt.timezone.utc)
            if content.created_at < start_datetime:
                return False

        if end_datetime:
            if end_datetime.tzinfo is None:
                end_datetime = end_datetime.replace(tzinfo=dt.timezone.utc)
            end_datetime = self._normalize_inclusive_end_datetime(end_datetime)
            if content.created_at > end_datetime:
                return False

        # Check keywords based on keyword_mode (match across common surfaced fields)
        if keywords:
            parts = []
            if content.title:
                parts.append(content.title.lower())
            if content.body:
                parts.append(content.body.lower())
            if content.url:
                parts.append(content.url.lower())
            if content.id:
                parts.append(content.id.lower())
            if content.username:
                parts.append(content.username.lower())
            if content.community:
                parts.append(content.community.lower())
            searchable_text = " ".join(parts)

            if keyword_mode == "all":
                if not all(keyword.lower() in searchable_text for keyword in keywords):
                    return False
            else:  # keyword_mode == "any"
                if not any(keyword.lower() in searchable_text for keyword in keywords):
                    return False

        return True

    def _get_sort_for_date_range(self, end_date: dt.datetime) -> str:
        """
        Determine the sort order based on the date range.
        """
        now = dt.datetime.now(tz=dt.timezone.utc)
        days_ago = (now - end_date).days

        if days_ago <= 1:
            return "new"
        else:
            return "top"


async def test_scrape():
    """Test the basic scrape functionality."""
    scraper = RedditJsonScraper()

    print("=" * 60)
    print("TESTING BASIC SCRAPE")
    print("=" * 60)

    entities = await scraper.scrape(
        ScrapeConfig(
            entity_limit=5,
            date_range=DateRange(
                start=dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=1),
                end=dt.datetime.now(tz=dt.timezone.utc),
            ),
            labels=[DataLabel(value="r/python")],
        )
    )

    print(f"\nScraped r/python: {len(entities)} entities")
    if entities:
        print(f"Sample URI: {entities[0].uri}")
        print(f"Sample datetime: {entities[0].datetime}")


async def test_on_demand_scrape():
    """Test the on_demand_scrape functionality."""
    scraper = RedditJsonScraper()

    print("\n" + "=" * 60)
    print("TESTING ON-DEMAND SCRAPE")
    print("=" * 60)

    # Test 1: Search by subreddit
    print("\n1. Testing subreddit search (r/python)...")
    entities = await scraper.on_demand_scrape(subreddit="r/python", limit=5)
    print(f"   Result: {len(entities)} entities from r/python")
    if entities:
        print(f"   Sample: {entities[0].uri}")

    # Test 2: Search by username
    print("\n2. Testing username search (spez)...")
    entities = await scraper.on_demand_scrape(usernames=["spez"], limit=3)
    print(f"   Result: {len(entities)} entities from user 'spez'")
    if entities:
        print(f"   Sample: {entities[0].uri}")

    # Test 3: Search with keywords
    print("\n3. Testing keyword search in r/python...")
    entities = await scraper.on_demand_scrape(
        subreddit="r/python",
        keywords=["django"],
        limit=3
    )
    print(f"   Result: {len(entities)} entities with 'django'")
    if entities:
        print(f"   Sample: {entities[0].uri}")

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


async def test_validation():
    """Test validation functionality and print all DataEntity details."""
    scraper = RedditJsonScraper()

    print("\n" + "=" * 60)
    print("TESTING VALIDATION")
    print("=" * 60)

    # First, scrape some data
    print("\n1. Scraping r/python to get test entities...")
    entities = await scraper.scrape(
        ScrapeConfig(
            entity_limit=3,
            date_range=DateRange(
                start=dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=1),
                end=dt.datetime.now(tz=dt.timezone.utc),
            ),
            labels=[DataLabel(value="r/python")],
        )
    )

    print(f"   Scraped {len(entities)} entities")

    # Print all DataEntity details
    print("\n2. Printing all DataEntity details:")
    print("-" * 60)
    for i, entity in enumerate(entities, 1):
        print(f"\n   Entity #{i}:")
        print(f"   URI: {entity.uri}")
        print(f"   Datetime: {entity.datetime}")
        print(f"   Source: {entity.source}")
        print(f"   Label: {entity.label}")
        print(f"   Content Size: {entity.content_size_bytes} bytes")

        # Decode and print the content
        try:
            content = RedditContent.from_data_entity(entity)
            print(f"   Content ID: {content.id}")
            print(f"   Username: {content.username}")
            print(f"   Community: {content.community}")
            print(f"   Data Type: {content.data_type}")
            print(f"   Title: {content.title[:80] + '...' if content.title and len(content.title) > 80 else content.title}")
            print(f"   Body: {content.body[:100] + '...' if content.body and len(content.body) > 100 else content.body}")
            print(f"   Created At: {content.created_at}")
            print(f"   Score: {content.score}")
            print(f"   Upvote Ratio: {content.upvote_ratio}")
            print(f"   Num Comments: {content.num_comments}")
            print(f"   NSFW: {content.is_nsfw}")
            print(f"   Media: {content.media}")
        except Exception as e:
            print(f"   Failed to decode content: {e}")

    # Validate the entities
    print("\n" + "-" * 60)
    print("3. Validating entities...")
    results = await scraper.validate(entities)

    print(f"\n   Validation Results:")
    for i, (entity, result) in enumerate(zip(entities, results), 1):
        print(f"\n   Entity #{i}: {entity.uri}")
        print(f"   Valid: {result.is_valid}")
        print(f"   Reason: {result.reason if hasattr(result, 'reason') and result.reason else 'N/A'}")
        print(f"   Content Size Validated: {result.content_size_bytes_validated} bytes")

    # Test with a known good entity (from bittensor_ subreddit)
    print("\n" + "-" * 60)
    print("4. Testing with a known good entity...")

    test_entity = DataEntity(
        uri="https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/",
        datetime=dt.datetime(2023, 12, 5, 15, 59, 13, tzinfo=dt.timezone.utc),
        source=DataSource.REDDIT,
        label=DataLabel(value="r/bittensor_"),
        content=b'{"id": "t3_18bf67l", "url": "https://www.reddit.com/r/bittensor_/comments/18bf67l/how_do_you_add_tao_to_metamask/", "username": "KOOLBREEZE144", "communityName": "r/bittensor_", "body": "Hey all!!\\n\\nHow do we add TAO to MetaMask? Online gives me these network configurations and still doesn\\u2019t work? \\n\\nHow are you all storing TAO? I wanna purchase on MEXC, but holding off until I can store it!  \\ud83d\\ude11 \\n\\nThanks in advance!!!\\n\\n=====\\n\\nhere is a manual way.\\nNetwork Name\\nTao Network\\n\\nRPC URL\\nhttp://rpc.testnet.tao.network\\n\\nChain ID\\n558\\n\\nCurrency Symbol\\nTAO", "createdAt": "2023-12-05T15:59:13+00:00", "dataType": "post", "title": "How do you add TAO to MetaMask?", "parentId": null}',
        content_size_bytes=775,
    )

    print(f"   Test Entity URI: {test_entity.uri}")

    validation_results = await scraper.validate([test_entity])
    print(f"   Validation Result:")
    print(f"   Valid: {validation_results[0].is_valid}")
    print(f"   Reason: {validation_results[0].reason if hasattr(validation_results[0], 'reason') and validation_results[0].reason else 'N/A'}")

    print("\n" + "=" * 60)
    print("VALIDATION TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_scrape())
    asyncio.run(test_on_demand_scrape())
    asyncio.run(test_validation())
