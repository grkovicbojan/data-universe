"""
Miner-side on-demand self-check aligned with ``vali_utils/on_demand/on_demand_validation.py``.

Filters scraped ``DataEntity`` rows before returning them on ``OnDemandRequest`` (validator
synapse) or before upload (API jobs that reuse the same scrape path).

Set ``MINER_OD_SELF_CHECK_SCRAPER=0`` to skip live ``scraper.validate`` (phase 4) while still
running format, job-match, and metadata checks — useful if X Apify credentials are unavailable.

Phases (per entity, in order):
  1) Parse / format (``XContent`` / ``RedditContent``), dedupe by URI
  2) Request-field + time window checks (same rules as validator)
  3) ``validate_metadata_completeness`` (``vali_utils/on_demand/output_models``)
  4) Optional live ``scraper.validate`` (``ApiDojoTwitterScraper`` / ``RedditJsonScraper``)
"""

from __future__ import annotations

import json
import datetime as dt
from typing import List, Optional

import bittensor as bt

from common import constants, utils
from common.data import DataEntity, DataSource
from common.protocol import OnDemandRequest
from scraping.reddit.model import RedditContent
from scraping.reddit.reddit_json_scraper import RedditJsonScraper
from scraping.x.apidojo_scraper import ApiDojoTwitterScraper
from scraping.x.model import XContent
from vali_utils.on_demand import utils as on_demand_utils
from vali_utils.on_demand.on_demand_validation import ValidationContext
from vali_utils.on_demand.output_models import validate_metadata_completeness


def validation_context_from_on_demand_request(synapse: OnDemandRequest) -> ValidationContext:
    """Build ``ValidationContext`` from a dendrite ``OnDemandRequest`` (matches job shape)."""
    if synapse.source == DataSource.X:
        src = "x"
    elif synapse.source == DataSource.REDDIT:
        src = "reddit"
    else:
        src = DataSource(synapse.source).name.lower()

    return ValidationContext(
        source=src,
        usernames=list(synapse.usernames or []),
        keywords=list(synapse.keywords or []),
        url=synapse.url,
        keyword_mode=synapse.keyword_mode,
        start_date=synapse.start_date,
        end_date=synapse.end_date,
        limit=synapse.limit,
    )


def _get_post_id(post: DataEntity) -> str:
    return getattr(post, "uri", None) or str(hash(str(post)))


def _validate_time_range(
    ctx: ValidationContext, post_timestamp: dt.datetime
) -> bool:
    try:
        if ctx.start_date:
            start_dt = utils.parse_iso_date(ctx.start_date)
            if start_dt is not None and post_timestamp < start_dt:
                return False
        if ctx.end_date:
            end_dt = utils.parse_iso_date(ctx.end_date)
            if end_dt is not None and post_timestamp > end_dt:
                return False
        return True
    except Exception as e:
        bt.logging.error(f"OD self-check time range error: {e}")
        return False


def _validate_x_request_fields(
    ctx: ValidationContext, x_entity: DataEntity, log_hotkey: str
) -> bool:
    try:
        x_content_dict = json.loads(x_entity.content.decode("utf-8"))

        if ctx.url:
            from scraping.x import utils as x_utils

            post_url = x_content_dict.get("url", "")
            if x_utils.normalize_url(post_url) != x_utils.normalize_url(ctx.url):
                bt.logging.trace(
                    f"OD self-check X [{log_hotkey}] URL mismatch: {post_url} != {ctx.url}"
                )
                return False
            return _validate_time_range(ctx, x_entity.datetime)

        if ctx.usernames:
            requested_usernames = [u.strip("@").lower() for u in ctx.usernames]
            if "user" in x_content_dict:
                user_dict = x_content_dict.get("user", {})
                post_username = user_dict.get("username", "").strip("@").lower()
            else:
                post_username = x_content_dict.get("username", "").strip("@").lower()

            if not post_username or post_username not in requested_usernames:
                bt.logging.trace(
                    f"OD self-check X [{log_hotkey}] username mismatch: {post_username}"
                )
                return False

        if ctx.keywords:
            post_text = x_content_dict.get("text", "").lower()
            if ctx.keyword_mode == "all":
                if not all(kw.lower() in post_text for kw in ctx.keywords):
                    return False
            else:
                if not any(kw.lower() in post_text for kw in ctx.keywords):
                    return False

        return _validate_time_range(ctx, x_entity.datetime)
    except Exception as e:
        bt.logging.trace(f"OD self-check X request fields error: {e}")
        return False


def _validate_reddit_request_fields(
    ctx: ValidationContext, reddit_entity: DataEntity, log_hotkey: str
) -> bool:
    try:
        reddit_content_dict = json.loads(reddit_entity.content.decode("utf-8"))

        if ctx.usernames:
            requested_usernames = [u.lower() for u in ctx.usernames]
            post_username = reddit_content_dict.get("username")
            if not post_username or post_username.lower() not in requested_usernames:
                bt.logging.trace(
                    f"OD self-check Reddit [{log_hotkey}] username mismatch: {post_username}"
                )
                return False

        if ctx.keywords:
            post_community = reddit_content_dict.get("communityName")
            if post_community:
                post_community = post_community.lower().removeprefix("r/")
                subreddit_match = any(
                    kw.lower().removeprefix("r/") == post_community for kw in ctx.keywords
                )
            else:
                subreddit_match = False

            body_text = reddit_content_dict.get("body") or ""
            title_text = reddit_content_dict.get("title") or ""
            content_text = (body_text + " " + title_text).lower().strip()

            if ctx.keyword_mode == "all":
                keyword_in_content = (
                    all(kw.lower() in content_text for kw in ctx.keywords)
                    if content_text
                    else False
                )
            else:
                keyword_in_content = (
                    any(kw.lower() in content_text for kw in ctx.keywords)
                    if content_text
                    else False
                )

            if not (subreddit_match or keyword_in_content):
                return False

        return _validate_time_range(ctx, reddit_entity.datetime)
    except Exception as e:
        bt.logging.trace(f"OD self-check Reddit request fields error: {e}")
        return False


def _validate_request_fields(
    ctx: ValidationContext, entity: DataEntity, log_hotkey: str
) -> bool:
    src = ctx.source.upper()
    if src == "X":
        return _validate_x_request_fields(ctx, entity, log_hotkey)
    if src == "REDDIT":
        return _validate_reddit_request_fields(ctx, entity, log_hotkey)
    return False


def _coerce_reddit_od_metadata(entity: DataEntity) -> DataEntity:
    """Ensure score / upvote_ratio / num_comments are non-null for OD metadata completeness."""
    try:
        c = RedditContent.from_data_entity(entity)
    except Exception:
        return entity

    updates = {}
    if c.score is None:
        updates["score"] = 0
    if c.upvote_ratio is None:
        updates["upvote_ratio"] = 0.0
    if c.num_comments is None:
        updates["num_comments"] = 0
    if not updates:
        return entity

    try:
        fixed = c.copy(update=updates)
        return RedditContent.to_data_entity(fixed)
    except Exception:
        return entity


def _format_parse_ok(entity: DataEntity, ctx: ValidationContext) -> bool:
    """Single-item format check (validator ``_validate_miner_data_format`` parse step)."""
    src = ctx.source.upper()
    try:
        if not isinstance(entity, DataEntity) or not entity.uri or not entity.content:
            return False
        if src == "X":
            XContent.from_data_entity(entity)
        elif src == "REDDIT":
            RedditContent.from_data_entity(entity)
        else:
            return False
        return True
    except Exception:
        return False


def _entity_for_scraper_validate(entity: DataEntity, ctx: ValidationContext) -> DataEntity:
    if ctx.source.upper() != "X":
        return entity
    if (
        on_demand_utils.is_nested_format(entity)
        and dt.datetime.now(tz=dt.timezone.utc)
        < constants.X_ENHANCED_FORMAT_COMPATIBILITY_EXPIRATION_DATE
    ):
        x_content = XContent.from_data_entity(entity)
        return XContent.to_data_entity(content=x_content)
    return entity


async def _scraper_validate_one(
    ctx: ValidationContext,
    entity: DataEntity,
    x_scraper: Optional[ApiDojoTwitterScraper],
    reddit_scraper: Optional[RedditJsonScraper],
) -> bool:
    ent = _entity_for_scraper_validate(entity, ctx)
    try:
        if ctx.source.upper() == "X" and x_scraper is not None:
            results = await x_scraper.validate(
                entities=[ent], allow_low_engagement=True
            )
        elif ctx.source.upper() == "REDDIT" and reddit_scraper is not None:
            results = await reddit_scraper.validate([ent])
        else:
            return False

        if not results:
            return False
        r = results[0]
        return r.is_valid if hasattr(r, "is_valid") else bool(r)
    except Exception as e:
        bt.logging.trace(f"OD self-check scraper validate error: {e}")
        return False


async def self_check_and_filter_od_entities(
    ctx: ValidationContext,
    entities: List[DataEntity],
    max_count: int,
    log_hotkey: str,
    *,
    run_scraper_validation: bool = True,
) -> List[DataEntity]:
    """
    Return up to ``max_count`` entities that pass the same checks the OD validator applies
    (format, request fields, metadata completeness, optional scraper validate).
    """
    if not entities or max_count <= 0:
        return []

    x_scraper: Optional[ApiDojoTwitterScraper] = None
    reddit_scraper: Optional[RedditJsonScraper] = None
    if run_scraper_validation:
        if ctx.source.upper() == "X":
            x_scraper = ApiDojoTwitterScraper()
        elif ctx.source.upper() == "REDDIT":
            reddit_scraper = RedditJsonScraper()

    seen: set[str] = set()
    out: List[DataEntity] = []

    for raw in entities:
        if len(out) >= max_count:
            break

        entity = raw
        if ctx.source.upper() == "REDDIT":
            entity = _coerce_reddit_od_metadata(raw)

        if not _format_parse_ok(entity, ctx):
            continue

        pid = _get_post_id(entity)
        if pid in seen:
            continue

        if not _validate_request_fields(ctx, entity, log_hotkey):
            continue

        ok_meta, missing = validate_metadata_completeness(entity)
        if not ok_meta:
            bt.logging.trace(
                f"OD self-check [{log_hotkey}] drop metadata uri={entity.uri} missing={missing}"
            )
            continue

        if run_scraper_validation:
            if not await _scraper_validate_one(ctx, entity, x_scraper, reddit_scraper):
                bt.logging.trace(
                    f"OD self-check [{log_hotkey}] scraper failed uri={entity.uri}"
                )
                continue

        seen.add(pid)
        out.append(entity)

    if len(out) < len(entities):
        bt.logging.info(
            f"OD self-check [{log_hotkey}]: {len(entities)} scraped -> {len(out)} "
            f"passed (cap={max_count}, scraper={run_scraper_validation})"
        )
    return out
