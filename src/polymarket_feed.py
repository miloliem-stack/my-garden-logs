"""Polymarket market discovery and quote retrieval helpers."""
import calendar
import json
import os
import re
import time
from datetime import timezone
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests


POLY_GAMMA_BASE = os.getenv("POLY_GAMMA_BASE", "https://gamma-api.polymarket.com").rstrip("/")
POLY_CLOB_BASE = os.getenv("POLY_CLOB_BASE", "https://clob.polymarket.com").rstrip("/")
QUOTE_MAX_AGE_SEC = float(os.getenv("QUOTE_MAX_AGE_SEC", "5"))
QUOTE_MAX_SPREAD = float(os.getenv("QUOTE_MAX_SPREAD", "0.10"))
QUOTE_REQUIRE_BOTH_SIDES = str(os.getenv("QUOTE_REQUIRE_BOTH_SIDES", "true")).lower() in ("1", "true", "yes", "on")
QUOTE_MIN_DEPTH = float(os.getenv("QUOTE_MIN_DEPTH", "0"))
QUOTE_CACHE_TTL_SEC = float(os.getenv("QUOTE_CACHE_TTL_SEC", "1"))
_QUOTE_CACHE: Dict[str, Dict] = {}
_ET_TZ = ZoneInfo("America/New_York")
_BTC_SERIES_IDS = {"bitcoin-up-or-down", "btc-hourly", "btc-hourly-up-down", "btc"}
_BTC_HOURLY_SLUG_RE = re.compile(
    r"^bitcoin-up-or-down-([a-z]+)-(\d{1,2})-(\d{4})-(\d{1,2})(am|pm)-et$",
    re.IGNORECASE,
)
_BTC_HOURLY_TITLE_RE = re.compile(
    r"^bitcoin up or down ([A-Za-z]+) (\d{1,2}), (\d{4}) (\d{1,2})(am|pm) et$",
    re.IGNORECASE,
)


def _parse_dt(value):
    if value in (None, ""):
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts, tz="UTC")


def _request_json(path: str, *, params: Optional[dict] = None, timeout: float = 4.0):
    try:
        response = requests.get(f"{POLY_GAMMA_BASE}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _normalize_series_id(series_id: Optional[str]) -> str:
    return str(series_id or "").strip().lower()


def _coerce_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _coerce_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "markets", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def build_btc_hourly_event_slug(dt_et) -> str:
    ts = pd.Timestamp(dt_et)
    if ts.tzinfo is None:
        ts = ts.tz_localize(_ET_TZ)
    else:
        ts = ts.tz_convert(_ET_TZ)
    month = calendar.month_name[ts.month].lower()
    hour = ts.hour
    hour_12 = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    return f"bitcoin-up-or-down-{month}-{ts.day}-{ts.year}-{hour_12}{suffix}-et"


def _parse_btc_hourly_start_from_slug_or_title(slug: Optional[str], title: Optional[str]):
    for value, pattern in ((slug, _BTC_HOURLY_SLUG_RE), (title, _BTC_HOURLY_TITLE_RE)):
        text = str(value or "").strip()
        if not text:
            continue
        match = pattern.match(text)
        if not match:
            continue
        month_name, day, year, hour_12, suffix = match.groups()
        try:
            month = list(calendar.month_name).index(month_name.capitalize())
        except ValueError:
            continue
        hour = int(hour_12) % 12
        if suffix.lower() == "pm":
            hour += 12
        return pd.Timestamp(
            year=int(year),
            month=month,
            day=int(day),
            hour=hour,
            tz=_ET_TZ,
        ).tz_convert("UTC")
    return None


def _normalize_btc_hourly_window(slug: Optional[str], title: Optional[str], start, end):
    duration = (end - start).total_seconds() if start is not None and end is not None else None
    if duration is not None and abs(duration - 3600) <= 300:
        return start, end
    derived_start = _parse_btc_hourly_start_from_slug_or_title(slug, title)
    if derived_start is not None:
        return derived_start, derived_start + pd.Timedelta(hours=1)
    if end is not None:
        derived_start = end - pd.Timedelta(hours=1)
        return derived_start, end
    return start, end


def candidate_btc_hourly_event_slugs(now_utc=None) -> list[str]:
    now = _parse_dt(now_utc) or pd.Timestamp.now(tz="UTC")
    base_et = now.tz_convert(_ET_TZ).floor("h")
    ordered = [base_et, base_et - pd.Timedelta(hours=1), base_et + pd.Timedelta(hours=1)]
    slugs = []
    for dt_et in ordered:
        slug = build_btc_hourly_event_slug(dt_et)
        if slug not in slugs:
            slugs.append(slug)
    return slugs


def fetch_event_by_slug(slug) -> Optional[dict]:
    if not slug:
        return None
    payload = _request_json(f"/events/slug/{slug}")
    if isinstance(payload, dict):
        return payload
    payload = _request_json("/events", params={"slug": slug})
    items = _coerce_items(payload)
    return items[0] if items else None


def fetch_market_by_slug(slug) -> Optional[dict]:
    if not slug:
        return None
    payload = _request_json(f"/markets/slug/{slug}")
    if isinstance(payload, dict):
        return payload
    payload = _request_json("/markets", params={"slug": slug})
    items = _coerce_items(payload)
    return items[0] if items else None


def fetch_market_by_id(market_id) -> Optional[dict]:
    if not market_id:
        return None
    payload = _request_json(f"/markets/{market_id}")
    if isinstance(payload, dict):
        return payload
    payload = _request_json("/markets", params={"id": market_id})
    items = _coerce_items(payload)
    if items:
        return items[0]
    payload = _request_json("/markets", params={"market_id": market_id})
    items = _coerce_items(payload)
    return items[0] if items else None


def _fetch_generic_events(series_id: str) -> list[dict]:
    payload = _request_json("/events", params={"series": series_id, "limit": 200}, timeout=5.0)
    items = _coerce_items(payload)
    if items:
        return items
    payload = _request_json(f"/series/{series_id}/events", timeout=5.0)
    return _coerce_items(payload)


def _normalize_market_status(market: Dict, start, end, *, now=None) -> Optional[str]:
    raw_status = str(market.get("status") or market.get("state") or market.get("marketStatus") or "").strip().lower()
    if raw_status in ("open", "active", "tradable", "trading"):
        return "open"
    if raw_status in ("closed", "resolved", "redeemed", "archived", "paused", "settled", "finalized"):
        return None
    if market.get("closed") is True or market.get("archived") is True:
        return None
    if market.get("enableOrderBook") is True or market.get("acceptingOrders") is True or market.get("tradable") is True or market.get("active") is True:
        return "open"
    current = _parse_dt(now) or pd.Timestamp.now(tz="UTC")
    if start is not None and end is not None and start <= current < end and market.get("active") is True:
        return "open"
    return None


def _extract_tokens_from_outcomes(container: Dict) -> Dict[str, Optional[str]]:
    token_yes = None
    token_no = None
    clob_token_ids = _coerce_json_list(container.get("clobTokenIds"))
    outcomes = _coerce_json_list(container.get("outcomes"))
    if (
        isinstance(clob_token_ids, list)
        and isinstance(outcomes, list)
        and len(clob_token_ids) == len(outcomes)
    ):
        for label, token_id in zip(outcomes, clob_token_ids):
            normalized_label = str(label or "").strip().lower()
            if normalized_label in ("yes", "up"):
                token_yes = token_yes or token_id
            elif normalized_label in ("no", "down"):
                token_no = token_no or token_id
    outcome_groups = [
        container.get("tokens"),
        outcomes,
        container.get("outcomeTokens"),
        clob_token_ids,
    ]
    for group in outcome_groups:
        if isinstance(group, dict):
            group = list(group.values())
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                continue
            token_id = item.get("id") or item.get("tokenId") or item.get("token_id") or item.get("assetId")
            label = str(item.get("side") or item.get("name") or item.get("outcome") or item.get("label") or "").strip().upper()
            if label in ("YES", "UP"):
                token_yes = token_yes or token_id
            elif label in ("NO", "DOWN"):
                token_no = token_no or token_id
    return {"token_yes": token_yes, "token_no": token_no}


def _extract_tokens(container: Dict) -> Dict[str, Optional[str]]:
    tokens = _extract_tokens_from_outcomes(container)
    token_yes = tokens["token_yes"] or container.get("yesTokenId") or container.get("tokenYes") or container.get("token_yes")
    token_no = tokens["token_no"] or container.get("noTokenId") or container.get("tokenNo") or container.get("token_no")
    return {"token_yes": token_yes, "token_no": token_no}


def _iter_market_candidates(payload: Dict) -> list[Dict]:
    candidates = []
    if isinstance(payload, dict):
        candidates.append(payload)
        for key in ("market", "markets"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))
        for key in ("events",):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        candidates.extend(_iter_market_candidates(item))
    deduped = []
    seen = set()
    for item in candidates:
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _normalize_market_bundle(payload: Dict, *, detection_source: str, now=None) -> Optional[Dict]:
    candidates = _iter_market_candidates(payload)
    normalized_candidates = []
    for candidate in candidates:
        start = _parse_dt(candidate.get("startDate") or candidate.get("start") or payload.get("startDate") or payload.get("start"))
        end = _parse_dt(candidate.get("endDate") or candidate.get("end") or payload.get("endDate") or payload.get("end"))
        tokens = _extract_tokens(candidate)
        if not tokens["token_yes"] or not tokens["token_no"]:
            merged_tokens = _extract_tokens(payload)
            tokens["token_yes"] = tokens["token_yes"] or merged_tokens["token_yes"]
            tokens["token_no"] = tokens["token_no"] or merged_tokens["token_no"]
        market_id = (
            candidate.get("id")
            or candidate.get("marketId")
            or candidate.get("market_id")
            or candidate.get("questionID")
            or payload.get("marketId")
            or payload.get("market_id")
        )
        condition_id = (
            candidate.get("conditionId")
            or candidate.get("condition_id")
            or candidate.get("condition")
            or payload.get("conditionId")
            or payload.get("condition_id")
        )
        slug = candidate.get("slug") or payload.get("slug") or payload.get("eventSlug")
        title = candidate.get("title") or candidate.get("question") or candidate.get("name") or payload.get("title") or payload.get("name")
        start, end = _normalize_btc_hourly_window(slug, title, start, end)
        status = _normalize_market_status(candidate, start, end, now=now)
        market = {
            "market_id": market_id,
            "condition_id": condition_id,
            "slug": slug,
            "title": title,
            "token_yes": tokens["token_yes"],
            "token_no": tokens["token_no"],
            "startDate": start,
            "endDate": end,
            "status": status,
            "detection_source": detection_source,
            "raw": payload,
        }
        if any(market.get(field) is not None for field in ("market_id", "slug", "token_yes", "token_no", "startDate", "endDate")):
            score = sum(
                market.get(field) not in (None, "")
                for field in ("market_id", "condition_id", "token_yes", "token_no", "startDate", "endDate", "status")
            )
            normalized_candidates.append((score, market))
    if normalized_candidates:
        normalized_candidates.sort(key=lambda item: item[0], reverse=True)
        return normalized_candidates[0][1]
    return None


def _validate_market_bundle(market: Optional[Dict], *, now=None) -> str:
    if market is None:
        return "normalization_failed"
    if not market.get("market_id"):
        return "missing_market_id"
    if not market.get("token_yes") or not market.get("token_no"):
        return "missing_tokens"
    start = _parse_dt(market.get("startDate"))
    end = _parse_dt(market.get("endDate"))
    if start is None or end is None:
        return "missing_market_window"
    duration = (end - start).total_seconds()
    if abs(duration - 3600) > 300:
        return "invalid_duration"
    current = _parse_dt(now) or pd.Timestamp.now(tz="UTC")
    if not (start <= current < end):
        return "outside_market_window"
    if market.get("status") != "open":
        return "market_not_open"
    return "ok"


def locate_active_btc_hourly_market_with_debug(now=None) -> dict:
    current = _parse_dt(now) or pd.Timestamp.now(tz="UTC")
    candidate_slugs = candidate_btc_hourly_event_slugs(current)
    attempts = []
    if not candidate_slugs:
        return {"market": None, "reason": "no_candidate_slug_found", "candidate_slugs": [], "attempts": attempts}

    for slug in candidate_slugs:
        event = fetch_event_by_slug(slug)
        attempt = {"slug": slug, "event_found": event is not None, "market_found": False, "event_reason": None, "market_reason": None}
        if event is not None:
            normalized = _normalize_market_bundle(event, detection_source="gamma_event_slug", now=current)
            reason = _validate_market_bundle(normalized, now=current)
            attempt["event_reason"] = reason
            if reason == "ok":
                attempts.append(attempt)
                return {"market": normalized, "reason": "ok", "candidate_slugs": candidate_slugs, "attempts": attempts}
        else:
            attempt["event_reason"] = "event_fetch_failed"

        market = fetch_market_by_slug(slug)
        attempt["market_found"] = market is not None
        if market is not None:
            normalized = _normalize_market_bundle(market, detection_source="gamma_market_slug", now=current)
            reason = _validate_market_bundle(normalized, now=current)
            attempt["market_reason"] = reason
            attempts.append(attempt)
            if reason == "ok":
                return {"market": normalized, "reason": "ok", "candidate_slugs": candidate_slugs, "attempts": attempts}
        else:
            attempt["market_reason"] = "market_fetch_failed"
            attempts.append(attempt)

    preferred_reasons = []
    fallback_reasons = []
    for attempt in attempts:
        for reason in (attempt.get("event_reason"), attempt.get("market_reason")):
            if not reason or reason == "ok":
                continue
            if reason in ("event_fetch_failed", "market_fetch_failed"):
                fallback_reasons.append(reason)
            else:
                preferred_reasons.append(reason)
    reason = preferred_reasons[-1] if preferred_reasons else (fallback_reasons[-1] if fallback_reasons else "no_candidate_slug_found")
    return {"market": None, "reason": reason, "candidate_slugs": candidate_slugs, "attempts": attempts}


def locate_active_btc_hourly_market(now=None) -> Optional[Dict]:
    result = locate_active_btc_hourly_market_with_debug(now=now)
    return result.get("market")


def discover_current_hour_event(series_id: str) -> Optional[Dict]:
    if _normalize_series_id(series_id) in _BTC_SERIES_IDS:
        market = locate_active_btc_hourly_market(now=pd.Timestamp.now(tz="UTC"))
        return dict(market) if market is not None else None
    now = pd.Timestamp.now(tz="UTC")
    for item in _fetch_generic_events(series_id):
        start = _parse_dt(item.get("startDate") or item.get("start"))
        end = _parse_dt(item.get("endDate") or item.get("end"))
        if start is None or end is None:
            continue
        duration = (end - start).total_seconds()
        if abs(duration - 3600) > 300:
            continue
        if start <= now < end:
            result = dict(item)
            result["startDate"] = start
            result["endDate"] = end
            return result
    return None


def _detect_generic_active_hourly_market(series_id: str, now=None) -> Optional[Dict]:
    event = discover_current_hour_event(series_id)
    if event is None:
        return None
    start = _parse_dt(event.get("startDate"))
    end = _parse_dt(event.get("endDate"))
    status = _normalize_market_status(event, start, end, now=now)
    tokens = _extract_tokens(event)
    market = {
        "market_id": event.get("id") or event.get("marketId") or event.get("eventId"),
        "condition_id": event.get("conditionId") or event.get("condition_id") or event.get("condition"),
        "slug": event.get("slug") or event.get("series") or event.get("title"),
        "title": event.get("title") or event.get("name"),
        "token_yes": tokens["token_yes"],
        "token_no": tokens["token_no"],
        "startDate": start,
        "endDate": end,
        "status": status,
        "detection_source": "generic_series_discovery",
        "raw": event,
    }
    if _validate_market_bundle(market, now=now) != "ok":
        return None
    return market


def detect_active_hourly_market_with_debug(series_id: str, now=None) -> dict:
    normalized = _normalize_series_id(series_id)
    if not normalized:
        return {"market": None, "reason": "missing_series_id", "candidate_slugs": [], "attempts": []}
    if normalized in _BTC_SERIES_IDS:
        return locate_active_btc_hourly_market_with_debug(now=now)
    market = _detect_generic_active_hourly_market(series_id, now=now)
    if market is not None:
        return {
            "market": market,
            "reason": "ok",
            "candidate_slugs": [],
            "attempts": [{"series_id": series_id, "reason": "ok", "source": "generic_series_discovery"}],
        }
    return {
        "market": None,
        "reason": "unsupported_series_or_no_active_hourly_market",
        "candidate_slugs": [],
        "attempts": [{"series_id": series_id, "reason": "unsupported_series_or_no_active_hourly_market", "source": "generic_series_discovery"}],
    }


def detect_active_hourly_market(series_id: str, now=None) -> Optional[Dict]:
    return detect_active_hourly_market_with_debug(series_id, now=now).get("market")


def discover_current_hour_market(series_id: str) -> Optional[Dict]:
    market = detect_active_hourly_market(series_id)
    if market is None:
        return None
    return market


def _parse_book_level(level) -> Optional[tuple]:
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        return float(level[0]), float(level[1])
    if isinstance(level, dict):
        price = level.get("price") or level.get("p")
        size = level.get("size") or level.get("quantity") or level.get("qty") or level.get("q")
        if price is None or size is None:
            return None
        return float(price), float(size)
    return None


def _extract_orderbook_payload(raw):
    if not isinstance(raw, dict):
        return raw
    for key in ("book", "data", "orderbook"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _extract_numeric_field(raw, *keys) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except Exception:
            return None
    if isinstance(raw, dict):
        for key in keys:
            value = raw.get(key)
            if value is not None:
                parsed = _extract_numeric_field(value)
                if parsed is not None:
                    return parsed
        for key in ("data", "result", "price", "spread", "value"):
            value = raw.get(key)
            if value is not None:
                parsed = _extract_numeric_field(value, *keys)
                if parsed is not None:
                    return parsed
    return None


def _build_quote_snapshot_from_book(token_id: str, raw: Optional[Dict], *, source: str, fetch_failed: bool = False, fetched_at: Optional[float] = None, error: Optional[str] = None) -> Dict:
    fetched_at = fetched_at if fetched_at is not None else time.time()
    best_bid = best_ask = bid_size = ask_size = mid = spread = None
    is_empty = True
    is_crossed = False
    parse_error = None
    book = _extract_orderbook_payload(raw)
    if isinstance(book, dict):
        try:
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid_level = _parse_book_level(bids[0]) if bids else None
            ask_level = _parse_book_level(asks[0]) if asks else None
            if bid_level is not None:
                best_bid, bid_size = bid_level
            if ask_level is not None:
                best_ask, ask_size = ask_level
            is_empty = best_bid is None and best_ask is None
            if best_bid is not None and best_ask is not None:
                spread = float(best_ask - best_bid)
                mid = float((best_bid + best_ask) / 2.0)
                is_crossed = best_bid > best_ask
            elif best_bid is not None and not QUOTE_REQUIRE_BOTH_SIDES:
                mid = float(best_bid)
            elif best_ask is not None and not QUOTE_REQUIRE_BOTH_SIDES:
                mid = float(best_ask)
        except Exception as exc:
            parse_error = str(exc)
            is_empty = True
    else:
        parse_error = "non-dict-orderbook"

    return {
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "is_crossed": bool(is_crossed),
        "is_empty": bool(is_empty),
        "source": source,
        "fetched_at": pd.Timestamp.fromtimestamp(fetched_at, tz=timezone.utc).isoformat(),
        "age_seconds": max(0.0, time.time() - fetched_at),
        "fetch_failed": bool(fetch_failed),
        "error": error or parse_error,
        "raw": raw,
    }


def _build_quote_snapshot(token_id: str, raw: Optional[Dict], *, source: str, fetch_failed: bool = False, fetched_at: Optional[float] = None, error: Optional[str] = None) -> Dict:
    fetched_at = fetched_at if fetched_at is not None else time.time()
    error_parts = []
    buy_price = None
    sell_price = None
    spread = None
    book_snapshot = None
    composite_raw = raw if isinstance(raw, dict) else {}

    if isinstance(composite_raw, dict) and any(key in composite_raw for key in ("buy_price", "sell_price", "spread", "book")):
        buy_price = _extract_numeric_field(composite_raw.get("buy_price"), "price")
        sell_price = _extract_numeric_field(composite_raw.get("sell_price"), "price")
        spread = _extract_numeric_field(composite_raw.get("spread"), "spread")
        if composite_raw.get("book") is not None:
            book_snapshot = _build_quote_snapshot_from_book(token_id, composite_raw.get("book"), source="clob_orderbook", fetched_at=fetched_at)
    elif raw is not None:
        book_snapshot = _build_quote_snapshot_from_book(token_id, raw, source="clob_orderbook", fetched_at=fetched_at)

    # Public CLOB semantics: side=SELL is the top executable bid, side=BUY is the top executable ask.
    best_bid = sell_price
    best_ask = buy_price
    if best_bid is not None and best_ask is not None and best_bid > best_ask:
        best_bid, best_ask = best_ask, best_bid
        error_parts.append("price_side_swap_applied")

    bid_size = book_snapshot.get("bid_size") if book_snapshot else None
    ask_size = book_snapshot.get("ask_size") if book_snapshot else None
    is_crossed = False
    is_empty = best_bid is None and best_ask is None

    if best_bid is not None and best_ask is not None:
        computed_spread = float(best_ask - best_bid)
        if spread is None:
            spread = computed_spread
        else:
            spread = max(0.0, float(spread))
        mid = float((best_bid + best_ask) / 2.0)
        is_crossed = best_bid > best_ask
    elif book_snapshot is not None and not book_snapshot.get("is_empty"):
        best_bid = book_snapshot.get("best_bid")
        best_ask = book_snapshot.get("best_ask")
        bid_size = book_snapshot.get("bid_size")
        ask_size = book_snapshot.get("ask_size")
        mid = book_snapshot.get("mid")
        spread = book_snapshot.get("spread")
        is_crossed = bool(book_snapshot.get("is_crossed"))
        is_empty = bool(book_snapshot.get("is_empty"))
        source = "clob_orderbook_fallback"
    else:
        mid = None
        if best_bid is not None and not QUOTE_REQUIRE_BOTH_SIDES:
            mid = float(best_bid)
            is_empty = False
        elif best_ask is not None and not QUOTE_REQUIRE_BOTH_SIDES:
            mid = float(best_ask)
            is_empty = False

    if fetch_failed and error:
        error_parts.append(str(error))
    elif error:
        error_parts.append(str(error))
    if book_snapshot is not None and book_snapshot.get("error") and source != "clob_orderbook_fallback":
        error_parts.append(f"book:{book_snapshot['error']}")

    return {
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "is_crossed": bool(is_crossed),
        "is_empty": bool(is_empty),
        "source": source,
        "fetched_at": pd.Timestamp.fromtimestamp(fetched_at, tz=timezone.utc).isoformat(),
        "age_seconds": max(0.0, time.time() - fetched_at),
        "fetch_failed": bool(fetch_failed),
        "error": "; ".join(part for part in error_parts if part) or None,
        "raw": composite_raw if composite_raw else raw,
    }


def get_quote_snapshot(token_id: str, force_refresh: bool = False) -> Dict:
    now = time.time()
    cached = _QUOTE_CACHE.get(token_id)
    if cached and not force_refresh and (now - cached["fetched_at_epoch"]) <= QUOTE_CACHE_TTL_SEC:
        snap = dict(cached["snapshot"])
        snap["age_seconds"] = max(0.0, now - cached["fetched_at_epoch"])
        return snap
    raw = {}
    errors = []

    def _fetch(path: str, *, params: Optional[dict] = None):
        response = requests.get(f"{POLY_CLOB_BASE}{path}", params=params, timeout=5)
        response.raise_for_status()
        return response.json()

    try:
        raw["buy_price"] = _fetch("/price", params={"token_id": token_id, "side": "BUY"})
    except Exception as exc:
        errors.append(f"buy_price:{exc}")
    try:
        raw["sell_price"] = _fetch("/price", params={"token_id": token_id, "side": "SELL"})
    except Exception as exc:
        errors.append(f"sell_price:{exc}")
    try:
        raw["spread"] = _fetch("/spread", params={"token_id": token_id})
    except Exception as exc:
        errors.append(f"spread:{exc}")

    needs_book = QUOTE_MIN_DEPTH > 0
    if "buy_price" not in raw or "sell_price" not in raw:
        needs_book = True

    if needs_book:
        try:
            raw["book"] = _fetch("/book", params={"token_id": token_id})
        except Exception as exc:
            errors.append(f"book:{exc}")

    fetch_failed = "buy_price" not in raw and "sell_price" not in raw and "book" not in raw
    source = "clob_price" if "buy_price" in raw or "sell_price" in raw else "clob_orderbook"
    snapshot = _build_quote_snapshot(
        token_id,
        raw if raw else None,
        source=source,
        fetch_failed=fetch_failed,
        fetched_at=now,
        error="; ".join(errors) if errors else None,
    )
    _QUOTE_CACHE[token_id] = {"snapshot": snapshot, "fetched_at_epoch": now}
    return dict(snapshot)


def classify_quote_snapshot(snapshot: Dict) -> Dict:
    if snapshot.get("fetch_failed"):
        return {"tradable": False, "reason": "quote_fetch_failed"}
    if snapshot.get("age_seconds", 0) > QUOTE_MAX_AGE_SEC:
        return {"tradable": False, "reason": "quote_stale"}
    if snapshot.get("is_empty"):
        return {"tradable": False, "reason": "quote_empty"}
    if snapshot.get("is_crossed"):
        return {"tradable": False, "reason": "quote_crossed"}
    if QUOTE_REQUIRE_BOTH_SIDES and (snapshot.get("best_bid") is None or snapshot.get("best_ask") is None):
        return {"tradable": False, "reason": "quote_empty"}
    if snapshot.get("spread") is not None and snapshot["spread"] > QUOTE_MAX_SPREAD:
        return {"tradable": False, "reason": "quote_too_wide"}
    if QUOTE_MIN_DEPTH > 0:
        if snapshot.get("bid_size") is not None and snapshot["bid_size"] < QUOTE_MIN_DEPTH:
            return {"tradable": False, "reason": "quote_insufficient_depth"}
        if snapshot.get("ask_size") is not None and snapshot["ask_size"] < QUOTE_MIN_DEPTH:
            return {"tradable": False, "reason": "quote_insufficient_depth"}
    if snapshot.get("mid") is None or not (0 < float(snapshot["mid"]) < 1):
        return {"tradable": False, "reason": "quote_empty"}
    return {"tradable": True, "reason": None}


def get_orderbook(token_id: str) -> Dict:
    snapshot = get_quote_snapshot(token_id)
    raw = snapshot.get("raw") or {}
    if isinstance(raw, dict) and raw.get("book") is not None:
        return raw.get("book") or {}
    return _extract_orderbook_payload(raw) if raw else {}
