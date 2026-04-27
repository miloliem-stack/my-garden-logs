from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from . import storage
from .binance_feed import get_1h_open_for_timestamp
from .polymarket_feed import (
    classify_quote_snapshot,
    detect_active_hourly_market,
    detect_active_hourly_market_with_debug,
    get_quote_snapshot,
)


def _iso_or_none(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def build_epoch_key(market_id: Optional[str], start_time, end_time) -> Optional[str]:
    if not market_id or start_time is None or end_time is None:
        return None
    return f'{market_id}:{_iso_or_none(start_time)}:{_iso_or_none(end_time)}'


def _parse_ts(value):
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _compute_strike(start_time, now_ts: str) -> tuple[Optional[float], str, Optional[str]]:
    start_ts = _parse_ts(start_time)
    if start_ts is None:
        return None, 'missing_start_time', None
    strike_price = get_1h_open_for_timestamp(start_ts)
    if strike_price is None:
        return None, 'hour_open_lookup_failed', None
    return float(strike_price), 'binance_1h_open', now_ts


def _resolve_detection_debug(series_id: str, now=None) -> dict:
    try:
        result = detect_active_hourly_market_with_debug(series_id, now=now)
    except TypeError:
        market = detect_active_hourly_market(series_id)
        return {
            'market': market,
            'reason': 'ok' if market is not None else 'market_detection_failed',
            'candidate_slugs': [],
            'attempts': [],
        }
    if result.get('market') is None:
        try:
            market = detect_active_hourly_market(series_id, now=now)
        except TypeError:
            market = detect_active_hourly_market(series_id)
        if market is not None:
            return {
                'market': market,
                'reason': 'ok',
                'candidate_slugs': result.get('candidate_slugs') or [],
                'attempts': result.get('attempts') or [],
            }
    return result


def resolve_active_market_bundle_with_debug(series_id: str, now=None) -> dict:
    if not series_id:
        return {
            'bundle': None,
            'routing_reason': 'missing_series_id',
            'candidate_slugs': [],
            'attempt_count': 0,
            'detection_source': None,
            'attempts': [],
        }
    now_ts = _iso_or_none(now or datetime.now(timezone.utc))
    detection_debug = _resolve_detection_debug(series_id, now=now)
    detected = detection_debug.get('market')
    if detected is None:
        attempts = detection_debug.get('attempts') or []
        return {
            'bundle': None,
            'routing_reason': detection_debug.get('reason') or 'market_detection_failed',
            'candidate_slugs': detection_debug.get('candidate_slugs') or [],
            'attempt_count': len(attempts),
            'detection_source': None,
            'attempts': attempts,
        }

    start_time = detected.get('startDate')
    end_time = detected.get('endDate')
    market_id = detected.get('market_id')
    token_yes = detected.get('token_yes')
    token_no = detected.get('token_no')
    condition_id = detected.get('condition_id')

    storage.upsert_market(
        market_id=market_id,
        condition_id=condition_id,
        slug=detected.get('slug') or detected.get('title'),
        title=detected.get('title'),
        start_time=_iso_or_none(start_time),
        end_time=_iso_or_none(end_time),
        status=detected.get('status'),
        last_checked_ts=now_ts,
    )
    storage.upsert_market_tokens(
        market_id=market_id,
        condition_id=condition_id,
        token_yes=token_yes,
        token_no=token_no,
        start_time=_iso_or_none(start_time),
        end_time=_iso_or_none(end_time),
        discovered_ts=now_ts,
    )

    runtime_state = storage.get_series_runtime_state(series_id)
    current_epoch_key = build_epoch_key(market_id, start_time, end_time)
    previous_epoch_key = None
    previous_market_id = None
    previous_token_yes = None
    previous_token_no = None
    previous_strike_price = None
    switched = False
    if runtime_state:
        previous_market_id = runtime_state.get('active_market_id')
        previous_token_yes = runtime_state.get('active_token_yes')
        previous_token_no = runtime_state.get('active_token_no')
        previous_strike_price = runtime_state.get('strike_price')
        previous_epoch_key = build_epoch_key(
            runtime_state.get('active_market_id'),
            runtime_state.get('active_start_time'),
            runtime_state.get('active_end_time'),
        )
        switched = previous_epoch_key is not None and current_epoch_key != previous_epoch_key

    strike_price = runtime_state.get('strike_price') if runtime_state and not switched else None
    strike_source = runtime_state.get('strike_source') if runtime_state and not switched else None
    strike_fixed_ts = runtime_state.get('strike_fixed_ts') if runtime_state and not switched else None
    runtime_status = 'active'
    if strike_price is None:
        strike_price, strike_source, strike_fixed_ts = _compute_strike(start_time, now_ts)
        if strike_price is None:
            runtime_status = 'missing_strike'

    if switched:
        runtime_state = storage.mark_series_market_switch(
            series_id,
            active_market_id=market_id,
            active_token_yes=token_yes,
            active_token_no=token_no,
            active_start_time=_iso_or_none(start_time),
            active_end_time=_iso_or_none(end_time),
            strike_price=strike_price,
            strike_source=strike_source,
            strike_fixed_ts=strike_fixed_ts,
            switch_ts=now_ts,
            status=runtime_status,
            previous_market_id=previous_market_id,
        )
    else:
        runtime_state = storage.set_series_runtime_state(
            series_id,
            active_market_id=market_id,
            active_token_yes=token_yes,
            active_token_no=token_no,
            active_start_time=_iso_or_none(start_time),
            active_end_time=_iso_or_none(end_time),
            strike_price=strike_price,
            strike_source=strike_source,
            strike_fixed_ts=strike_fixed_ts,
            status=runtime_status,
            previous_market_id=runtime_state.get('previous_market_id') if runtime_state else None,
            last_switch_ts=runtime_state.get('last_switch_ts') if runtime_state else now_ts,
        )

    yes_quote = get_quote_snapshot(token_yes) if token_yes else {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None}
    no_quote = get_quote_snapshot(token_no) if token_no else {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None}

    bundle = {
        'series_id': series_id,
        'market_id': market_id,
        'previous_market_id': previous_market_id,
        'previous_token_yes': previous_token_yes,
        'previous_token_no': previous_token_no,
        'previous_strike_price': previous_strike_price,
        'switched': switched,
        'condition_id': condition_id,
        'token_yes': token_yes,
        'token_no': token_no,
        'start_time': _iso_or_none(start_time),
        'end_time': _iso_or_none(end_time),
        'status': detected.get('status'),
        'strike_price': runtime_state.get('strike_price'),
        'strike_source': runtime_state.get('strike_source'),
        'strike_fixed_ts': runtime_state.get('strike_fixed_ts'),
        'yes_quote': yes_quote,
        'no_quote': no_quote,
        'yes_quote_state': classify_quote_snapshot(yes_quote),
        'no_quote_state': classify_quote_snapshot(no_quote),
        'detection_source': detected.get('detection_source'),
        'epoch_key': current_epoch_key,
        'runtime_status': runtime_state.get('status'),
        'switch_ts': runtime_state.get('last_switch_ts'),
        'routing_reason': detection_debug.get('reason'),
        'routing_candidate_slugs': detection_debug.get('candidate_slugs') or [],
        'routing_attempt_count': len(detection_debug.get('attempts') or []),
        'routing_attempts': detection_debug.get('attempts') or [],
        'raw_market': detected,
    }
    return {
        'bundle': bundle,
        'routing_reason': bundle.get('routing_reason'),
        'candidate_slugs': bundle.get('routing_candidate_slugs'),
        'attempt_count': bundle.get('routing_attempt_count'),
        'detection_source': bundle.get('detection_source'),
        'attempts': bundle.get('routing_attempts'),
    }


def resolve_active_market_bundle(series_id: str, now=None) -> Optional[Dict]:
    return resolve_active_market_bundle_with_debug(series_id, now=now).get('bundle')
