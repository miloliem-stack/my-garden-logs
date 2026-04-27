import os
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import market_router, run_bot, storage


def setup_function(_fn):
    try:
        storage.get_db_path().unlink()
    except Exception:
        pass
    storage.ensure_db()


def _market(market_id, token_yes, token_no, start, end):
    return {
        'market_id': market_id,
        'condition_id': f'COND-{market_id}',
        'slug': f'slug-{market_id}',
        'title': f'title-{market_id}',
        'token_yes': token_yes,
        'token_no': token_no,
        'startDate': start,
        'endDate': end,
        'status': 'open',
        'detection_source': 'test',
        'raw': {'id': market_id},
    }


def _quote(token_id, mid):
    return {
        'token_id': token_id,
        'best_bid': mid - 0.01,
        'best_ask': mid + 0.01,
        'mid': mid,
        'spread': 0.02,
        'bid_size': 10.0,
        'ask_size': 10.0,
        'is_crossed': False,
        'is_empty': False,
        'source': 'test',
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'age_seconds': 0.0,
        'fetch_failed': False,
        'error': None,
        'raw': {'bids': [[mid - 0.01, 10]], 'asks': [[mid + 0.01, 10]]},
    }


def test_startup_active_market_resolution_persists_runtime_state(monkeypatch):
    now = datetime.now(timezone.utc)
    market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: market)
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: 64000.0)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4 if token_id == 'YES1' else 0.6))

    bundle = market_router.resolve_active_market_bundle('SERIES1', now=now)

    runtime_state = storage.get_series_runtime_state('SERIES1')
    token_state = storage.get_market_tokens('M1')
    assert bundle['market_id'] == 'M1'
    assert bundle['switched'] is False
    assert runtime_state['active_market_id'] == 'M1'
    assert runtime_state['active_token_yes'] == 'YES1'
    assert runtime_state['active_token_no'] == 'NO1'
    assert runtime_state['strike_price'] == 64000.0
    assert runtime_state['status'] == 'active'
    assert token_state['token_yes'] == 'YES1'
    assert token_state['token_no'] == 'NO1'


def test_router_uses_btc_hourly_locator_for_bitcoin_up_or_down(monkeypatch):
    now = datetime.now(timezone.utc)
    market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    monkeypatch.setattr(market_router, 'detect_active_hourly_market_with_debug', lambda series_id, now=None: {
        'market': market,
        'reason': 'ok',
        'candidate_slugs': ['bitcoin-up-or-down-march-22-2026-5pm-et'],
        'attempts': [{'slug': 'bitcoin-up-or-down-march-22-2026-5pm-et', 'event_reason': 'ok'}],
    })
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: 64000.0)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4 if token_id == 'YES1' else 0.6))

    bundle = market_router.resolve_active_market_bundle('bitcoin-up-or-down', now=now)

    assert bundle['market_id'] == 'M1'
    assert bundle['routing_reason'] == 'ok'
    assert bundle['routing_candidate_slugs'] == ['bitcoin-up-or-down-march-22-2026-5pm-et']
    assert bundle['routing_attempt_count'] == 1


def test_same_hour_repeated_resolution_does_not_create_false_switch(monkeypatch):
    now = datetime.now(timezone.utc)
    market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: market)
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: 64000.0)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4))

    first = market_router.resolve_active_market_bundle('SERIES1', now=now)
    second = market_router.resolve_active_market_bundle('SERIES1', now=now + timedelta(minutes=5))

    assert first['switched'] is False
    assert second['switched'] is False
    assert second['strike_price'] == 64000.0
    assert storage.get_series_runtime_state('SERIES1')['previous_market_id'] is None


def test_hour_switch_updates_active_market_tokens_and_strike(monkeypatch):
    now = datetime.now(timezone.utc)
    first_market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    second_market = _market('M2', 'YES2', 'NO2', now + timedelta(hours=1), now + timedelta(hours=2))
    detected = iter([first_market, second_market])
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: next(detected))
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.45))
    monkeypatch.setattr(
        market_router,
        'get_1h_open_for_timestamp',
        lambda ts: 64000.0 if pd.to_datetime(ts, utc=True).hour == pd.to_datetime(now, utc=True).hour else 64500.0,
    )

    first = market_router.resolve_active_market_bundle('SERIES1', now=now)
    second = market_router.resolve_active_market_bundle('SERIES1', now=now + timedelta(hours=1, minutes=1))

    runtime_state = storage.get_series_runtime_state('SERIES1')
    assert first['market_id'] == 'M1'
    assert second['switched'] is True
    assert second['previous_market_id'] == 'M1'
    assert second['market_id'] == 'M2'
    assert second['token_yes'] == 'YES2'
    assert second['token_no'] == 'NO2'
    assert runtime_state['active_market_id'] == 'M2'
    assert runtime_state['previous_market_id'] == 'M1'
    assert runtime_state['strike_price'] == 64500.0


def test_old_hour_inventory_remains_associated_with_old_market_on_switch(monkeypatch):
    now = datetime.now(timezone.utc)
    first_market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    second_market = _market('M2', 'YES2', 'NO2', now + timedelta(hours=1), now + timedelta(hours=2))
    detected = iter([first_market, second_market])
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: next(detected))
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: 64000.0)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4))

    market_router.resolve_active_market_bundle('SERIES1', now=now)
    storage.create_open_lot('M1', 'YES1', 'YES', 3.0, 0.4, now.isoformat())
    market_router.resolve_active_market_bundle('SERIES1', now=now + timedelta(hours=1, minutes=1))

    snapshot = {item['market_id']: item for item in storage.get_position_snapshot()}
    assert snapshot['M1']['tradable_open_inventory']['YES'] == 3.0
    assert storage.get_series_runtime_state('SERIES1')['active_market_id'] == 'M2'


def test_runner_series_mode_no_longer_uses_stale_startup_tokens_after_switch(monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    bundle = {
        'series_id': 'SERIES1',
        'market_id': 'M2',
        'token_yes': 'YES2',
        'token_no': 'NO2',
        'start_time': now.isoformat(),
        'end_time': (now + pd.Timedelta(hours=1)).isoformat(),
        'status': 'open',
        'strike_price': 64500.0,
        'strike_source': 'binance_1h_open',
        'runtime_status': 'active',
        'switched': True,
        'epoch_key': 'M2:epoch',
        'yes_quote': _quote('YES2', 0.42),
        'no_quote': _quote('NO2', 0.58),
    }
    monkeypatch.setattr(run_bot, 'resolve_active_market_bundle_with_debug', lambda series_id, now=None: {
        'bundle': bundle,
        'routing_reason': bundle.get('routing_reason'),
        'candidate_slugs': bundle.get('routing_candidate_slugs') or [],
        'attempt_count': bundle.get('routing_attempt_count') or 0,
        'detection_source': bundle.get('detection_source'),
        'attempts': bundle.get('routing_attempts') or [],
    })
    args = Namespace(series='SERIES1', token_yes='STALE_YES', token_no='STALE_NO', market_id='STALE_MARKET')

    resolved = run_bot.resolve_live_market_context(args, now=now)

    assert resolved['market_id'] == 'M2'
    assert resolved['token_yes'] == 'YES2'
    assert resolved['token_no'] == 'NO2'


def test_temporary_detection_failure_does_not_trade_stale_tokens(monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    monkeypatch.setattr(run_bot, 'resolve_active_market_bundle_with_debug', lambda series_id, now=None: {
        'bundle': None,
        'routing_reason': 'unsupported_series_or_no_active_hourly_market',
        'candidate_slugs': [],
        'attempt_count': 1,
        'detection_source': None,
        'attempts': [{'series_id': series_id, 'reason': 'unsupported_series_or_no_active_hourly_market'}],
    })
    args = Namespace(series='SERIES1', token_yes='STALE_YES', token_no='STALE_NO', market_id='STALE_MARKET')

    resolved = run_bot.resolve_live_market_context(args, now=now)
    ctx = run_bot.build_trade_context(resolved['market_meta'], resolved['yes_quote'], resolved['no_quote'], now=now, routing_bundle=resolved['routing_bundle'])
    ctx['routing_debug'] = resolved['routing_debug']
    ok, reason = run_bot.can_trade_context(ctx)
    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, ok, reason)

    assert resolved['market_id'] is None
    assert resolved['token_yes'] is None
    assert resolved['token_no'] is None
    assert resolved['routing_debug']['series_arg'] == 'SERIES1'
    assert resolved['routing_debug']['routing_bundle_present'] is False
    assert resolved['routing_debug']['routing_reason'] == 'unsupported_series_or_no_active_hourly_market'
    assert resolved['routing_debug']['missing_market_id'] is True
    assert resolved['routing_debug']['missing_token_yes'] is True
    assert resolved['routing_debug']['missing_token_no'] is True
    assert ok is False
    assert reason == 'market_detection_failed'
    assert heartbeat['routing_series_arg'] == 'SERIES1'
    assert heartbeat['routing_bundle_present'] is False
    assert heartbeat['routing_reason'] == 'unsupported_series_or_no_active_hourly_market'
    assert heartbeat['routing_missing_market_id'] is True
    assert heartbeat['routing_missing_token_yes'] is True
    assert heartbeat['routing_missing_token_no'] is True


def test_manual_mode_still_works_unchanged(monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    monkeypatch.setattr(run_bot, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.45 if token_id == 'YES1' else 0.55))
    args = Namespace(series=None, token_yes='YES1', token_no='NO1', market_id='M1')

    resolved = run_bot.resolve_live_market_context(args, now=now)

    assert resolved['routing_bundle'] is None
    assert resolved['routing_debug'] == {}
    assert resolved['market_id'] == 'M1'
    assert resolved['token_yes'] == 'YES1'
    assert resolved['token_no'] == 'NO1'


def test_missing_strike_price_blocks_trading_cleanly(monkeypatch):
    now = datetime.now(timezone.utc)
    market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: market)
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: None)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4))

    bundle = market_router.resolve_active_market_bundle('SERIES1', now=now)
    ctx = run_bot.build_trade_context({
        'market_id': bundle['market_id'],
        'token_yes': bundle['token_yes'],
        'token_no': bundle['token_no'],
        'status': bundle['status'],
        'startDate': pd.to_datetime(bundle['start_time'], utc=True),
        'endDate': pd.to_datetime(bundle['end_time'], utc=True),
    }, bundle['yes_quote'], bundle['no_quote'], now=now, routing_bundle=bundle)
    ok, reason = run_bot.can_trade_context(ctx)

    assert bundle['runtime_status'] == 'missing_strike'
    assert bundle['strike_price'] is None
    assert ok is False
    assert reason == 'missing_strike_price'


def test_heartbeat_and_switch_diagnostics_include_routing_fields(monkeypatch):
    now = datetime.now(timezone.utc)
    market = _market('M1', 'YES1', 'NO1', now, now + timedelta(hours=1))
    monkeypatch.setattr(market_router, 'detect_active_hourly_market', lambda series_id: market)
    monkeypatch.setattr(market_router, 'get_1h_open_for_timestamp', lambda ts: 64000.0)
    monkeypatch.setattr(market_router, 'get_quote_snapshot', lambda token_id: _quote(token_id, 0.4 if token_id == 'YES1' else 0.6))

    bundle = market_router.resolve_active_market_bundle('SERIES1', now=now)
    ctx = run_bot.build_trade_context(
        {
            'market_id': bundle['market_id'],
            'token_yes': bundle['token_yes'],
            'token_no': bundle['token_no'],
            'status': bundle['status'],
            'startDate': pd.to_datetime(bundle['start_time'], utc=True),
            'endDate': pd.to_datetime(bundle['end_time'], utc=True),
        },
        bundle['yes_quote'],
        bundle['no_quote'],
        now=now,
        routing_bundle=bundle,
    )
    ctx['routing_debug'] = run_bot.build_routing_debug('SERIES1', bundle)
    ok, reason = run_bot.can_trade_context(ctx)
    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, ok, reason)
    switch_event = run_bot.build_market_switch_event({
        **bundle,
        'switched': True,
        'previous_market_id': 'M0',
        'previous_token_yes': 'YES0',
        'previous_token_no': 'NO0',
        'previous_strike_price': 63500.0,
    })

    assert heartbeat['series_id'] == 'SERIES1'
    assert heartbeat['active_market_id'] == 'M1'
    assert heartbeat['active_token_yes'] == 'YES1'
    assert heartbeat['active_token_no'] == 'NO1'
    assert heartbeat['strike_price'] == 64000.0
    assert heartbeat['routing_series_arg'] == 'SERIES1'
    assert heartbeat['routing_bundle_present'] is True
    assert heartbeat['routing_runtime_status'] == bundle['runtime_status']
    assert heartbeat['routing_strike_price'] == 64000.0
    assert heartbeat['routing_epoch_key'] == bundle['epoch_key']
    assert heartbeat['routing_detection_source'] == bundle['detection_source']
    assert heartbeat['routing_reason'] == 'ok'
    assert heartbeat['routing_attempt_count'] == bundle['routing_attempt_count']
    assert 'new_market_id' in switch_event
    assert switch_event['new_market_id'] == 'M1'


def test_runner_auto_routes_btc_hourly_when_series_is_omitted(monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    bundle = {
        'series_id': 'bitcoin-up-or-down',
        'market_id': 'MBTC',
        'token_yes': 'YBTC',
        'token_no': 'NBTC',
        'start_time': now.isoformat(),
        'end_time': (now + pd.Timedelta(hours=1)).isoformat(),
        'status': 'open',
        'strike_price': 65000.0,
        'strike_source': 'binance_1h_open',
        'runtime_status': 'active',
        'epoch_key': 'MBTC:epoch',
        'routing_reason': 'ok',
        'routing_candidate_slugs': ['bitcoin-up-or-down-march-22-2026-6pm-et'],
        'routing_attempt_count': 1,
        'routing_attempts': [{'slug': 'bitcoin-up-or-down-march-22-2026-6pm-et', 'event_reason': 'ok'}],
        'yes_quote': _quote('YBTC', 0.47),
        'no_quote': _quote('NBTC', 0.53),
    }
    monkeypatch.setattr(run_bot, 'resolve_active_market_bundle_with_debug', lambda series_id, now=None: {
        'bundle': bundle if series_id == 'bitcoin-up-or-down' else None,
        'routing_reason': 'ok' if series_id == 'bitcoin-up-or-down' else 'market_detection_failed',
        'candidate_slugs': bundle.get('routing_candidate_slugs') if series_id == 'bitcoin-up-or-down' else [],
        'attempt_count': bundle.get('routing_attempt_count') if series_id == 'bitcoin-up-or-down' else 0,
        'detection_source': bundle.get('detection_source'),
        'attempts': bundle.get('routing_attempts') if series_id == 'bitcoin-up-or-down' else [],
    })
    args = Namespace(series=None, token_yes=None, token_no=None, market_id=None)

    resolved = run_bot.resolve_live_market_context(args, now=now)

    assert resolved['market_id'] == 'MBTC'
    assert resolved['token_yes'] == 'YBTC'
    assert resolved['token_no'] == 'NBTC'
    assert resolved['routing_debug']['series_arg'] == 'bitcoin-up-or-down'
    assert resolved['routing_debug']['routing_reason'] == 'ok'
    assert resolved['routing_debug']['candidate_slugs'] == ['bitcoin-up-or-down-march-22-2026-6pm-et']
    assert resolved['routing_debug']['attempt_count'] == 1
