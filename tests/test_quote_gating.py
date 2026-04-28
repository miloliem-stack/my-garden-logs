import os
import sys
import json
from datetime import datetime, timezone, timedelta
import asyncio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src import polymarket_feed, run_bot, storage, polymarket_client


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()
    polymarket_feed._QUOTE_CACHE.clear()


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'http {self.status_code}')

    def json(self):
        return self._payload


def test_normalized_quote_snapshot_on_good_orderbook(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        if url.endswith('/price') and params == {'token_id': 'TOK', 'side': 'BUY'}:
            return _FakeResponse({'price': '0.43'})
        if url.endswith('/price') and params == {'token_id': 'TOK', 'side': 'SELL'}:
            return _FakeResponse({'price': '0.41'})
        if url.endswith('/spread') and params == {'token_id': 'TOK'}:
            return _FakeResponse({'spread': '0.02'})
        raise AssertionError(f'unexpected call: {url} {params}')

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)
    snap = polymarket_feed.get_quote_snapshot('TOK', force_refresh=True)
    assert calls[0][0] == f'{polymarket_feed.POLY_CLOB_BASE}/price'
    assert calls[0][1] == {'token_id': 'TOK', 'side': 'BUY'}
    assert calls[1][0] == f'{polymarket_feed.POLY_CLOB_BASE}/price'
    assert calls[1][1] == {'token_id': 'TOK', 'side': 'SELL'}
    assert calls[2][0] == f'{polymarket_feed.POLY_CLOB_BASE}/spread'
    assert calls[2][1] == {'token_id': 'TOK'}
    assert snap['best_bid'] == 0.41
    assert snap['best_ask'] == 0.43
    assert snap['mid'] == 0.42
    assert abs(snap['spread'] - 0.02) < 1e-12
    assert snap['source'] == 'clob_price'
    assert polymarket_feed.classify_quote_snapshot(snap) == {'tradable': True, 'reason': None}


def test_quote_snapshot_returns_fetch_failed_on_404(monkeypatch):
    monkeypatch.setattr(polymarket_feed.requests, 'get', lambda *args, **kwargs: _FakeResponse({'error': 'not found'}, status_code=404))
    snap = polymarket_feed.get_quote_snapshot('MISSING', force_refresh=True)
    assert snap['fetch_failed'] is True
    assert polymarket_feed.classify_quote_snapshot(snap) == {'tradable': False, 'reason': 'quote_fetch_failed'}


def test_invalid_quote_classification_empty_crossed_and_malformed(monkeypatch):
    monkeypatch.setattr(polymarket_feed.requests, 'get', lambda *args, **kwargs: _FakeResponse({'bids': [], 'asks': []}) if args[0].endswith('/book') else _FakeResponse({'error': 'missing'}, status_code=404))
    empty = polymarket_feed.get_quote_snapshot('EMPTY', force_refresh=True)
    assert polymarket_feed.classify_quote_snapshot(empty)['reason'] == 'quote_empty'

    monkeypatch.setattr(polymarket_feed.requests, 'get', lambda *args, **kwargs: _FakeResponse({'bids': [[0.7, 1]], 'asks': [[0.6, 1]]}) if args[0].endswith('/book') else _FakeResponse({'error': 'missing'}, status_code=404))
    crossed = polymarket_feed.get_quote_snapshot('CROSS', force_refresh=True)
    assert polymarket_feed.classify_quote_snapshot(crossed)['reason'] == 'quote_crossed'

    monkeypatch.setattr(polymarket_feed.requests, 'get', lambda *args, **kwargs: _FakeResponse({'bids': ['bad'], 'asks': []}) if args[0].endswith('/book') else _FakeResponse({'error': 'missing'}, status_code=404))
    malformed = polymarket_feed.get_quote_snapshot('BAD', force_refresh=True)
    assert polymarket_feed.classify_quote_snapshot(malformed)['reason'] == 'quote_empty'


def test_quote_snapshot_falls_back_to_price_diff_when_spread_unavailable(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith('/price') and params == {'token_id': 'TOK2', 'side': 'BUY'}:
            return _FakeResponse({'price': '0.90'})
        if url.endswith('/price') and params == {'token_id': 'TOK2', 'side': 'SELL'}:
            return _FakeResponse({'price': '0.89'})
        if url.endswith('/spread') and params == {'token_id': 'TOK2'}:
            return _FakeResponse({'error': 'missing'}, status_code=404)
        raise AssertionError(f'unexpected call: {url} {params}')

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)
    snap = polymarket_feed.get_quote_snapshot('TOK2', force_refresh=True)

    assert snap['best_bid'] == 0.89
    assert snap['best_ask'] == 0.90
    assert abs(snap['mid'] - 0.895) < 1e-12
    assert abs(snap['spread'] - 0.01) < 1e-12
    assert polymarket_feed.classify_quote_snapshot(snap) == {'tradable': True, 'reason': None}


def test_quote_snapshot_falls_back_to_book_when_price_unavailable(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith('/price'):
            return _FakeResponse({'error': 'missing'}, status_code=404)
        if url.endswith('/spread'):
            return _FakeResponse({'error': 'missing'}, status_code=404)
        if url.endswith('/book') and params == {'token_id': 'TOK3'}:
            return _FakeResponse({'bids': [{'price': '0.41', 'size': '20'}], 'asks': [{'price': '0.43', 'size': '30'}]})
        raise AssertionError(f'unexpected call: {url} {params}')

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)
    snap = polymarket_feed.get_quote_snapshot('TOK3', force_refresh=True)

    assert snap['source'] == 'clob_orderbook_fallback'
    assert snap['best_bid'] == 0.41
    assert snap['best_ask'] == 0.43
    assert abs(snap['spread'] - 0.02) < 1e-12
    assert polymarket_feed.classify_quote_snapshot(snap) == {'tradable': True, 'reason': None}


def test_ghost_orderbook_does_not_override_valid_price_quote(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith('/price') and params == {'token_id': 'TOK4', 'side': 'BUY'}:
            return _FakeResponse({'price': '0.90'})
        if url.endswith('/price') and params == {'token_id': 'TOK4', 'side': 'SELL'}:
            return _FakeResponse({'price': '0.89'})
        if url.endswith('/spread') and params == {'token_id': 'TOK4'}:
            return _FakeResponse({'spread': '0.01'})
        if url.endswith('/book') and params == {'token_id': 'TOK4'}:
            return _FakeResponse({'bids': [{'price': '0.01', 'size': '20'}], 'asks': [{'price': '0.99', 'size': '30'}]})
        raise AssertionError(f'unexpected call: {url} {params}')

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)
    old_min_depth = polymarket_feed.QUOTE_MIN_DEPTH
    polymarket_feed.QUOTE_MIN_DEPTH = 1
    try:
        snap = polymarket_feed.get_quote_snapshot('TOK4', force_refresh=True)
    finally:
        polymarket_feed.QUOTE_MIN_DEPTH = old_min_depth

    assert snap['best_bid'] == 0.89
    assert snap['best_ask'] == 0.90
    assert abs(snap['spread'] - 0.01) < 1e-12
    assert snap['bid_size'] == 20.0
    assert snap['ask_size'] == 30.0
    assert polymarket_feed.classify_quote_snapshot(snap) == {'tradable': True, 'reason': None}


def test_quote_staleness_gating_blocks_strategy_execution():
    now = datetime.now(timezone.utc)
    market = {'market_id': 'M1', 'token_yes': 'Y', 'token_no': 'N', 'status': 'open', 'startDate': now - timedelta(minutes=5), 'endDate': now + timedelta(minutes=5)}
    storage.create_market('M1', status='open', start_time=(now - timedelta(minutes=5)).isoformat(), end_time=(now + timedelta(minutes=5)).isoformat())
    yes_quote = {'mid': 0.5, 'age_seconds': 999, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.01, 'best_bid': 0.49, 'best_ask': 0.51}
    no_quote = {'mid': 0.5, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.01, 'best_bid': 0.49, 'best_ask': 0.51}
    ctx = run_bot.build_trade_context(market, yes_quote, no_quote, now=now)
    ok, reason = run_bot.can_trade_context(ctx)
    assert ok is False
    assert reason == 'quote_stale'


def test_runner_does_not_trade_when_market_window_expired():
    now = datetime.now(timezone.utc)
    storage.create_market('M2', status='open', start_time=(now - timedelta(hours=2)).isoformat(), end_time=(now - timedelta(minutes=1)).isoformat())
    market = {'market_id': 'M2', 'token_yes': 'Y2', 'token_no': 'N2', 'status': 'open', 'startDate': now - timedelta(hours=2), 'endDate': now - timedelta(minutes=1)}
    quote = {'mid': 0.5, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.01, 'best_bid': 0.49, 'best_ask': 0.51}
    ctx = run_bot.build_trade_context(market, quote, quote, now=now)
    ok, reason = run_bot.can_trade_context(ctx)
    assert ok is False
    assert reason == 'market_not_open' or reason == 'market_window_expired'


def test_lifecycle_refresh_transitions_open_to_closed():
    now = datetime.now(timezone.utc)
    storage.create_market('M4', status='open', start_time=(now - timedelta(hours=1)).isoformat(), end_time=(now - timedelta(minutes=1)).isoformat())
    refreshed = storage.refresh_market_lifecycle('M4', checked_ts=now.isoformat())
    assert refreshed['status'] == 'closed'


def test_lifecycle_refresh_handles_resolved_market_and_redeemable_inventory():
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    storage.create_market('M5', status='open', start_time=(now - timedelta(hours=1)).isoformat(), end_time=(now - timedelta(minutes=1)).isoformat())
    storage.create_open_lot('M5', 'TY', 'YES', 4.0, 0.5, ts)
    storage.refresh_market_lifecycle('M5', source_data={'status': 'resolved', 'winning_outcome': 'YES'}, checked_ts=ts)
    snap = {item['market_id']: item for item in storage.get_position_snapshot()}['M5']
    assert snap['status'] == 'resolved'
    assert snap['resolved_redeemable_qty'] == 4.0


def test_lifecycle_refresh_promotes_expired_btc_hourly_market_to_resolved(monkeypatch):
    now = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc)
    start = datetime(2026, 3, 27, 11, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
    storage.create_market(
        'MBTC1',
        slug='bitcoin-up-or-down-march-27-2026-7am-et',
        title='Bitcoin Up or Down March 27, 2026 7am ET',
        status='closed',
        start_time=start.isoformat(),
        end_time=end.isoformat(),
    )
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86950.0)

    refreshed = storage.refresh_market_lifecycle('MBTC1', checked_ts=now.isoformat())

    assert refreshed['status'] == 'resolved'
    assert refreshed['winning_outcome'] == 'NO'


def test_lifecycle_refresh_leaves_closed_when_btc_resolution_source_unavailable(monkeypatch):
    now = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc)
    start = datetime(2026, 3, 27, 11, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
    storage.create_market(
        'MBTC2',
        slug='bitcoin-up-or-down-march-27-2026-7am-et',
        title='Bitcoin Up or Down March 27, 2026 7am ET',
        status='closed',
        start_time=start.isoformat(),
        end_time=end.isoformat(),
    )
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: None)

    refreshed = storage.refresh_market_lifecycle('MBTC2', checked_ts=now.isoformat())

    assert refreshed['status'] == 'closed'
    assert refreshed['winning_outcome'] is None
    assert refreshed['resolution_diagnostics'] == {
        'has_slug': True,
        'has_title': True,
        'has_start_time': True,
        'has_end_time': True,
        'binance_open_found': True,
        'binance_close_found': False,
    }


def test_lifecycle_refresh_keeps_already_resolved_market_stable(monkeypatch):
    now = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc)
    storage.create_market(
        'MBTC3',
        slug='bitcoin-up-or-down-march-27-2026-7am-et',
        title='Bitcoin Up or Down March 27, 2026 7am ET',
        status='resolved',
        start_time=datetime(2026, 3, 27, 11, 0, tzinfo=timezone.utc).isoformat(),
        end_time=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc).isoformat(),
    )
    storage.update_market_status('MBTC3', 'resolved', winning_outcome='YES')
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86000.0)

    refreshed = storage.refresh_market_lifecycle('MBTC3', checked_ts=now.isoformat())

    assert refreshed['status'] == 'resolved'
    assert refreshed['winning_outcome'] == 'YES'


def test_normalized_order_response_parsing_preserves_tx_hash_and_order_id():
    normalized = polymarket_client.normalize_client_response({'status': 'ok', 'txHash': '0xabc', 'orderId': 'OID1'})
    assert normalized['tx_hash'] == '0xabc'
    assert normalized['order_id'] == 'OID1'
    normalized_alt = polymarket_client.normalize_client_response({'state': 'filled', 'transactionHash': '0xdef', 'id': 'OID2'})
    assert normalized_alt['tx_hash'] == '0xdef'
    assert normalized_alt['order_id'] == 'OID2'


def test_diagnostic_heartbeat_returns_structured_reason_when_disabled():
    now = datetime.now(timezone.utc)
    market = {'market_id': 'M6', 'token_yes': 'Y6', 'token_no': 'N6', 'status': 'open', 'startDate': now - timedelta(minutes=5), 'endDate': now + timedelta(minutes=5)}
    storage.create_market('M6', status='open', start_time=(now - timedelta(minutes=5)).isoformat(), end_time=(now + timedelta(minutes=5)).isoformat())
    bad_quote = {'mid': None, 'age_seconds': 0, 'fetch_failed': True, 'is_empty': True, 'is_crossed': False, 'spread': None}
    ctx = run_bot.build_trade_context(market, bad_quote, bad_quote, now=now)
    ok, reason = run_bot.can_trade_context(ctx)
    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, ok, reason)
    assert heartbeat['trading_allowed'] is False
    assert heartbeat['disabled_reason'] == 'quote_fetch_failed'


def test_runner_invokes_stale_order_management_on_cadence(monkeypatch):
    calls = []
    monkeypatch.setattr(run_bot.execution, 'manage_stale_orders', lambda now_ts=None, dry_run=False: calls.append((now_ts, dry_run)) or [{'order_id': 1, 'action': 'refreshed'}])
    now = datetime.now(timezone.utc)
    first_ts, first_actions = run_bot.maybe_manage_stale_orders(None, now=now, dry_run=True)
    second_ts, second_actions = run_bot.maybe_manage_stale_orders(first_ts, now=now + timedelta(seconds=1), dry_run=True)
    assert len(calls) == 1
    assert first_actions[0]['action'] == 'refreshed'
    assert second_actions == []


def test_binance_consumer_retries_after_open_timeout_without_crashing(monkeypatch):
    events = []

    class _FakeSocket:
        def __init__(self):
            self._messages = [json.dumps({'k': {'x': True, 'c': '101.0', 'T': 1711972800000}})]

        async def recv(self):
            if self._messages:
                return self._messages.pop(0)
            raise asyncio.TimeoutError()

    class _FakeConnect:
        def __init__(self):
            self.calls = 0

        def __call__(self, url, open_timeout=None):
            outer = self

            class _Ctx:
                async def __aenter__(self_inner):
                    outer.calls += 1
                    if outer.calls == 1:
                        raise TimeoutError('timed out during opening handshake')
                    return _FakeSocket()

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    fake_connect = _FakeConnect()
    monkeypatch.setattr(run_bot.websockets, 'connect', fake_connect)
    monkeypatch.setattr(run_bot, 'emit_console_status', lambda label, **fields: events.append((label, fields)))
    monkeypatch.setattr(run_bot, 'BINANCE_WS_RETRY_DELAY_SEC', 0.01)
    monkeypatch.setattr(run_bot, 'BINANCE_WS_OPEN_TIMEOUT_SEC', 0.01)

    class _Engine:
        def __init__(self):
            self.calls = []

        def observe_bar(self, price, ts=None, finalized=False):
            self.calls.append({'price': price, 'ts': ts, 'finalized': finalized})

    engine = _Engine()

    asyncio.run(
        run_bot.consume_binance_klines(
            live_engine=engine,
            state_lock=asyncio.Lock(),
            duration=1,
            console_mode='debug',
        )
    )

    assert fake_connect.calls >= 2
    assert any(label == 'feed' and fields.get('status') == 'disconnected' for label, fields in events)
    assert any(label == 'feed' and fields.get('status') == 'reconnected' for label, fields in events)
    assert engine.calls


def test_diagnostics_stay_coherent_after_stale_order_maintenance(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M7', status='open')
    order = storage.create_order('diag-maint', 'M7', 'TOK7', 'YES', 'buy', 3.0, 0.2, 'submitted', ts)
    storage.create_reservation(order['id'], 'M7', 'TOK7', 'YES', 'exposure', 0.6, ts)
    monkeypatch.setattr(run_bot.execution, 'manage_stale_orders', lambda now_ts=None, dry_run=False: [{'order_id': order['id'], 'action': 'refreshed', 'status': 'submitted'}])
    _, actions = run_bot.maybe_manage_stale_orders(None, now=datetime.now(timezone.utc), dry_run=True)
    report = storage.get_active_order_diagnostics()
    assert actions[0]['status'] == 'submitted'
    assert report['orders'][0]['state'] == 'submitted'
