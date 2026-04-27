import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage
from scripts import first_live_order_smoke, check_live_order_readiness


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    for key in ('BOT_DB_PATH', 'POLY_API_KEY', 'POLY_API_SECRET', 'POLY_API_PASSPHRASE', 'LIVE'):
        os.environ.pop(key, None)
    storage.ensure_db()


def _market_meta():
    now = datetime.now(timezone.utc)
    return {
        'market_id': 'LIVE1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'condition_id': 'COND1',
        'slug': 'slug-1',
        'title': 'Hourly',
        'startDate': now - timedelta(minutes=5),
        'endDate': now + timedelta(minutes=5),
        'status': 'open',
    }


def _quote(mid=0.4):
    return {
        'best_bid': mid - 0.01,
        'best_ask': mid + 0.01,
        'mid': mid,
        'spread': 0.02,
        'bid_size': 10.0,
        'ask_size': 10.0,
        'is_crossed': False,
        'is_empty': False,
        'source': 'test',
        'fetched_at': '2026-03-16T00:00:00+00:00',
        'age_seconds': 0.1,
        'fetch_failed': False,
        'raw': {'bids': [[mid - 0.01, 10.0]], 'asks': [[mid + 0.01, 10.0]]},
    }


def test_first_live_order_smoke_defaults_to_dry_run_and_refuses_live_without_opt_in(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(first_live_order_smoke.run_bot, 'enforce_startup_gate', lambda allow_dirty_start=False: {'clean_start': True})
    monkeypatch.setattr(first_live_order_smoke.polymarket_feed, 'detect_active_hourly_market', lambda series_id: _market_meta())
    monkeypatch.setattr(first_live_order_smoke.polymarket_feed, 'get_quote_snapshot', lambda token_id: _quote(0.4 if token_id == 'YES1' else 0.6))
    monkeypatch.setattr(first_live_order_smoke.polymarket_client, 'place_limit_order', lambda *args, **kwargs: {'dry_run': True, 'status': 'dry_run'})
    monkeypatch.setattr(sys, 'argv', ['first_live_order_smoke.py', '--series-id', 'SERIES1', '--artifact-dir', str(tmp_path)])
    first_live_order_smoke.main()
    out = capsys.readouterr().out
    assert '"artifact_dir"' in out

    monkeypatch.setattr(sys, 'argv', ['first_live_order_smoke.py', '--series-id', 'SERIES1', '--live'])
    try:
        first_live_order_smoke.main()
        assert False, 'expected live smoke to require explicit confirmation'
    except SystemExit as exc:
        assert 'confirm-live' in str(exc)


def test_readiness_script_fails_cleanly_when_credentials_missing(monkeypatch, capsys):
    monkeypatch.setattr(check_live_order_readiness.run_bot, 'enforce_startup_gate', lambda allow_dirty_start=False: {'clean_start': True})
    monkeypatch.setattr(check_live_order_readiness.polymarket_feed, 'detect_active_hourly_market', lambda series_id: _market_meta())
    monkeypatch.setattr(check_live_order_readiness.polymarket_feed, 'get_quote_snapshot', lambda token_id: _quote(0.4 if token_id == 'YES1' else 0.6))
    monkeypatch.setattr(sys, 'argv', ['check_live_order_readiness.py', '--series-id', 'SERIES1'])
    try:
        check_live_order_readiness.main()
        assert False, 'expected missing credentials to fail'
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert '"market_detected": true' in out.lower()
    assert '"trade_gate_ok": true' in out.lower()


def test_readiness_script_reports_market_quote_and_storage_gate(monkeypatch, capsys):
    monkeypatch.setenv('POLY_API_KEY', 'k')
    monkeypatch.setenv('POLY_API_SECRET', 's')
    monkeypatch.setenv('POLY_API_PASSPHRASE', 'p')
    monkeypatch.setattr(check_live_order_readiness.run_bot, 'enforce_startup_gate', lambda allow_dirty_start=False: {'clean_start': True, 'db_path': 'x'})
    monkeypatch.setattr(check_live_order_readiness.polymarket_feed, 'detect_active_hourly_market', lambda series_id: _market_meta())
    monkeypatch.setattr(check_live_order_readiness.polymarket_feed, 'get_quote_snapshot', lambda token_id: _quote(0.4 if token_id == 'YES1' else 0.6))
    monkeypatch.setattr(sys, 'argv', ['check_live_order_readiness.py', '--series-id', 'SERIES1'])
    check_live_order_readiness.main()
    out = capsys.readouterr().out
    assert '"storage_gate"' in out
    assert '"yes_quote_valid": true' in out.lower()
    assert '"no_quote_valid": true' in out.lower()
