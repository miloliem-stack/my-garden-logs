import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage, run_bot
from scripts import check_clean_start, check_first_trade_readiness, init_fresh_db, bootstrap_first_trade_dry_run


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    os.environ.pop('BOT_DB_PATH', None)
    os.environ.pop('BOT_REQUIRE_CLEAN_START', None)
    os.environ.pop('BOT_ALLOW_DIRTY_START', None)
    storage.ensure_db()


def test_clean_start_status_helper():
    status = storage.get_clean_start_status()
    assert status['clean_start'] is True
    assert status['open_lots'] == 0
    assert status['merged_lots'] == 0
    assert status['redeemed_lots'] == 0
    assert status['reconciliation_issues'] == 0


def test_check_clean_start_script_passes(capsys):
    check_clean_start.main()
    out = capsys.readouterr().out
    assert 'result: clean DB confirmed' in out


def test_check_first_trade_readiness_script(monkeypatch, capsys):
    monkeypatch.setattr(
        check_first_trade_readiness.polymarket_feed,
        'detect_active_hourly_market',
        lambda series_id: {
            'market_id': 'DISCOVERED1',
            'token_yes': 'YES1',
            'token_no': 'NO1',
            'condition_id': 'COND1',
            'slug': 'slug-1',
            'title': 'Hourly 1',
            'startDate': None,
            'endDate': None,
            'status': 'open',
        },
    )
    monkeypatch.setattr(sys, 'argv', ['check_first_trade_readiness.py', '--series-id', 'SERIES1'])
    check_first_trade_readiness.main()
    out = capsys.readouterr().out
    assert '"db_schema_present": true' in out.lower()
    assert '"market_discovery_works": true' in out.lower()
    assert '"market_hydration_works": true' in out.lower()
    assert '"snapshot_empty_before_first_trade": true' in out.lower()
    assert 'result: ready for first trade' in out


def test_bot_db_path_respected_across_storage_and_runner(monkeypatch, tmp_path):
    db_path = tmp_path / 'prod.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    storage.ensure_db()
    assert db_path.exists()
    assert storage.get_db_path() == db_path
    status = run_bot.enforce_startup_gate()
    assert status['clean_start'] is True


def test_init_fresh_db_creates_clean_db(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / 'fresh.db'
    monkeypatch.setattr(sys, 'argv', ['init_fresh_db.py', '--db-path', str(db_path)])
    init_fresh_db.main()
    out = capsys.readouterr().out
    assert db_path.exists()
    assert 'result: clean DB ready' in out


def test_runner_refuses_dirty_db_when_clean_start_required(monkeypatch, tmp_path):
    db_path = tmp_path / 'dirty.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    monkeypatch.setenv('BOT_REQUIRE_CLEAN_START', 'true')
    storage.ensure_db()
    ts = '2026-03-15T00:00:00+00:00'
    storage.create_market('DIRTY', status='open')
    storage.create_open_lot('DIRTY', 'TOK', 'YES', 1.0, 0.5, ts)
    try:
        run_bot.enforce_startup_gate()
        assert False, 'expected startup gate to refuse dirty DB'
    except SystemExit as exc:
        assert 'Refusing startup' in str(exc)


def test_runner_starts_with_clean_db(monkeypatch, tmp_path):
    db_path = tmp_path / 'clean.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    monkeypatch.setenv('BOT_REQUIRE_CLEAN_START', 'true')
    storage.ensure_db()
    status = run_bot.enforce_startup_gate()
    assert status['clean_start'] is True


def test_bootstrap_first_trade_dry_run_works(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / 'bootstrap.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    storage.ensure_db()
    monkeypatch.setattr(
        bootstrap_first_trade_dry_run.polymarket_feed,
        'detect_active_hourly_market',
        lambda series_id: {
            'market_id': 'DISCOVERED2',
            'token_yes': 'YES2',
            'token_no': 'NO2',
            'condition_id': 'COND2',
            'slug': 'slug-2',
            'title': 'Hourly 2',
            'startDate': None,
            'endDate': None,
            'status': 'open',
        },
    )
    monkeypatch.setattr(sys, 'argv', ['bootstrap_first_trade_dry_run.py', '--series-id', 'SERIES2', '--q-market', '0.4'])
    bootstrap_first_trade_dry_run.main()
    out = capsys.readouterr().out
    assert 'snapshot_before=' in out
    assert 'dry_run_action=' in out
    assert 'snapshot_after=' in out
    assert storage.get_clean_start_status()['open_lots'] == 0
