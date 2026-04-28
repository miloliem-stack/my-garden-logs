import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage
from src import strategy_manager
from src.strategy_manager import decide_and_execute
from datetime import datetime, timezone


def setup_module(module):
    try:
        import os
        os.remove('bot_state.db')
    except Exception:
        pass
    storage.ensure_db()


def test_missing_market_id_rejected():
    # missing market_id should raise
    try:
        decide_and_execute(0.6, 0.5, 'T_YES', 'T_NO', market_id=None, dry_run=True)
        assert False, 'Expected ValueError for missing market_id'
    except ValueError:
        pass


def test_no_trade_when_market_closed():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_TEST', slug='test', status='closed')
    # even with positive edge, closed market should not trade
    res = decide_and_execute(0.7, 0.5, 'T_YES', 'T_NO', market_id='MKT_TEST', dry_run=True)
    assert res is None


def test_strategy_skips_incomplete_market_metadata():
    storage.create_market('MKT_OPEN', slug='test-open', status='open')
    res = decide_and_execute(0.7, 0.5, None, 'T_NO', market_id='MKT_OPEN', dry_run=True)
    assert res['action'] == 'skipped_incomplete_market_metadata'


def test_exposure_cap_blocks_new_buys(monkeypatch):
    storage.create_market('MKT_CAP', slug='cap', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    order = storage.create_order('cap-order', 'MKT_CAP', 'TOK', 'YES', 'buy', 1.0, 80.0, 'open', ts)
    storage.create_reservation(order['id'], 'MKT_CAP', 'TOK', 'YES', 'exposure', 80.0, ts)
    monkeypatch.setattr(strategy_manager, 'BOT_BANKROLL', 100.0)
    monkeypatch.setattr(strategy_manager, 'PER_TRADE_CAP_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'TOTAL_EXPOSURE_CAP', 0.5)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_trade'})

    res = decide_and_execute(0.8, 0.4, 'TOKY', 'TOKN', market_id='MKT_CAP', dry_run=True)
    assert res['action'] == 'skipped_due_to_exposure_cap'


def test_strategy_sizing_uses_runtime_effective_bankroll_instead_of_static_env(monkeypatch):
    storage.create_market('MKT_BANKROLL', slug='bankroll', status='open')
    monkeypatch.setattr(strategy_manager, 'BOT_BANKROLL', 1000.0)
    monkeypatch.setattr(strategy_manager, 'PER_TRADE_CAP_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'TOTAL_EXPOSURE_CAP', 1.0)
    monkeypatch.setattr(strategy_manager, 'get_inflight_exposure', lambda: 0.0)
    monkeypatch.setattr(strategy_manager, 'get_open_orders', lambda market_id=None: [])
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda token_id, qty, limit_price, dry_run, market_id, outcome_side, **kwargs: {'qty': qty, 'limit_price': limit_price})

    decision_state = {
        'p_yes': 0.9,
        'p_no': 0.1,
        'q_yes': 0.5,
        'q_no': 0.5,
        'edge_yes': 0.4,
        'edge_no': -0.4,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': {'allow_new_entries': True, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
    }
    wallet_snapshot = {'effective_bankroll': 10.0, 'bankroll_source': 'wallet_live'}

    res = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', market_id='MKT_BANKROLL', dry_run=True, wallet_state=wallet_snapshot)

    assert abs(res['qty'] - 1.6) < 1e-9
    assert res['effective_bankroll'] == 10.0
    assert res['bankroll_source'] == 'wallet_live'


def test_strategy_manager_no_longer_exports_midpoint_fallback():
    assert not hasattr(strategy_manager, 'get_market_mid')
