import os
from datetime import datetime, timezone

from src import storage, strategy_manager


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_duplicate_active_order_blocks_new_entry(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DED1', status='open')
    storage.create_order('active-buy', 'DED1', 'YES1', 'YES', 'buy', 1.0, 0.4, 'open', ts, venue_order_id='OID-D1')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'q_yes': 0.4,
        'q_no': 0.6,
        'p_yes': 0.6,
        'p_no': 0.4,
        'edge_yes': 0.2,
        'edge_no': -0.2,
        'trade_allowed': True,
        'policy': {'allow_new_entries': True, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
    }
    result = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'DED1', dry_run=True)
    assert result['action'] == 'skipped_existing_active_order'
    assert result['active_order_ids']


def test_trade_action_reports_quantized_submitted_qty(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DED2', status='open')
    monkeypatch.setattr(
        strategy_manager,
        'place_marketable_buy',
        lambda *args, **kwargs: {
            'status': 'submitted',
            'submitted_qty': 2.0,
            'submitted_notional': 0.8,
            'raw_requested_qty': 2.0833333333333326,
        },
    )
    decision_state = {
        'q_yes': 0.4,
        'q_no': 0.6,
        'p_yes': 0.6,
        'p_no': 0.4,
        'edge_yes': 0.2,
        'edge_no': -0.2,
        'trade_allowed': True,
        'policy': {'allow_new_entries': True, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
    }
    result = strategy_manager.build_trade_action(decision_state, 'YES2', 'NO2', 'DED2', dry_run=False)
    assert result['qty'] == 2.0
    assert result['quantized_qty'] == 2.0
    assert result['quantized_notional'] == 0.8
    assert result['raw_requested_qty'] >= result['quantized_qty']


def test_same_side_inventory_blocks_new_entry_when_disabled(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DED3', status='open')
    storage.create_open_lot('DED3', 'YES3', 'YES', 3.0, 0.42, ts, tx_hash='tx-held')
    monkeypatch.setattr(strategy_manager, 'ALLOW_SAME_SIDE_ENTRY', False)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'off')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'q_yes': 0.4,
        'q_no': 0.6,
        'p_yes': 0.6,
        'p_no': 0.4,
        'edge_yes': 0.2,
        'edge_no': -0.2,
        'trade_allowed': True,
        'policy': {'allow_new_entries': True, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
    }
    result = strategy_manager.build_trade_action(decision_state, 'YES3', 'NO3', 'DED3', dry_run=True)
    assert result['action'] == 'skipped_existing_same_side_exposure'
    assert result['reason'] == 'same_side_entry_disabled'
    assert result['blocking_qty'] == 3.0
