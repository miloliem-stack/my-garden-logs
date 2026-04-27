from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sys
from pathlib import Path

from src import execution, storage, strategy_manager

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers_execution_integration import (
    decision_state_yes,
    patch_cancel_fixture,
    patch_status_fixture,
    patch_submit_sequence,
    reset_db,
)


def setup_function(_fn):
    reset_db()


def test_signal_to_accepted_then_open(monkeypatch):
    storage.create_market('INT-A', status='open')
    patch_submit_sequence(monkeypatch, execution, ['sdk_status_accepted.json'])
    decision = decision_state_yes()

    action = strategy_manager.build_trade_action(decision, 'TOKY', 'TOKN', 'INT-A', dry_run=False)
    order_id = action['resp']['order_id']
    order = storage.get_order(order_id=order_id)
    assert order['status'] == 'submitted'

    patch_status_fixture(monkeypatch, execution, 'sdk_status_open.json')
    refreshed = execution.refresh_order_status(order_id, dry_run=False)
    assert refreshed['order']['status'] == 'open'


def test_accepted_to_partial_fill_to_fill(monkeypatch):
    storage.create_market('INT-P', status='open')
    patch_submit_sequence(monkeypatch, execution, ['sdk_status_accepted.json'])

    resp = execution.place_marketable_buy('TOKP', 5.0, limit_price=0.41, dry_run=False, market_id='INT-P', outcome_side='YES')
    order_id = resp['order_id']
    assert storage.get_order(order_id=order_id)['status'] == 'submitted'

    execution.process_order_update(order_id, execution.polymarket_client.replay_sdk_fixture('sdk_status_partial.json', default_status='unknown'))
    assert storage.get_order(order_id=order_id)['status'] == 'partially_filled'
    assert storage.get_total_qty_by_token('TOKP', market_id='INT-P') == 2.0

    execution.process_order_update(order_id, execution.polymarket_client.replay_sdk_fixture('sdk_status_filled.json', default_status='unknown'))
    assert storage.get_order(order_id=order_id)['status'] == 'filled'
    assert storage.get_total_qty_by_token('TOKP', market_id='INT-P') == 5.0


def test_accepted_then_cancel(monkeypatch):
    storage.create_market('INT-C', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('INT-C', 'TOKC', 'NO', 8.0, 0.1, ts)
    patch_submit_sequence(monkeypatch, execution, ['sdk_status_accepted.json'])

    resp = execution.place_marketable_sell('TOKC', 4.0, limit_price=0.3, dry_run=False, market_id='INT-C', outcome_side='NO')
    order_id = resp['order_id']
    assert storage.get_reserved_qty('INT-C', 'TOKC', 'NO') == 4.0

    patch_cancel_fixture(monkeypatch, execution, 'sdk_status_canceled.json')
    canceled = execution.cancel_and_reconcile_order(order_id, dry_run=False)
    assert canceled['order']['status'] == 'canceled'
    assert storage.get_reserved_qty('INT-C', 'TOKC', 'NO') == 0.0


def test_ambiguous_submit_then_restart_recovery(monkeypatch):
    storage.create_market('INT-U', status='open')
    patch_submit_sequence(monkeypatch, execution, ['sdk_timeout_ambiguous.json'], default_status='unknown')

    resp = execution.place_marketable_buy('TOKU', 3.0, limit_price=0.4, dry_run=False, market_id='INT-U', outcome_side='YES')
    order_id = resp['order_id']
    assert storage.get_order(order_id=order_id)['status'] == 'unknown'

    patch_status_fixture(monkeypatch, execution, 'sdk_status_open.json')
    recovered = execution.recover_active_orders_on_startup(dry_run=False)
    assert any(item['order_id'] == order_id and item['refreshed'] is True for item in recovered)
    assert storage.get_order(order_id=order_id)['status'] == 'open'


def test_repeated_signal_while_active_order_exists_skips(monkeypatch):
    storage.create_market('INT-D', status='open')
    patch_submit_sequence(monkeypatch, execution, ['sdk_status_accepted.json'])
    decision = decision_state_yes()

    first = strategy_manager.build_trade_action(decision, 'TOKY', 'TOKN', 'INT-D', dry_run=False)
    assert storage.get_order(order_id=first['resp']['order_id'])['status'] == 'submitted'

    second = strategy_manager.build_trade_action(decision, 'TOKY', 'TOKN', 'INT-D', dry_run=False)
    assert second['action'] == 'skipped_existing_active_order'
    assert second['market_id'] == 'INT-D'


def test_stale_order_management_and_recovery_cycle(monkeypatch):
    storage.create_market('INT-S', status='open')
    patch_submit_sequence(monkeypatch, execution, ['sdk_status_accepted.json'])
    resp = execution.place_marketable_buy('TOKS', 3.0, limit_price=0.4, dry_run=False, market_id='INT-S', outcome_side='YES')
    order_id = resp['order_id']
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
    storage.update_order(order_id, updated_ts=old_ts)

    patch_status_fixture(monkeypatch, execution, 'sdk_status_open.json')
    patch_cancel_fixture(monkeypatch, execution, 'sdk_status_canceled.json')
    actions = execution.manage_stale_orders(now_ts=datetime.now(timezone.utc).isoformat(), dry_run=False)
    assert any(item['order_id'] == order_id for item in actions)
    assert storage.get_order(order_id=order_id)['status'] == 'canceled'



