from __future__ import annotations

import os
from typing import Iterable, Optional

from src import polymarket_client, storage


def reset_db() -> None:
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def decision_state_yes(*, p_yes: float = 0.7, q_yes: float = 0.4, q_no: float = 0.6) -> dict:
    return {
        'p_yes': p_yes,
        'p_no': 1.0 - p_yes,
        'q_yes': q_yes,
        'q_no': q_no,
        'edge_yes': p_yes - q_yes,
        'edge_no': (1.0 - p_yes) - q_no,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': {
            'edge_threshold_yes': 0.02,
            'edge_threshold_no': 0.02,
            'kelly_multiplier': 1.0,
            'max_trade_notional_multiplier': 1.0,
            'allow_new_entries': True,
            'policy_bucket': 'far',
        },
    }


def patch_submit_sequence(monkeypatch, execution_module, fixture_names: Iterable[str], *, default_status: str = 'submitted'):
    queue = list(fixture_names)

    def fake_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=True, retries=1, client_order_id=None, compiled_intent=None):
        if not queue:
            raise AssertionError('submit fixture queue exhausted')
        fixture_name = queue.pop(0)
        return polymarket_client.replay_sdk_fixture(
            fixture_name,
            default_status=default_status,
            client_order_id=client_order_id,
        )

    monkeypatch.setattr(execution_module, 'place_marketable_order', fake_place_marketable_order)
    return queue


def patch_status_fixture(monkeypatch, execution_module, fixture_name: str, *, default_status: str = 'unknown'):
    def fake_get_order_status(order_id=None, client_order_id=None, dry_run=False):
        return polymarket_client.replay_sdk_fixture(fixture_name, default_status=default_status, client_order_id=client_order_id)

    monkeypatch.setattr(execution_module, 'get_order_status', fake_get_order_status)


def patch_cancel_fixture(monkeypatch, execution_module, fixture_name: str, *, default_status: str = 'unknown'):
    def fake_cancel_order(order_id=None, client_order_id=None, dry_run=False):
        return polymarket_client.replay_sdk_fixture(fixture_name, default_status=default_status, client_order_id=client_order_id)

    monkeypatch.setattr(execution_module, 'cancel_order', fake_cancel_order)


def latest_order():
    orders = storage.get_open_orders(
        statuses=['pending_submit', 'submitted', 'open', 'partially_filled', 'cancel_requested', 'unknown', 'filled', 'canceled']
    )
    if not orders:
        return None
    return max(orders, key=lambda row: row['id'])
