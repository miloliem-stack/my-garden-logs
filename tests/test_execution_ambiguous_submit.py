import os
from datetime import datetime, timezone

from src import execution, storage


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_ambiguous_submit_goes_to_unknown_not_failed(monkeypatch):
    storage.create_market('AMB1', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'error', 'reason': 'timeout'})
    resp = execution.place_live_limit('buy', 2.0, 0.2, 'TOKA', 'AMB1', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'unknown'
    assert order['filled_qty'] == 0.0
    events = storage.list_order_events(order['id'])
    assert any(event['event_type'] == 'submit_ambiguous' for event in events)


def test_submit_with_fill_evidence_is_not_forced_into_unknown(monkeypatch):
    storage.create_market('AMB2', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'accepted', 'filledQuantity': 2.0, 'price': 0.2})
    resp = execution.place_live_limit('buy', 2.0, 0.2, 'TOKB', 'AMB2', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'filled'
    assert order['filled_qty'] == 2.0
    assert storage.get_total_qty_by_token('TOKB', market_id='AMB2') == 2.0
