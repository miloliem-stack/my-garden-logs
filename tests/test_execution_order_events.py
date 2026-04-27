import os

from src import execution, storage


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_order_events_are_appended_for_submit_refresh_and_cancel(monkeypatch):
    storage.create_market('EV1', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'open', 'orderId': 'OID-E1', 'filled_qty': 0.0})
    resp = execution.place_live_limit('buy', 1.0, 0.2, 'TOKE', 'EV1', 'YES')
    order_id = resp['order_id']
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'open', 'order_id': 'OID-E1', 'filled_qty': 0.0})
    monkeypatch.setattr(execution, 'cancel_order', lambda **kwargs: {'status': 'canceled', 'order_id': 'OID-E1', 'remaining_qty': 1.0, 'filled_qty': 0.0})
    execution.refresh_order_status(order_id, dry_run=False)
    execution.cancel_and_reconcile_order(order_id, dry_run=False)
    event_types = [event['event_type'] for event in storage.list_order_events(order_id)]
    assert 'submit_before' in event_types
    assert 'submit_after' in event_types
    assert 'refresh_before' in event_types
    assert 'refresh_after' in event_types
    assert 'cancel_before' in event_types
    assert 'cancel_after' in event_types

