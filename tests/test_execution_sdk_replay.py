import os
from datetime import datetime, timezone

from src import execution, polymarket_client, storage


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_sdk_fixture_replay_normalizes_lifecycle_shapes():
    accepted = polymarket_client.replay_sdk_fixture('sdk_status_accepted.json', default_status='submitted')
    partial = polymarket_client.replay_sdk_fixture('sdk_status_partial.json', default_status='unknown')
    auth_error = polymarket_client.replay_sdk_fixture('sdk_auth_error.json', default_status='unknown')

    assert accepted['order_id'] == 'OID_SDK_ACCEPTED'
    assert accepted['client_order_id'] == 'CID_SDK_ACCEPTED'
    assert partial['fills'][0]['size'] == 2.0
    assert partial['status'] == 'partially_filled'
    assert auth_error['status'] == 'rejected'
    assert auth_error['http_status'] == 401


def test_sdk_fixture_ambiguous_submit_goes_to_unknown(monkeypatch):
    storage.create_market('SDKA', status='open')
    monkeypatch.setattr(
        execution,
        'place_limit_order',
        lambda *args, **kwargs: polymarket_client.replay_sdk_fixture(
            'sdk_timeout_ambiguous.json',
            default_status='unknown',
            client_order_id=kwargs.get('client_order_id'),
        ),
    )
    resp = execution.place_live_limit('buy', 2.0, 0.2, 'TOKA', 'SDKA', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'unknown'
    assert order['filled_qty'] == 0.0


def test_sdk_fixture_partial_fill_applies_incremental_inventory(monkeypatch):
    storage.create_market('SDKP', status='open')
    monkeypatch.setattr(
        execution,
        'place_limit_order',
        lambda *args, **kwargs: polymarket_client.replay_sdk_fixture(
            'sdk_status_partial.json',
            default_status='submitted',
            client_order_id=kwargs.get('client_order_id'),
        ),
    )
    resp = execution.place_live_limit('buy', 5.0, 0.41, 'TOKP', 'SDKP', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'partially_filled'
    assert storage.get_total_qty_by_token('TOKP', market_id='SDKP') == 2.0

    execution.process_order_update(
        order['id'],
        polymarket_client.replay_sdk_fixture('sdk_status_partial.json', default_status='unknown'),
    )
    assert storage.get_total_qty_by_token('TOKP', market_id='SDKP') == 2.0

    execution.process_order_update(
        order['id'],
        polymarket_client.replay_sdk_fixture('sdk_status_filled.json', default_status='unknown'),
    )
    assert storage.get_total_qty_by_token('TOKP', market_id='SDKP') == 5.0
    assert storage.get_order(order_id=order['id'])['status'] == 'filled'


def test_sdk_restart_recovery_for_submitted_open_and_unknown(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('SDKR', status='open')
    submitted = storage.create_order('sdk-submitted', 'SDKR', 'TOKR1', 'YES', 'buy', 2.0, 0.2, 'submitted', ts, venue_order_id='OID_SDK_OPEN')
    open_order = storage.create_order('sdk-open', 'SDKR', 'TOKR2', 'YES', 'buy', 5.0, 0.2, 'open', ts, venue_order_id='OID_SDK_PARTIAL')
    unknown = storage.create_order('sdk-unknown', 'SDKR', 'TOKR3', 'YES', 'buy', 2.0, 0.2, 'unknown', ts, venue_order_id='OID_SDK_OPEN')
    for order in (submitted, open_order, unknown):
        storage.create_reservation(order['id'], 'SDKR', order['token_id'], 'YES', 'exposure', 0.4, ts)

    def fake_get_order_status(order_id=None, client_order_id=None, dry_run=False):
        if order_id == 'OID_SDK_PARTIAL':
            return polymarket_client.replay_sdk_fixture('sdk_status_partial.json', default_status='unknown')
        return polymarket_client.replay_sdk_fixture('sdk_status_open.json', default_status='unknown')

    monkeypatch.setattr(execution, 'get_order_status', fake_get_order_status)
    report = execution.recover_active_orders_on_startup(dry_run=False)
    statuses = {storage.get_order(order_id=o['id'])['status'] for o in (submitted, open_order, unknown)}
    assert 'open' in statuses
    assert 'partially_filled' in statuses
    assert len(report) == 3


def test_sdk_cancel_reconciliation_releases_remaining_reservation(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('SDKC', status='open')
    storage.create_open_lot('SDKC', 'TOKC', 'NO', 8.0, 0.1, ts)
    monkeypatch.setattr(
        execution,
        'place_limit_order',
        lambda *args, **kwargs: polymarket_client.replay_sdk_fixture('sdk_status_accepted.json', default_status='submitted', client_order_id=kwargs.get('client_order_id')),
    )
    resp = execution.place_live_limit('sell', 4.0, 0.25, 'TOKC', 'SDKC', 'NO')
    order = storage.get_order(order_id=resp['order_id'])
    monkeypatch.setattr(
        execution,
        'cancel_order',
        lambda **kwargs: polymarket_client.replay_sdk_fixture('sdk_status_canceled.json', default_status='unknown'),
    )
    execution.cancel_and_reconcile_order(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    assert refreshed['status'] == 'canceled'
    assert storage.get_reserved_qty('SDKC', 'TOKC', 'NO') == 0.0


def test_sdk_client_order_id_is_preserved_locally_across_normalization():
    normalized = polymarket_client.replay_sdk_fixture(
        'sdk_status_accepted.json',
        default_status='submitted',
        client_order_id='LOCAL-CID-1',
    )
    assert normalized['client_order_id'] == 'CID_SDK_ACCEPTED'

    timeout_normalized = polymarket_client.replay_sdk_fixture(
        'sdk_timeout_ambiguous.json',
        default_status='unknown',
        client_order_id='LOCAL-CID-2',
    )
    assert timeout_normalized['client_order_id'] == 'LOCAL-CID-2'
