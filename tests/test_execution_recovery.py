import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import execution, storage
from src.tools import repair_misclassified_clob_order_ids


def setup_function(_fn):
    os.environ['BOT_DB_PATH'] = str(Path('/tmp') / f'btc_1h_test_execution_recovery_{_fn.__name__}.db')
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_startup_recovery_refreshes_active_orders(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC1', status='open')
    order = storage.create_order('recover-1', 'REC1', 'TOKR', 'YES', 'buy', 2.0, 0.2, 'submitted', ts, venue_order_id='OID-1')
    storage.create_reservation(order['id'], 'REC1', 'TOKR', 'YES', 'exposure', 0.4, ts)
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'open', 'order_id': 'OID-1', 'filled_qty': 0.0})
    report = execution.recover_active_orders_on_startup(dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    assert refreshed['status'] in ('submitted', 'open')
    assert report[0]['refreshed'] is True


def test_startup_recovery_null_order_marks_not_found_and_releases_reservation(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC2', status='open')
    order = storage.create_order('recover-2', 'REC2', 'TOKR2', 'YES', 'buy', 2.0, 0.2, 'submitted', ts, venue_order_id='OID-MISSING')
    storage.create_reservation(order['id'], 'REC2', 'TOKR2', 'YES', 'exposure', 0.4, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'result': None,
            'order_id': kwargs.get('order_id'),
            'raw_probe': {'path': f"/order/{kwargs.get('order_id')}"},
        },
    )

    report = execution.recover_active_orders_on_startup(dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]

    assert report[0]['status'] == 'not_found_on_venue'
    assert refreshed['status'] == 'not_found_on_venue'
    assert reservation['status'] == 'released'


def test_startup_recovery_duplicate_runs_do_not_repeat_not_found_order(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC3', status='open')
    order = storage.create_order('recover-3', 'REC3', 'TOKR3', 'YES', 'buy', 2.0, 0.2, 'submitted', ts, venue_order_id='OID-MISSING-2')
    storage.create_reservation(order['id'], 'REC3', 'TOKR3', 'YES', 'exposure', 0.4, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'result': None,
            'order_id': kwargs.get('order_id'),
            'raw_probe': {'path': f"/order/{kwargs.get('order_id')}"},
        },
    )

    first = execution.recover_active_orders_on_startup(dry_run=False)
    second = execution.recover_active_orders_on_startup(dry_run=False)

    assert len(first) == 1
    assert first[0]['status'] == 'not_found_on_venue'
    assert second == []


def test_startup_recovery_reconstructs_inventory_from_tx_receipt(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    tx_hash = '0x' + 'b' * 64
    storage.create_market('REC4', status='open')
    order = storage.create_order(
        'recover-4',
        'REC4',
        'TOKR4',
        'NO',
        'buy',
        3.0,
        0.33,
        'unknown',
        ts,
        tx_hash=tx_hash,
    )
    storage.create_reservation(order['id'], 'REC4', 'TOKR4', 'NO', 'exposure', 0.99, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'result': None,
            'order_id': kwargs.get('order_id'),
            'raw_probe': {'path': f"/order/{kwargs.get('order_id')}"},
        },
    )
    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: {
            'txHash': tx,
            'logs': [
                {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TOKR4', 'value': 3.0}}
            ],
        } if tx == tx_hash else None,
    )
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    report = execution.recover_active_orders_on_startup(dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert report[0]['refreshed'] is True
    assert report[0]['result']['response']['recovered_via'] == 'tx_receipt_reconciliation'
    assert refreshed['tx_hash'] == tx_hash
    assert refreshed['status'] == 'filled'
    assert refreshed['filled_qty'] == 3.0
    assert storage.get_total_qty_by_token('TOKR4', market_id='REC4') == 3.0
    assert len(storage.get_order_fill_events(order['id'])) == 1


def test_refresh_uses_hash_like_venue_order_id_for_lookup_and_recovers_fill(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    venue_order_id = '0x' + 'd' * 64
    storage.create_market('REC4B', status='open')
    order = storage.create_order('recover-4b', 'REC4B', 'TOKR4B', 'YES', 'buy', 2.0, 0.35, 'unknown', ts, venue_order_id=venue_order_id)
    storage.create_reservation(order['id'], 'REC4B', 'TOKR4B', 'YES', 'exposure', 0.7, ts)

    def fake_get_order_status(order_id=None, client_order_id=None, dry_run=False):
        assert order_id == venue_order_id
        assert client_order_id is None
        return {
            'status': 'filled',
            'order_id': venue_order_id,
            'filled_qty': 2.0,
        }

    monkeypatch.setattr(execution, 'get_order_status', fake_get_order_status)

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert out['order']['status'] == 'filled'
    assert refreshed['venue_order_id'] == venue_order_id
    assert refreshed['tx_hash'] is None
    assert refreshed['filled_qty'] == 2.0
    assert storage.get_total_qty_by_token('TOKR4B', market_id='REC4B') == 2.0


def test_receipt_recovery_is_idempotent(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    tx_hash = '0x' + 'c' * 64
    storage.create_market('REC5', status='open')
    order = storage.create_order('recover-5', 'REC5', 'TOKR5', 'YES', 'buy', 2.0, 0.25, 'unknown', ts, tx_hash=tx_hash)
    storage.create_reservation(order['id'], 'REC5', 'TOKR5', 'YES', 'exposure', 0.5, ts)

    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: {
            'txHash': tx,
            'logs': [
                {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TOKR5', 'value': 2.0}}
            ],
        } if tx == tx_hash else None,
    )
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    first = storage.reconcile_tx(tx_hash)
    second = storage.reconcile_tx(tx_hash)

    assert first['status'] == 'ok'
    assert second['status'] == 'ok'
    assert storage.get_total_qty_by_token('TOKR5', market_id='REC5') == 2.0
    assert len(storage.get_order_fill_events(order['id'])) == 1


def test_refresh_recovers_buy_from_user_trade_history_when_only_client_order_id_exists(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC6', status='open')
    order = storage.create_order('live-cid-1', 'REC6', 'TOKR6', 'YES', 'buy', 1.75, 0.4, 'unknown', ts)
    storage.create_reservation(order['id'], 'REC6', 'TOKR6', 'YES', 'exposure', 0.7, ts)

    monkeypatch.setattr(
        execution.polymarket_client,
        'get_user_trades',
        lambda market_id=None, token_id=None, dry_run=False: {
            'status': 'ok',
            'trades': [
                {'market': market_id, 'asset_id': token_id, 'size': 1.75, 'price': 0.4, 'timestamp': ts}
            ],
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert out['order']['status'] == 'filled'
    assert refreshed['filled_qty'] == 1.75
    assert storage.get_total_qty_by_token('TOKR6', market_id='REC6') == 1.75


def test_refresh_prefers_tx_receipt_recovery_before_provisional_client_order_lookup(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    tx_hash = '0x' + 'f' * 64
    storage.create_market('REC7', status='open')
    order = storage.create_order('live-cid-2', 'REC7', 'TOKR7', 'NO', 'buy', 1.25, 0.6, 'unknown', ts, tx_hash=tx_hash)
    storage.create_reservation(order['id'], 'REC7', 'TOKR7', 'NO', 'exposure', 0.75, ts)

    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: {
            'txHash': tx,
            'logs': [
                {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TOKR7', 'value': 1.25}}
            ],
        } if tx == tx_hash else None,
    )
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('trade fallback should not run first')))
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: (_ for _ in ()).throw(AssertionError('client_order_id lookup should not run first')))

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert out['order']['status'] == 'filled'
    assert refreshed['filled_qty'] == 1.25
    assert storage.get_total_qty_by_token('TOKR7', market_id='REC7') == 1.25


def test_refresh_recovers_unknown_buy_via_trade_history_even_with_tx_hash(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    tx_hash = '0x' + '9' * 64
    storage.create_market('REC7A', status='open')
    order = storage.create_order('recover-7a', 'REC7A', 'TOKR7A', 'YES', 'buy', 1.5, 0.4, 'unknown', ts, tx_hash=tx_hash)
    storage.create_reservation(order['id'], 'REC7A', 'TOKR7A', 'YES', 'exposure', 0.6, ts)

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: None)
    monkeypatch.setattr(
        execution.polymarket_client,
        'get_user_trades',
        lambda market_id=None, token_id=None, dry_run=False: {
            'status': 'ok',
            'trades': [
                {'market': market_id, 'asset_id': token_id, 'size': 1.5, 'price': 0.4, 'timestamp': ts}
            ],
        },
    )
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: (_ for _ in ()).throw(AssertionError('venue/client lookup should not run after trade recovery')))

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert out['response']['recovered_via'] == 'user_trade_history'
    assert refreshed['status'] == 'filled'
    assert refreshed['filled_qty'] == 1.5
    assert storage.get_total_qty_by_token('TOKR7A', market_id='REC7A') == 1.5


def test_marketable_buy_matched_submit_and_refresh_payloads_recover_fill(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    venue_order_id = '0x' + '4' * 64
    tx_hash = '0x' + '5' * 64
    storage.create_market('REC7B', status='open')

    monkeypatch.setattr(
        execution,
        'place_marketable_order',
        lambda *args, **kwargs: {
            'status': 'matched',
            'orderID': venue_order_id,
            'transactionsHashes': [tx_hash],
            'clientOrderId': kwargs.get('client_order_id'),
        },
    )
    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda order_id=None, client_order_id=None, dry_run=False: {
            'status': 'MATCHED',
            'orderID': order_id,
            'size_matched': '2.0',
            'original_size': '2.0',
            'side': 'BUY',
        },
    )

    resp = execution.place_marketable_buy('TOKR7B', 2.0, limit_price=0.51, dry_run=False, market_id='REC7B', outcome_side='YES')
    submitted = storage.get_order(order_id=resp['order_id'])

    assert submitted['venue_order_id'] == venue_order_id
    assert submitted['tx_hash'] == tx_hash
    assert submitted['status'] == 'submitted'
    assert submitted['filled_qty'] == 0.0

    out = execution.refresh_order_status(submitted['id'], dry_run=False)
    refreshed = storage.get_order(order_id=submitted['id'])

    assert out['order']['status'] == 'filled'
    assert refreshed['venue_order_id'] == venue_order_id
    assert refreshed['tx_hash'] == tx_hash
    assert refreshed['filled_qty'] == 2.0
    assert refreshed['remaining_qty'] == 0.0
    assert storage.get_total_qty_by_token('TOKR7B', market_id='REC7B') == 2.0
    lots = storage.get_open_lots(token_id='TOKR7B', market_id='REC7B')
    assert len(lots) == 1
    assert lots[0]['qty'] == 2.0


def test_refresh_clamps_tiny_matched_overfill_to_requested_qty(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    venue_order_id = 'OID-CLAMP-1'
    tx_hash = '0x' + '6' * 64
    storage.create_market('REC7C', status='open')
    order = storage.create_order('recover-7c', 'REC7C', 'TOKR7C', 'YES', 'buy', 1.2065, 0.41, 'unknown', ts, venue_order_id=venue_order_id)
    storage.create_reservation(order['id'], 'REC7C', 'TOKR7C', 'YES', 'exposure', 1.2065 * 0.41, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda order_id=None, client_order_id=None, dry_run=False: {
            'status': 'MATCHED',
            'orderID': order_id,
            'transactionsHashes': [tx_hash],
            'size_matched': '1.206512',
            'original_size': '1.2065',
            'side': 'BUY',
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    lots = storage.get_open_lots(token_id='TOKR7C', market_id='REC7C')
    events = storage.list_order_events(order['id'])
    fill_event = next(event for event in events if event['event_type'] == 'fill_applied')

    assert out['order']['status'] == 'filled'
    assert refreshed['filled_qty'] == 1.2065
    assert refreshed['remaining_qty'] == 0.0
    assert refreshed['tx_hash'] == tx_hash
    assert lots[0]['qty'] == 1.2065
    assert out['response']['fill_clamped'] is True
    assert out['response']['fill_clamp_from'] == 1.206512
    assert out['response']['fill_clamp_to'] == 1.2065
    assert fill_event['response']['fill_clamped'] is True
    assert fill_event['response']['fill_clamp_to'] == 1.2065


def test_refresh_exact_matched_fill_has_no_clamp_metadata(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC7D', status='open')
    order = storage.create_order('recover-7d', 'REC7D', 'TOKR7D', 'YES', 'buy', 1.2065, 0.41, 'unknown', ts, venue_order_id='OID-EXACT-1')
    storage.create_reservation(order['id'], 'REC7D', 'TOKR7D', 'YES', 'exposure', 1.2065 * 0.41, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda order_id=None, client_order_id=None, dry_run=False: {
            'status': 'MATCHED',
            'orderID': order_id,
            'size_matched': '1.2065',
            'original_size': '1.2065',
            'side': 'BUY',
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    events = storage.list_order_events(order['id'])
    fill_event = next(event for event in events if event['event_type'] == 'fill_applied')

    assert out['order']['status'] == 'filled'
    assert out['order']['filled_qty'] == 1.2065
    assert 'fill_clamped' not in out['response']
    assert 'fill_clamped' not in fill_event['response']


def test_refresh_partial_matched_fill_remains_unchanged_without_clamp(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC7E', status='open')
    order = storage.create_order('recover-7e', 'REC7E', 'TOKR7E', 'YES', 'buy', 1.2065, 0.41, 'unknown', ts, venue_order_id='OID-PART-1')
    storage.create_reservation(order['id'], 'REC7E', 'TOKR7E', 'YES', 'exposure', 1.2065 * 0.41, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda order_id=None, client_order_id=None, dry_run=False: {
            'status': 'MATCHED',
            'orderID': order_id,
            'size_matched': '1.0',
            'original_size': '1.2065',
            'side': 'BUY',
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    lots = storage.get_open_lots(token_id='TOKR7E', market_id='REC7E')

    assert out['order']['status'] == 'partially_filled'
    assert refreshed['filled_qty'] == 1.0
    assert abs(refreshed['remaining_qty'] - 0.2065) < 1e-12
    assert lots[0]['qty'] == 1.0
    assert 'fill_clamped' not in out['response']


def test_refresh_null_lookup_preserves_partially_filled_economic_state(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC7F', status='open')
    order = storage.create_order('recover-7f', 'REC7F', 'TOKR7F', 'YES', 'buy', 5.0, 0.41, 'open', ts, venue_order_id='OID-NULL-1')
    storage.create_reservation(order['id'], 'REC7F', 'TOKR7F', 'YES', 'exposure', 5.0 * 0.41, ts)
    storage.apply_incremental_order_fill(order['id'], 2.0, fill_ts=ts, price=0.41, raw={'source': 'test'})
    storage.transition_order_state(order['id'], 'partially_filled', reason='test_seed_partial_fill', ts=ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'result': None,
            'order_id': kwargs.get('order_id'),
            'raw_probe': {'path': f"/order/{kwargs.get('order_id')}"},
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert out['response']['ledger_fill_preserved'] is True
    assert refreshed['status'] == 'partially_filled'
    assert refreshed['filled_qty'] == 2.0
    assert refreshed['remaining_qty'] == 3.0


def test_repair_not_found_on_venue_fill_evidence_repairs_partial_and_filled():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC10A', status='open')

    partial = storage.create_order('repair-partial', 'REC10A', 'TOKP', 'YES', 'buy', 4.0, 0.25, 'open', ts)
    storage.create_reservation(partial['id'], 'REC10A', 'TOKP', 'YES', 'exposure', 1.0, ts)
    storage.apply_incremental_order_fill(partial['id'], 1.5, fill_ts=ts, price=0.25, raw={'source': 'test'})
    storage.transition_order_state(partial['id'], 'partially_filled', reason='test_seed_partial_fill', ts=ts)
    storage.transition_order_state(partial['id'], 'not_found_on_venue', reason='test_seed_regression', ts=ts)

    filled = storage.create_order('repair-filled', 'REC10A', 'TOKF', 'NO', 'buy', 2.0, 0.35, 'open', ts)
    storage.create_reservation(filled['id'], 'REC10A', 'TOKF', 'NO', 'exposure', 0.7, ts)
    storage.apply_incremental_order_fill(filled['id'], 2.0, fill_ts=ts, price=0.35, raw={'source': 'test'})
    storage.transition_order_state(filled['id'], 'filled', reason='test_seed_full_fill', ts=ts)
    storage.update_order(filled['id'], status='not_found_on_venue', updated_ts=ts)

    repaired = storage.repair_not_found_on_venue_fill_evidence(updated_ts=ts)
    partial_after = storage.get_order(order_id=partial['id'])
    filled_after = storage.get_order(order_id=filled['id'])

    assert repaired['repaired'] == 2
    assert partial_after['status'] == 'partially_filled'
    assert partial_after['filled_qty'] == 1.5
    assert partial_after['remaining_qty'] == 2.5
    assert filled_after['status'] == 'filled'
    assert filled_after['filled_qty'] == 2.0
    assert filled_after['remaining_qty'] == 0.0


def test_startup_recovery_does_not_treat_hash_like_venue_order_id_as_tx_hash(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    venue_order_id = '0x' + 'e' * 64
    storage.create_market('REC8', status='open')
    order = storage.create_order('recover-8', 'REC8', 'TOKR8', 'YES', 'buy', 1.0, 0.4, 'unknown', ts, venue_order_id=venue_order_id)
    storage.create_reservation(order['id'], 'REC8', 'TOKR8', 'YES', 'exposure', 0.4, ts)

    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'status': 'unknown',
            'order_id': kwargs.get('order_id'),
            'filled_qty': 0.0,
        },
    )
    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: (_ for _ in ()).throw(AssertionError('hash-like venue_order_id must not be reconciled as tx_hash')),
    )

    report = execution.recover_active_orders_on_startup(dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])

    assert all(item.get('reconciled_tx') != venue_order_id for item in report)
    assert refreshed['venue_order_id'] == venue_order_id
    assert refreshed['tx_hash'] is None
    assert refreshed['status'] == 'unknown'


def test_repair_helper_moves_safe_misclassified_hash_into_venue_order_id(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    bad_hash = '0x' + 'f' * 64
    storage.create_market('REC9', status='open')
    order = storage.create_order(
        'recover-9',
        'REC9',
        'TOKR9',
        'NO',
        'buy',
        1.5,
        0.42,
        'unknown',
        ts,
        tx_hash=bad_hash,
        raw_response={'status': 'unknown', 'orderId': bad_hash},
    )

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: None)

    preview = repair_misclassified_clob_order_ids.repair_misclassified_clob_order_ids(dry_run=True)
    assert preview['examined'] == 1
    assert preview['repaired'] == 0
    assert preview['orders'][0]['id'] == order['id']

    applied = repair_misclassified_clob_order_ids.repair_misclassified_clob_order_ids(dry_run=False)
    repaired = storage.get_order(order_id=order['id'])

    assert applied['repaired'] == 1
    assert repaired['venue_order_id'] == bad_hash
    assert repaired['tx_hash'] is None


def test_stale_unknown_buy_escalates_to_manual_review_and_releases_reservation(monkeypatch):
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(hours=2)).isoformat()
    storage.create_market('REC11', status='open')
    order = storage.create_order('recover-11', 'REC11', 'TOKR11', 'YES', 'buy', 2.0, 0.2, 'unknown', ts)
    storage.create_reservation(order['id'], 'REC11', 'TOKR11', 'YES', 'exposure', 0.4, ts)
    storage.update_order(order['id'], updated_ts=ts)

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'unknown'})

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'unknown_review_age_sec': 1800},
    )
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]

    assert managed[0]['action'] == 'manual_review'
    assert refreshed['status'] == 'manual_review'
    assert reservation['status'] == 'released'


def test_recent_unknown_buy_does_not_escalate_to_manual_review(monkeypatch):
    now = datetime.now(timezone.utc)
    ts = now.isoformat()
    storage.create_market('REC12', status='open')
    order = storage.create_order('recover-12', 'REC12', 'TOKR12', 'YES', 'buy', 2.0, 0.2, 'unknown', ts)
    storage.create_reservation(order['id'], 'REC12', 'TOKR12', 'YES', 'exposure', 0.4, ts)

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'unknown'})

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'max_open_age_sec': 0, 'unknown_review_age_sec': 1800},
    )
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]

    assert managed[0]['action'] == 'reported_unknown_pending_review_window'
    assert refreshed['status'] == 'unknown'
    assert reservation['status'] == 'active'


def test_unknown_order_uses_stable_unknown_since_across_repeated_refreshes(monkeypatch):
    now = datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)
    created_ts = (now - timedelta(hours=1)).isoformat()
    refresh_ts = (now - timedelta(minutes=1)).isoformat()
    storage.create_market('REC12A', status='open')
    order = storage.create_order('recover-12a', 'REC12A', 'TOKR12A', 'YES', 'buy', 2.0, 0.2, 'unknown', created_ts)
    storage.create_reservation(order['id'], 'REC12A', 'TOKR12A', 'YES', 'exposure', 0.4, created_ts)
    storage.append_order_event(order['id'], 'submit_ambiguous', old_status='pending_submit', new_status='unknown', response={'http_status': 500}, ts=created_ts)
    storage.update_order(order['id'], updated_ts=refresh_ts)
    storage.append_order_event(order['id'], 'refresh_after', old_status='unknown', new_status='unknown', response={'lookup': 'provisional'}, ts=refresh_ts)

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'unknown', 'client_order_id': kwargs.get('client_order_id')})

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'max_open_age_sec': 0, 'unknown_review_age_sec': 1800},
    )
    refreshed = storage.get_order(order_id=order['id'])
    assert managed[0]['action'] == 'manual_review'
    assert refreshed['status'] == 'manual_review'


def test_unknown_order_manual_review_payload_records_both_ages(monkeypatch):
    now = datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)
    created_ts = (now - timedelta(hours=2)).isoformat()
    recent_refresh_ts = (now - timedelta(seconds=30)).isoformat()
    storage.create_market('REC12B', status='open')
    order = storage.create_order('recover-12b', 'REC12B', 'TOKR12B', 'YES', 'buy', 2.0, 0.2, 'unknown', created_ts)
    storage.create_reservation(order['id'], 'REC12B', 'TOKR12B', 'YES', 'exposure', 0.4, created_ts)
    storage.append_order_event(order['id'], 'submit_ambiguous', old_status='pending_submit', new_status='unknown', response={'http_status': 500, 'error': 'could not run the execution'}, ts=created_ts)
    storage.update_order(order['id'], updated_ts=recent_refresh_ts)

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'unknown'})

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'max_open_age_sec': 0, 'unknown_review_age_sec': 1800},
    )
    assert managed[0]['action'] == 'manual_review'
    refreshed = storage.get_order(order_id=order['id'])
    payload = json.loads(refreshed['raw_response_json'])
    assert payload['stale_refresh_age_sec'] < 1800
    assert payload['unknown_age_sec'] >= 1800
    assert payload['unknown_since_ts'] == created_ts


def test_order_9662_regression_provisional_client_lookup_eventually_escalates(monkeypatch):
    submit_ts = datetime(2026, 4, 4, 8, 0, tzinfo=timezone.utc)
    now = submit_ts + timedelta(hours=1)
    storage.create_market('REC9662', status='open')
    order = storage.create_order(
        'recover-9662',
        'REC9662',
        'TOK9662',
        'YES',
        'buy',
        1.0,
        0.45,
        'unknown',
        submit_ts.isoformat(),
        raw_response={'status': 'unknown', 'http_status': 500, 'error': 'could not run the execution'},
    )
    storage.create_reservation(order['id'], 'REC9662', 'TOK9662', 'YES', 'exposure', 0.45, submit_ts.isoformat())
    storage.append_order_event(
        order['id'],
        'submit_ambiguous',
        old_status='pending_submit',
        new_status='unknown',
        response={'status': 'unknown', 'http_status': 500, 'error': 'could not run the execution'},
        ts=submit_ts.isoformat(),
    )

    refresh_points = [submit_ts + timedelta(minutes=5), submit_ts + timedelta(minutes=25), submit_ts + timedelta(minutes=50)]
    provisional_response = {
        'status': 'unknown',
        'reason': 'client_order_id lookup is provisional',
        'http_status': 200,
    }

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: dict(provisional_response))

    for refresh_point in refresh_points:
        storage.update_order(order['id'], updated_ts=refresh_point.isoformat())
        storage.append_order_event(
            order['id'],
            'refresh_after',
            old_status='unknown',
            new_status='unknown',
            response=provisional_response,
            ts=refresh_point.isoformat(),
        )

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'max_open_age_sec': 0, 'unknown_review_age_sec': 1800},
    )
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]

    assert managed[0]['action'] == 'manual_review'
    assert refreshed['status'] == 'manual_review'
    assert refreshed['venue_order_id'] is None
    assert refreshed['tx_hash'] is None
    assert reservation['status'] == 'released'


def test_stale_unknown_sell_without_venue_id_escalates_to_manual_review(monkeypatch):
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(hours=2)).isoformat()
    storage.create_market('REC12S', status='open')
    storage.create_open_lot('REC12S', 'TOKR12S', 'NO', 3.0, 0.2, ts)
    order = storage.create_order('recover-12-sell', 'REC12S', 'TOKR12S', 'NO', 'sell', 2.0, 0.25, 'unknown', ts)
    storage.create_reservation(order['id'], 'REC12S', 'TOKR12S', 'NO', 'inventory', 2.0, ts)
    storage.update_order(order['id'], updated_ts=ts)

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'unknown'})

    managed = execution.manage_stale_orders(
        now_ts=now.isoformat(),
        dry_run=False,
        thresholds={'unknown_review_age_sec': 1800},
    )
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]

    assert managed[0]['action'] == 'manual_review'
    assert refreshed['status'] == 'manual_review'
    assert reservation['status'] == 'released'


def test_manual_review_is_terminal_and_releases_reservations():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC13', status='open')
    order = storage.create_order('recover-13', 'REC13', 'TOKR13', 'YES', 'buy', 1.0, 0.2, 'unknown', ts)
    storage.create_reservation(order['id'], 'REC13', 'TOKR13', 'YES', 'exposure', 0.2, ts)

    transitioned = storage.transition_order_state(order['id'], 'manual_review', reason='manual_intervention_required', ts=ts)
    reservation = storage.get_order_reservations(order['id'])[0]

    assert transitioned['status'] == 'manual_review'
    assert reservation['status'] == 'released'
    assert storage.can_transition_order_state('manual_review', 'filled') is False


def test_repair_helper_skips_rows_when_receipt_exists(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    tx_hash = '0x' + '1' * 64
    storage.create_market('REC10', status='open')
    order = storage.create_order(
        'recover-10',
        'REC10',
        'TOKR10',
        'YES',
        'buy',
        2.0,
        0.25,
        'unknown',
        ts,
        tx_hash=tx_hash,
        raw_response={'status': 'unknown', 'orderId': tx_hash},
    )

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: {'txHash': tx, 'logs': []} if tx == tx_hash else None)

    applied = repair_misclassified_clob_order_ids.repair_misclassified_clob_order_ids(dry_run=False)
    unchanged = storage.get_order(order_id=order['id'])

    assert applied['repaired'] == 0
    assert unchanged['tx_hash'] == tx_hash
    assert unchanged['venue_order_id'] is None
