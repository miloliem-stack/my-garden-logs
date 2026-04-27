import os
import sys
import json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import storage, execution, run_bot, polymarket_client


def setup_db():
    if storage.get_db_path().exists():
        try:
            storage.get_db_path().unlink()
        except Exception:
            pass
    storage.ensure_db()


def test_live_buy_creates_inventory(monkeypatch):
    setup_db()
    storage.create_market('MK')
    # mock place_limit_order to simulate filled buy
    def mock_place_limit_order(token_id, side, qty, price, post_only=True, dry_run=False):
        return {'status': 'ok', 'txHash': '0xBUYTX', 'filledQuantity': qty}
    monkeypatch.setattr(execution, 'place_limit_order', mock_place_limit_order)

    resp = execution.place_live_limit('buy', 5.0, 0.2, 'tkn1', 'MK', 'YES')
    assert resp.get('status') == 'ok'
    # inventory should show 5.0 for market MK
    have = storage.get_total_qty_by_token('tkn1', market_id='MK')
    assert have == 5.0
    # fills should record tx hash
    fills = storage.get_unreconciled_fills()
    assert any(f['tx_hash'] == '0xBUYTX' for f in fills)
    orders = storage.get_open_orders(statuses=['filled'])
    assert len(orders) == 1
    assert orders[0]['filled_qty'] == 5.0


def test_live_sell_consumes_inventory(monkeypatch):
    setup_db()
    storage.create_market('MK2')
    ts = datetime.now(timezone.utc).isoformat()
    # create initial inventory: open lot 10
    storage.create_open_lot(market_id='MK2', token_id='tkns', outcome_side='NO', qty=10.0, avg_price=0.1, ts=ts)

    # mock place_limit_order to simulate filled sell of 4
    def mock_place_limit_order(token_id, side, qty, price, post_only=True, dry_run=False):
        return {'status': 'ok', 'txHash': '0xSELLTX', 'filledQuantity': 4.0}
    monkeypatch.setattr(execution, 'place_limit_order', mock_place_limit_order)

    resp = execution.place_live_limit('sell', 4.0, 0.15, 'tkns', 'MK2', 'NO')
    assert resp.get('status') == 'ok'
    # inventory should be reduced to 6.0
    have_after = storage.get_total_qty_by_token('tkns', market_id='MK2')
    assert abs(have_after - 6.0) < 1e-9
    # fills should include the sell tx
    fills = storage.get_unreconciled_fills()
    assert any(f['tx_hash'] == '0xSELLTX' for f in fills)


def test_reserve_inventory_before_sell_and_release_on_cancel(monkeypatch):
    setup_db()
    storage.create_market('RSV', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('RSV', 'TOK', 'YES', 10.0, 0.1, ts)

    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 0.0, 'orderId': 'VENUE1'})
    resp = execution.place_live_limit('sell', 6.0, 0.2, 'TOK', 'RSV', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'submitted'
    assert storage.get_reserved_qty('RSV', 'TOK', 'YES') == 6.0
    assert storage.get_available_qty('RSV', 'TOK', 'YES') == 4.0

    execution.process_order_update(order['id'], {'status': 'canceled', 'filledQuantity': 0.0, 'orderId': 'VENUE1'})
    order = storage.get_order(order_id=order['id'])
    assert order['status'] == 'canceled'
    assert storage.get_reserved_qty('RSV', 'TOK', 'YES') == 0.0
    assert storage.get_available_qty('RSV', 'TOK', 'YES') == 10.0


def test_partial_fill_applies_only_incremental_delta(monkeypatch):
    setup_db()
    storage.create_market('PART', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 2.0, 'orderId': 'VENUE2', 'txHash': '0xPART'})

    resp = execution.place_live_limit('buy', 5.0, 0.2, 'TOKB', 'PART', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'partially_filled'
    assert storage.get_total_qty_by_token('TOKB', market_id='PART') == 2.0
    consistency = storage.assert_order_reservation_consistency(order['id'])
    assert consistency['ok'] is True
    assert abs(consistency['actual_exposure'] - 0.6) < 1e-9

    execution.process_order_update(order['id'], {'status': 'ok', 'filledQuantity': 2.0, 'orderId': 'VENUE2', 'txHash': '0xPART'})
    assert storage.get_total_qty_by_token('TOKB', market_id='PART') == 2.0

    execution.process_order_update(order['id'], {'status': 'filled', 'filledQuantity': 5.0, 'orderId': 'VENUE2', 'txHash': '0xPART'})
    order = storage.get_order(order_id=order['id'])
    assert order['status'] == 'filled'
    assert order['filled_qty'] == 5.0
    assert storage.get_total_qty_by_token('TOKB', market_id='PART') == 5.0


def test_full_fill_clears_sell_reservation(monkeypatch):
    setup_db()
    storage.create_market('FULL', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('FULL', 'TOKS', 'NO', 8.0, 0.1, ts)
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 3.0, 'orderId': 'VENUE3', 'txHash': '0xFULL'})

    resp = execution.place_live_limit('sell', 3.0, 0.2, 'TOKS', 'FULL', 'NO')
    order = storage.get_order(order_id=resp['order_id'])
    assert order['status'] == 'filled'
    assert storage.get_reserved_qty('FULL', 'TOKS', 'NO') == 0.0
    assert storage.get_available_qty('FULL', 'TOKS', 'NO') == 5.0


def test_partial_sell_fill_keeps_residual_inventory_and_records_realized_pnl(monkeypatch):
    setup_db()
    storage.create_market('EXITPART', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('EXITPART', 'TOKX', 'YES', 5.0, 0.20, ts)
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 3.0, 'orderId': 'VENUEX', 'txHash': '0xEXITPART'})

    resp = execution.place_live_limit('sell', 5.0, 0.35, 'TOKX', 'EXITPART', 'YES')
    order = storage.get_order(order_id=resp['order_id'])

    assert order['status'] == 'partially_filled'
    assert storage.get_total_qty_by_token('TOKX', market_id='EXITPART') == 2.0
    assert storage.get_reserved_qty('EXITPART', 'TOKX', 'YES') == 2.0

    conn = storage.sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT qty, price, extra_json FROM fills WHERE tx_hash = ? AND kind = 'sell'", ('0xEXITPART',))
    qty, price, extra_json = cur.fetchone()
    conn.close()
    extra = json.loads(extra_json)

    assert qty == -3.0
    assert price == 0.35
    assert extra['entry_price'] == 0.20
    assert extra['exit_price'] == 0.35
    assert abs(extra['profit_per_share'] - 0.15) < 1e-9
    assert abs(extra['profit_total'] - 0.45) < 1e-9


def test_failed_submit_does_not_create_fake_inventory_changes(monkeypatch):
    setup_db()
    storage.create_market('FAIL', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'error', 'reason': 'rejected'})
    resp = execution.place_live_limit('buy', 4.0, 0.2, 'TOKF', 'FAIL', 'YES')
    assert resp['status'] == 'error'
    assert storage.get_total_qty_by_token('TOKF', market_id='FAIL') == 0.0
    orders = storage.get_open_orders()
    assert len(orders) == 1
    assert orders[0]['status'] == 'unknown'


def test_duplicate_order_update_does_not_double_apply(monkeypatch):
    setup_db()
    storage.create_market('DUP', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 1.0, 'orderId': 'VENUE4', 'txHash': '0xDUP'})
    resp = execution.place_live_limit('buy', 1.0, 0.25, 'TOKD', 'DUP', 'YES')
    order_id = resp['order_id']
    execution.process_order_update(order_id, {'status': 'filled', 'filledQuantity': 1.0, 'orderId': 'VENUE4', 'txHash': '0xDUP'})
    execution.process_order_update(order_id, {'status': 'filled', 'filledQuantity': 1.0, 'orderId': 'VENUE4', 'txHash': '0xDUP'})
    assert storage.get_total_qty_by_token('TOKD', market_id='DUP') == 1.0
    assert len(storage.get_order_fill_events(order_id)) == 1


def test_sell_uses_available_qty_not_raw_open_qty(monkeypatch):
    setup_db()
    storage.create_market('AVL', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('AVL', 'TOKA', 'YES', 5.0, 0.1, ts)
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 0.0, 'orderId': 'VENUE5'})
    execution.place_live_limit('sell', 4.0, 0.2, 'TOKA', 'AVL', 'YES')
    try:
        execution.place_live_limit('sell', 2.0, 0.2, 'TOKA', 'AVL', 'YES')
        assert False, 'expected reservation-aware inventory check to fail'
    except RuntimeError as exc:
        assert 'Not enough available inventory' in str(exc)


def test_cross_market_reservations_do_not_interfere(monkeypatch):
    setup_db()
    ts = datetime.now(timezone.utc).isoformat()
    for market_id in ('MKA', 'MKB'):
        storage.create_market(market_id, status='open')
        storage.create_open_lot(market_id, 'TOK', 'YES', 5.0, 0.1, ts)
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': 0.0, 'orderId': 'VENUE6'})
    execution.place_live_limit('sell', 4.0, 0.2, 'TOK', 'MKA', 'YES')
    assert storage.get_available_qty('MKA', 'TOK', 'YES') == 1.0
    assert storage.get_available_qty('MKB', 'TOK', 'YES') == 5.0


def test_dirty_start_recovery_preserves_inventory_and_normalizes_reservations():
    setup_db()
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('REC', status='open')
    storage.create_open_lot('REC', 'TOKR', 'YES', 10.0, 0.1, ts)
    sell_order = storage.create_order('recover-sell', 'REC', 'TOKR', 'YES', 'sell', 6.0, 0.2, 'open', ts)
    storage.create_reservation(sell_order['id'], 'REC', 'TOKR', 'YES', 'inventory', 9.0, ts)
    filled_order = storage.create_order('recover-filled', 'REC', 'TOKR', 'YES', 'sell', 1.0, 0.2, 'filled', ts)
    storage.create_reservation(filled_order['id'], 'REC', 'TOKR', 'YES', 'inventory', 1.0, ts)

    report = run_bot.recover_dirty_start()

    assert storage.get_total_qty_by_token('TOKR', market_id='REC') == 10.0
    assert storage.get_reserved_qty('REC', 'TOKR', 'YES') == 6.0
    assert any(item['order_id'] == sell_order['id'] for item in report['reservation_repairs'])
    assert storage.get_order_reservations(filled_order['id'], active_only=True) == []


def test_legal_and_illegal_order_state_transitions():
    setup_db()
    ts = datetime.now(timezone.utc).isoformat()
    order = storage.create_order('transition-1', 'MT', 'TOK', 'YES', 'buy', 1.0, 0.2, 'pending_submit', ts)
    storage.transition_order_state(order['id'], 'submitted', ts=ts)
    storage.transition_order_state(order['id'], 'open', ts=ts)
    try:
        storage.transition_order_state(order['id'], 'pending_submit', ts=ts)
        assert False, 'expected illegal backward transition to fail'
    except RuntimeError:
        pass


def test_normalized_place_and_cancel_parsing():
    place = polymarket_client.normalize_client_response({'status': 'accepted', 'orderId': 'OID1', 'filledQuantity': 0.0})
    cancel = polymarket_client.normalize_client_response({'state': 'canceled', 'id': 'OID1'})
    assert place['order_id'] == 'OID1'
    assert place['status'] == 'accepted'
    assert cancel['order_id'] == 'OID1'
    assert cancel['status'] == 'canceled'


def test_stale_pending_submit_order_is_refreshed_before_replacement(monkeypatch):
    setup_db()
    storage.create_market('STALEP', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'submitted', 'orderId': 'OIDP', 'filled_qty': 0.0})
    resp = execution.place_live_limit('buy', 1.0, 0.2, 'TOKP', 'STALEP', 'YES', client_order_id='retry-me')
    order = storage.get_order(order_id=resp['order_id'])
    storage.update_order(order['id'], updated_ts=(datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat())
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'open', 'order_id': 'OIDP', 'filled_qty': 0.0})
    managed = execution.manage_stale_orders()
    assert managed[0]['action'] == 'refreshed'
    assert storage.get_order(order_id=order['id'])['status'] in ('submitted', 'open', 'unknown')


def test_unknown_submit_outcome_blocks_duplicate_resubmission(monkeypatch):
    setup_db()
    storage.create_market('UNK', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'unknown'})
    first = execution.place_live_limit('buy', 1.0, 0.2, 'TOKU', 'UNK', 'YES', client_order_id='dup-check')
    second = execution.place_live_limit('buy', 1.0, 0.2, 'TOKU', 'UNK', 'YES', client_order_id='dup-check')
    assert first['order_id'] == second['order_id']
    assert second['status'] == 'deduped_existing_order'


def test_tx_like_submit_id_is_stored_as_venue_order_id_not_tx_hash(monkeypatch):
    setup_db()
    storage.create_market('TXID', status='open')
    venue_order_id = '0x' + 'a' * 64
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'unknown', 'orderId': venue_order_id})

    resp = execution.place_live_limit('buy', 1.5, 0.2, 'TOKT', 'TXID', 'YES')
    order = storage.get_order(order_id=resp['order_id'])

    assert order['status'] == 'unknown'
    assert order['tx_hash'] is None
    assert order['venue_order_id'] == venue_order_id


def test_marketable_buy_below_one_dollar_is_skipped_locally(monkeypatch):
    setup_db()
    storage.create_market('MINBUY1', status='open')
    monkeypatch.setattr(execution, 'place_marketable_order', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('venue submit should not be called')))

    resp = execution.place_marketable_buy('TOKMIN1', 0.5, limit_price=0.98, dry_run=False, market_id='MINBUY1', outcome_side='YES')

    assert resp['status'] == 'skipped_below_min_market_buy_notional'
    assert resp['min_notional'] == 1.0
    assert abs(resp['submitted_notional'] - 0.49) < 1e-9
    assert resp['token_id'] == 'TOKMIN1'
    assert resp['side'] == 'buy'
    assert storage.get_open_orders() == []


def test_marketable_buy_at_one_dollar_is_allowed(monkeypatch):
    setup_db()
    storage.create_market('MINBUY2', status='open')
    monkeypatch.setattr(execution, 'place_marketable_order', lambda *args, **kwargs: {'status': 'ok', 'filledQuantity': args[2], 'txHash': '0xMINBUYOK'})

    resp = execution.place_marketable_buy('TOKMIN2', 1.02, limit_price=0.99, dry_run=False, market_id='MINBUY2', outcome_side='YES')

    assert resp['status'] == 'ok'
    assert storage.get_total_qty_by_token('TOKMIN2', market_id='MINBUY2') > 1.0


def test_marketable_buy_uses_fak_by_default(monkeypatch):
    setup_db()
    storage.create_market('MKTBUY', status='open')
    monkeypatch.delenv('MARKETABLE_ORDER_TYPE', raising=False)
    captured = {}

    def mock_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=False, client_order_id=None, compiled_intent=None):
        captured['order_type'] = order_type
        captured['compiled_intent'] = compiled_intent
        return {'status': 'accepted', 'orderId': 'OID-MKTBUY', 'filledQuantity': 0.0}

    monkeypatch.setattr(execution, 'place_marketable_order', mock_place_marketable_order)
    execution.place_marketable_buy('TOKMKTBUY', 3.0, limit_price=0.41, dry_run=False, market_id='MKTBUY', outcome_side='YES')

    assert captured['order_type'] == 'FAK'
    assert captured['compiled_intent']['order_type'] == 'FAK'
    assert captured['compiled_intent']['request_body']['orderType'] == 'FAK'


def test_marketable_sell_uses_fak_by_default(monkeypatch):
    setup_db()
    storage.create_market('MKTSELL', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('MKTSELL', 'TOKMKTSELL', 'NO', 5.0, 0.1, ts)
    monkeypatch.delenv('MARKETABLE_ORDER_TYPE', raising=False)
    captured = {}

    def mock_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=False, client_order_id=None, compiled_intent=None):
        captured['order_type'] = order_type
        captured['compiled_intent'] = compiled_intent
        return {'status': 'accepted', 'orderId': 'OID-MKTSELL', 'filledQuantity': 0.0}

    monkeypatch.setattr(execution, 'place_marketable_order', mock_place_marketable_order)
    execution.place_marketable_sell('TOKMKTSELL', 2.0, limit_price=0.3, dry_run=False, market_id='MKTSELL', outcome_side='NO')

    assert captured['order_type'] == 'FAK'
    assert captured['compiled_intent']['order_type'] == 'FAK'
    assert captured['compiled_intent']['request_body']['orderType'] == 'FAK'


def test_marketable_order_type_can_be_overridden_to_fok(monkeypatch):
    setup_db()
    storage.create_market('MKTFOK', status='open')
    monkeypatch.setenv('MARKETABLE_ORDER_TYPE', 'FOK')
    captured = {}

    def mock_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=False, client_order_id=None, compiled_intent=None):
        captured['order_type'] = order_type
        captured['compiled_intent'] = compiled_intent
        return {'status': 'accepted', 'orderId': 'OID-MKTFOK', 'filledQuantity': 0.0}

    monkeypatch.setattr(execution, 'place_marketable_order', mock_place_marketable_order)
    execution.place_marketable_buy('TOKMKTFOK', 3.0, limit_price=0.41, dry_run=False, market_id='MKTFOK', outcome_side='YES')

    assert captured['order_type'] == 'FOK'
    assert captured['compiled_intent']['order_type'] == 'FOK'
    assert captured['compiled_intent']['request_body']['orderType'] == 'FOK'


def test_stale_open_order_goes_through_conservative_cancel_path(monkeypatch):
    setup_db()
    storage.create_market('CAN', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_open_lot('CAN', 'TOKC', 'YES', 4.0, 0.1, ts)
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'open', 'orderId': 'OIDC', 'filled_qty': 0.0})
    resp = execution.place_live_limit('sell', 2.0, 0.2, 'TOKC', 'CAN', 'YES')
    order = storage.get_order(order_id=resp['order_id'])
    storage.update_order(order['id'], updated_ts=(datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat())
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'open', 'order_id': 'OIDC', 'filled_qty': 0.0})
    monkeypatch.setattr(execution, 'cancel_order', lambda **kwargs: {'status': 'canceled', 'order_id': 'OIDC', 'filled_qty': 0.0, 'remaining_qty': 2.0})
    managed = execution.manage_stale_orders()
    assert managed[0]['action'] == 'cancel_requested'
    assert storage.get_order(order_id=order['id'])['status'] == 'canceled'
    assert storage.get_reserved_qty('CAN', 'TOKC', 'YES') == 0.0


def test_marketable_partial_immediate_fill_does_not_leave_resting_order(monkeypatch):
    setup_db()
    storage.create_market('FAKPART', status='open')
    monkeypatch.delenv('MARKETABLE_ORDER_TYPE', raising=False)

    monkeypatch.setattr(
        execution,
        'place_marketable_order',
        lambda *args, **kwargs: {
            'status': 'partially_filled',
            'orderId': 'OID-FAKPART',
            'filledQuantity': 1.25,
            'remainingQuantity': max(0.0, args[2] - 1.25),
            'order_type': kwargs.get('order_type'),
            'marketable': True,
        },
    )

    resp = execution.place_marketable_buy('TOKFAKPART', 3.0, limit_price=0.41, dry_run=False, market_id='FAKPART', outcome_side='YES')
    order = storage.get_order(order_id=resp['order_id'])

    assert order['status'] == 'canceled'
    assert order['filled_qty'] == 1.25
    assert storage.get_open_orders(market_id='FAKPART') == []

    storage.update_order(order['id'], updated_ts=(datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat())
    assert execution.manage_stale_orders(now_ts=datetime.now(timezone.utc).isoformat(), dry_run=False) == []


def test_marketable_buy_residual_terminalizes_on_not_found_refresh(monkeypatch):
    setup_db()
    storage.create_market('FAKNULL', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    order = storage.create_order(
        'fak-null-1',
        'FAKNULL',
        'TOKFAKNULL',
        'YES',
        'buy',
        3.0,
        0.41,
        'partially_filled',
        ts,
        venue_order_id='OID-FAKNULL',
        raw_response={'order_type': 'FAK', 'marketable': True, 'venue_intent_mode': 'amount'},
    )
    storage.create_reservation(order['id'], 'FAKNULL', 'TOKFAKNULL', 'YES', 'exposure', (3.0 - 1.25) * 0.41, ts)
    storage.apply_incremental_order_fill(order['id'], 1.25, fill_ts=ts, tx_hash='0xFAKNULL', price=0.41, raw={'status': 'partial_fill_bootstrap'})

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'status': 'not_found_on_venue',
            'orderId': kwargs.get('order_id'),
        },
    )

    out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    reservation = storage.get_order_reservations(order['id'])[0]
    payload = json.loads(refreshed['raw_response_json'])

    assert out['order']['status'] == 'not_found_on_venue'
    assert refreshed['status'] == 'not_found_on_venue'
    assert refreshed['filled_qty'] == 1.25
    assert refreshed['remaining_qty'] == 1.75
    assert reservation['status'] == 'released'
    assert storage.get_total_qty_by_token('TOKFAKNULL', market_id='FAKNULL') == 1.25
    assert payload['residual_terminalized'] is True
    assert payload['residual_terminalization_reason'] == 'marketable_buy_dead_residual'
    assert payload['ledger_fill_preserved'] is True
    assert payload['ledger_fill_preserved_qty'] == 1.25
    assert storage.get_open_orders(market_id='FAKNULL') == []


def test_marketable_buy_matched_partial_terminalizes_without_lingering(monkeypatch):
    setup_db()
    storage.create_market('FAKMATCH', status='open')
    monkeypatch.delenv('MARKETABLE_ORDER_TYPE', raising=False)

    monkeypatch.setattr(
        execution,
        'place_marketable_order',
        lambda *args, **kwargs: {
            'status': 'matched',
            'orderId': 'OID-FAKMATCH',
            'filledQuantity': 1.5,
            'remainingQuantity': max(0.0, args[2] - 1.5),
            'order_type': kwargs.get('order_type'),
            'marketable': True,
        },
    )

    resp = execution.place_marketable_buy('TOKFAKMATCH', 3.0, limit_price=0.41, dry_run=False, market_id='FAKMATCH', outcome_side='YES')
    order = storage.get_order(order_id=resp['order_id'])
    payload = json.loads(order['raw_response_json'])

    assert order['status'] == 'canceled'
    assert order['filled_qty'] == 1.5
    assert order['remaining_qty'] > 0
    assert storage.get_total_qty_by_token('TOKFAKMATCH', market_id='FAKMATCH') == 1.5
    assert payload['residual_terminalized'] is True
    assert payload['residual_terminalization_reason'] == 'marketable_buy_dead_residual'
    assert payload['ledger_fill_preserved'] is True
    assert payload['ledger_fill_preserved_qty'] == 1.5
    assert storage.get_open_orders(market_id='FAKMATCH') == []


def test_non_marketable_partial_fill_can_remain_partially_filled_on_matched_status():
    setup_db()
    storage.create_market('RESTPART', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    order = storage.create_order('rest-part-1', 'RESTPART', 'TOKREST', 'YES', 'buy', 4.0, 0.3, 'submitted', ts, venue_order_id='OID-REST')
    storage.create_reservation(order['id'], 'RESTPART', 'TOKREST', 'YES', 'exposure', 1.2, ts)

    out = execution.process_order_update(
        order['id'],
        {
            'status': 'matched',
            'orderId': 'OID-REST',
            'filledQuantity': 1.0,
            'remainingQuantity': 3.0,
        },
    )
    refreshed = storage.get_order(order_id=order['id'])

    assert out['order']['status'] == 'partially_filled'
    assert refreshed['status'] == 'partially_filled'
    assert storage.get_total_qty_by_token('TOKREST', market_id='RESTPART') == 1.0
    assert storage.get_open_orders(market_id='RESTPART')[0]['id'] == order['id']


def test_regression_ledger_fill_preserved_not_found_residual_does_not_stay_partially_filled(monkeypatch):
    setup_db()
    storage.create_market('RESREG', status='open')
    ts = datetime.now(timezone.utc).isoformat()
    order = storage.create_order(
        'res-reg-1',
        'RESREG',
        'TOKRESREG',
        'YES',
        'buy',
        2.0,
        0.4,
        'partially_filled',
        ts,
        venue_order_id='OID-RESREG',
        raw_response={'policy': 'marketable_buy_entry', 'venue_intent_mode': 'amount'},
    )
    storage.create_reservation(order['id'], 'RESREG', 'TOKRESREG', 'YES', 'exposure', 0.4, ts)
    storage.apply_incremental_order_fill(order['id'], 1.0, fill_ts=ts, tx_hash='0xRESREG', price=0.4, raw={'status': 'partial_fill_bootstrap'})

    monkeypatch.setattr(execution.polymarket_client, 'get_user_trades', lambda *args, **kwargs: {'status': 'ok', 'trades': []})
    monkeypatch.setattr(
        execution,
        'get_order_status',
        lambda **kwargs: {
            'status': 'not_found_on_venue',
            'result': None,
            'ledger_fill_preserved': True,
            'ledger_fill_preserved_qty': 1.0,
        },
    )

    refreshed_out = execution.refresh_order_status(order['id'], dry_run=False)
    refreshed = storage.get_order(order_id=order['id'])
    payload = json.loads(refreshed['raw_response_json'])

    assert refreshed_out['order']['status'] == 'not_found_on_venue'
    assert refreshed['status'] == 'not_found_on_venue'
    assert refreshed['filled_qty'] == 1.0
    assert payload['ledger_fill_preserved'] is True
    assert payload['ledger_fill_preserved_qty'] == 1.0
    assert payload['residual_terminalized'] is True


def test_restart_recovery_preserves_ambiguous_active_orders():
    setup_db()
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('AMB', status='open')
    order = storage.create_order('amb-order', 'AMB', 'TOKA', 'YES', 'buy', 2.0, 0.2, 'unknown', ts)
    storage.create_reservation(order['id'], 'AMB', 'TOKA', 'YES', 'exposure', 0.4, ts)
    report = run_bot.recover_dirty_start()
    recovered = storage.get_order(order_id=order['id'])
    assert recovered['status'] == 'unknown'
    assert any(item['client_order_id'] == 'amb-order' for item in report['stale_orders_for_review'])


def test_client_order_id_generation_is_traceable():
    cid = execution._new_client_order_id('live', market_id='MID', side='buy', outcome_side='YES')
    assert cid.startswith('live-')
    assert 'MID' in cid
    assert 'buy' in cid
    assert 'YES' in cid


def test_active_order_diagnostics_reflect_remaining_qty_and_state():
    setup_db()
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DIAG', status='open')
    order = storage.create_order('diag-1', 'DIAG', 'TOKD', 'YES', 'buy', 5.0, 0.2, 'partially_filled', ts)
    storage.update_order(order['id'], filled_qty=2.0, remaining_qty=3.0, updated_ts=ts)
    storage.create_reservation(order['id'], 'DIAG', 'TOKD', 'YES', 'exposure', 0.6, ts)
    report = storage.get_active_order_diagnostics(now_ts=ts)
    entry = report['orders'][0]
    assert entry['remaining_qty'] == 3.0
    assert entry['state'] == 'partially_filled'
    assert entry['reservation_amount'] == 0.6


def test_marketable_buy_uses_quantized_qty_and_exposure(monkeypatch):
    setup_db()
    storage.create_market('QB', status='open')

    captured = {}

    def mock_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=False, client_order_id=None, compiled_intent=None):
        captured['qty'] = qty
        captured['limit_price'] = limit_price
        captured['compiled_intent'] = compiled_intent
        return {
            'status': 'accepted',
            'orderId': 'OID-QB',
            'filledQuantity': 0.0,
            'clientOrderId': client_order_id,
            'quantized_qty': qty,
            'quantized_notional': 1.69,
            'raw_requested_qty': 3.141592,
        }

    monkeypatch.setattr(execution, 'place_marketable_order', mock_place_marketable_order)
    resp = execution.place_marketable_buy('TOKQB', 3.141592, limit_price=0.53827, dry_run=False, market_id='QB', outcome_side='YES')
    order = storage.get_order(order_id=resp['order_id'])
    assert captured['qty'] == 3.1396
    assert captured['compiled_intent']['venue_intent_mode'] == 'amount'
    assert captured['compiled_intent']['request_body']['amount'] == 1.69
    assert order['requested_qty'] == 3.1396
    assert abs(storage.get_inflight_exposure('QB') - 1.69) < 1e-9
    assert resp['submitted_qty'] == 3.1396
    assert resp['submitted_notional'] == 1.69
    assert resp['submitted_amount'] == 1.69


def test_marketable_buy_skips_too_small_quantized_order():
    setup_db()
    storage.create_market('QS', status='open')
    resp = execution.place_marketable_buy('TOKQS', 0.00011, limit_price=0.11, dry_run=False, market_id='QS', outcome_side='YES')
    assert resp['status'] == 'skipped_below_min_market_buy_notional'
    assert resp['reason'] == 'skipped_below_min_market_buy_notional'
    assert storage.get_open_orders(market_id='QS') == []


def test_marketable_buy_storage_and_events_match_submitted_venue_values(monkeypatch):
    setup_db()
    storage.create_market('QW', status='open')

    def mock_place_marketable_order(token_id, side, qty, limit_price=None, order_type='FAK', dry_run=False, client_order_id=None, compiled_intent=None):
        return {
            'status': 'accepted',
            'orderId': 'OID-QW',
            'filledQuantity': 0.0,
            'clientOrderId': client_order_id,
            'raw_probe': {'request_body': dict(compiled_intent['request_body'])},
        }

    monkeypatch.setattr(execution, 'place_marketable_order', mock_place_marketable_order)
    resp = execution.place_marketable_buy('TOKQW', 2.5, limit_price=0.41, dry_run=False, market_id='QW', outcome_side='YES')
    order = storage.get_order(order_id=resp['order_id'])
    raw_response = json.loads(order['raw_response_json'])
    events = storage.list_order_events(order['id'])
    submit_before = next(event for event in events if event['event_type'] == 'submit_before')

    assert order['requested_qty'] == 2.4878
    assert abs(storage.get_inflight_exposure('QW') - 1.02) < 1e-9
    assert submit_before['request']['venue_request_body']['amount'] == 1.02
    assert submit_before['request']['submitted_notional'] == 1.02
    assert raw_response['raw_probe']['request_body']['amount'] == 1.02
    assert raw_response['submitted_notional'] == 1.02
    assert raw_response['submitted_qty'] == 2.4878


def test_marketable_buy_hash_like_submit_id_is_stored_as_venue_order_id(monkeypatch):
    setup_db()
    storage.create_market('QHASH', status='open')
    venue_order_id = '0x' + '2' * 64

    monkeypatch.setattr(
        execution,
        'place_marketable_order',
        lambda *args, **kwargs: {
            'status': 'unknown',
            'orderId': venue_order_id,
            'clientOrderId': kwargs.get('client_order_id'),
        },
    )

    resp = execution.place_marketable_buy('TOKQHASH', 2.0, limit_price=0.51, dry_run=False, market_id='QHASH', outcome_side='YES')
    order = storage.get_order(order_id=resp['order_id'])

    assert order['status'] == 'unknown'
    assert order['venue_order_id'] == venue_order_id
    assert order['tx_hash'] is None


def test_stale_order_management_does_not_cancel_fresh_active_orders(monkeypatch):
    setup_db()
    storage.create_market('FRESH', status='open')
    monkeypatch.setattr(execution, 'place_limit_order', lambda *args, **kwargs: {'status': 'submitted', 'filled_qty': 0.0, 'orderId': 'OIDF'})
    resp = execution.place_live_limit('buy', 1.0, 0.2, 'TOKFRESH', 'FRESH', 'YES')
    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: {'status': 'submitted', 'order_id': 'OIDF', 'filled_qty': 0.0})
    managed = execution.manage_stale_orders(now_ts=datetime.now(timezone.utc).isoformat())
    assert managed == []
    assert storage.get_order(order_id=resp['order_id'])['status'] == 'submitted'
