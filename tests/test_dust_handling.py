import os
from datetime import datetime, timezone
from pathlib import Path

from src import execution, run_bot, storage


def setup_function(fn):
    os.environ['BOT_DB_PATH'] = str(Path('/tmp') / f'btc_1h_test_dust_{fn.__name__}.db')
    os.environ['DUST_QTY_THRESHOLD'] = '0.1'
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_tiny_order_remainder_becomes_dust_finalized_and_is_ignored_by_stale_order_management(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DUST1', status='open', title='Dust One')
    order = storage.create_order('dust-1', 'DUST1', 'TOKD1', 'YES', 'sell', 5.0023, 0.55, 'partially_filled', ts, venue_order_id='OID-DUST1')
    storage.update_order(order['id'], filled_qty=5.0, remaining_qty=0.0023, updated_ts=ts)
    storage.create_reservation(order['id'], 'DUST1', 'TOKD1', 'YES', 'inventory', 0.0023, ts)

    review = run_bot.collect_startup_dust_review(refresh_markets=False)
    results = run_bot.apply_startup_dust_action(review, action='finalize_all', ts=ts)
    refreshed = storage.get_order(order_id=order['id'])

    monkeypatch.setattr(execution, 'get_order_status', lambda **kwargs: (_ for _ in ()).throw(AssertionError('dust order should not be refreshed')))
    managed = execution.manage_stale_orders(now_ts=ts, dry_run=False)

    assert any(item['item_key'] == f'order:{order["id"]}' for item in results)
    assert refreshed['status'] == 'dust_finalized'
    assert managed == []


def test_dust_finalization_releases_reservation():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DUST2', status='open')
    order = storage.create_order('dust-2', 'DUST2', 'TOKD2', 'YES', 'buy', 1.0, 0.2, 'partially_filled', ts)
    storage.update_order(order['id'], filled_qty=0.9977, remaining_qty=0.0023, updated_ts=ts)
    storage.create_reservation(order['id'], 'DUST2', 'TOKD2', 'YES', 'exposure', 0.00046, ts)

    storage.move_order_to_dust_status(order['id'], dust_status='dust_finalized', reason='unit_test', ts=ts)

    reservations = storage.get_order_reservations(order['id'])
    events = storage.list_order_events(order['id'])

    assert reservations[0]['status'] == 'released'
    assert any(event['event_type'] == 'dust_finalized' for event in events)


def test_startup_dust_report_tags_winning_losing_and_unknown():
    ts = datetime.now(timezone.utc).isoformat()
    storage.upsert_market('MWIN', status='resolved', winning_outcome='YES', title='Win Market')
    storage.upsert_market('MLOSS', status='resolved', winning_outcome='NO', title='Loss Market')
    storage.upsert_market('MUNK', status='open', title='Unknown Market')

    order_win = storage.create_order('dust-win', 'MWIN', 'TOKW', 'YES', 'sell', 1.0, 0.5, 'partially_filled', ts)
    storage.update_order(order_win['id'], filled_qty=0.95, remaining_qty=0.05, updated_ts=ts)
    order_loss = storage.create_order('dust-loss', 'MLOSS', 'TOKL', 'YES', 'sell', 1.0, 0.5, 'partially_filled', ts)
    storage.update_order(order_loss['id'], filled_qty=0.95, remaining_qty=0.05, updated_ts=ts)
    storage.create_open_lot('MUNK', 'TOKU', 'YES', 0.05, 0.5, ts)

    review = run_bot.collect_startup_dust_review(refresh_markets=False)
    items = {item['market_id']: item for item in review['items'] if item['market_id'] in {'MWIN', 'MLOSS', 'MUNK'}}

    assert items['MWIN']['winning_tag'] == 'winning'
    assert items['MLOSS']['winning_tag'] == 'losing'
    assert items['MUNK']['winning_tag'] == 'unknown'


def test_restore_path_returns_items_to_active_consideration():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DUST4', status='open', title='Restore Market')
    order = storage.create_order('dust-restore', 'DUST4', 'TOKD4', 'YES', 'sell', 1.0, 0.5, 'partially_filled', ts)
    storage.update_order(order['id'], filled_qty=0.95, remaining_qty=0.05, updated_ts=ts)
    storage.create_reservation(order['id'], 'DUST4', 'TOKD4', 'YES', 'inventory', 0.05, ts)
    storage.create_open_lot('DUST4', 'TOKD4', 'YES', 0.05, 0.4, ts)

    review = run_bot.collect_startup_dust_review(refresh_markets=False)
    run_bot.apply_startup_dust_action(review, action='keep_all_dormant', ts=ts)

    dormant_review = run_bot.collect_startup_dust_review(refresh_markets=False)
    run_bot.apply_startup_dust_action(dormant_review, action='restore_selected', targets=[order['id'], 'DUST4'], ts=ts)

    restored_order = storage.get_order(order_id=order['id'])
    open_lots = storage.get_open_lots(market_id='DUST4')
    active_orders = storage.get_open_orders(market_id='DUST4')

    assert restored_order['status'] == 'unknown'
    assert any(item['id'] == order['id'] for item in active_orders)
    assert abs(sum(float(lot['qty']) for lot in open_lots) - 0.05) < 1e-12


def test_startup_dust_review_excludes_terminal_rejected_orders():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DUST5', status='open', title='Rejected Dust Market')
    rejected = storage.create_order('dust-rejected', 'DUST5', 'TOKD5', 'YES', 'buy', 0.05, 0.2, 'rejected', ts)
    storage.update_order(rejected['id'], remaining_qty=0.05, updated_ts=ts)

    review = run_bot.collect_startup_dust_review(refresh_markets=False)
    item_keys = {item['item_key'] for item in review['items']}

    assert f'order:{rejected["id"]}' not in item_keys
