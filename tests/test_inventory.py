import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import storage
import os
import sqlite3
from datetime import datetime, timezone


def setup_function(fn):
    os.environ['BOT_DB_PATH'] = str(Path('/tmp') / f'btc_1h_inventory_{fn.__name__}.db')
    try:
        os.remove(storage.get_db_path())
    except FileNotFoundError:
        pass
    storage.ensure_db()


def test_market_scoped_buys_and_sells_fifo():
    ts = datetime.now(timezone.utc).isoformat()
    # create market A
    storage.create_market('MKT_A', slug='mkt-a', status='open')
    # buy YES token twice
    storage.insert_fill('MKT_A', 'T_YES_A', 'YES', 10.0, 0.6, ts, tx_hash='tx1', kind='buy')
    storage.create_open_lot('MKT_A', 'T_YES_A', 'YES', 10.0, 0.6, ts, tx_hash='tx1')
    storage.insert_fill('MKT_A', 'T_YES_A', 'YES', 5.0, 0.65, ts, tx_hash='tx2', kind='buy')
    storage.create_open_lot('MKT_A', 'T_YES_A', 'YES', 5.0, 0.65, ts, tx_hash='tx2')

    bal = storage.get_pair_balance('MKT_A')
    assert bal['YES'] == 15.0 and bal['NO'] == 0.0

    # sell partial 12 -> consumes FIFO: first 10 then 2 from second lot
    consumed = storage.consume_open_lots_fifo('MKT_A', 'T_YES_A', 'YES', 12.0, consume_tx='selltx', ts=ts)
    assert consumed == 12.0
    bal2 = storage.get_pair_balance('MKT_A')
    assert abs(bal2['YES'] - 3.0) < 1e-9


def test_market_isolation_and_closed_unresolved():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_B', slug='mkt-b', status='open')
    storage.insert_fill('MKT_B', 'T_YES_B', 'YES', 4.0, 0.6, ts, tx_hash='txb1', kind='buy')
    storage.create_open_lot('MKT_B', 'T_YES_B', 'YES', 4.0, 0.6, ts, tx_hash='txb1')

    # close market but not resolved
    storage.update_market_status('MKT_B', 'closed')
    m = storage.get_market('MKT_B')
    assert m['status'] == 'closed'

    # ensure closed inventory still present but cannot be consumed by trading logic (we just check classification)
    open_lots = storage.get_open_lots_for_market('MKT_B')
    assert len(open_lots) == 1


def test_resolve_and_redeem():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_C', slug='mkt-c', status='open')
    storage.insert_fill('MKT_C', 'T_YES_C', 'YES', 7.0, 0.5, ts, tx_hash='txc1', kind='buy')
    storage.create_open_lot('MKT_C', 'T_YES_C', 'YES', 7.0, 0.5, ts, tx_hash='txc1')
    storage.insert_fill('MKT_C', 'T_NO_C', 'NO', 3.0, 0.5, ts, tx_hash='txc2', kind='buy')
    storage.create_open_lot('MKT_C', 'T_NO_C', 'NO', 3.0, 0.5, ts, tx_hash='txc2')

    # resolve market with YES winning
    storage.update_market_status('MKT_C', 'resolved', winning_outcome='YES')
    # redeem
    redeemed = storage.redeem_market('MKT_C', 'YES', redeem_tx_hash='redeemtx', ts=ts)
    assert redeemed == 7.0
    # with the current redeemer behavior all open lots are archived into redeemed_lots
    bal = storage.get_pair_balance('MKT_C')
    assert bal['YES'] == 0.0 and bal['NO'] == 0.0
    # redeemed_lots should include both sides
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ?', ('MKT_C',))
    assert float(cur.fetchone()[0]) == 10.0
    conn.close()


def test_merge_consumes_equal_yes_and_no_and_preserves_market_status():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_M', slug='mkt-m', status='open')
    storage.insert_fill('MKT_M', 'T_YES_M', 'YES', 7.0, 0.51, ts, tx_hash='txmy', kind='buy')
    storage.create_open_lot('MKT_M', 'T_YES_M', 'YES', 7.0, 0.51, ts, tx_hash='txmy')
    storage.insert_fill('MKT_M', 'T_NO_M', 'NO', 5.0, 0.49, ts, tx_hash='txmn', kind='buy')
    storage.create_open_lot('MKT_M', 'T_NO_M', 'NO', 5.0, 0.49, ts, tx_hash='txmn')

    merged = storage.merge_market_pair('MKT_M', 4.0, merge_tx_hash='mergetx', ts=ts, collateral_returned={'token': 'USDC', 'amount': 4.0})
    assert merged == 4.0
    bal = storage.get_pair_balance('MKT_M')
    assert bal['YES'] == 3.0
    assert bal['NO'] == 1.0
    assert storage.get_market('MKT_M')['status'] == 'open'

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT outcome_side, SUM(qty) FROM merged_lots WHERE market_id = ? GROUP BY outcome_side ORDER BY outcome_side", ('MKT_M',))
    assert cur.fetchall() == [('NO', 4.0), ('YES', 4.0)]
    cur.execute("SELECT kind, outcome_side, qty, tx_hash FROM fills WHERE tx_hash = ? ORDER BY outcome_side", ('mergetx',))
    assert cur.fetchall() == [('merge', 'NO', -4.0, 'mergetx'), ('merge', 'YES', -4.0, 'mergetx')]
    conn.close()


def test_merge_does_not_affect_other_markets():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_M1', status='open')
    storage.create_market('MKT_M2', status='open')
    for market_id, suffix in [('MKT_M1', '1'), ('MKT_M2', '2')]:
        storage.insert_fill(market_id, f'T_YES_{suffix}', 'YES', 3.0, 0.5, ts, tx_hash=f'buy-y-{suffix}', kind='buy')
        storage.create_open_lot(market_id, f'T_YES_{suffix}', 'YES', 3.0, 0.5, ts, tx_hash=f'buy-y-{suffix}')
        storage.insert_fill(market_id, f'T_NO_{suffix}', 'NO', 3.0, 0.5, ts, tx_hash=f'buy-n-{suffix}', kind='buy')
        storage.create_open_lot(market_id, f'T_NO_{suffix}', 'NO', 3.0, 0.5, ts, tx_hash=f'buy-n-{suffix}')

    storage.merge_market_pair('MKT_M1', 2.0, merge_tx_hash='merge-m1', ts=ts)
    assert storage.get_pair_balance('MKT_M1') == {'YES': 1.0, 'NO': 1.0}
    assert storage.get_pair_balance('MKT_M2') == {'YES': 3.0, 'NO': 3.0}


def test_merge_insufficient_one_sided_inventory_fails():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_M3', status='open')
    storage.insert_fill('MKT_M3', 'T_YES_3', 'YES', 5.0, 0.5, ts, tx_hash='buy-y-3', kind='buy')
    storage.create_open_lot('MKT_M3', 'T_YES_3', 'YES', 5.0, 0.5, ts, tx_hash='buy-y-3')
    storage.insert_fill('MKT_M3', 'T_NO_3', 'NO', 1.0, 0.5, ts, tx_hash='buy-n-3', kind='buy')
    storage.create_open_lot('MKT_M3', 'T_NO_3', 'NO', 1.0, 0.5, ts, tx_hash='buy-n-3')

    try:
        storage.merge_market_pair('MKT_M3', 2.0, merge_tx_hash='merge-fail', ts=ts)
        assert False, 'expected merge to fail on insufficient NO inventory'
    except RuntimeError as exc:
        assert 'Not enough paired inventory' in str(exc)
    assert storage.get_pair_balance('MKT_M3') == {'YES': 5.0, 'NO': 1.0}


def test_position_snapshot_includes_mergeable_pair_qty():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_S', status='resolved')
    storage.update_market_status('MKT_S', 'resolved', winning_outcome='YES')
    storage.insert_fill('MKT_S', 'T_YES_S', 'YES', 6.0, 0.5, ts, tx_hash='buy-y-s', kind='buy')
    storage.create_open_lot('MKT_S', 'T_YES_S', 'YES', 6.0, 0.5, ts, tx_hash='buy-y-s')
    storage.insert_fill('MKT_S', 'T_NO_S', 'NO', 4.0, 0.5, ts, tx_hash='buy-n-s', kind='buy')
    storage.create_open_lot('MKT_S', 'T_NO_S', 'NO', 4.0, 0.5, ts, tx_hash='buy-n-s')

    snap = {item['market_id']: item for item in storage.get_position_snapshot()}
    assert snap['MKT_S']['mergeable_pair_qty'] == 4.0
    assert snap['MKT_S']['resolved_redeemable_qty'] == 6.0


def test_position_snapshot_mergeable_pair_qty_uses_available_inventory():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MKT_AV', status='open')
    storage.create_open_lot('MKT_AV', 'T_YES_AV', 'YES', 6.0, 0.5, ts, tx_hash='buy-y-av')
    storage.create_open_lot('MKT_AV', 'T_NO_AV', 'NO', 4.0, 0.5, ts, tx_hash='buy-n-av')
    order = storage.create_order('merge-reserve', 'MKT_AV', 'T_YES_AV', 'YES', 'sell', 5.0, 0.5, 'open', ts)
    storage.create_reservation(order['id'], 'MKT_AV', 'T_YES_AV', 'YES', 'inventory', 5.0, ts)

    snap = {item['market_id']: item for item in storage.get_position_snapshot()}
    assert snap['MKT_AV']['available_inventory']['YES'] == 1.0
    assert snap['MKT_AV']['mergeable_pair_qty'] == 1.0
