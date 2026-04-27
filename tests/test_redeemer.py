import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage
import src.redeemer as redeemer_mod
import src.polymarket_client as poly
import sqlite3
import os
import builtins
from datetime import datetime, timezone


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_closed_but_unresolved_skipped():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', slug='m1', status='closed')
    storage.insert_fill('M1', 'T1', 'YES', 2.0, 0.6, ts, tx_hash='tx1', kind='buy')
    storage.create_open_lot('M1', 'T1', 'YES', 2.0, 0.6, ts, tx_hash='tx1')

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=True)
    r.redeem_once()
    # market should remain closed and open_lots intact
    assert storage.get_total_qty_by_market_and_side('M1', 'YES') == 2.0


def test_redeemer_promotes_expired_btc_hourly_market_and_enters_dry_run(monkeypatch):
    ts = datetime(2026, 3, 26, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market(
        'MBTC-R',
        slug='bitcoin-up-or-down-march-26-2026-7am-et',
        title='Bitcoin Up or Down March 26, 2026 7am ET',
        status='closed',
        start_time='2026-03-26T11:00:00+00:00',
        end_time='2026-03-26T12:00:00+00:00',
        condition_id='0x' + '13' * 32,
    )
    storage.insert_fill('MBTC-R', 'TBR', 'NO', 2.17, 0.4, ts, tx_hash='tx-br', kind='buy')
    storage.create_open_lot('MBTC-R', 'TBR', 'NO', 2.17, 0.4, ts, tx_hash='tx-br')
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86900.0)

    called = {'v': False}

    def fake(market_id, dry_run=False, **kwargs):
        called['v'] = True
        assert market_id == 'MBTC-R'
        assert kwargs.get('winning_outcome') == 'NO'
        assert kwargs.get('redeemable_qty') == 2.17
        return {'status': 'dry_run', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=True)
    r.redeem_once()

    market = storage.get_market('MBTC-R')
    assert called['v'] is True
    assert market['status'] == 'resolved'
    assert market['winning_outcome'] == 'NO'


def test_resolved_market_dry_run_leaves_inventory(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MD', slug='md', status='resolved', condition_id='0x' + '11' * 32)
    storage.update_market_status('MD', 'resolved', winning_outcome='YES')
    storage.insert_fill('MD', 'TD', 'YES', 7.0, 0.4, ts, tx_hash='txd', kind='buy')
    storage.create_open_lot('MD', 'TD', 'YES', 7.0, 0.4, ts, tx_hash='txd')

    called = {'v': False}

    def fake(market_id, dry_run=False, **kwargs):
        called['v'] = True
        return {'status': 'dry_run', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=True)
    r.redeem_once()

    assert called['v'] is True
    assert storage.get_total_qty_by_market_and_side('MD', 'YES') == 7.0
    # no redeemed rows
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ?', ('MD',))
    assert cur.fetchone()[0] is None
    conn.close()


def test_resolved_market_redeem_success(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M2', slug='m2', status='resolved')
    storage.update_market_status('M2', 'resolved', winning_outcome='YES')
    storage.insert_fill('M2', 'T2', 'YES', 5.0, 0.5, ts, tx_hash='tx2', kind='buy')
    storage.create_open_lot('M2', 'T2', 'YES', 5.0, 0.5, ts, tx_hash='tx2')

    # monkeypatch polymarket_client.redeem_market_onchain to simulate success
    def fake_redeem(market_id, dry_run=False, **kwargs):
        return {'status': 'ok', 'tx_hash': 'redeemtx', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_redeem)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    # after successful redeem, open lots for winning side should be removed
    assert storage.get_total_qty_by_market_and_side('M2', 'YES') == 0.0
    # redeemed_lots should include the qty (both sides moved; only YES exists here)
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ?", ('M2',))
    assert cur.fetchone()[0] == 5.0
    conn.close()


def test_settler_retires_entire_market_inventory_and_archives_losers(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MSET', slug='mset', status='resolved', condition_id='0x' + '21' * 32)
    storage.update_market_status('MSET', 'resolved', winning_outcome='YES')
    storage.insert_fill('MSET', 'TY', 'YES', 4.0, 0.55, ts, tx_hash='buy-y', kind='buy')
    storage.insert_fill('MSET', 'TN', 'NO', 3.0, 0.45, ts, tx_hash='buy-n', kind='buy')
    storage.create_open_lot('MSET', 'TY', 'YES', 4.0, 0.55, ts, tx_hash='buy-y')
    storage.create_open_lot('MSET', 'TN', 'NO', 3.0, 0.45, ts, tx_hash='buy-n')
    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': 'redeem-settle', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'})
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash, 'status': '0x1'})

    result = redeemer_mod.settle_resolved_market_inventory('MSET', dry_run=False)

    assert result['status'] == 'redeemed'
    assert storage.get_total_qty_by_market_and_side('MSET', 'YES') == 0.0
    assert storage.get_total_qty_by_market_and_side('MSET', 'NO') == 0.0
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT kind, outcome_side, qty FROM fills WHERE market_id = ? AND kind IN ('redeem', 'settle') ORDER BY kind, outcome_side", ('MSET',))
    assert cur.fetchall() == [('redeem', 'YES', 4.0), ('settle', 'NO', 0.0)]
    conn.close()
    disposals = storage.list_inventory_disposals('MSET')
    assert disposals[-1]['policy_type'] == 'settler'
    assert disposals[-1]['tx_hash'] == 'redeem-settle'
    assert disposals[-1]['receipt']['transactionHash'] == 'redeem-settle'


def test_settler_ignores_diagnostic_not_found_order_status_and_uses_inventory_truth(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MDIAG', slug='mdiag', status='resolved', condition_id='0x' + '31' * 32)
    storage.update_market_status('MDIAG', 'resolved', winning_outcome='YES')
    storage.create_order('diag-order', 'MDIAG', 'TYD', 'YES', 'buy', 2.0, 0.5, 'not_found_on_venue', ts, tx_hash='0xdiag')
    storage.insert_fill('MDIAG', 'TYD', 'YES', 2.0, 0.5, ts, tx_hash='fill-diag', kind='buy')
    storage.create_open_lot('MDIAG', 'TYD', 'YES', 2.0, 0.5, ts, tx_hash='fill-diag')
    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': 'redeem-diag', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'})
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash})

    result = redeemer_mod.settle_resolved_market_inventory('MDIAG', dry_run=False)

    assert result['status'] == 'redeemed'
    assert storage.get_total_qty_by_market_and_side('MDIAG', 'YES') == 0.0


def test_unresolved_settler_leaves_inventory_and_records_skip():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MOPEN', slug='mopen', status='closed')
    storage.create_open_lot('MOPEN', 'TOPEN', 'YES', 1.0, 0.5, ts, tx_hash='tx-open')

    result = redeemer_mod.settle_resolved_market_inventory('MOPEN', dry_run=False)

    assert result['reason'] == 'market_not_resolved'
    assert storage.get_total_qty_by_market_and_side('MOPEN', 'YES') == 1.0
    assert storage.list_inventory_disposals('MOPEN')[-1]['classification'] == 'market_not_resolved'


def test_btc_hourly_market_hydrates_by_market_id_and_promotes_to_resolved(monkeypatch):
    now = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MHYDRATE', status='closed')
    monkeypatch.setattr(
        redeemer_mod.storage,
        'hydrate_market_metadata_by_id',
        lambda market_id: (
            storage.upsert_market(
                market_id=market_id,
                condition_id='0x' + '44' * 32,
                slug='bitcoin-up-or-down-march-27-2026-7am-et',
                title='Bitcoin Up or Down March 27, 2026 7am ET',
                start_time='2026-03-27T11:00:00+00:00',
                end_time='2026-03-27T12:00:00+00:00',
                status='closed',
            ) or storage.get_market(market_id)
        ),
    )
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86950.0)

    refreshed = redeemer_mod._refresh_market_for_redemption('MHYDRATE', checked_ts=now)

    assert refreshed['slug'] == 'bitcoin-up-or-down-march-27-2026-7am-et'
    assert refreshed['status'] == 'resolved'
    assert refreshed['winning_outcome'] == 'NO'


def test_redeemed_market_returns_already_redeemed_not_market_not_resolved():
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MRED', slug='mred', status='redeemed')
    storage.upsert_market('MRED', winning_outcome='YES', last_redeem_ts=ts)

    result = redeemer_mod.settle_resolved_market_inventory('MRED', dry_run=False, checked_ts=ts)

    assert result['status'] == 'skipped'
    assert result['reason'] == 'already_redeemed'
    disposal = storage.list_inventory_disposals('MRED')[-1]
    assert disposal['classification'] == 'already_redeemed'
    assert disposal['failure_reason'] is None


def test_redeemer_redeems_winner_after_market_id_hydration(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MHR', status='closed')
    storage.create_open_lot('MHR', 'THR', 'NO', 2.0, 0.4, ts, tx_hash='tx-hr')

    def fake_hydrate(market_id):
        storage.upsert_market(
            market_id=market_id,
            condition_id='0x' + '55' * 32,
            slug='bitcoin-up-or-down-march-27-2026-7am-et',
            title='Bitcoin Up or Down March 27, 2026 7am ET',
            start_time='2026-03-27T11:00:00+00:00',
            end_time='2026-03-27T12:00:00+00:00',
            status='closed',
        )
        return storage.get_market(market_id)

    monkeypatch.setattr(redeemer_mod.storage, 'hydrate_market_metadata_by_id', fake_hydrate)
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86950.0)
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash})
    monkeypatch.setattr(
        'src.polymarket_client.redeem_market_onchain',
        lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': 'redeem-hydrated', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'},
    )

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()

    assert storage.get_total_qty_by_market_and_side('MHR', 'NO') == 0.0
    assert storage.get_market('MHR')['status'] == 'redeemed'


def test_hydrated_market_with_only_loser_inventory_returns_no_redeemable_qty(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MLOSER', status='closed')
    storage.create_open_lot('MLOSER', 'TL', 'NO', 2.0, 0.4, ts, tx_hash='tx-loser')

    def fake_hydrate(market_id):
        storage.upsert_market(
            market_id=market_id,
            condition_id='0x' + '66' * 32,
            slug='bitcoin-up-or-down-march-27-2026-7am-et',
            title='Bitcoin Up or Down March 27, 2026 7am ET',
            start_time='2026-03-27T11:00:00+00:00',
            end_time='2026-03-27T12:00:00+00:00',
            status='closed',
        )
        return storage.get_market(market_id)

    monkeypatch.setattr(redeemer_mod.storage, 'hydrate_market_metadata_by_id', fake_hydrate)
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 87100.0)

    result = redeemer_mod.settle_resolved_market_inventory('MLOSER', dry_run=False, checked_ts=ts)

    assert result['status'] == 'finalized_loss'
    assert result['reason'] == 'no_redeemable_qty'
    assert result['winning_outcome'] == 'YES'
    assert storage.get_total_qty_by_market_and_side('MLOSER', 'NO') == 0.0
    assert storage.get_market('MLOSER')['status'] == 'redeemed'
    assert storage.list_inventory_disposals('MLOSER')[-1]['classification'] == 'settled_loss'


def test_redeemer_console_hides_tx_hash_outside_debug(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MCI', slug='mci', status='resolved')
    storage.update_market_status('MCI', 'resolved', winning_outcome='YES')
    storage.create_open_lot('MCI', 'TCI', 'YES', 2.0, 0.5, ts, tx_hash='tx-console')

    monkeypatch.setenv('LOG_MODE', 'info')
    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': '0xhide', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'})
    lines = []
    monkeypatch.setattr(builtins, 'print', lambda message: lines.append(message))

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()

    assert lines
    assert all('tx_hash=' not in line for line in lines)


def test_redeemer_console_debug_can_show_tx_hash(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MCD', slug='mcd', status='resolved')
    storage.update_market_status('MCD', 'resolved', winning_outcome='YES')
    storage.create_open_lot('MCD', 'TCD', 'YES', 2.0, 0.5, ts, tx_hash='tx-console-debug')

    monkeypatch.setenv('LOG_MODE', 'debug')
    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': '0xshow', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'})
    lines = []
    monkeypatch.setattr(builtins, 'print', lambda message: lines.append(message))

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()

    assert any('tx_hash=0xshow' in line for line in lines)


def test_error_with_tx_hash_does_not_redeem(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('ME', slug='me', status='resolved')
    storage.update_market_status('ME', 'resolved', winning_outcome='YES')
    storage.insert_fill('ME', 'TE', 'YES', 3.0, 0.5, ts, tx_hash='txe', kind='buy')
    storage.create_open_lot('ME', 'TE', 'YES', 3.0, 0.5, ts, tx_hash='txe')

    def fake_bad(market_id, dry_run=False, **kwargs):
        # contains a tx_hash but status != ok
        return {'status': 'error', 'tx_hash': 'badtx', 'error_reason': 'rpc_or_submission_failure', 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_bad)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()

    # inventory should remain
    assert storage.get_total_qty_by_market_and_side('ME', 'YES') == 3.0
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ?', ('ME',))
    assert cur.fetchone()[0] is None
    conn.close()


def test_missing_winning_outcome_skips(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MM', slug='mm', status='resolved')
    # do NOT set winning_outcome
    storage.insert_fill('MM', 'TM', 'YES', 1.0, 0.5, ts, tx_hash='txm', kind='buy')
    storage.create_open_lot('MM', 'TM', 'YES', 1.0, 0.5, ts, tx_hash='txm')

    called = {'v': False}

    def fake(market_id, dry_run=False, **kwargs):
        called['v'] = True
        return {'status': 'ok', 'tx_hash': 'tx'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()

    assert called['v'] is False
    assert storage.get_total_qty_by_market_and_side('MM', 'YES') == 1.0


def test_redeem_idempotent(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MI', slug='mi', status='resolved')
    storage.update_market_status('MI', 'resolved', winning_outcome='YES')
    storage.insert_fill('MI', 'TI', 'YES', 6.0, 0.5, ts, tx_hash='txi', kind='buy')
    storage.create_open_lot('MI', 'TI', 'YES', 6.0, 0.5, ts, tx_hash='txi')

    call_count = {'n': 0}

    def fake_ok(market_id, dry_run=False, **kwargs):
        call_count['n'] += 1
        return {'status': 'ok', 'tx_hash': f'redeem-{call_count["n"]}', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_ok)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    # first redeem should clear inventory
    assert storage.get_total_qty_by_market_and_side('MI', 'YES') == 0.0
    # run again; should not attempt redeem since no open inventory
    r.redeem_once()
    assert call_count['n'] == 1


def test_cross_market_isolation(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    # two markets, only one resolved
    storage.create_market('MA', slug='a', status='resolved')
    storage.update_market_status('MA', 'resolved', winning_outcome='YES')
    storage.insert_fill('MA', 'TA', 'YES', 2.0, 0.5, ts, tx_hash='txa', kind='buy')
    storage.create_open_lot('MA', 'TA', 'YES', 2.0, 0.5, ts, tx_hash='txa')

    storage.create_market('MB', slug='b', status='open')
    storage.insert_fill('MB', 'TB', 'YES', 3.0, 0.5, ts, tx_hash='txb', kind='buy')
    storage.create_open_lot('MB', 'TB', 'YES', 3.0, 0.5, ts, tx_hash='txb')

    def fake_redeem_ok(market_id, dry_run=False, **kwargs):
        return {'status': 'ok', 'tx_hash': f'redeem-{market_id}', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_redeem_ok)

    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    # MA should be redeemed, MB untouched
    assert storage.get_total_qty_by_market_and_side('MA', 'YES') == 0.0
    assert storage.get_total_qty_by_market_and_side('MB', 'YES') == 3.0


def test_redeemer_requires_explicit_success_status(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MX', slug='mx', status='resolved')
    storage.update_market_status('MX', 'resolved', winning_outcome='YES')
    storage.create_open_lot('MX', 'TX', 'YES', 2.0, 0.5, ts, tx_hash='txx')

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', lambda market_id, dry_run=False, **kwargs: {'status': 'accepted', 'tx_hash': '0xnotfinal', 'path_used': 'direct_onchain'})
    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    assert storage.get_total_qty_by_market_and_side('MX', 'YES') == 2.0


def test_redeemer_cooldown_prevents_immediate_retry(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MC', slug='mc', status='resolved')
    storage.update_market_status('MC', 'resolved', winning_outcome='YES')
    storage.create_open_lot('MC', 'TC', 'YES', 2.0, 0.5, ts, tx_hash='txc')

    calls = {'n': 0}

    def fake_fail(market_id, dry_run=False, **kwargs):
        calls['n'] += 1
        return {'status': 'error', 'reason': 'temporary', 'error_reason': 'rpc_or_submission_failure', 'path_used': 'direct_onchain'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_fail)
    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    r.redeem_once()
    assert calls['n'] == 1


def test_resolved_market_with_no_redeemable_qty_skips(monkeypatch):
    storage.create_market('MZ', slug='mz', status='resolved', condition_id='0x' + '12' * 32)
    storage.update_market_status('MZ', 'resolved', winning_outcome='YES')

    called = {'v': False}

    def fake_redeem(market_id, dry_run=False, **kwargs):
        called['v'] = True
        return {'status': 'ok', 'tx_hash': 'should-not-send'}

    monkeypatch.setattr('src.polymarket_client.redeem_market_onchain', fake_redeem)
    r = redeemer_mod.Redeemer(interval_minutes=1, dry_run=False)
    r.redeem_once()
    assert called['v'] is False


def test_inventory_sweep_hydrates_resolves_and_redeems_ended_btc_hourly_winner(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MSWEEPWIN', status='closed')
    storage.create_open_lot('MSWEEPWIN', 'TSWEEPWIN', 'NO', 2.0, 0.4, ts, tx_hash='tx-sweep-win')

    def fake_hydrate(market_id):
        storage.upsert_market(
            market_id=market_id,
            condition_id='0x' + '77' * 32,
            slug='bitcoin-up-or-down-march-27-2026-7am-et',
            title='Bitcoin Up or Down March 27, 2026 7am ET',
            start_time='2026-03-27T11:00:00+00:00',
            end_time='2026-03-27T12:00:00+00:00',
            status='closed',
        )
        return storage.get_market(market_id)

    monkeypatch.setattr(redeemer_mod.storage, 'hydrate_market_metadata_by_id', fake_hydrate)
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 86950.0)
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash})
    monkeypatch.setattr(
        'src.polymarket_client.redeem_market_onchain',
        lambda market_id, dry_run=False, **kwargs: {'status': 'ok', 'tx_hash': 'redeem-sweep-win', 'redeemed_qty': kwargs.get('redeemable_qty'), 'path_used': 'direct_onchain'},
    )

    results = redeemer_mod.settle_inventory_candidates(dry_run=False, checked_ts=ts)

    assert len(results) == 1
    assert results[0]['market_id'] == 'MSWEEPWIN'
    assert results[0]['status'] == 'redeemed'
    assert storage.get_total_qty_by_market_and_side('MSWEEPWIN', 'NO') == 0.0
    assert storage.get_market('MSWEEPWIN')['status'] == 'redeemed'


def test_inventory_sweep_finalizes_ended_btc_hourly_loser_only_inventory(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MSWEEPLOSS', status='closed')
    storage.create_open_lot('MSWEEPLOSS', 'TSWEEPLOSS', 'NO', 2.0, 0.4, ts, tx_hash='tx-sweep-loss')

    def fake_hydrate(market_id):
        storage.upsert_market(
            market_id=market_id,
            condition_id='0x' + '78' * 32,
            slug='bitcoin-up-or-down-march-27-2026-7am-et',
            title='Bitcoin Up or Down March 27, 2026 7am ET',
            start_time='2026-03-27T11:00:00+00:00',
            end_time='2026-03-27T12:00:00+00:00',
            status='closed',
        )
        return storage.get_market(market_id)

    called = {'redeem': False}

    monkeypatch.setattr(redeemer_mod.storage, 'hydrate_market_metadata_by_id', fake_hydrate)
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 87100.0)
    monkeypatch.setattr(
        'src.polymarket_client.redeem_market_onchain',
        lambda market_id, dry_run=False, **kwargs: called.__setitem__('redeem', True) or {'status': 'ok', 'tx_hash': 'unexpected'},
    )

    results = redeemer_mod.settle_inventory_candidates(dry_run=False, checked_ts=ts)

    assert len(results) == 1
    assert results[0]['status'] == 'finalized_loss'
    assert results[0]['reason'] == 'no_redeemable_qty'
    assert called['redeem'] is False
    assert storage.get_total_qty_by_market_and_side('MSWEEPLOSS', 'NO') == 0.0
    assert storage.get_market('MSWEEPLOSS')['status'] == 'redeemed'


def test_inventory_sweep_returns_already_redeemed(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MSWEEPRED', slug='sweep-red', title='Sweep Redeemed', condition_id='0x' + '79' * 32, start_time='2026-03-27T11:00:00+00:00', end_time='2026-03-27T12:00:00+00:00', status='redeemed')
    storage.create_open_lot('MSWEEPRED', 'TSWEEPRED', 'YES', 1.0, 0.4, ts, tx_hash='tx-sweep-red')

    results = redeemer_mod.settle_inventory_candidates(dry_run=False, checked_ts=ts)

    assert len(results) == 1
    assert results[0]['status'] == 'skipped'
    assert results[0]['reason'] == 'already_redeemed'
    assert results[0]['classification'] == 'already_redeemed'


def test_inventory_sweep_hydrates_missing_metadata_before_lifecycle_decision(monkeypatch):
    ts = datetime(2026, 3, 27, 12, 5, tzinfo=timezone.utc).isoformat()
    storage.create_market('MSWEEPHYD', status='closed')
    storage.create_open_lot('MSWEEPHYD', 'TSWEEPHYD', 'YES', 1.0, 0.4, ts, tx_hash='tx-sweep-hyd')

    hydrate_calls = {'n': 0}

    def fake_hydrate(market_id):
        hydrate_calls['n'] += 1
        storage.upsert_market(
            market_id=market_id,
            condition_id='0x' + '80' * 32,
            slug='bitcoin-up-or-down-march-27-2026-7am-et',
            title='Bitcoin Up or Down March 27, 2026 7am ET',
            start_time='2026-03-27T11:00:00+00:00',
            end_time='2026-03-27T12:00:00+00:00',
            status='closed',
        )
        return storage.get_market(market_id)

    monkeypatch.setattr(redeemer_mod.storage, 'hydrate_market_metadata_by_id', fake_hydrate)
    monkeypatch.setattr('src.binance_feed.get_1h_open_for_timestamp', lambda ts: 87000.0)
    monkeypatch.setattr('src.binance_feed.get_1h_close_for_timestamp', lambda ts: 87100.0)

    results = redeemer_mod.settle_inventory_candidates(dry_run=False, checked_ts=ts)

    assert len(results) == 1
    assert hydrate_calls['n'] == 1
    assert results[0]['market']['slug'] == 'bitcoin-up-or-down-march-27-2026-7am-et'
    assert results[0]['market']['condition_id'] == '0x' + '80' * 32
