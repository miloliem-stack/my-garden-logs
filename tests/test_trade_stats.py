import json
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src import storage, trade_stats


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / 'bot_state.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    storage.ensure_db()
    return db_path


def _seed_buy_order(ts: str, *, market_id: str = 'M1', token_id: str = 'YES1', outcome_side: str = 'YES', price: float = 0.45, qty: float = 2.0, tx_hash: str = '0x' + '1' * 64):
    order = storage.create_order(
        client_order_id=f'buy-{tx_hash[-6:]}',
        market_id=market_id,
        token_id=token_id,
        outcome_side=outcome_side,
        side='buy',
        requested_qty=qty,
        limit_price=price,
        status='submitted',
        created_ts=ts,
        tx_hash=tx_hash,
    )
    storage.apply_incremental_order_fill(order['id'], qty, fill_ts=ts, tx_hash=tx_hash, price=price)
    return order


def _seed_sell_order(ts: str, *, market_id: str = 'M1', token_id: str = 'YES1', outcome_side: str = 'YES', price: float = 0.60, qty: float = 2.0, tx_hash: str = '0x' + '2' * 64):
    order = storage.create_order(
        client_order_id=f'sell-{tx_hash[-6:]}',
        market_id=market_id,
        token_id=token_id,
        outcome_side=outcome_side,
        side='sell',
        requested_qty=qty,
        limit_price=price,
        status='submitted',
        created_ts=ts,
        tx_hash=tx_hash,
    )
    storage.create_reservation(order['id'], market_id, token_id, outcome_side, 'inventory', qty, ts)
    storage.apply_incremental_order_fill(order['id'], qty, fill_ts=ts, tx_hash=tx_hash, price=price)
    return order


def test_trade_journal_row_creation_from_buy_fill(temp_db):
    ts = '2026-04-02T10:00:00+00:00'
    order = _seed_buy_order(ts)
    fill = storage.list_fills(kind='buy')[0]

    row = trade_stats.build_trade_journal_row(fill, {'order': order}, None)

    assert row['kind'] == 'buy'
    assert row['side'] == 'buy'
    assert row['qty'] == 2.0
    assert row['price'] == 0.45
    assert row['notional'] == pytest.approx(0.9)
    assert row['client_order_id'] == f"buy-{'1' * 6}"
    assert row['realized_pnl'] is None


def test_trade_journal_row_creation_from_sell_fill_with_realized_profit(temp_db):
    _seed_buy_order('2026-04-02T10:00:00+00:00')
    order = _seed_sell_order('2026-04-02T10:05:00+00:00')
    fill = storage.list_fills(kind='sell')[0]

    row = trade_stats.build_trade_journal_row(fill, {'order': order}, None)

    assert row['kind'] == 'sell'
    assert row['side'] == 'sell'
    assert row['qty'] == 2.0
    assert row['realized_pnl'] == pytest.approx(0.3)
    assert row['notional'] == pytest.approx(1.2)


def test_trade_journal_row_creation_from_redeem_fill(temp_db):
    fill = {
        'market_id': 'M1',
        'token_id': 'YES1',
        'outcome_side': 'YES',
        'qty': 1.0,
        'price': 1.0,
        'ts': '2026-04-02T11:00:00+00:00',
        'kind': 'redeem',
        'tx_hash': '0x' + '3' * 64,
        'extra': {},
        'extra_json': None,
    }

    row = trade_stats.build_trade_journal_row(fill, None, None)

    assert row['kind'] == 'redeem'
    assert row['side'] is None
    assert row['qty'] == 1.0
    assert row['notional'] == pytest.approx(1.0)


def test_journal_rebuild_works_without_decision_log(temp_db):
    _seed_buy_order('2026-04-02T10:00:00+00:00')
    _seed_sell_order('2026-04-02T10:05:00+00:00')

    result = trade_stats.rebuild_trade_journal()
    rows = storage.list_trade_journal()

    assert result['journal_rows_inserted'] == 2
    assert len(rows) == 2
    assert all(row['decision_ts'] is None for row in rows)


def test_decision_metadata_enrichment_works_for_nearby_decision_log(temp_db, tmp_path):
    _seed_buy_order('2026-04-02T10:00:00+00:00')
    log_path = tmp_path / 'decision_state.jsonl'
    decision = {
        'timestamp': '2026-04-02T09:58:00+00:00',
        'market_id': 'M1',
        'chosen_side': 'YES',
        'reason': 'ok',
        'tau_minutes': 12,
        'policy_bucket': 'mid',
        'raw_p_yes': 0.59,
        'p_yes': 0.61,
        'p_no': 0.39,
        'raw_edge_yes': 0.05,
        'edge_yes': 0.06,
        'edge_no': -0.02,
        'polarized_tail_penalty': 0.1,
        'polarized_tail_blocked': False,
        'position_reeval_action': 'hold',
        'position_reeval_reason': 'still_valid',
    }
    log_path.write_text(json.dumps(decision) + '\n', encoding='utf-8')

    trade_stats.rebuild_trade_journal(decision_log_path=str(log_path))
    row = storage.list_trade_journal()[0]

    assert row['decision_ts'] == '2026-04-02T09:58:00+00:00'
    assert row['policy_bucket'] == 'mid'
    assert row['tau_minutes'] == 12
    assert row['adjusted_p_yes'] == pytest.approx(0.61)
    assert row['adjusted_edge_yes'] == pytest.approx(0.06)
    assert row['reeval_action'] == 'hold'


def test_report_summary_totals_are_correct_on_small_fixture_set(temp_db, tmp_path):
    _seed_buy_order('2026-04-02T10:00:00+00:00', market_id='M1', token_id='YES1', outcome_side='YES', price=0.45, qty=2.0, tx_hash='0x' + '4' * 64)
    _seed_sell_order('2026-04-02T10:05:00+00:00', market_id='M1', token_id='YES1', outcome_side='YES', price=0.60, qty=2.0, tx_hash='0x' + '5' * 64)
    storage.insert_fill('M1', 'YES1', 'YES', 1.0, 1.0, '2026-04-02T10:10:00+00:00', tx_hash='0x' + '6' * 64, kind='redeem')

    decision_path = tmp_path / 'decision_state.jsonl'
    decision_path.write_text(json.dumps({'timestamp': '2026-04-02T09:59:00+00:00', 'market_id': 'M1', 'policy_bucket': 'mid'}) + '\n', encoding='utf-8')
    trade_stats.rebuild_trade_journal(decision_log_path=str(decision_path))

    report = trade_stats.build_trade_stats_report()
    summary = report['summary']

    assert summary['total_journal_rows'] == 3
    assert summary['total_buy_notional'] == pytest.approx(0.9)
    assert summary['total_sell_notional'] == pytest.approx(1.2)
    assert summary['total_redeem_rows'] == 1
    assert summary['total_realized_pnl'] == pytest.approx(0.3)
    assert summary['realized_pnl_row_count'] == 1
    assert summary['average_realized_pnl_per_exit'] == pytest.approx(0.3)
    assert summary['win_count'] == 1
    assert summary['loss_count'] == 0
    assert summary['flat_count'] == 0
    assert summary['counts_by_kind']['buy'] == 1
    assert summary['counts_by_kind']['sell'] == 1
    assert summary['counts_by_kind']['redeem'] == 1
    assert summary['counts_by_market_id']['M1'] == 3


def test_cumulative_realized_pnl_series_is_monotonic_in_time_and_sums_correctly(temp_db):
    _seed_buy_order('2026-04-02T10:00:00+00:00', tx_hash='0x' + '7' * 64)
    _seed_sell_order('2026-04-02T10:05:00+00:00', price=0.60, qty=2.0, tx_hash='0x' + '8' * 64)
    _seed_buy_order('2026-04-02T10:10:00+00:00', market_id='M2', token_id='NO2', outcome_side='NO', price=0.50, qty=1.0, tx_hash='0x' + '9' * 64)
    sell2 = storage.create_order(
        client_order_id='sell-2',
        market_id='M2',
        token_id='NO2',
        outcome_side='NO',
        side='sell',
        requested_qty=1.0,
        limit_price=0.40,
        status='submitted',
        created_ts='2026-04-02T10:15:00+00:00',
        tx_hash='0x' + 'a' * 64,
    )
    storage.create_reservation(sell2['id'], 'M2', 'NO2', 'NO', 'inventory', 1.0, '2026-04-02T10:15:00+00:00')
    storage.apply_incremental_order_fill(sell2['id'], 1.0, fill_ts='2026-04-02T10:15:00+00:00', tx_hash='0x' + 'a' * 64, price=0.40)

    trade_stats.rebuild_trade_journal()
    series = trade_stats.build_cumulative_realized_pnl_series()

    timestamps = [row['ts'] for row in series]
    assert timestamps == sorted(timestamps)
    assert series[-1]['cumulative_realized_pnl'] == pytest.approx(0.2)
    assert sum(row['realized_pnl'] for row in series) == pytest.approx(0.2)
