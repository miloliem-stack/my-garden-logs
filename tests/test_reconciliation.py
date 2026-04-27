import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import storage
import src.polymarket_client as poly
import pytest
import sqlite3
import os
from datetime import datetime, timezone


def setup_function(fn):
    os.environ['BOT_DB_PATH'] = str(Path('/tmp') / f'btc_1h_test_reconciliation_{fn.__name__}.db')
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_trade_receipt_reconciliation(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    # set wallet address for tests
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    # create a buy fill with tx_hash
    storage.insert_fill('M1', 'T1', 'YES', 2.0, 0.5, ts, tx_hash='tx_trade_1', kind='buy')

    # simulate receipt with TransferSingle into wallet for token T1 value 2
    receipt = {
        'txHash': 'tx_trade_1',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'T1', 'value': 2.0}}
        ]
    }

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_trade_1' else None)

    res = storage.reconcile_tx('tx_trade_1')
    assert res['status'] == 'ok'
    assert res['observed']['wallet_effects'] == [{'token_id': 'T1', 'direction': 'in', 'qty': 2.0}]
    assert res['expected']['expected_effects'] == [{'token_id': 'T1', 'direction': 'in', 'qty': 2.0}]
    # fill should be marked processed
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT receipt_processed FROM fills WHERE tx_hash = ?', ('tx_trade_1',))
    assert cur.fetchone()[0] == 1
    conn.close()


def test_redeem_receipt_reconciliation(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    # create a redeem fill with tx_hash
    storage.insert_fill('M2', 'T2', 'YES', 5.0, 0.5, ts, tx_hash='tx_redeem_1', kind='redeem')

    receipt = {
        'txHash': 'tx_redeem_1',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0xMYWALLET', 'to': '0x0', 'id': 'T2', 'value': 5.0}}
        ]
    }

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_redeem_1' else None)
    res = storage.reconcile_tx('tx_redeem_1')
    assert res['status'] == 'ok'


def test_trade_receipt_reconciliation_normalizes_micro_buy_fill(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    storage.insert_fill('M1M', 'T1M', 'YES', 1.444442, 0.5, ts, tx_hash='tx_trade_micro_buy', kind='buy')

    receipt = {
        'txHash': 'tx_trade_micro_buy',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'T1M', 'value': 1444442.0}}
        ]
    }

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_trade_micro_buy' else None)

    res = storage.reconcile_tx('tx_trade_micro_buy')
    assert res['status'] == 'ok'
    assert res['observed']['wallet_effects'] == [{'token_id': 'T1M', 'direction': 'in', 'qty': 1.444442}]
    assert res['observed']['decoded_transfers'][0]['raw_value'] == 1444442.0
    assert res['observed']['decoded_transfers'][0]['value'] == 1.444442


def test_trade_receipt_reconciliation_normalizes_micro_sell_fill(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    storage.insert_fill('M2M', 'T2M', 'YES', 1.17, 0.5, ts, tx_hash='tx_trade_micro_sell', kind='sell')

    receipt = {
        'txHash': 'tx_trade_micro_sell',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0xMYWALLET', 'to': '0x0', 'id': 'T2M', 'value': 1170000.0}}
        ]
    }

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_trade_micro_sell' else None)

    res = storage.reconcile_tx('tx_trade_micro_sell')
    assert res['status'] == 'ok'
    assert res['observed']['wallet_effects'] == [{'token_id': 'T2M', 'direction': 'out', 'qty': 1.17}]
    assert res['expected']['expected_effects'] == [{'token_id': 'T2M', 'direction': 'out', 'qty': 1.17}]

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM reconciliation_issues WHERE tx_hash = ? AND reason = ?', ('tx_trade_micro_sell', 'mismatch'))
    assert cur.fetchone()[0] == 0
    conn.close()


def test_merge_reconciliation_expects_out_on_both_token_ids(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    wallet = '0x' + 'a' * 40
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', wallet)

    storage.create_market('MM1', status='open')
    storage.insert_fill('MM1', 'TY', 'YES', 4.0, 0.5, ts, tx_hash='buy-merge-y', kind='buy')
    storage.create_open_lot('MM1', 'TY', 'YES', 4.0, 0.5, ts, tx_hash='buy-merge-y')
    storage.insert_fill('MM1', 'TN', 'NO', 4.0, 0.5, ts, tx_hash='buy-merge-n', kind='buy')
    storage.create_open_lot('MM1', 'TN', 'NO', 4.0, 0.5, ts, tx_hash='buy-merge-n')
    storage.merge_market_pair('MM1', 4.0, merge_tx_hash='tx_merge_1', ts=ts, collateral_returned={'token': 'USDC', 'amount': 4.0})

    receipt = {
        'txHash': 'tx_merge_1',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': wallet, 'to': '0x' + 'b' * 40, 'id': 'TY', 'value': 4.0}},
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': wallet, 'to': '0x' + 'b' * 40, 'id': 'TN', 'value': 4.0}},
            {'address': '0x' + 'c' * 40, 'topics': [storage.ERC20_TRANSFER_SIG, '0x' + '00' * 24 + ('f' * 40), '0x' + '00' * 24 + ('a' * 40)], 'data': hex(4)},
        ],
    }
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_merge_1' else None)

    res = storage.reconcile_tx('tx_merge_1')
    assert res['status'] == 'ok'
    assert sorted(res['expected']['expected_effects'], key=lambda item: item['token_id']) == [
        {'token_id': 'TN', 'direction': 'out', 'qty': 4.0},
        {'token_id': 'TY', 'direction': 'out', 'qty': 4.0},
    ]
    assert res['observed']['merge_classification']['status'] == 'merge_confirmed'
    assert any(t['direction'] == 'in' for t in res['observed']['collateral_transfers'])


def test_mismatch_flagged_not_erased(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')

    storage.insert_fill('M3', 'T3', 'YES', 3.0, 0.5, ts, tx_hash='tx_mismatch', kind='buy')

    # observed only 2.0
    receipt = {'txHash': 'tx_mismatch', 'logs': [
        {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'T3', 'value': 2.0}}
    ]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_mismatch' else None)

    res = storage.reconcile_tx('tx_mismatch')
    assert res['status'] == 'mismatch'
    # fill should remain receipt_processed = 0
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT receipt_processed FROM fills WHERE tx_hash = ?', ('tx_mismatch',))
    assert cur.fetchone()[0] == 0
    # an issue should be recorded
    cur.execute('SELECT COUNT(*) FROM reconciliation_issues WHERE tx_hash = ?', ('tx_mismatch',))
    assert cur.fetchone()[0] >= 1
    conn.close()


def test_observed_wallet_effects_without_expected_rows_flagged(monkeypatch):
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    receipt = {'txHash': 'tx_unexpected', 'logs': [
        {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'T9', 'value': 7.0}}
    ]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_unexpected' else None)

    res = storage.reconcile_tx('tx_unexpected')
    assert res['status'] == 'unexpected_observed'
    assert res['observed']['wallet_effects'] == [{'token_id': 'T9', 'direction': 'in', 'qty': 7.0}]
    assert res['expected']['expected_effects'] == []

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT reason FROM reconciliation_issues WHERE tx_hash = ?', ('tx_unexpected',))
    reasons = [row[0] for row in cur.fetchall()]
    assert 'unexpected_observed' in reasons
    conn.close()


def test_decoded_receipt_without_wallet_effect(monkeypatch):
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    receipt = {'txHash': 'tx_other_wallet', 'logs': [
        {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0xALICE', 'to': '0xBOB', 'id': 'T10', 'value': 3.0}}
    ]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_other_wallet' else None)

    res = storage.reconcile_tx('tx_other_wallet')
    assert res['status'] == 'decoded_no_wallet_effect'
    assert res['observed']['wallet_effects'] == []
    assert res['observed']['all_effects'] == [{'token_id': 'T10', 'direction': 'other', 'qty': 3.0}]


def test_missing_receipt_pending(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.insert_fill('M4', 'T4', 'YES', 1.0, 0.5, ts, tx_hash='tx_missing', kind='buy')
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: None)
    res = storage.reconcile_tx('tx_missing')
    assert res['status'] == 'pending'


def test_raw_transfer_single_decoding(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0x' + 'a'*40)
    # prepare a fill
    storage.insert_fill('MR', str(11), 'YES', 9.0, 0.5, ts, tx_hash='tx_raw_single', kind='buy')

    # require canonical signature constant
    single_sig = getattr(poly, 'TRANSFER_SINGLE_SIG', None)
    if single_sig is None:
        pytest.skip('keccak unavailable; skipping strict raw TransferSingle test')
    # create raw topics: [sig, operator, from, to]
    # topics: [sig, operator, from, to] -> place wallet in 'to' (topics[3])
    topics = [
        single_sig,
        '0x' + '00' * 32,
        '0x' + '00' * 32,
        '0x' + '00' * 24 + ('a' * 40)
    ]
    # data: id=11, value=9
    id_hex = format(11, '064x')
    val_hex = format(9, '064x')
    data = '0x' + id_hex + val_hex
    receipt = {'txHash': 'tx_raw_single', 'logs': [{'topics': topics, 'data': data}]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_raw_single' else None)
    res = storage.reconcile_tx('tx_raw_single')
    assert res['status'] == 'ok'


def test_raw_transfer_single_wrong_topic_does_not_decode(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0x' + 'a'*40)
    storage.insert_fill('MRX', str(12), 'YES', 1.0, 0.5, ts, tx_hash='tx_raw_single_bad', kind='buy')
    # construct a raw log shaped like ERC-1155 but with wrong topic0
    topics = [
        '0x' + '00' * 32,
        '0x' + '00' * 32,
        '0x' + '00' * 32,
        '0x' + '00' * 24 + ('a' * 40)
    ]
    id_hex = format(12, '064x')
    val_hex = format(1, '064x')
    data = '0x' + id_hex + val_hex
    receipt = {'txHash': 'tx_raw_single_bad', 'logs': [{'topics': topics, 'data': data}]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_raw_single_bad' else None)
    res = storage.reconcile_tx('tx_raw_single_bad')
    assert res['status'] == 'mismatch'


def test_raw_transfer_batch_decoding(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0x' + 'b'*40)
    storage.insert_fill('MB', str(21), 'YES', 4.0, 0.5, ts, tx_hash='tx_raw_batch', kind='buy')

    # require canonical signature constant
    batch_sig = getattr(poly, 'TRANSFER_BATCH_SIG', None)
    if batch_sig is None:
        pytest.skip('keccak unavailable; skipping strict raw TransferBatch test')
    # topics with operator/from/to (sig in topics[0])
    topics = [batch_sig, '0x' + '00' * 32, '0x' + '00' * 32, '0x' + '00' * 24 + ('b' * 40)]
    # Construct batch data: offsets (0x40,0x... etc), then at offset: length + items
    # For simplicity construct data where offset 0 points to ids array and offset 1 to values array
    # ids: [21,22], values: [4,5]
    # Build dynamic layout: word0=0x40 (64 bytes), word1=0xa0 (160 bytes)
    w0 = format(64, '064x')
    w1 = format(160, '064x')
    # at offset 0x40: length 2, id1,id2
    l = format(2, '064x')
    id1 = format(21, '064x')
    id2 = format(22, '064x')
    # at offset 0x80: length 2, val1,val2
    v1 = format(4, '064x')
    v2 = format(5, '064x')
    data = '0x' + w0 + w1 + l + id1 + id2 + l + v1 + v2
    receipt = {'txHash': 'tx_raw_batch', 'logs': [{'topics': topics, 'data': data}]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_raw_batch' else None)
    res = storage.reconcile_tx('tx_raw_batch')
    assert res['status'] == 'ok' or res['status'] == 'mismatch'


def test_raw_transfer_batch_wrong_topic_does_not_decode(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0x' + 'b'*40)
    storage.insert_fill('MBX', str(31), 'YES', 2.0, 0.5, ts, tx_hash='tx_raw_batch_bad', kind='buy')
    # wrong topic0
    topics = ['0x' + '00' * 32, '0x' + '00' * 32, '0x' + '00' * 32, '0x' + '00' * 24 + ('b' * 40)]
    # reuse batch data construction from earlier
    w0 = format(64, '064x')
    w1 = format(160, '064x')
    l = format(2, '064x')
    id1 = format(31, '064x')
    id2 = format(32, '064x')
    v1 = format(2, '064x')
    v2 = format(3, '064x')
    data = '0x' + w0 + w1 + l + id1 + id2 + l + v1 + v2
    receipt = {'txHash': 'tx_raw_batch_bad', 'logs': [{'topics': topics, 'data': data}]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_raw_batch_bad' else None)
    res = storage.reconcile_tx('tx_raw_batch_bad')
    assert res['status'] == 'mismatch'


def test_decode_failure_is_explicit(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0x' + 'c' * 40)
    storage.insert_fill('MDF', str(41), 'YES', 2.0, 0.5, ts, tx_hash='tx_decode_fail', kind='buy')

    batch_sig = getattr(poly, 'TRANSFER_BATCH_SIG', None)
    if batch_sig is None:
        pytest.skip('keccak unavailable; skipping strict decode failure test')

    topics = [batch_sig, '0x' + '00' * 32, '0x' + '00' * 32, '0x' + '00' * 24 + ('c' * 40)]
    # malformed dynamic data: batch signature but invalid payload layout
    data = '0x' + '00' * 10
    receipt = {'txHash': 'tx_decode_fail', 'logs': [{'topics': topics, 'data': data}]}
    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', lambda tx: receipt if tx == 'tx_decode_fail' else None)

    res = storage.reconcile_tx('tx_decode_fail')
    assert res['status'] == 'decode_failed'
    assert len(res['observed']['decode_failures']) == 1

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT reason FROM reconciliation_issues WHERE tx_hash = ?', ('tx_decode_fail',))
    reasons = [row[0] for row in cur.fetchall()]
    assert 'decode_failed' in reasons
    conn.close()


def test_merge_like_receipt_classification():
    wallet = '0x' + 'd' * 40
    receipt = {
        'txHash': 'tx_hist_merge',
        'logs': [
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': wallet, 'to': '0x' + 'e' * 40, 'id': 'TYES', 'value': 6.0}},
            {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': wallet, 'to': '0x' + 'e' * 40, 'id': 'TNO', 'value': 6.0}},
            {'address': '0x' + 'f' * 40, 'topics': [storage.ERC20_TRANSFER_SIG, '0x' + '00' * 24 + ('1' * 40), '0x' + '00' * 24 + ('d' * 40)], 'data': hex(6)},
        ],
    }
    out = storage.classify_merge_receipt(receipt, wallet, market_tokens={'YES': 'TYES', 'NO': 'TNO'})
    assert out['status'] == 'merge_confirmed'
    assert out['erc1155_outflows'] == [{'token_id': 'TNO', 'qty': 6.0}, {'token_id': 'TYES', 'qty': 6.0}]


def test_reconciliation_sweep_aggregates_without_mutating_mismatch(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    storage.insert_fill('SW1', 'TS1', 'YES', 1.0, 0.5, ts, tx_hash='tx_sweep_ok', kind='buy')
    storage.insert_fill('SW2', 'TS2', 'YES', 2.0, 0.5, ts, tx_hash='tx_sweep_mismatch', kind='buy')

    def fake_receipt(tx):
        if tx == 'tx_sweep_ok':
            return {'txHash': tx, 'logs': [{'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TS1', 'value': 1.0}}]}
        if tx == 'tx_sweep_mismatch':
            return {'txHash': tx, 'logs': [{'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TS2', 'value': 1.0}}]}
        return None

    monkeypatch.setattr('src.polymarket_client.get_tx_receipt', fake_receipt)
    sweep = storage.run_reconciliation_sweep()
    assert sweep['summary']['ok'] == 1
    assert sweep['summary']['mismatch'] == 1

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT receipt_processed FROM fills WHERE tx_hash = ?', ('tx_sweep_mismatch',))
    assert cur.fetchone()[0] == 0
    conn.close()
