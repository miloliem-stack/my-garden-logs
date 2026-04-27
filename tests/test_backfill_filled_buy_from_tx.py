import os
from datetime import datetime, timezone

import pytest

from src import storage
from src.tools import backfill_filled_buy_from_tx


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_backfill_tool_creates_synthetic_order_fill_and_open_lot(monkeypatch):
    tx_hash = '0x' + 'd' * 64
    ts = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: {
            'txHash': tx,
            'logs': [
                {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TOKBF', 'value': 2.5}}
            ],
        } if tx == tx_hash else None,
    )

    result = backfill_filled_buy_from_tx.backfill_filled_buy_from_tx(
        tx_hash=tx_hash,
        market_id='MBF1',
        token_id='TOKBF',
        outcome_side='YES',
        price=0.44,
        ts=ts,
    )

    assert result['observed_qty'] == 2.5
    assert result['order']['status'] == 'filled'
    assert result['order']['tx_hash'] == tx_hash
    assert result['order_fills'][0]['fill_qty'] == 2.5
    assert storage.get_total_qty_by_token('TOKBF', market_id='MBF1') == 2.5


def test_duplicate_tx_hash_backfill_refused_without_force(monkeypatch):
    tx_hash = '0x' + 'e' * 64
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('MBF2', status='open')
    storage.create_order('existing-backfill', 'MBF2', 'TOKBE', 'NO', 'buy', 1.0, 0.5, 'filled', ts, tx_hash=tx_hash)
    monkeypatch.setattr('src.polymarket_client.WALLET_ADDRESS', '0xMYWALLET')
    monkeypatch.setattr(
        'src.polymarket_client.get_tx_receipt',
        lambda tx: {
            'txHash': tx,
            'logs': [
                {'event': 'TransferSingle', 'args': {'operator': '0xop', 'from': '0x0', 'to': '0xMYWALLET', 'id': 'TOKBE', 'value': 1.0}}
            ],
        } if tx == tx_hash else None,
    )

    with pytest.raises(RuntimeError, match='already exist'):
        backfill_filled_buy_from_tx.backfill_filled_buy_from_tx(
            tx_hash=tx_hash,
            market_id='MBF2',
            token_id='TOKBE',
            outcome_side='NO',
            price=0.5,
            ts=ts,
        )
