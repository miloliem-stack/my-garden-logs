from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Optional

from .. import polymarket_client, storage


def _normalize_addr(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    addr = str(addr).lower()
    return addr if addr.startswith('0x') else f'0x{addr}'


def _wallet_inflow_qty_for_token(receipt: dict, token_id: str, wallet_address: Optional[str] = None) -> float:
    wallet = _normalize_addr(wallet_address or polymarket_client.WALLET_ADDRESS)
    total = 0.0
    for log in receipt.get('logs') or []:
        decoded = None
        try:
            decoded = polymarket_client._try_decode_erc1155_log(log)
        except Exception:
            decoded = None
        if not decoded or not decoded.get('event') or not decoded.get('args'):
            continue
        args = decoded['args']
        to_addr = _normalize_addr(args.get('to'))
        if wallet is None or to_addr != wallet:
            continue
        if decoded['event'] == 'TransferSingle':
            if str(args.get('id')) == str(token_id):
                total += float(args.get('value') or 0.0)
        else:
            for log_token_id, value in zip(args.get('ids') or [], args.get('values') or []):
                if str(log_token_id) == str(token_id):
                    total += float(value or 0.0)
    return total


def backfill_filled_buy_from_tx(
    *,
    tx_hash: str,
    market_id: str,
    token_id: str,
    outcome_side: str,
    price: float,
    ts: Optional[str] = None,
    client_order_id: Optional[str] = None,
    force: bool = False,
) -> dict:
    storage.ensure_db()
    ts = ts or datetime.now(timezone.utc).isoformat()
    receipt = polymarket_client.get_tx_receipt(tx_hash)
    if not receipt:
        raise RuntimeError(f'No receipt found for tx_hash={tx_hash}')
    observed_qty = _wallet_inflow_qty_for_token(receipt, token_id)
    if observed_qty <= 0:
        raise RuntimeError(f'No wallet inflow observed for token_id={token_id} tx_hash={tx_hash}')

    existing_orders = storage.get_orders_by_tx_hash(tx_hash)
    if existing_orders and not force:
        raise RuntimeError(f'Order(s) already exist for tx_hash={tx_hash}: {[order["id"] for order in existing_orders]}')

    if existing_orders and force:
        order = existing_orders[0]
    else:
        short_tx = tx_hash[2:10] if tx_hash.startswith('0x') else tx_hash[:8]
        order = storage.create_order(
            client_order_id=client_order_id or f'manual-backfill-{market_id}-{outcome_side}-{short_tx}',
            market_id=market_id,
            token_id=token_id,
            outcome_side=outcome_side,
            side='buy',
            requested_qty=observed_qty,
            limit_price=float(price),
            status='submitted',
            created_ts=ts,
            tx_hash=tx_hash,
            raw_response={
                'status': 'submitted',
                'source': 'manual_backfill',
                'tx_hash': tx_hash,
                'observed_qty': observed_qty,
            },
        )

    apply_res = storage.apply_incremental_order_fill(
        order['id'],
        observed_qty,
        fill_ts=ts,
        tx_hash=tx_hash,
        price=float(price),
        raw={'source': 'manual_backfill', 'tx_hash': tx_hash, 'observed_qty': observed_qty},
    )
    current = storage.get_order(order_id=order['id'])
    if current['status'] != 'filled' and storage.can_transition_order_state(current['status'], 'filled'):
        current = storage.transition_order_state(
            order['id'],
            'filled',
            reason='manual_backfill_from_tx',
            raw={'source': 'manual_backfill', 'tx_hash': tx_hash, 'observed_qty': observed_qty},
            ts=ts,
        )
    storage.append_order_event(
        order['id'],
        'manual_backfill_from_tx',
        old_status=order['status'],
        new_status=current['status'],
        response={'source': 'manual_backfill', 'tx_hash': tx_hash, 'observed_qty': observed_qty, 'applied_qty': apply_res.get('applied_qty', 0.0)},
        ts=ts,
    )
    return {
        'tx_hash': tx_hash,
        'observed_qty': observed_qty,
        'order': storage.get_order(order_id=order['id']),
        'order_fills': storage.get_order_fill_events(order['id']),
        'open_lots': storage.get_open_lots(token_id=token_id, market_id=market_id),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tx-hash', required=True)
    parser.add_argument('--market-id', required=True)
    parser.add_argument('--token-id', required=True)
    parser.add_argument('--outcome-side', required=True, choices=['YES', 'NO'])
    parser.add_argument('--price', required=True, type=float)
    parser.add_argument('--ts', default=None)
    parser.add_argument('--client-order-id', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    result = backfill_filled_buy_from_tx(
        tx_hash=args.tx_hash,
        market_id=args.market_id,
        token_id=args.token_id,
        outcome_side=args.outcome_side,
        price=args.price,
        ts=args.ts,
        client_order_id=args.client_order_id,
        force=args.force,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
