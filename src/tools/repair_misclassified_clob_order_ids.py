from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .. import polymarket_client, storage


def _safe_candidate_from_raw(raw_response_json: Optional[str]) -> Optional[str]:
    if not raw_response_json:
        return None
    try:
        payload = json.loads(raw_response_json)
    except Exception:
        return None
    return payload.get('order_id') or payload.get('orderId') or payload.get('id')


def find_repairable_orders(*, confirm_no_receipt: bool = True) -> List[Dict]:
    storage.ensure_db()
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, client_order_id, venue_order_id, status, tx_hash, raw_response_json
            FROM {storage.ORDER_TABLE}
            WHERE status IN ('unknown', 'not_found_on_venue')
              AND tx_hash IS NOT NULL
              AND tx_hash != ''
              AND venue_order_id IS NULL'''
    )
    rows = cur.fetchall()
    conn.close()

    candidates = []
    for row in rows:
        order_id, client_order_id, venue_order_id, status, tx_hash, raw_response_json = row
        if not isinstance(tx_hash, str) or not storage.TX_HASH_RE.match(tx_hash):
            continue
        raw_order_id = _safe_candidate_from_raw(raw_response_json)
        if raw_order_id != tx_hash:
            continue
        receipt = polymarket_client.get_tx_receipt(tx_hash) if confirm_no_receipt else None
        if confirm_no_receipt and receipt is not None:
            continue
        candidates.append(
            {
                'id': order_id,
                'client_order_id': client_order_id,
                'status': status,
                'tx_hash': tx_hash,
                'venue_order_id': venue_order_id,
                'raw_order_id': raw_order_id,
                'receipt_found': receipt is not None if confirm_no_receipt else None,
            }
        )
    return candidates


def repair_misclassified_clob_order_ids(*, dry_run: bool = True, confirm_no_receipt: bool = True) -> Dict:
    candidates = find_repairable_orders(confirm_no_receipt=confirm_no_receipt)
    if dry_run:
        return {
            'dry_run': True,
            'examined': len(candidates),
            'repaired': 0,
            'orders': candidates,
        }

    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    repaired = []
    ts = datetime.now(timezone.utc).isoformat()
    for candidate in candidates:
        cur.execute(
            f'''UPDATE {storage.ORDER_TABLE}
                SET venue_order_id = ?, tx_hash = NULL, updated_ts = ?
                WHERE id = ? AND venue_order_id IS NULL AND tx_hash = ?''',
            (candidate['tx_hash'], ts, candidate['id'], candidate['tx_hash']),
        )
        if cur.rowcount:
            repaired.append(candidate)
    conn.commit()
    conn.close()

    return {
        'dry_run': False,
        'examined': len(candidates),
        'repaired': len(repaired),
        'orders': repaired,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='persist repairs instead of dry-run')
    parser.add_argument('--skip-receipt-check', action='store_true', help='do not verify that the stored tx_hash lacks a chain receipt')
    args = parser.parse_args(argv)

    result = repair_misclassified_clob_order_ids(
        dry_run=not args.apply,
        confirm_no_receipt=not args.skip_receipt_check,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
