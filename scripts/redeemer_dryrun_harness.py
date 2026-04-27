"""End-to-end dry-run harness for redeemer/inventory lifecycle.

Usage: python scripts/redeemer_dryrun_harness.py

This script runs a single market through: open -> closed -> resolved -> redeem (dry-run then live with fake client).
It prints position snapshots at each step and writes example JSONL metrics via the redeemer.
"""
import os
import json
from datetime import datetime, timezone
from time import sleep

ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.insert(0, ROOT)

from src import storage
from src.redeemer import Redeemer
import src.polymarket_client as poly


def print_snap(prefix):
    snap = storage.get_position_snapshot()
    print('\n===', prefix, 'snapshot ===')
    print(json.dumps(snap, indent=2))


def main():
    # fresh DB
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()

    ts = datetime.now(timezone.utc).isoformat()
    market = 'E2E_MKT'

    # 1) OPEN: create market and buys
    storage.create_market(market, slug='e2e', status='open')
    storage.insert_fill(market, 'TOK_Y', 'YES', 8.0, 0.5, ts, tx_hash='tx-open-y', kind='buy')
    storage.create_open_lot(market, 'TOK_Y', 'YES', 8.0, 0.5, ts, tx_hash='tx-open-y')
    storage.insert_fill(market, 'TOK_N', 'NO', 2.0, 0.5, ts, tx_hash='tx-open-n', kind='buy')
    storage.create_open_lot(market, 'TOK_N', 'NO', 2.0, 0.5, ts, tx_hash='tx-open-n')

    print_snap('after open')

    # 2) CLOSED: change to closed (unresolved)
    storage.update_market_status(market, 'closed')
    print_snap('after closed (unresolved)')

    # 3) RESOLVED: resolve YES as winner
    storage.update_market_status(market, 'resolved', winning_outcome='YES')
    print_snap('after resolved (before redeem)')

    # 4) Redeemer dry-run: should not mutate inventory
    # ensure metrics file absent
    mfile = 'redeemer_metrics.jsonl'
    try:
        os.remove(mfile)
    except Exception:
        pass

    r = Redeemer(interval_minutes=1, dry_run=True)
    r.redeem_once()

    print('\nAfter redeemer dry-run:')
    print_snap('post dry-run')

    # confirm metrics written (dry-run writes a metrics entry)
    if os.path.exists(mfile):
        print('\nSample metrics (dry-run):')
        with open(mfile) as fh:
            lines = fh.read().strip().splitlines()
            for l in lines[-5:]:
                print(l)
    else:
        print('\nNo metrics file found after dry-run')

    # 5) Redeemer live with fake polymarket client
    def fake_redeem_ok(market_id, dry_run=False):
        return {'status': 'ok', 'tx_hash': f'redeem-{market_id}'}

    poly.redeem_market_onchain = fake_redeem_ok

    r_live = Redeemer(interval_minutes=1, dry_run=False)
    r_live.redeem_once()

    print('\nAfter redeemer live (fake success):')
    print_snap('post live redeem')

    # show metrics file
    if os.path.exists(mfile):
        print('\nMetrics file content:')
        with open(mfile) as fh:
            for l in fh:
                print(l.strip())

    # assertions (simple checks)
    snap = storage.get_position_snapshot()
    m = [s for s in snap if s['market_id'] == market][0]
    assert m['tradable_open_inventory']['YES'] == 0.0 and m['tradable_open_inventory']['NO'] == 0.0
    assert m['resolved_unredeemed_inventory']['YES'] == 0.0
    assert m['redeemed_inventory']['YES'] == 8.0

    print('\nE2E harness completed successfully')


if __name__ == '__main__':
    main()
