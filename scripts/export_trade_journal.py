#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src import storage, trade_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild and export derived trade journal CSV.')
    parser.add_argument('--market-id')
    parser.add_argument('--kind')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', default='artifacts/trade_journal.csv')
    parser.add_argument('--decision-log-path')
    parser.add_argument('--clear-existing', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage.ensure_db()
    rebuild = trade_stats.rebuild_trade_journal(
        clear_existing=args.clear_existing,
        decision_log_path=args.decision_log_path,
    )
    rows = storage.list_trade_journal(market_id=args.market_id, kind=args.kind, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'ts',
        'market_id',
        'kind',
        'side',
        'outcome_side',
        'qty',
        'price',
        'notional',
        'realized_pnl',
        'policy_bucket',
        'tau_minutes',
        'adjusted_edge_yes',
        'adjusted_edge_no',
        'decision_reason',
        'reeval_action',
        'reeval_reason',
        'tx_hash',
        'client_order_id',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    print(f'exported {len(rows)} trade journal rows to {output_path}')
    print(f'rebuild summary: {rebuild}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
