#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src import storage, trade_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build simple realized trade statistics artifacts.')
    parser.add_argument('--decision-log-path')
    parser.add_argument('--clear-existing', action='store_true')
    parser.add_argument('--output-dir', default='artifacts')
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as handle:
        if not fieldnames:
            handle.write('')
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    storage.ensure_db()
    trade_stats.rebuild_trade_journal(
        clear_existing=args.clear_existing,
        decision_log_path=args.decision_log_path,
    )
    report = trade_stats.build_trade_stats_report()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / 'trade_stats_summary.json'
    by_market_path = output_dir / 'trade_stats_by_market.csv'
    cumulative_path = output_dir / 'cumulative_realized_pnl.csv'

    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    _write_csv(by_market_path, report['by_market'])
    _write_csv(cumulative_path, report['cumulative_realized_pnl'])

    print(f'wrote {summary_path}')
    print(f'wrote {by_market_path}')
    print(f'wrote {cumulative_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
