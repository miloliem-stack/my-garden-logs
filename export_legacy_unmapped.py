#!/usr/bin/env python3
"""Export legacy_unmapped_rows to CSV for manual reconciliation.

Usage:
  python export_legacy_unmapped.py --db bot_state.db --out legacy_unmapped.csv

The script fails if the `legacy_unmapped_rows` table does not exist.
Exports all columns and preserves raw JSON payload for audit.
"""
from pathlib import Path
import argparse
import sqlite3
import csv
import sys


def table_exists(conn, table):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def export(db_path: Path, out_path: Path):
    conn = sqlite3.connect(str(db_path))
    if not table_exists(conn, 'legacy_unmapped_rows'):
        print('Error: table legacy_unmapped_rows does not exist in', db_path)
        conn.close()
        sys.exit(2)

    cur = conn.cursor()
    cur.execute('SELECT * FROM legacy_unmapped_rows')
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols + ['human_reason'])
        exported = 0
        for r in rows:
            row = list(r)
            # attempt to parse raw_json to include a short human_reason extracted
            # raw_json is assumed to be in a column; we preserve it and add a brief reason
            try:
                # find raw_json column index if present
                if 'raw_json' in cols:
                    raw_idx = cols.index('raw_json')
                    raw = r[raw_idx]
                    # keep raw as-is, and also derive human_reason from reason column if present
                    human_reason = None
                    if 'reason' in cols:
                        human_reason = r[cols.index('reason')]
                    row.append(human_reason)
                else:
                    row.append(None)
            except Exception:
                row.append(None)
            writer.writerow(row)
            exported += 1

    print(f'Exported {exported} rows to {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='bot_state.db')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print('Error: DB not found:', db_path)
        sys.exit(1)

    export(db_path, out_path)


if __name__ == '__main__':
    main()
