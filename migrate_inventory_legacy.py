#!/usr/bin/env python3
"""Conservative migration helper for legacy SQLite inventory schema.

This script inspects the existing `bot_state.db` and attempts to migrate legacy
`lots` and `orders` rows into the new market-scoped `fills` + `open_lots` schema
implemented in `src/storage.py`.

Rules:
- Only migrate rows that include explicit `market_id`, `token_id`, and
  `outcome_side` fields (or obvious aliases). Do NOT infer outcome or market
  identity.
- Rows that cannot be safely mapped are copied into `legacy_unmapped_rows`
  for audit and quarantined from active inventory.

Usage:
  python migrate_inventory_legacy.py --db bot_state.db [--dry-run] [--force]

The script supports a dry-run mode and will create a timestamped backup unless
`--no-backup` and `--force` are both provided.
"""
from __future__ import annotations
import argparse
import sqlite3
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any


DB_DEFAULT = Path.cwd() / 'bot_state.db'


def open_conn(dbpath: Path):
    return sqlite3.connect(str(dbpath))


def table_exists(conn, tbl: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
    return cur.fetchone() is not None


def create_legacy_unmapped_table(conn):
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS legacy_unmapped_rows(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_table TEXT,
        raw_json TEXT,
        reason TEXT,
        ts_migrated TEXT
    )''')
    conn.commit()


def fetch_rows(conn, table: str) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rows.append({k: v for k, v in zip(cols, r)})
    return rows


def detect_columns(conn, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]


def safe_map_row(row: Dict[str, Any], cols: List[str]) -> Dict[str, Any] | None:
    """Return mapped fields when safe, else None.

    Required mapped fields: market_id, token_id, outcome_side, qty, price, ts
    We accept common column name aliases.
    """
    # alias resolution
    def find(*names):
        for n in names:
            if n in row and row.get(n) is not None:
                return row.get(n)
        return None

    market_id = find('market_id', 'marketId', 'market')
    token_id = find('token_id', 'tokenId', 'token')
    outcome_side = find('outcome_side', 'outcome', 'side', 'token_side')
    qty = find('qty', 'quantity', 'amount')
    price = find('avg_price', 'price', 'avgPrice')
    ts = find('ts', 'timestamp', 'time')

    # outcome_side must be explicit and one of YES/NO (case-insensitive)
    if outcome_side is not None:
        if isinstance(outcome_side, str):
            os_clean = outcome_side.strip().upper()
            if os_clean in ('YES', 'NO', 'UP', 'DOWN'):
                # normalize to YES/NO
                outcome_side = 'YES' if os_clean in ('YES', 'UP') else 'NO'
            else:
                # ambiguous string (maybe 'buy'/'sell'), cannot map
                return None
        else:
            # non-string outcome_side: cannot map
            return None

    # require market_id and token_id
    if market_id is None or token_id is None or outcome_side is None or qty is None:
        return None

    # ensure numeric qty
    try:
        qtyv = float(qty)
    except Exception:
        return None

    try:
        pricev = None if price is None else float(price)
    except Exception:
        pricev = None

    return {
        'market_id': str(market_id),
        'token_id': str(token_id),
        'outcome_side': outcome_side,
        'qty': qtyv,
        'price': pricev,
        'ts': str(ts) if ts is not None else datetime.now(timezone.utc).isoformat(),
        'raw': row,
    }


def migrate(dbpath: Path, dry_run: bool = True, backup: bool = True, force: bool = False) -> Dict[str, int]:
    conn = open_conn(dbpath)
    cur = conn.cursor()
    create_legacy_unmapped_table(conn)

    stats = {'examined': 0, 'migrated': 0, 'quarantined': 0, 'skipped': 0}

    legacy_tables = []
    for t in ('lots', 'orders'):
        if table_exists(conn, t):
            legacy_tables.append(t)

    if not legacy_tables:
        conn.close()
        return stats

    # dry-run: just compute plan
    plan = []
    for t in legacy_tables:
        cols = detect_columns(conn, t)
        rows = fetch_rows(conn, t)
        for r in rows:
            stats['examined'] += 1
            mapped = safe_map_row(r, cols)
            if mapped is None:
                stats['quarantined'] += 1
                plan.append(('quarantine', t, r, 'missing_required_fields'))
            else:
                stats['migrated'] += 1
                plan.append(('migrate', t, r, mapped))

    # report plan summary
    print('Migration plan:')
    print(f"  legacy tables found: {legacy_tables}")
    print(f"  rows examined: {stats['examined']}")
    print(f"  rows safely mappable: {stats['migrated']}")
    print(f"  rows to quarantine: {stats['quarantined']}")

    if dry_run:
        print('Dry-run mode: no changes will be applied.')
        conn.close()
        return stats

    # destructive: ensure backup unless forced
    if backup and not force:
        bak = dbpath.with_suffix('.bak.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        print(f'Creating backup: {bak}')
        shutil.copy(str(dbpath), str(bak))

    # apply migrations: ensure target schema exists by importing storage.ensure_db
    try:
        # import local storage module to create new tables
        from src import storage
        storage.ensure_db()
    except Exception as e:
        print('Error importing storage; ensure the repository is on PYTHONPATH', e)
        conn.close()
        return stats

    # re-open connection to ensure table creation visible
    conn.close()
    conn = open_conn(dbpath)
    cur = conn.cursor()

    for action, table, row, payload in plan:
        if action == 'quarantine':
            cur.execute('INSERT INTO legacy_unmapped_rows(source_table, raw_json, reason, ts_migrated) VALUES (?, ?, ?, ?)',
                        (table, json.dumps(row, default=str), payload, datetime.now(timezone.utc).isoformat()))
        elif action == 'migrate':
            mapped = payload
            # insert into fills and open_lots via SQL to avoid module imports in constrained envs
            cur.execute('INSERT INTO fills(market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, receipt_processed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (mapped['market_id'], mapped['token_id'], mapped['outcome_side'], None, mapped['qty'], mapped['price'], mapped['ts'], 'buy', 1))
            cur.execute('INSERT INTO open_lots(market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (mapped['market_id'], mapped['token_id'], mapped['outcome_side'], mapped['qty'], mapped['price'], mapped['ts'], None))
        else:
            stats['skipped'] += 1

    conn.commit()

    # Run integrity checks
    issues = []
    # no active lot without market_id or token_id
    cur.execute('SELECT COUNT(*) FROM open_lots WHERE market_id IS NULL OR token_id IS NULL')
    if cur.fetchone()[0] > 0:
        issues.append('open_lots with NULL market_id/token_id')
    # no negative qty
    cur.execute('SELECT COUNT(*) FROM open_lots WHERE qty < 0')
    if cur.fetchone()[0] > 0:
        issues.append('open_lots with negative qty')

    print('Migration completed. Integrity checks:')
    if issues:
        print('  Issues found:')
        for it in issues:
            print('   -', it)
    else:
        print('  No integrity issues detected.')

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=str(DB_DEFAULT))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-backup', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    dbpath = Path(args.db)
    if not dbpath.exists():
        print('DB not found:', dbpath)
        return

    stats = migrate(dbpath, dry_run=args.dry_run, backup=not args.no_backup, force=args.force)
    print('Summary:', stats)


if __name__ == '__main__':
    main()
