import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3
import os
from src import storage
import migrate_inventory_legacy as mig


def setup_function(fn):
    os.environ['BOT_DB_PATH'] = str((ROOT / 'bot_state.db').resolve())
    try:
        os.remove(storage.get_db_path())
    except FileNotFoundError:
        pass
    # create legacy tables for testing
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    # legacy lots table (simple)
    cur.execute('''CREATE TABLE lots(id INTEGER PRIMARY KEY AUTOINCREMENT, token_id TEXT, market_id TEXT, qty REAL, avg_price REAL, ts TEXT, tx_hash TEXT, outcome_side TEXT)''')
    # legacy orders table (simple)
    cur.execute('''CREATE TABLE orders(id INTEGER PRIMARY KEY AUTOINCREMENT, side TEXT, qty REAL, price REAL, ts TEXT, token_id TEXT, market_id TEXT)''')
    conn.commit()
    conn.close()


def test_dry_run_leaves_db_unchanged():
    # insert a legacy row lacking outcome_side -> should be quarantined
    conn = sqlite3.connect('bot_state.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO lots(token_id, market_id, qty, avg_price, ts) VALUES (?, ?, ?, ?, ?)", ('TOK1', 'M1', 5.0, 0.6, 'ts1'))
    conn.commit()
    conn.close()

    stats = mig.migrate(storage.get_db_path(), dry_run=True)
    assert stats['examined'] == 1
    # ensure open_lots still not created
    storage.ensure_db()
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='open_lots'")
    assert cur.fetchone() is not None
    cur.execute('SELECT COUNT(*) FROM open_lots')
    assert cur.fetchone()[0] == 0
    conn.close()


def test_migrate_clearly_mappable_row():
    # create a mappable row with explicit outcome_side
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("INSERT INTO lots(token_id, market_id, qty, avg_price, ts, outcome_side) VALUES (?, ?, ?, ?, ?, ?)", ('TOK2', 'M2', 3.0, 0.55, 'ts2', 'YES'))
    conn.commit()
    conn.close()

    stats = mig.migrate(storage.get_db_path(), dry_run=False, force=True)
    assert stats['migrated'] >= 1
    # ensure open_lots has migrated entry
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM open_lots WHERE market_id = ? AND token_id = ?', ('M2', 'TOK2'))
    assert cur.fetchone()[0] == 1
    conn.close()


def test_quarantine_missing_fields():
    # insert an ambiguous row missing token_id
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute("INSERT INTO lots(token_id, market_id, qty, avg_price, ts) VALUES (?, ?, ?, ?, ?)", (None, None, 2.0, 0.5, 'ts3'))
    conn.commit()
    conn.close()

    stats = mig.migrate(storage.get_db_path(), dry_run=False, force=True)
    # ensure quarantine table contains entries
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM legacy_unmapped_rows')
    assert cur.fetchone()[0] >= 1
    conn.close()
