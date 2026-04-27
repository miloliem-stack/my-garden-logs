"""Simple SQLite storage stub for persistence (lots, prices, model state).
This is a minimal placeholder; expand schema as needed.
"""
import sqlite3
import os
from pathlib import Path
import json
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone


DB_PATH = Path.cwd() / 'bot_state.db'
ERC20_TRANSFER_SIG = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
ORDER_TABLE = 'bot_orders'
ORDER_FILL_TABLE = 'bot_order_fills'
RESERVATION_TABLE = 'bot_reservations'
ORDER_EVENT_TABLE = 'bot_order_events'
TRADE_JOURNAL_TABLE = 'trade_journal'
INVENTORY_DISPOSAL_TABLE = 'inventory_disposals'
POSITION_MANAGEMENT_STATE_TABLE = 'position_management_state'
REGIME_OBSERVATION_TABLE = 'regime_observations'
LOT_REGIME_ATTRIBUTION_TABLE = 'lot_regime_attribution'
REALIZED_PNL_EVENT_TABLE = 'realized_pnl_events'
ORDER_STATES = {
    'pending_submit',
    'submitted',
    'open',
    'partially_filled',
    'filled',
    'cancel_requested',
    'canceled',
    'expired',
    'rejected',
    'failed',
    'unknown',
    'not_found_on_venue',
    'manual_review',
    'dust_ignored',
    'dust_finalized',
}
ACTIVE_ORDER_STATUSES = {'pending_submit', 'submitted', 'open', 'partially_filled', 'cancel_requested', 'unknown'}
TERMINAL_ORDER_STATUSES = {'filled', 'canceled', 'expired', 'rejected', 'failed', 'not_found_on_venue', 'manual_review', 'dust_ignored', 'dust_finalized'}
ORDER_STATE_TRANSITIONS = {
    'pending_submit': {'submitted', 'open', 'partially_filled', 'filled', 'canceled', 'failed', 'rejected', 'unknown', 'not_found_on_venue', 'dust_ignored', 'dust_finalized'},
    'submitted': {'open', 'partially_filled', 'filled', 'cancel_requested', 'canceled', 'failed', 'rejected', 'expired', 'unknown', 'not_found_on_venue', 'dust_ignored', 'dust_finalized'},
    'open': {'partially_filled', 'filled', 'cancel_requested', 'canceled', 'expired', 'unknown', 'not_found_on_venue', 'dust_ignored', 'dust_finalized'},
    'partially_filled': {'filled', 'cancel_requested', 'canceled', 'expired', 'unknown', 'not_found_on_venue', 'dust_ignored', 'dust_finalized'},
    'cancel_requested': {'canceled', 'filled', 'partially_filled', 'expired', 'unknown', 'not_found_on_venue', 'dust_ignored', 'dust_finalized'},
    'unknown': {'submitted', 'open', 'partially_filled', 'filled', 'cancel_requested', 'canceled', 'expired', 'failed', 'rejected', 'unknown', 'not_found_on_venue', 'manual_review', 'dust_ignored', 'dust_finalized'},
    'not_found_on_venue': {'partially_filled', 'filled', 'manual_review', 'dust_ignored', 'dust_finalized'},
    'manual_review': {'unknown'},
    'dust_ignored': {'unknown'},
    'dust_finalized': {'unknown'},
    'filled': set(),
    'canceled': set(),
    'expired': set(),
    'rejected': set(),
    'failed': set(),
}
MARKET_STATUS_RANK = {
    'open': 1,
    'closed': 2,
    'resolved': 3,
    'redeemed': 4,
    'archived': 5,
}
TX_HASH_RE = re.compile(r'^0x[a-fA-F0-9]{64}$')
ERC1155_SHARE_SCALE = 1_000_000.0
ORDER_QTY_TOLERANCE = 1e-12
DORMANT_LOT_TABLE = 'dormant_lots'
DORMANT_LOT_ACTIVE_STATUSES = {'dust_ignored', 'dust_finalized'}


def get_dust_qty_threshold() -> float:
    return float(os.getenv('DUST_QTY_THRESHOLD', '0.1'))


def is_dust_qty(qty: float, threshold: Optional[float] = None) -> bool:
    dust_threshold = get_dust_qty_threshold() if threshold is None else float(threshold)
    value = abs(float(qty or 0.0))
    return ORDER_QTY_TOLERANCE < value < dust_threshold


def is_dust_order(order: Dict, threshold: Optional[float] = None) -> bool:
    if not isinstance(order, dict):
        return False
    remaining_qty = float(order.get('remaining_qty') or 0.0)
    return remaining_qty > ORDER_QTY_TOLERANCE and is_dust_qty(remaining_qty, threshold=threshold)


def is_dust_inventory_lot(net_qty: float, threshold: Optional[float] = None) -> bool:
    return is_dust_qty(net_qty, threshold=threshold)


def _json_dumps(value):
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _json_loads(value, default=None):
    if value in (None, ''):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _ensure_column(cur, table: str, column: str, ddl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cur.fetchall()}
    if column not in columns:
        cur.execute(f'ALTER TABLE {table} ADD COLUMN {column} {ddl}')


def _normalize_erc1155_transfer_qty(raw_qty: float, reference_qtys: Optional[List[float]] = None) -> float:
    qty = float(raw_qty or 0.0)
    scaled_qty = qty / ERC1155_SHARE_SCALE
    refs = [abs(float(ref)) for ref in (reference_qtys or []) if ref is not None and abs(float(ref)) > ORDER_QTY_TOLERANCE]
    if refs:
        if any(abs(qty - ref) <= 1e-9 for ref in refs):
            return qty
        if any(abs(scaled_qty - ref) <= 1e-9 for ref in refs):
            return scaled_qty
        if qty > max(refs) + 1e-9 and scaled_qty <= max(refs) + 1e-9:
            return scaled_qty
    if abs(qty - round(qty)) <= 1e-9 and qty >= ERC1155_SHARE_SCALE:
        return scaled_qty
    return qty


def get_db_path() -> Path:
    override = os.getenv('BOT_DB_PATH')
    if override:
        return Path(override).expanduser()
    return DB_PATH


def ensure_db():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # prices table retained
    cur.execute('''CREATE TABLE IF NOT EXISTS prices(ts TEXT PRIMARY KEY, price REAL)''')

    # markets table: track lifecycle and metadata
    cur.execute('''CREATE TABLE IF NOT EXISTS markets(
        market_id TEXT PRIMARY KEY,
        condition_id TEXT,
        slug TEXT,
        title TEXT,
        start_time TEXT,
        end_time TEXT,
        status TEXT,
        winning_outcome TEXT,
        last_checked_ts TEXT,
        last_redeem_ts TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS market_tokens(
        market_id TEXT PRIMARY KEY,
        condition_id TEXT,
        token_yes TEXT,
        token_no TEXT,
        start_time TEXT,
        end_time TEXT,
        discovered_ts TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS series_runtime_state(
        series_id TEXT PRIMARY KEY,
        active_market_id TEXT,
        active_token_yes TEXT,
        active_token_no TEXT,
        active_start_time TEXT,
        active_end_time TEXT,
        strike_price REAL,
        strike_source TEXT,
        strike_fixed_ts TEXT,
        last_switch_ts TEXT,
        status TEXT,
        previous_market_id TEXT
    )''')

    # fills: append-only execution facts (buys/sells/redeems). Must include market_id and token_id
    cur.execute('''CREATE TABLE IF NOT EXISTS fills(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        tx_hash TEXT,
        qty REAL,
        price REAL,
        ts TEXT,
        kind TEXT,
        receipt_processed INTEGER DEFAULT 0,
        extra_json TEXT
    )''')

    # open_lots: current live inventory per market/outcome (consumed on sells/redeem via FIFO)
    cur.execute('''CREATE TABLE IF NOT EXISTS open_lots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        qty REAL,
        avg_price REAL,
        ts TEXT,
        tx_hash TEXT
    )''')

    # redeemed_lots: history of redeemed inventory for audit
    cur.execute('''CREATE TABLE IF NOT EXISTS redeemed_lots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        qty REAL,
        avg_price REAL,
        ts TEXT,
        tx_hash TEXT,
        redeem_tx_hash TEXT,
        redeem_receipt_json TEXT
    )''')

    # merged_lots: history of paired token merges for audit
    cur.execute('''CREATE TABLE IF NOT EXISTS merged_lots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        qty REAL,
        avg_price REAL,
        ts TEXT,
        tx_hash TEXT,
        merge_tx_hash TEXT,
        extra_json TEXT
    )''')

    # receipts: raw on-chain receipts for reconciliation
    cur.execute('''CREATE TABLE IF NOT EXISTS receipts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT,
        raw_json TEXT,
        parsed INTEGER DEFAULT 0,
        ts TEXT
    )''')

    # track reconciliation issues for operator review
    cur.execute('''CREATE TABLE IF NOT EXISTS reconciliation_issues(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash TEXT,
        fill_id INTEGER,
        observed_json TEXT,
        expected_json TEXT,
        reason TEXT,
        ts TEXT
    )''')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {ORDER_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_order_id TEXT NOT NULL UNIQUE,
        venue_order_id TEXT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        side TEXT NOT NULL,
        requested_qty REAL NOT NULL,
        limit_price REAL,
        filled_qty REAL DEFAULT 0,
        remaining_qty REAL NOT NULL,
        status TEXT NOT NULL,
        tx_hash TEXT,
        created_ts TEXT NOT NULL,
        updated_ts TEXT NOT NULL,
        raw_response_json TEXT,
        decision_context_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{ORDER_TABLE}_market_status ON {ORDER_TABLE}(market_id, status)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{ORDER_TABLE}_tx_hash ON {ORDER_TABLE}(tx_hash)')
    _ensure_column(cur, ORDER_TABLE, 'decision_context_json', 'TEXT')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {ORDER_FILL_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        venue_fill_id TEXT,
        tx_hash TEXT,
        fill_qty REAL NOT NULL,
        cumulative_filled_qty REAL NOT NULL,
        price REAL,
        fill_ts TEXT NOT NULL,
        raw_json TEXT,
        UNIQUE(order_id, cumulative_filled_qty, tx_hash)
    )''')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {RESERVATION_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        market_id TEXT NOT NULL,
        token_id TEXT,
        outcome_side TEXT,
        reservation_type TEXT NOT NULL,
        qty REAL NOT NULL,
        status TEXT NOT NULL,
        created_ts TEXT NOT NULL,
        updated_ts TEXT NOT NULL,
        released_ts TEXT,
        extra_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{RESERVATION_TABLE}_active ON {RESERVATION_TABLE}(market_id, token_id, outcome_side, status)')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {ORDER_EVENT_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        old_status TEXT,
        new_status TEXT,
        request_json TEXT,
        response_json TEXT,
        error_text TEXT,
        ts TEXT NOT NULL
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{ORDER_EVENT_TABLE}_order_ts ON {ORDER_EVENT_TABLE}(order_id, ts)')
    ensure_trade_journal_schema(cur)

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {INVENTORY_DISPOSAL_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        policy_type TEXT NOT NULL,
        action TEXT NOT NULL,
        request_json TEXT,
        response_json TEXT,
        tx_hash TEXT,
        receipt_json TEXT,
        classification TEXT,
        failure_reason TEXT,
        ts TEXT NOT NULL
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{INVENTORY_DISPOSAL_TABLE}_market_ts ON {INVENTORY_DISPOSAL_TABLE}(market_id, ts)')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {POSITION_MANAGEMENT_STATE_TABLE}(
        market_id TEXT PRIMARY KEY,
        add_count INTEGER NOT NULL DEFAULT 0,
        reduce_count INTEGER NOT NULL DEFAULT 0,
        flip_count INTEGER NOT NULL DEFAULT 0,
        last_action TEXT,
        last_action_ts TEXT,
        last_action_reason TEXT,
        last_seen_side TEXT,
        persistence_target_action TEXT,
        persistence_count INTEGER NOT NULL DEFAULT 0,
        updated_ts TEXT NOT NULL
    )''')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {DORMANT_LOT_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        qty REAL NOT NULL,
        avg_price REAL,
        ts TEXT,
        tx_hash TEXT,
        source_open_lot_id INTEGER,
        linked_order_id INTEGER,
        dormant_status TEXT NOT NULL,
        dormant_reason TEXT,
        created_ts TEXT NOT NULL,
        updated_ts TEXT NOT NULL,
        restored_ts TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{DORMANT_LOT_TABLE}_market_status ON {DORMANT_LOT_TABLE}(market_id, dormant_status, outcome_side)')

    cur.execute('''CREATE TABLE IF NOT EXISTS venue_probe_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_ts TEXT NOT NULL,
        finished_ts TEXT,
        host TEXT,
        api_base TEXT,
        wallet_address TEXT,
        live_mode INTEGER NOT NULL DEFAULT 0,
        write_enabled INTEGER NOT NULL DEFAULT 0,
        summary_json TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS venue_probe_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_name TEXT NOT NULL,
        transport TEXT NOT NULL,
        direction TEXT NOT NULL,
        method TEXT,
        path TEXT,
        url TEXT,
        channel TEXT,
        http_status INTEGER,
        ok INTEGER,
        latency_ms REAL,
        request_headers_json TEXT,
        request_body_json TEXT,
        response_headers_json TEXT,
        response_body_json TEXT,
        error_text TEXT,
        classification TEXT,
        ts TEXT NOT NULL
    )''')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_venue_probe_events_run_step ON venue_probe_events(run_id, step_name, ts)')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {REGIME_OBSERVATION_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        market_id TEXT NOT NULL,
        token_yes TEXT,
        token_no TEXT,
        regime_label TEXT NOT NULL,
        trend_score REAL,
        tail_score REAL,
        reversal_score REAL,
        regime_reason TEXT,
        persistence_count INTEGER NOT NULL DEFAULT 0,
        decision_state_json TEXT,
        source_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{REGIME_OBSERVATION_TABLE}_market_ts ON {REGIME_OBSERVATION_TABLE}(market_id, ts)')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {LOT_REGIME_ATTRIBUTION_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        open_lot_id INTEGER NOT NULL,
        order_id INTEGER,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        entry_ts TEXT NOT NULL,
        entry_regime_label TEXT,
        entry_regime_scores_json TEXT,
        decision_state_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{LOT_REGIME_ATTRIBUTION_TABLE}_lot ON {LOT_REGIME_ATTRIBUTION_TABLE}(open_lot_id)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{LOT_REGIME_ATTRIBUTION_TABLE}_market ON {LOT_REGIME_ATTRIBUTION_TABLE}(market_id, token_id, outcome_side)')

    cur.execute(f'''CREATE TABLE IF NOT EXISTS {REALIZED_PNL_EVENT_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        market_id TEXT NOT NULL,
        token_id TEXT NOT NULL,
        outcome_side TEXT NOT NULL,
        qty REAL NOT NULL,
        disposition_type TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        gross_pnl REAL,
        net_pnl REAL,
        fee_slippage_buffer REAL,
        source_fill_id INTEGER,
        source_disposal_id INTEGER,
        source_tx_hash TEXT,
        open_lot_id INTEGER,
        entry_regime_label TEXT,
        entry_regime_scores_json TEXT,
        exit_regime_label TEXT,
        exit_regime_scores_json TEXT,
        extra_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{REALIZED_PNL_EVENT_TABLE}_market_ts ON {REALIZED_PNL_EVENT_TABLE}(market_id, ts)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{REALIZED_PNL_EVENT_TABLE}_entry_regime ON {REALIZED_PNL_EVENT_TABLE}(entry_regime_label)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{REALIZED_PNL_EVENT_TABLE}_exit_regime ON {REALIZED_PNL_EVENT_TABLE}(exit_regime_label)')

    conn.commit()
    conn.close()


def ensure_trade_journal_schema(cur=None):
    owns_connection = cur is None
    conn = None
    if owns_connection:
        conn = sqlite3.connect(get_db_path())
        cur = conn.cursor()
    cur.execute(f'''CREATE TABLE IF NOT EXISTS {TRADE_JOURNAL_TABLE}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        market_id TEXT NOT NULL,
        token_id TEXT,
        outcome_side TEXT,
        side TEXT,
        kind TEXT NOT NULL,
        qty REAL NOT NULL,
        price REAL,
        notional REAL,
        tx_hash TEXT,
        order_id INTEGER,
        client_order_id TEXT,
        venue_order_id TEXT,
        decision_ts TEXT,
        decision_reason TEXT,
        policy_bucket TEXT,
        tau_minutes INTEGER,
        raw_p_yes REAL,
        raw_p_no REAL,
        adjusted_p_yes REAL,
        adjusted_p_no REAL,
        raw_edge_yes REAL,
        raw_edge_no REAL,
        adjusted_edge_yes REAL,
        adjusted_edge_no REAL,
        tail_penalty_score REAL,
        tail_hard_block INTEGER,
        reeval_action TEXT,
        reeval_reason TEXT,
        realized_pnl REAL,
        extra_json TEXT
    )''')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{TRADE_JOURNAL_TABLE}_ts ON {TRADE_JOURNAL_TABLE}(ts)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{TRADE_JOURNAL_TABLE}_market_ts ON {TRADE_JOURNAL_TABLE}(market_id, ts)')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{TRADE_JOURNAL_TABLE}_kind_ts ON {TRADE_JOURNAL_TABLE}(kind, ts)')
    if owns_connection and conn is not None:
        conn.commit()
        conn.close()


def insert_price(ts: str, price: float):
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO prices(ts, price) VALUES (?, ?)', (ts, price))
    conn.commit()
    conn.close()


def create_probe_run(
    *,
    started_ts: Optional[str] = None,
    host: Optional[str] = None,
    api_base: Optional[str] = None,
    wallet_address: Optional[str] = None,
    live_mode: bool = False,
    write_enabled: bool = False,
    summary: Optional[Dict] = None,
) -> int:
    started_ts = started_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO venue_probe_runs(started_ts, host, api_base, wallet_address, live_mode, write_enabled, summary_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            started_ts,
            host,
            api_base,
            wallet_address,
            int(bool(live_mode)),
            int(bool(write_enabled)),
            _json_dumps(summary),
        ),
    )
    run_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return run_id


def append_probe_event(
    run_id: int,
    *,
    step_name: str,
    transport: str,
    direction: str,
    method: Optional[str] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
    channel: Optional[str] = None,
    http_status: Optional[int] = None,
    ok: Optional[bool] = None,
    latency_ms: Optional[float] = None,
    request_headers: Optional[Dict] = None,
    request_body: Optional[Dict] = None,
    response_headers: Optional[Dict] = None,
    response_body: Optional[Dict] = None,
    error_text: Optional[str] = None,
    classification: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    ts = ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''INSERT INTO venue_probe_events(
            run_id, step_name, transport, direction, method, path, url, channel, http_status, ok, latency_ms,
            request_headers_json, request_body_json, response_headers_json, response_body_json, error_text, classification, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            run_id,
            step_name,
            transport,
            direction,
            method,
            path,
            url,
            channel,
            http_status,
            None if ok is None else int(bool(ok)),
            latency_ms,
            _json_dumps(request_headers),
            _json_dumps(request_body),
            _json_dumps(response_headers),
            _json_dumps(response_body),
            error_text,
            classification,
            ts,
        ),
    )
    event_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return event_id


def finish_probe_run(run_id: int, *, finished_ts: Optional[str] = None, summary: Optional[Dict] = None) -> None:
    finished_ts = finished_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        'UPDATE venue_probe_runs SET finished_ts = ?, summary_json = ? WHERE id = ?',
        (finished_ts, _json_dumps(summary), run_id),
    )
    conn.commit()
    conn.close()


def get_probe_run(run_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        'SELECT id, started_ts, finished_ts, host, api_base, wallet_address, live_mode, write_enabled, summary_json FROM venue_probe_runs WHERE id = ?',
        (run_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row[0],
        'started_ts': row[1],
        'finished_ts': row[2],
        'host': row[3],
        'api_base': row[4],
        'wallet_address': row[5],
        'live_mode': bool(row[6]),
        'write_enabled': bool(row[7]),
        'summary_json': row[8],
        'summary': json.loads(row[8]) if row[8] else None,
    }


def list_probe_events(run_id: int, step_name: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    if step_name is None:
        cur.execute(
            '''SELECT id, run_id, step_name, transport, direction, method, path, url, channel, http_status, ok, latency_ms,
                      request_headers_json, request_body_json, response_headers_json, response_body_json, error_text, classification, ts
               FROM venue_probe_events WHERE run_id = ? ORDER BY id''',
            (run_id,),
        )
    else:
        cur.execute(
            '''SELECT id, run_id, step_name, transport, direction, method, path, url, channel, http_status, ok, latency_ms,
                      request_headers_json, request_body_json, response_headers_json, response_body_json, error_text, classification, ts
               FROM venue_probe_events WHERE run_id = ? AND step_name = ? ORDER BY id''',
            (run_id, step_name),
        )
    rows = cur.fetchall()
    conn.close()
    events = []
    for row in rows:
        events.append(
            {
                'id': row[0],
                'run_id': row[1],
                'step_name': row[2],
                'transport': row[3],
                'direction': row[4],
                'method': row[5],
                'path': row[6],
                'url': row[7],
                'channel': row[8],
                'http_status': row[9],
                'ok': None if row[10] is None else bool(row[10]),
                'latency_ms': row[11],
                'request_headers_json': row[12],
                'request_body_json': row[13],
                'response_headers_json': row[14],
                'response_body_json': row[15],
                'error_text': row[16],
                'classification': row[17],
                'ts': row[18],
                'request_headers': json.loads(row[12]) if row[12] else None,
                'request_body': json.loads(row[13]) if row[13] else None,
                'response_headers': json.loads(row[14]) if row[14] else None,
                'response_body': json.loads(row[15]) if row[15] else None,
            }
        )
    return events


def insert_fill(market_id: str, token_id: str, outcome_side: str, qty: float, price: float, ts: str, tx_hash: Optional[str] = None, kind: str = 'trade', extra: Optional[Dict] = None):
    if not market_id or not token_id:
        raise ValueError('market_id and token_id are required for fills')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('INSERT INTO fills(market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, json.dumps(extra) if extra is not None else None))
    conn.commit()
    conn.close()


def list_fills(market_id: Optional[str] = None, kind: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = 'SELECT id, market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, receipt_processed, extra_json FROM fills'
    params: List = []
    clauses = []
    if market_id is not None:
        clauses.append('market_id = ?')
        params.append(market_id)
    if kind is not None:
        clauses.append('kind = ?')
        params.append(kind)
    if clauses:
        q += ' WHERE ' + ' AND '.join(clauses)
    q += ' ORDER BY ts ASC, id ASC'
    if limit is not None:
        q += ' LIMIT ?'
        params.append(int(limit))
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'market_id': row[1],
            'token_id': row[2],
            'outcome_side': row[3],
            'tx_hash': row[4],
            'qty': float(row[5] or 0.0),
            'price': None if row[6] is None else float(row[6]),
            'ts': row[7],
            'kind': row[8],
            'receipt_processed': int(row[9] or 0),
            'extra_json': row[10],
            'extra': _json_loads(row[10], default={}) or {},
        }
        for row in rows
    ]


def create_order(client_order_id: str, market_id: str, token_id: str, outcome_side: str, side: str, requested_qty: float, limit_price: Optional[float], status: str, created_ts: str, venue_order_id: Optional[str] = None, tx_hash: Optional[str] = None, raw_response: Optional[Dict] = None, decision_context: Optional[Dict] = None) -> Dict:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT OR IGNORE INTO {ORDER_TABLE}(
            client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price,
            filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json, decision_context_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            client_order_id,
            venue_order_id,
            market_id,
            token_id,
            outcome_side,
            side,
            float(requested_qty),
            limit_price,
            0.0,
            float(requested_qty),
            status,
            tx_hash,
            created_ts,
            created_ts,
            json.dumps(raw_response) if raw_response is not None else None,
            _json_dumps(decision_context),
        ),
    )
    conn.commit()
    conn.close()
    return get_order(client_order_id=client_order_id)


def get_order(order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Optional[Dict]:
    if order_id is None and client_order_id is None:
        raise ValueError('order_id or client_order_id is required')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    if order_id is not None:
        cur.execute(f'SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json, decision_context_json FROM {ORDER_TABLE} WHERE id = ?', (order_id,))
    else:
        cur.execute(f'SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json, decision_context_json FROM {ORDER_TABLE} WHERE client_order_id = ?', (client_order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row[0],
        'client_order_id': row[1],
        'venue_order_id': row[2],
        'market_id': row[3],
        'token_id': row[4],
        'outcome_side': row[5],
        'side': row[6],
        'requested_qty': float(row[7]),
        'limit_price': None if row[8] is None else float(row[8]),
        'filled_qty': float(row[9]),
        'remaining_qty': float(row[10]),
        'status': row[11],
        'tx_hash': row[12],
        'created_ts': row[13],
        'updated_ts': row[14],
        'raw_response_json': row[15],
        'raw_response': _json_loads(row[15], default={}) or {},
        'decision_context_json': row[16],
        'decision_context': _json_loads(row[16], default={}) or {},
    }


def list_orders(market_id: Optional[str] = None, tx_hash: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price,
                   filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json
            FROM {ORDER_TABLE}'''
    params: List = []
    clauses = []
    if market_id is not None:
        clauses.append('market_id = ?')
        params.append(market_id)
    if tx_hash is not None:
        clauses.append('tx_hash = ?')
        params.append(tx_hash)
    if clauses:
        q += ' WHERE ' + ' AND '.join(clauses)
    q += ' ORDER BY created_ts ASC, id ASC'
    if limit is not None:
        q += ' LIMIT ?'
        params.append(int(limit))
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'client_order_id': r[1], 'venue_order_id': r[2], 'market_id': r[3], 'token_id': r[4],
            'outcome_side': r[5], 'side': r[6], 'requested_qty': float(r[7]), 'limit_price': None if r[8] is None else float(r[8]),
            'filled_qty': float(r[9]), 'remaining_qty': float(r[10]), 'status': r[11], 'tx_hash': r[12], 'created_ts': r[13], 'updated_ts': r[14],
            'raw_response_json': r[15], 'raw_response': _json_loads(r[15], default={}) or {}
        }
        for r in rows
    ]


def _order_raw_response(order: Dict) -> Dict:
    try:
        return json.loads(order.get('raw_response_json') or '{}')
    except Exception:
        return {}


def _expected_buy_exposure(order: Dict, remaining_qty: Optional[float] = None, fallback_price: Optional[float] = None) -> float:
    qty = max(0.0, float(order['remaining_qty'] if remaining_qty is None else remaining_qty))
    if qty <= 0:
        return 0.0
    raw = _order_raw_response(order)
    submitted_notional = raw.get('quantized_notional')
    requested_qty = max(0.0, float(order.get('requested_qty') or 0.0))
    if submitted_notional is not None and requested_qty > 0:
        return max(0.0, qty * (float(submitted_notional) / requested_qty))
    price = order.get('limit_price')
    if price is None:
        price = fallback_price
    return max(0.0, qty * float(price or 0.0))


def update_order(order_id: int, *, venue_order_id: Optional[str] = None, status: Optional[str] = None, tx_hash: Optional[str] = None, raw_response: Optional[Dict] = None, filled_qty: Optional[float] = None, remaining_qty: Optional[float] = None, updated_ts: Optional[str] = None):
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''UPDATE {ORDER_TABLE} SET venue_order_id = ?, status = ?, tx_hash = ?, raw_response_json = ?, filled_qty = ?, remaining_qty = ?, updated_ts = ? WHERE id = ?''',
        (
            venue_order_id if venue_order_id is not None else order['venue_order_id'],
            status if status is not None else order['status'],
            tx_hash if tx_hash is not None else order['tx_hash'],
            json.dumps(raw_response) if raw_response is not None else order['raw_response_json'],
            float(filled_qty) if filled_qty is not None else order['filled_qty'],
            float(remaining_qty) if remaining_qty is not None else order['remaining_qty'],
            updated_ts or datetime.now(timezone.utc).isoformat(),
            order_id,
        ),
    )
    conn.commit()
    conn.close()
    return get_order(order_id=order_id)


def append_order_event(order_id: int, event_type: str, old_status: Optional[str] = None, new_status: Optional[str] = None, request: Optional[Dict] = None, response: Optional[Dict] = None, error_text: Optional[str] = None, ts: Optional[str] = None) -> int:
    ts = ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {ORDER_EVENT_TABLE}(order_id, event_type, old_status, new_status, request_json, response_json, error_text, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            order_id,
            event_type,
            old_status,
            new_status,
            _json_dumps(request),
            _json_dumps(response),
            error_text,
            ts,
        ),
    )
    event_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return event_id


def list_order_events(order_id: int) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, order_id, event_type, old_status, new_status, request_json, response_json, error_text, ts
            FROM {ORDER_EVENT_TABLE} WHERE order_id = ? ORDER BY id ASC''',
        (order_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'order_id': row[1],
            'event_type': row[2],
            'old_status': row[3],
            'new_status': row[4],
            'request_json': row[5],
            'response_json': row[6],
            'error_text': row[7],
            'ts': row[8],
            'request': json.loads(row[5]) if row[5] else None,
            'response': json.loads(row[6]) if row[6] else None,
        }
        for row in rows
    ]


def get_unknown_since_ts(order_id: int) -> Optional[str]:
    order = get_order(order_id=order_id)
    if order is None:
        return None
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT ts
            FROM {ORDER_EVENT_TABLE}
            WHERE order_id = ?
              AND (
                event_type = 'submit_ambiguous'
                OR (new_status = 'unknown' AND COALESCE(old_status, '') != 'unknown')
              )
            ORDER BY ts ASC, id ASC
            LIMIT 1''',
        (order_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return order.get('created_ts')


def record_inventory_disposal(
    market_id: str,
    *,
    policy_type: str,
    action: str,
    request: Optional[Dict] = None,
    response: Optional[Dict] = None,
    tx_hash: Optional[str] = None,
    receipt: Optional[Dict] = None,
    classification: Optional[str] = None,
    failure_reason: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    ts = ts or datetime.now(timezone.utc).isoformat()
    payload = (
        market_id,
        policy_type,
        action,
        _json_dumps(request),
        _json_dumps(response),
        tx_hash,
        _json_dumps(receipt),
        classification,
        failure_reason,
        ts,
    )
    for attempt in range(2):
        conn = sqlite3.connect(get_db_path())
        cur = conn.cursor()
        try:
            cur.execute(
                f'''INSERT INTO {INVENTORY_DISPOSAL_TABLE}(market_id, policy_type, action, request_json, response_json, tx_hash, receipt_json, classification, failure_reason, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                payload,
            )
            row_id = int(cur.lastrowid)
            conn.commit()
            conn.close()
            return row_id
        except sqlite3.OperationalError as exc:
            conn.close()
            if attempt == 0 and 'no such table' in str(exc).lower():
                ensure_db()
                continue
            raise
    raise RuntimeError('failed to record inventory disposal')


def get_position_management_state(market_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT market_id, add_count, reduce_count, flip_count, last_action, last_action_ts, last_action_reason,
                   last_seen_side, persistence_target_action, persistence_count, updated_ts
            FROM {POSITION_MANAGEMENT_STATE_TABLE}
            WHERE market_id = ?''',
        (market_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        'market_id': row[0],
        'add_count': int(row[1] or 0),
        'reduce_count': int(row[2] or 0),
        'flip_count': int(row[3] or 0),
        'last_action': row[4],
        'last_action_ts': row[5],
        'last_action_reason': row[6],
        'last_seen_side': row[7],
        'persistence_target_action': row[8],
        'persistence_count': int(row[9] or 0),
        'updated_ts': row[10],
    }


def upsert_position_management_state(
    market_id: str,
    *,
    add_count: Optional[int] = None,
    reduce_count: Optional[int] = None,
    flip_count: Optional[int] = None,
    last_action: Optional[str] = None,
    last_action_ts: Optional[str] = None,
    last_action_reason: Optional[str] = None,
    last_seen_side: Optional[str] = None,
    persistence_target_action: Optional[str] = None,
    persistence_count: Optional[int] = None,
    updated_ts: Optional[str] = None,
) -> Dict:
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    existing = get_position_management_state(market_id) or {
        'market_id': market_id,
        'add_count': 0,
        'reduce_count': 0,
        'flip_count': 0,
        'last_action': None,
        'last_action_ts': None,
        'last_action_reason': None,
        'last_seen_side': None,
        'persistence_target_action': None,
        'persistence_count': 0,
        'updated_ts': ts,
    }
    payload = {
        'market_id': market_id,
        'add_count': existing['add_count'] if add_count is None else int(add_count),
        'reduce_count': existing['reduce_count'] if reduce_count is None else int(reduce_count),
        'flip_count': existing['flip_count'] if flip_count is None else int(flip_count),
        'last_action': existing['last_action'] if last_action is None else last_action,
        'last_action_ts': existing['last_action_ts'] if last_action_ts is None else last_action_ts,
        'last_action_reason': existing['last_action_reason'] if last_action_reason is None else last_action_reason,
        'last_seen_side': existing['last_seen_side'] if last_seen_side is None else last_seen_side,
        'persistence_target_action': existing['persistence_target_action'] if persistence_target_action is None else persistence_target_action,
        'persistence_count': existing['persistence_count'] if persistence_count is None else int(persistence_count),
        'updated_ts': ts,
    }
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {POSITION_MANAGEMENT_STATE_TABLE}(
                market_id, add_count, reduce_count, flip_count, last_action, last_action_ts, last_action_reason,
                last_seen_side, persistence_target_action, persistence_count, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                add_count = excluded.add_count,
                reduce_count = excluded.reduce_count,
                flip_count = excluded.flip_count,
                last_action = excluded.last_action,
                last_action_ts = excluded.last_action_ts,
                last_action_reason = excluded.last_action_reason,
                last_seen_side = excluded.last_seen_side,
                persistence_target_action = excluded.persistence_target_action,
                persistence_count = excluded.persistence_count,
                updated_ts = excluded.updated_ts''',
        (
            payload['market_id'],
            payload['add_count'],
            payload['reduce_count'],
            payload['flip_count'],
            payload['last_action'],
            payload['last_action_ts'],
            payload['last_action_reason'],
            payload['last_seen_side'],
            payload['persistence_target_action'],
            payload['persistence_count'],
            payload['updated_ts'],
        ),
    )
    conn.commit()
    conn.close()
    return get_position_management_state(market_id) or payload


def reset_position_management_persistence(market_id: str, updated_ts: Optional[str] = None) -> None:
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    upsert_position_management_state(
        market_id,
        persistence_target_action='hold',
        persistence_count=0,
        updated_ts=ts,
    )


def record_position_management_action(
    market_id: str,
    action: str,
    reason: str,
    ts: Optional[str] = None,
    increment_add: bool = False,
    increment_reduce: bool = False,
    increment_flip: bool = False,
    last_seen_side: Optional[str] = None,
) -> Dict:
    action_ts = ts or datetime.now(timezone.utc).isoformat()
    state = get_position_management_state(market_id) or {
        'add_count': 0,
        'reduce_count': 0,
        'flip_count': 0,
    }
    return upsert_position_management_state(
        market_id,
        add_count=int(state.get('add_count') or 0) + (1 if increment_add else 0),
        reduce_count=int(state.get('reduce_count') or 0) + (1 if increment_reduce else 0),
        flip_count=int(state.get('flip_count') or 0) + (1 if increment_flip else 0),
        last_action=action,
        last_action_ts=action_ts,
        last_action_reason=reason,
        last_seen_side=last_seen_side,
        persistence_target_action='hold',
        persistence_count=0,
        updated_ts=action_ts,
    )


def list_inventory_disposals(market_id: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, market_id, policy_type, action, request_json, response_json, tx_hash, receipt_json, classification, failure_reason, ts
            FROM {INVENTORY_DISPOSAL_TABLE}'''
    params: List = []
    if market_id is not None:
        q += ' WHERE market_id = ?'
        params.append(market_id)
    q += ' ORDER BY id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'market_id': row[1],
            'policy_type': row[2],
            'action': row[3],
            'request': json.loads(row[4]) if row[4] else None,
            'response': json.loads(row[5]) if row[5] else None,
            'tx_hash': row[6],
            'receipt': json.loads(row[7]) if row[7] else None,
            'classification': row[8],
            'failure_reason': row[9],
            'ts': row[10],
        }
        for row in rows
    ]


def record_regime_observation(
    *,
    ts: str,
    market_id: str,
    token_yes: Optional[str],
    token_no: Optional[str],
    regime_label: str,
    trend_score: Optional[float],
    tail_score: Optional[float],
    reversal_score: Optional[float],
    regime_reason: Optional[str],
    persistence_count: int,
    decision_state: Optional[Dict] = None,
    source: Optional[Dict] = None,
) -> int:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {REGIME_OBSERVATION_TABLE}(
            ts, market_id, token_yes, token_no, regime_label, trend_score, tail_score, reversal_score,
            regime_reason, persistence_count, decision_state_json, source_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            ts,
            market_id,
            token_yes,
            token_no,
            regime_label,
            trend_score,
            tail_score,
            reversal_score,
            regime_reason,
            int(persistence_count or 0),
            _json_dumps(decision_state),
            _json_dumps(source),
        ),
    )
    row_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return row_id


def list_regime_observations(market_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, ts, market_id, token_yes, token_no, regime_label, trend_score, tail_score, reversal_score,
                   regime_reason, persistence_count, decision_state_json, source_json
            FROM {REGIME_OBSERVATION_TABLE}'''
    params: List = []
    clauses = []
    if market_id is not None:
        clauses.append('market_id = ?')
        params.append(market_id)
    if clauses:
        q += ' WHERE ' + ' AND '.join(clauses)
    q += ' ORDER BY ts DESC, id DESC'
    if limit is not None:
        q += ' LIMIT ?'
        params.append(int(limit))
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    observations = []
    for row in rows:
        decision_state = _json_loads(row[11], default={}) or {}
        source = _json_loads(row[12], default={}) or {}
        observations.append(
            {
                'id': row[0],
                'ts': row[1],
                'market_id': row[2],
                'token_yes': row[3],
                'token_no': row[4],
                'regime_label': row[5],
                'trend_score': None if row[6] is None else float(row[6]),
                'tail_score': None if row[7] is None else float(row[7]),
                'reversal_score': None if row[8] is None else float(row[8]),
                'regime_reason': row[9],
                'persistence_count': int(row[10] or 0),
                'decision_state_json': row[11],
                'decision_state': decision_state,
                'source_json': row[12],
                'source': source,
                **_extract_microstructure_fields(decision_state.get('regime_state'), decision_state, source),
            }
        )
    return observations


def get_latest_regime_observation(market_id: str) -> Optional[Dict]:
    rows = list_regime_observations(market_id=market_id, limit=1)
    return rows[0] if rows else None


def create_lot_regime_attribution(
    *,
    open_lot_id: int,
    order_id: Optional[int],
    market_id: str,
    token_id: str,
    outcome_side: str,
    entry_ts: str,
    entry_regime_label: Optional[str],
    entry_regime_scores: Optional[Dict],
    decision_state: Optional[Dict] = None,
) -> int:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {LOT_REGIME_ATTRIBUTION_TABLE}(
            open_lot_id, order_id, market_id, token_id, outcome_side, entry_ts, entry_regime_label, entry_regime_scores_json, decision_state_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            open_lot_id,
            order_id,
            market_id,
            token_id,
            outcome_side,
            entry_ts,
            entry_regime_label,
            _json_dumps(entry_regime_scores),
            _json_dumps(decision_state),
        ),
    )
    row_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return row_id


def get_lot_regime_attribution(open_lot_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, open_lot_id, order_id, market_id, token_id, outcome_side, entry_ts, entry_regime_label, entry_regime_scores_json, decision_state_json
            FROM {LOT_REGIME_ATTRIBUTION_TABLE}
            WHERE open_lot_id = ?
            ORDER BY id DESC
            LIMIT 1''',
        (open_lot_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        'id': row[0],
        'open_lot_id': row[1],
        'order_id': row[2],
        'market_id': row[3],
        'token_id': row[4],
        'outcome_side': row[5],
        'entry_ts': row[6],
        'entry_regime_label': row[7],
        'entry_regime_scores_json': row[8],
        'entry_regime_scores': _json_loads(row[8], default={}) or {},
        'decision_state_json': row[9],
        'decision_state': _json_loads(row[9], default={}) or {},
    }


def _insert_realized_pnl_event_cur(
    cur,
    *,
    ts: str,
    market_id: str,
    token_id: str,
    outcome_side: str,
    qty: float,
    disposition_type: str,
    entry_price: Optional[float],
    exit_price: Optional[float],
    gross_pnl: Optional[float],
    net_pnl: Optional[float],
    fee_slippage_buffer: Optional[float] = None,
    source_fill_id: Optional[int] = None,
    source_disposal_id: Optional[int] = None,
    source_tx_hash: Optional[str] = None,
    open_lot_id: Optional[int] = None,
    entry_regime_label: Optional[str] = None,
    entry_regime_scores: Optional[Dict] = None,
    exit_regime_label: Optional[str] = None,
    exit_regime_scores: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> int:
    cur.execute(
        f'''INSERT INTO {REALIZED_PNL_EVENT_TABLE}(
            ts, market_id, token_id, outcome_side, qty, disposition_type, entry_price, exit_price, gross_pnl, net_pnl,
            fee_slippage_buffer, source_fill_id, source_disposal_id, source_tx_hash, open_lot_id, entry_regime_label,
            entry_regime_scores_json, exit_regime_label, exit_regime_scores_json, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            ts,
            market_id,
            token_id,
            outcome_side,
            float(qty or 0.0),
            disposition_type,
            entry_price,
            exit_price,
            gross_pnl,
            net_pnl,
            fee_slippage_buffer,
            source_fill_id,
            source_disposal_id,
            source_tx_hash,
            open_lot_id,
            entry_regime_label,
            _json_dumps(entry_regime_scores),
            exit_regime_label,
            _json_dumps(exit_regime_scores),
            _json_dumps(extra),
        ),
    )
    return int(cur.lastrowid)


def record_realized_pnl_event(
    *,
    ts: str,
    market_id: str,
    token_id: str,
    outcome_side: str,
    qty: float,
    disposition_type: str,
    entry_price: Optional[float],
    exit_price: Optional[float],
    gross_pnl: Optional[float],
    net_pnl: Optional[float],
    fee_slippage_buffer: Optional[float] = None,
    source_fill_id: Optional[int] = None,
    source_disposal_id: Optional[int] = None,
    source_tx_hash: Optional[str] = None,
    open_lot_id: Optional[int] = None,
    entry_regime_label: Optional[str] = None,
    entry_regime_scores: Optional[Dict] = None,
    exit_regime_label: Optional[str] = None,
    exit_regime_scores: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> int:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    row_id = _insert_realized_pnl_event_cur(
        cur,
        ts=ts,
        market_id=market_id,
        token_id=token_id,
        outcome_side=outcome_side,
        qty=qty,
        disposition_type=disposition_type,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fee_slippage_buffer=fee_slippage_buffer,
        source_fill_id=source_fill_id,
        source_disposal_id=source_disposal_id,
        source_tx_hash=source_tx_hash,
        open_lot_id=open_lot_id,
        entry_regime_label=entry_regime_label,
        entry_regime_scores=entry_regime_scores,
        exit_regime_label=exit_regime_label,
        exit_regime_scores=exit_regime_scores,
        extra=extra,
    )
    conn.commit()
    conn.close()
    return row_id


def list_realized_pnl_events(market_id: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, ts, market_id, token_id, outcome_side, qty, disposition_type, entry_price, exit_price, gross_pnl,
                   net_pnl, fee_slippage_buffer, source_fill_id, source_disposal_id, source_tx_hash, open_lot_id,
                   entry_regime_label, entry_regime_scores_json, exit_regime_label, exit_regime_scores_json, extra_json
            FROM {REALIZED_PNL_EVENT_TABLE}'''
    params: List = []
    if market_id is not None:
        q += ' WHERE market_id = ?'
        params.append(market_id)
    q += ' ORDER BY ts ASC, id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'ts': row[1],
            'market_id': row[2],
            'token_id': row[3],
            'outcome_side': row[4],
            'qty': float(row[5] or 0.0),
            'disposition_type': row[6],
            'entry_price': None if row[7] is None else float(row[7]),
            'exit_price': None if row[8] is None else float(row[8]),
            'gross_pnl': None if row[9] is None else float(row[9]),
            'net_pnl': None if row[10] is None else float(row[10]),
            'fee_slippage_buffer': None if row[11] is None else float(row[11]),
            'source_fill_id': row[12],
            'source_disposal_id': row[13],
            'source_tx_hash': row[14],
            'open_lot_id': row[15],
            'entry_regime_label': row[16],
            'entry_regime_scores_json': row[17],
            'entry_regime_scores': _json_loads(row[17], default={}) or {},
            'exit_regime_label': row[18],
            'exit_regime_scores_json': row[19],
            'exit_regime_scores': _json_loads(row[19], default={}) or {},
            'extra_json': row[20],
            'extra': _json_loads(row[20], default={}) or {},
        }
        for row in rows
    ]


def _summarize_realized_events(rows: List[Dict], key_name: str, key_value) -> Dict:
    count = len(rows)
    pnl_total = sum(float(row.get('net_pnl') or 0.0) for row in rows)
    qty_total = sum(float(row.get('qty') or 0.0) for row in rows)
    wins = sum(1 for row in rows if float(row.get('net_pnl') or 0.0) > 0)
    return {
        key_name: key_value,
        'realized_pnl_total': pnl_total,
        'count_realized_events': count,
        'avg_pnl_per_event': (pnl_total / count) if count else 0.0,
        'win_rate': (wins / count) if count else 0.0,
        'qty_realized': qty_total,
    }


def get_realized_pnl_summary(market_id: Optional[str] = None) -> Dict:
    rows = list_realized_pnl_events(market_id=market_id)
    summary = _summarize_realized_events(rows, 'bucket', 'all')
    summary.pop('bucket', None)
    return summary


def _group_realized_events(field: str, *, market_id: Optional[str] = None) -> List[Dict]:
    rows = list_realized_pnl_events(market_id=market_id)
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        key = row.get(field) or 'unknown'
        buckets.setdefault(str(key), []).append(row)
    return [_summarize_realized_events(bucket_rows, field, bucket) for bucket, bucket_rows in sorted(buckets.items())]


def get_realized_pnl_by_entry_regime(market_id: Optional[str] = None) -> List[Dict]:
    return _group_realized_events('entry_regime_label', market_id=market_id)


def get_realized_pnl_by_exit_regime(market_id: Optional[str] = None) -> List[Dict]:
    return _group_realized_events('exit_regime_label', market_id=market_id)


def get_realized_pnl_by_regime_pair(market_id: Optional[str] = None) -> List[Dict]:
    rows = list_realized_pnl_events(market_id=market_id)
    buckets: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (str(row.get('entry_regime_label') or 'unknown'), str(row.get('exit_regime_label') or 'unknown'))
        buckets.setdefault(key, []).append(row)
    return [
        {
            **_summarize_realized_events(bucket_rows, 'regime_pair', f'{entry}->{exit}'),
            'entry_regime_label': entry,
            'exit_regime_label': exit,
        }
        for (entry, exit), bucket_rows in sorted(buckets.items())
    ]


def get_regime_trade_counts(market_id: Optional[str] = None) -> List[Dict]:
    rows = list_realized_pnl_events(market_id=market_id)
    buckets: Dict[str, Dict[str, float]] = {}
    for row in rows:
        key = str(row.get('entry_regime_label') or 'unknown')
        item = buckets.setdefault(key, {'entry_regime_label': key, 'count_realized_events': 0, 'qty_realized': 0.0})
        item['count_realized_events'] += 1
        item['qty_realized'] += float(row.get('qty') or 0.0)
    return [buckets[key] for key in sorted(buckets)]


def get_order_by_venue_order_id(venue_order_id: str) -> Optional[Dict]:
    if venue_order_id is None:
        raise ValueError('venue_order_id is required')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json
            FROM {ORDER_TABLE} WHERE venue_order_id = ? ORDER BY id DESC LIMIT 1''',
        (venue_order_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row[0],
        'client_order_id': row[1],
        'venue_order_id': row[2],
        'market_id': row[3],
        'token_id': row[4],
        'outcome_side': row[5],
        'side': row[6],
        'requested_qty': float(row[7]),
        'limit_price': None if row[8] is None else float(row[8]),
        'filled_qty': float(row[9]),
        'remaining_qty': float(row[10]),
        'status': row[11],
        'tx_hash': row[12],
        'created_ts': row[13],
        'updated_ts': row[14],
        'raw_response_json': row[15],
    }


def can_transition_order_state(old_state: str, new_state: str) -> bool:
    if old_state == new_state:
        return True
    if old_state not in ORDER_STATE_TRANSITIONS:
        return False
    return new_state in ORDER_STATE_TRANSITIONS[old_state]


def transition_order_state(order_id: int, new_state: str, *, reason: Optional[str] = None, raw: Optional[Dict] = None, ts: Optional[str] = None) -> Dict:
    if new_state not in ORDER_STATES:
        raise RuntimeError(f'Unknown order state: {new_state}')
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    old_state = order['status']
    if not can_transition_order_state(old_state, new_state):
        raise RuntimeError(f'Illegal order-state transition: {old_state} -> {new_state}')
    payload = raw
    if reason is not None:
        existing_raw = None
        try:
            existing_raw = json.loads(order['raw_response_json']) if order['raw_response_json'] else {}
        except Exception:
            existing_raw = {}
        payload = {**(existing_raw or {}), **(raw or {}), 'transition_reason': reason}
    updated = update_order(order_id, status=new_state, raw_response=payload, updated_ts=ts)
    if new_state in TERMINAL_ORDER_STATUSES:
        release_reservation(order_id, updated_ts=ts)
    return updated


def move_order_to_dust_status(
    order_id: int,
    *,
    dust_status: str = 'dust_ignored',
    reason: str = 'dust_detected',
    raw: Optional[Dict] = None,
    ts: Optional[str] = None,
) -> Dict:
    if dust_status not in {'dust_ignored', 'dust_finalized'}:
        raise ValueError(f'unsupported dust status: {dust_status}')
    ts = ts or datetime.now(timezone.utc).isoformat()
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    if order['status'] == dust_status:
        return order
    if not can_transition_order_state(order['status'], dust_status):
        raise RuntimeError(f'Illegal order-state transition: {order["status"]} -> {dust_status}')
    payload = {
        'dust_threshold': get_dust_qty_threshold(),
        'remaining_qty': float(order.get('remaining_qty') or 0.0),
        'dust_reason': reason,
        **(raw or {}),
    }
    updated = transition_order_state(order_id, dust_status, reason=reason, raw=payload, ts=ts)
    append_order_event(
        order_id,
        dust_status,
        old_status=order['status'],
        new_status=dust_status,
        response=payload,
        ts=ts,
    )
    return updated


def restore_dust_order(order_id: int, *, restored_state: str = 'unknown', ts: Optional[str] = None) -> Dict:
    ts = ts or datetime.now(timezone.utc).isoformat()
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    if order['status'] not in {'dust_ignored', 'dust_finalized'}:
        return order
    if not can_transition_order_state(order['status'], restored_state):
        raise RuntimeError(f'Illegal order-state transition: {order["status"]} -> {restored_state}')
    updated = transition_order_state(order_id, restored_state, reason='dust_restored', raw={'restored_from': order['status']}, ts=ts)
    reservation_type = 'inventory' if updated['side'] == 'sell' else 'exposure'
    reservation_qty = max(0.0, float(updated['remaining_qty'] or 0.0))
    if reservation_type == 'exposure':
        reservation_qty = _expected_buy_exposure(updated)
    if reservation_qty > ORDER_QTY_TOLERANCE:
        create_reservation(updated['id'], updated['market_id'], updated['token_id'], updated['outcome_side'], reservation_type, reservation_qty, ts)
    append_order_event(
        order_id,
        'dust_restored',
        old_status=order['status'],
        new_status=restored_state,
        response={'restored_from': order['status'], 'reservation_type': reservation_type, 'reservation_qty': reservation_qty},
        ts=ts,
    )
    return get_order(order_id=order_id)


def get_open_orders(market_id: Optional[str] = None, statuses: Optional[List[str]] = None) -> List[Dict]:
    statuses = statuses or ['pending_submit', 'submitted', 'open', 'partially_filled', 'cancel_requested', 'unknown']
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    placeholders = ','.join('?' for _ in statuses)
    params: List = list(statuses)
    q = f'''SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json
            FROM {ORDER_TABLE} WHERE status IN ({placeholders})'''
    if market_id is not None:
        q += ' AND market_id = ?'
        params.append(market_id)
    q += ' ORDER BY created_ts ASC, id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'client_order_id': r[1], 'venue_order_id': r[2], 'market_id': r[3], 'token_id': r[4],
            'outcome_side': r[5], 'side': r[6], 'requested_qty': float(r[7]), 'limit_price': None if r[8] is None else float(r[8]),
            'filled_qty': float(r[9]), 'remaining_qty': float(r[10]), 'status': r[11], 'tx_hash': r[12], 'created_ts': r[13], 'updated_ts': r[14], 'raw_response_json': r[15]
        }
        for r in rows
    ]


def get_order_dust_candidates(
    *,
    threshold: Optional[float] = None,
    statuses: Optional[List[str]] = None,
) -> List[Dict]:
    threshold = get_dust_qty_threshold() if threshold is None else float(threshold)
    candidate_statuses = statuses or sorted(
        state
        for state, allowed in ORDER_STATE_TRANSITIONS.items()
        if 'dust_ignored' in allowed or 'dust_finalized' in allowed
    )
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    placeholders = ','.join('?' for _ in candidate_statuses)
    cur.execute(
        f'''SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price,
                  filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json
            FROM {ORDER_TABLE}
            WHERE status IN ({placeholders}) AND remaining_qty > ? AND remaining_qty < ?
            ORDER BY updated_ts ASC, id ASC''',
        tuple(candidate_statuses) + (ORDER_QTY_TOLERANCE, threshold),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'client_order_id': r[1], 'venue_order_id': r[2], 'market_id': r[3], 'token_id': r[4],
            'outcome_side': r[5], 'side': r[6], 'requested_qty': float(r[7]), 'limit_price': None if r[8] is None else float(r[8]),
            'filled_qty': float(r[9]), 'remaining_qty': float(r[10]), 'status': r[11], 'tx_hash': r[12], 'created_ts': r[13], 'updated_ts': r[14], 'raw_response_json': r[15]
        }
        for r in rows
    ]


def get_inventory_dust_candidates(*, threshold: Optional[float] = None) -> List[Dict]:
    threshold = get_dust_qty_threshold() if threshold is None else float(threshold)
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''SELECT market_id, token_id, outcome_side, SUM(qty) AS net_qty
           FROM open_lots
           GROUP BY market_id, token_id, outcome_side
           HAVING ABS(SUM(qty)) > ? AND ABS(SUM(qty)) < ?
           ORDER BY market_id ASC, outcome_side ASC, token_id ASC''',
        (ORDER_QTY_TOLERANCE, threshold),
    )
    groups = cur.fetchall()
    candidates = []
    for market_id, token_id, outcome_side, net_qty in groups:
        cur.execute(
            '''SELECT id, qty, avg_price, ts, tx_hash
               FROM open_lots
               WHERE market_id = ? AND token_id = ? AND outcome_side = ?
               ORDER BY ts ASC, id ASC''',
            (market_id, token_id, outcome_side),
        )
        lots = cur.fetchall()
        linked_order_id = _infer_linked_order_id(market_id, token_id, outcome_side, tx_hash=(lots[0][4] if lots else None))
        candidates.append(
            {
                'market_id': market_id,
                'token_id': token_id,
                'outcome_side': outcome_side,
                'net_qty': float(net_qty),
                'qty': float(net_qty),
                'linked_order_id': linked_order_id,
                'lot_ids': [int(row[0]) for row in lots],
                'lots': [
                    {
                        'id': int(row[0]),
                        'qty': float(row[1]),
                        'avg_price': None if row[2] is None else float(row[2]),
                        'ts': row[3],
                        'tx_hash': row[4],
                    }
                    for row in lots
                ],
            }
        )
    conn.close()
    return candidates


def get_orders_by_tx_hash(tx_hash: str) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(f'SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json FROM {ORDER_TABLE} WHERE tx_hash = ? ORDER BY created_ts ASC, id ASC', (tx_hash,))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'client_order_id': r[1], 'venue_order_id': r[2], 'market_id': r[3], 'token_id': r[4],
            'outcome_side': r[5], 'side': r[6], 'requested_qty': float(r[7]), 'limit_price': None if r[8] is None else float(r[8]),
            'filled_qty': float(r[9]), 'remaining_qty': float(r[10]), 'status': r[11], 'tx_hash': r[12], 'created_ts': r[13], 'updated_ts': r[14], 'raw_response_json': r[15]
        }
        for r in rows
    ]


def get_order_fill_events(order_id: int) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(f'SELECT id, order_id, venue_fill_id, tx_hash, fill_qty, cumulative_filled_qty, price, fill_ts, raw_json FROM {ORDER_FILL_TABLE} WHERE order_id = ? ORDER BY cumulative_filled_qty ASC, id ASC', (order_id,))
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'order_id': r[1], 'venue_fill_id': r[2], 'tx_hash': r[3], 'fill_qty': float(r[4]), 'cumulative_filled_qty': float(r[5]), 'price': None if r[6] is None else float(r[6]), 'fill_ts': r[7], 'raw_json': r[8]} for r in rows]


def list_order_fill_events(tx_hash: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, order_id, venue_fill_id, tx_hash, fill_qty, cumulative_filled_qty, price, fill_ts, raw_json
            FROM {ORDER_FILL_TABLE}'''
    params: List = []
    if tx_hash is not None:
        q += ' WHERE tx_hash = ?'
        params.append(tx_hash)
    q += ' ORDER BY fill_ts ASC, id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0],
            'order_id': r[1],
            'venue_fill_id': r[2],
            'tx_hash': r[3],
            'fill_qty': float(r[4]),
            'cumulative_filled_qty': float(r[5]),
            'price': None if r[6] is None else float(r[6]),
            'fill_ts': r[7],
            'raw_json': r[8],
            'raw': _json_loads(r[8], default={}) or {},
        }
        for r in rows
    ]


def clear_trade_journal() -> None:
    ensure_db()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(f'DELETE FROM {TRADE_JOURNAL_TABLE}')
    conn.commit()
    conn.close()


def insert_trade_journal_row(**row) -> int:
    ensure_db()
    columns = [
        'ts', 'market_id', 'token_id', 'outcome_side', 'side', 'kind', 'qty', 'price', 'notional', 'tx_hash',
        'order_id', 'client_order_id', 'venue_order_id', 'decision_ts', 'decision_reason', 'policy_bucket',
        'tau_minutes', 'raw_p_yes', 'raw_p_no', 'adjusted_p_yes', 'adjusted_p_no', 'raw_edge_yes', 'raw_edge_no',
        'adjusted_edge_yes', 'adjusted_edge_no', 'tail_penalty_score', 'tail_hard_block', 'reeval_action',
        'reeval_reason', 'realized_pnl', 'extra_json',
    ]
    payload = {}
    for column in columns:
        value = row.get(column)
        if column == 'extra_json' and isinstance(value, dict):
            value = _json_dumps(value)
        if column == 'tail_hard_block' and value is not None:
            value = int(bool(value))
        payload[column] = value
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {TRADE_JOURNAL_TABLE}({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})''',
        tuple(payload[column] for column in columns),
    )
    row_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return row_id


def list_trade_journal(market_id: Optional[str] = None, kind: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    ensure_db()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'''SELECT id, ts, market_id, token_id, outcome_side, side, kind, qty, price, notional, tx_hash, order_id,
                   client_order_id, venue_order_id, decision_ts, decision_reason, policy_bucket, tau_minutes,
                   raw_p_yes, raw_p_no, adjusted_p_yes, adjusted_p_no, raw_edge_yes, raw_edge_no, adjusted_edge_yes,
                   adjusted_edge_no, tail_penalty_score, tail_hard_block, reeval_action, reeval_reason, realized_pnl,
                   extra_json
            FROM {TRADE_JOURNAL_TABLE}'''
    params: List = []
    clauses = []
    if market_id is not None:
        clauses.append('market_id = ?')
        params.append(market_id)
    if kind is not None:
        clauses.append('kind = ?')
        params.append(kind)
    if clauses:
        q += ' WHERE ' + ' AND '.join(clauses)
    q += ' ORDER BY ts ASC, id ASC'
    if limit is not None:
        q += ' LIMIT ?'
        params.append(int(limit))
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    journal = []
    for row in rows:
        journal.append(
            {
                'id': row[0],
                'ts': row[1],
                'market_id': row[2],
                'token_id': row[3],
                'outcome_side': row[4],
                'side': row[5],
                'kind': row[6],
                'qty': float(row[7] or 0.0),
                'price': None if row[8] is None else float(row[8]),
                'notional': None if row[9] is None else float(row[9]),
                'tx_hash': row[10],
                'order_id': row[11],
                'client_order_id': row[12],
                'venue_order_id': row[13],
                'decision_ts': row[14],
                'decision_reason': row[15],
                'policy_bucket': row[16],
                'tau_minutes': None if row[17] is None else int(row[17]),
                'raw_p_yes': None if row[18] is None else float(row[18]),
                'raw_p_no': None if row[19] is None else float(row[19]),
                'adjusted_p_yes': None if row[20] is None else float(row[20]),
                'adjusted_p_no': None if row[21] is None else float(row[21]),
                'raw_edge_yes': None if row[22] is None else float(row[22]),
                'raw_edge_no': None if row[23] is None else float(row[23]),
                'adjusted_edge_yes': None if row[24] is None else float(row[24]),
                'adjusted_edge_no': None if row[25] is None else float(row[25]),
                'tail_penalty_score': None if row[26] is None else float(row[26]),
                'tail_hard_block': None if row[27] is None else bool(row[27]),
                'reeval_action': row[28],
                'reeval_reason': row[29],
                'realized_pnl': None if row[30] is None else float(row[30]),
                'extra_json': row[31],
                'extra': _json_loads(row[31], default={}) or {},
            }
        )
    return journal


def get_trade_journal_summary(market_id: Optional[str] = None, kind: Optional[str] = None) -> Dict:
    rows = list_trade_journal(market_id=market_id, kind=kind)
    total_rows = len(rows)
    buy_rows = [row for row in rows if row.get('side') == 'buy']
    sell_rows = [row for row in rows if row.get('side') == 'sell']
    realized_rows = [row for row in rows if row.get('realized_pnl') is not None]
    realized_values = [float(row['realized_pnl']) for row in realized_rows]
    counts_by_kind: Dict[str, int] = {}
    counts_by_outcome_side: Dict[str, int] = {}
    counts_by_policy_bucket: Dict[str, int] = {}
    counts_by_reeval_action: Dict[str, int] = {}
    counts_by_market_id: Dict[str, int] = {}
    for row in rows:
        counts_by_kind[row.get('kind') or 'unknown'] = counts_by_kind.get(row.get('kind') or 'unknown', 0) + 1
        counts_by_outcome_side[row.get('outcome_side') or 'unknown'] = counts_by_outcome_side.get(row.get('outcome_side') or 'unknown', 0) + 1
        counts_by_policy_bucket[row.get('policy_bucket') or 'unknown'] = counts_by_policy_bucket.get(row.get('policy_bucket') or 'unknown', 0) + 1
        counts_by_reeval_action[row.get('reeval_action') or 'unknown'] = counts_by_reeval_action.get(row.get('reeval_action') or 'unknown', 0) + 1
        counts_by_market_id[row.get('market_id') or 'unknown'] = counts_by_market_id.get(row.get('market_id') or 'unknown', 0) + 1
    average_trade_size = sum(abs(float(row.get('qty') or 0.0)) for row in rows) / total_rows if total_rows else 0.0
    return {
        'total_journal_rows': total_rows,
        'total_buy_notional': sum(float(row.get('notional') or 0.0) for row in buy_rows),
        'total_sell_notional': sum(float(row.get('notional') or 0.0) for row in sell_rows),
        'total_redeem_rows': sum(1 for row in rows if row.get('kind') == 'redeem'),
        'total_realized_pnl': sum(realized_values),
        'realized_pnl_row_count': len(realized_rows),
        'average_realized_pnl_per_exit': (sum(realized_values) / len(realized_values)) if realized_values else 0.0,
        'win_count': sum(1 for value in realized_values if value > 0),
        'loss_count': sum(1 for value in realized_values if value < 0),
        'flat_count': sum(1 for value in realized_values if value == 0),
        'average_trade_size': average_trade_size,
        'counts_by_kind': counts_by_kind,
        'counts_by_outcome_side': counts_by_outcome_side,
        'counts_by_policy_bucket': counts_by_policy_bucket,
        'counts_by_reeval_action': counts_by_reeval_action,
        'counts_by_market_id': counts_by_market_id,
        'first_trade_ts': rows[0]['ts'] if rows else None,
        'last_trade_ts': rows[-1]['ts'] if rows else None,
    }


def create_reservation(order_id: int, market_id: str, token_id: Optional[str], outcome_side: Optional[str], reservation_type: str, qty: float, created_ts: str, extra: Optional[Dict] = None) -> Dict:
    if qty < 0:
        raise ValueError('reservation qty must be >= 0')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, order_id, market_id, token_id, outcome_side, reservation_type, qty, status, created_ts, updated_ts, released_ts, extra_json
           FROM {RESERVATION_TABLE} WHERE order_id = ? AND reservation_type = ? AND status = ?''',
        (order_id, reservation_type, 'active'),
    )
    row = cur.fetchone()
    if row is not None:
        conn.close()
        return {
            'id': row[0], 'order_id': row[1], 'market_id': row[2], 'token_id': row[3], 'outcome_side': row[4],
            'reservation_type': row[5], 'qty': float(row[6]), 'status': row[7], 'created_ts': row[8], 'updated_ts': row[9], 'released_ts': row[10], 'extra_json': row[11]
        }
    cur.execute(
        f'''INSERT INTO {RESERVATION_TABLE}(order_id, market_id, token_id, outcome_side, reservation_type, qty, status, created_ts, updated_ts, released_ts, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (order_id, market_id, token_id, outcome_side, reservation_type, float(qty), 'active', created_ts, created_ts, None, json.dumps(extra) if extra is not None else None),
    )
    reservation_id = cur.lastrowid
    conn.commit()
    conn.close()
    return get_reservation(reservation_id)


def get_reservation(reservation_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(f'SELECT id, order_id, market_id, token_id, outcome_side, reservation_type, qty, status, created_ts, updated_ts, released_ts, extra_json FROM {RESERVATION_TABLE} WHERE id = ?', (reservation_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        'id': row[0], 'order_id': row[1], 'market_id': row[2], 'token_id': row[3], 'outcome_side': row[4], 'reservation_type': row[5],
        'qty': float(row[6]), 'status': row[7], 'created_ts': row[8], 'updated_ts': row[9], 'released_ts': row[10], 'extra_json': row[11]
    }


def get_order_reservations(order_id: int, active_only: bool = False) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'SELECT id, order_id, market_id, token_id, outcome_side, reservation_type, qty, status, created_ts, updated_ts, released_ts, extra_json FROM {RESERVATION_TABLE} WHERE order_id = ?'
    params: List = [order_id]
    if active_only:
        q += ' AND status = ?'
        params.append('active')
    q += ' ORDER BY id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'order_id': r[1], 'market_id': r[2], 'token_id': r[3], 'outcome_side': r[4], 'reservation_type': r[5],
            'qty': float(r[6]), 'status': r[7], 'created_ts': r[8], 'updated_ts': r[9], 'released_ts': r[10], 'extra_json': r[11]
        }
        for r in rows
    ]


def set_reservation_qty(order_id: int, reservation_type: str, qty: float, updated_ts: Optional[str] = None):
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'UPDATE {RESERVATION_TABLE} SET qty = ?, updated_ts = ? WHERE order_id = ? AND reservation_type = ? AND status = ?',
        (float(max(0.0, qty)), updated_ts or datetime.now(timezone.utc).isoformat(), order_id, reservation_type, 'active'),
    )
    conn.commit()
    conn.close()


def repair_order_reservations(order_id: int, updated_ts: Optional[str] = None) -> Dict:
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    reservations = get_order_reservations(order_id)
    active_reservations = [reservation for reservation in reservations if reservation['status'] == 'active']
    released_reservations = []
    adjusted_reservations = []

    if order['status'] in TERMINAL_ORDER_STATUSES:
        for reservation in active_reservations:
            release_reservation(order_id, reservation['reservation_type'], updated_ts=ts)
            released_reservations.append({
                'reservation_id': reservation['id'],
                'reservation_type': reservation['reservation_type'],
                'before_qty': reservation['qty'],
                'after_qty': 0.0,
            })
        return {
            'order_id': order_id,
            'status': order['status'],
            'action': 'released_terminal_reservations',
            'released': released_reservations,
            'adjusted': adjusted_reservations,
        }

    expected_inventory_qty = max(0.0, float(order['remaining_qty'])) if order['side'] == 'sell' else 0.0
    expected_exposure_qty = _expected_buy_exposure(order) if order['side'] == 'buy' else 0.0

    for reservation in active_reservations:
        if reservation['reservation_type'] == 'inventory':
            expected_qty = expected_inventory_qty
        elif reservation['reservation_type'] == 'exposure':
            expected_qty = expected_exposure_qty
        else:
            expected_qty = reservation['qty']
        if abs(reservation['qty'] - expected_qty) > 1e-9:
            set_reservation_qty(order_id, reservation['reservation_type'], expected_qty, updated_ts=ts)
            adjusted_reservations.append({
                'reservation_id': reservation['id'],
                'reservation_type': reservation['reservation_type'],
                'before_qty': reservation['qty'],
                'after_qty': expected_qty,
            })

    return {
        'order_id': order_id,
        'status': order['status'],
        'action': 'normalized_active_reservations',
        'released': released_reservations,
        'adjusted': adjusted_reservations,
    }


def repair_all_active_order_reservations(updated_ts: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    statuses = tuple(ACTIVE_ORDER_STATUSES | TERMINAL_ORDER_STATUSES)
    placeholders = ','.join('?' for _ in statuses)
    cur.execute(f'SELECT id FROM {ORDER_TABLE} WHERE status IN ({placeholders}) ORDER BY created_ts ASC, id ASC', statuses)
    order_ids = [int(row[0]) for row in cur.fetchall()]
    conn.close()
    return [repair_order_reservations(order_id, updated_ts=updated_ts) for order_id in order_ids]


def assert_order_reservation_consistency(order_id: int) -> Dict:
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    reservations = get_order_reservations(order_id, active_only=True)
    active_inventory = sum(r['qty'] for r in reservations if r['reservation_type'] == 'inventory')
    active_exposure = sum(r['qty'] for r in reservations if r['reservation_type'] == 'exposure')
    expected_inventory = max(0.0, order['remaining_qty']) if order['side'] == 'sell' and order['status'] in ACTIVE_ORDER_STATUSES else 0.0
    expected_exposure = _expected_buy_exposure(order) if order['side'] == 'buy' and order['status'] in ACTIVE_ORDER_STATUSES else 0.0
    return {
        'order_id': order_id,
        'status': order['status'],
        'ok': abs(active_inventory - expected_inventory) <= 1e-9 and abs(active_exposure - expected_exposure) <= 1e-9,
        'expected_inventory': expected_inventory,
        'actual_inventory': active_inventory,
        'expected_exposure': expected_exposure,
        'actual_exposure': active_exposure,
    }


def assert_all_active_order_reservation_consistency() -> List[Dict]:
    return [assert_order_reservation_consistency(order['id']) for order in get_open_orders()]


def release_reservation(order_id: int, reservation_type: Optional[str] = None, updated_ts: Optional[str] = None):
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    if reservation_type is None:
        cur.execute(f'UPDATE {RESERVATION_TABLE} SET status = ?, updated_ts = ?, released_ts = ? WHERE order_id = ? AND status = ?', ('released', ts, ts, order_id, 'active'))
    else:
        cur.execute(f'UPDATE {RESERVATION_TABLE} SET status = ?, updated_ts = ?, released_ts = ? WHERE order_id = ? AND reservation_type = ? AND status = ?', ('released', ts, ts, order_id, reservation_type, 'active'))
    conn.commit()
    conn.close()


def get_reserved_qty(market_id: str, token_id: str, outcome_side: str) -> float:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT SUM(qty) FROM {RESERVATION_TABLE}
           WHERE market_id = ? AND token_id = ? AND outcome_side = ? AND reservation_type = ? AND status = ?''',
        (market_id, token_id, outcome_side, 'inventory', 'active'),
    )
    row = cur.fetchone()
    conn.close()
    return 0.0 if row is None or row[0] is None else float(row[0])


def get_available_qty(market_id: str, token_id: str, outcome_side: str) -> float:
    open_qty = get_total_qty_by_token(token_id, market_id=market_id)
    reserved_qty = get_reserved_qty(market_id, token_id, outcome_side)
    available_qty = open_qty - reserved_qty
    if available_qty < -1e-9:
        raise RuntimeError(f'Negative available inventory detected for {market_id}/{token_id}/{outcome_side}: open={open_qty}, reserved={reserved_qty}')
    return max(0.0, available_qty)


def get_inflight_exposure(market_id: Optional[str] = None) -> float:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = f'SELECT SUM(qty) FROM {RESERVATION_TABLE} WHERE reservation_type = ? AND status = ?'
    params: List = ['exposure', 'active']
    if market_id is not None:
        q += ' AND market_id = ?'
        params.append(market_id)
    cur.execute(q, tuple(params))
    row = cur.fetchone()
    conn.close()
    return 0.0 if row is None or row[0] is None else float(row[0])


def _record_order_fill_event(order_id: int, fill_qty: float, cumulative_filled_qty: float, price: Optional[float], fill_ts: str, tx_hash: Optional[str] = None, venue_fill_id: Optional[str] = None, raw: Optional[Dict] = None) -> Optional[int]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT OR IGNORE INTO {ORDER_FILL_TABLE}(order_id, venue_fill_id, tx_hash, fill_qty, cumulative_filled_qty, price, fill_ts, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (order_id, venue_fill_id, tx_hash, float(fill_qty), float(cumulative_filled_qty), price, fill_ts, json.dumps(raw) if raw is not None else None),
    )
    fill_event_id = int(cur.lastrowid) if cur.rowcount > 0 else None
    conn.commit()
    conn.close()
    return fill_event_id


def apply_incremental_order_fill(order_id: int, cumulative_filled_qty: float, fill_ts: Optional[str] = None, tx_hash: Optional[str] = None, price: Optional[float] = None, raw: Optional[Dict] = None) -> Dict:
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    fill_ts = fill_ts or datetime.now(timezone.utc).isoformat()
    cumulative = max(0.0, float(cumulative_filled_qty))
    requested = float(order['requested_qty'] or 0.0)
    if requested > 0 and cumulative > requested:
        cumulative = requested
    already_filled = float(order['filled_qty'])
    delta = cumulative - already_filled
    if requested > 0:
        delta = min(delta, max(0.0, requested - already_filled))
    if delta <= 1e-12:
        return {'applied_qty': 0.0, 'order': order, 'duplicate': True}

    fill_event_id = _record_order_fill_event(order_id, delta, cumulative, price if price is not None else order['limit_price'], fill_ts, tx_hash=tx_hash, raw=raw)
    if fill_event_id is None:
        return {'applied_qty': 0.0, 'order': get_order(order_id=order_id), 'duplicate': True}

    if order['side'] == 'buy':
        insert_fill(order['market_id'], order['token_id'], order['outcome_side'], delta, price if price is not None else (order['limit_price'] or 0.0), fill_ts, tx_hash=tx_hash, kind='buy', extra={'client_order_id': order['client_order_id'], 'order_id': order_id})
        create_open_lot(
            order['market_id'],
            order['token_id'],
            order['outcome_side'],
            delta,
            price if price is not None else (order['limit_price'] or 0.0),
            fill_ts,
            tx_hash=tx_hash,
            source_order_id=order_id,
            source_fill_id=fill_event_id,
            decision_context=order.get('decision_context'),
        )
    elif order['side'] == 'sell':
        execution_price = price if price is not None else (order['limit_price'] or 0.0)
        consume_open_lots_fifo(
            order['market_id'],
            order['token_id'],
            order['outcome_side'],
            delta,
            consume_tx=tx_hash,
            ts=fill_ts,
            execution_price=execution_price,
            extra={'client_order_id': order['client_order_id'], 'order_id': order_id, 'source_fill_id': fill_event_id, 'decision_context': order.get('decision_context')},
        )
    else:
        raise RuntimeError(f"Unsupported order side for fill application: {order['side']}")

    remaining_qty = max(0.0, requested - cumulative)
    set_reservation_qty(
        order_id,
        'inventory' if order['side'] == 'sell' else 'exposure',
        remaining_qty if order['side'] == 'sell' else _expected_buy_exposure(order, remaining_qty=remaining_qty, fallback_price=price),
        updated_ts=fill_ts,
    )
    updated = update_order(order_id, filled_qty=cumulative, remaining_qty=remaining_qty, tx_hash=tx_hash or order['tx_hash'], updated_ts=fill_ts)
    return {'applied_qty': delta, 'order': updated, 'duplicate': False}


def finalize_order_state(order_id: int, status: str, updated_ts: Optional[str] = None, venue_order_id: Optional[str] = None, tx_hash: Optional[str] = None, raw_response: Optional[Dict] = None):
    order = get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    updated = update_order(order_id, status=status, venue_order_id=venue_order_id, tx_hash=tx_hash, raw_response=raw_response, updated_ts=updated_ts)
    if status in TERMINAL_ORDER_STATUSES:
        release_reservation(order_id, updated_ts=updated_ts)
    return updated

def create_open_lot(
    market_id: str,
    token_id: str,
    outcome_side: str,
    qty: float,
    avg_price: float,
    ts: str,
    tx_hash: Optional[str] = None,
    source_order_id: Optional[int] = None,
    source_fill_id: Optional[int] = None,
    decision_context: Optional[Dict] = None,
):
    if not market_id or not token_id:
        raise ValueError('market_id and token_id required to create lots')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('INSERT INTO open_lots(market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash))
    lot_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    lot = get_open_lot(lot_id)
    if lot is not None:
        regime_state = (decision_context or {}).get('regime_state') or {}
        create_lot_regime_attribution(
            open_lot_id=lot_id,
            order_id=source_order_id,
            market_id=market_id,
            token_id=token_id,
            outcome_side=outcome_side,
            entry_ts=ts,
            entry_regime_label=regime_state.get('regime_label'),
            entry_regime_scores={
                'trend_score': regime_state.get('trend_score'),
                'tail_score': regime_state.get('tail_score'),
                'reversal_score': regime_state.get('reversal_score'),
                'market_polarization': regime_state.get('market_polarization'),
                'source_fill_id': source_fill_id,
            },
            decision_state=decision_context,
        )
    return lot


def get_open_lot(lot_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id, market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash FROM open_lots WHERE id = ?', (lot_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        'id': row[0],
        'market_id': row[1],
        'token_id': row[2],
        'outcome_side': row[3],
        'qty': float(row[4]),
        'avg_price': None if row[5] is None else float(row[5]),
        'ts': row[6],
        'tx_hash': row[7],
    }


def insert_lot_with_ids(token_id: str, market_id: str, qty: float, avg_price: float, ts: str, tx_hash: Optional[str] = None, outcome_side: Optional[str] = None):
    # convenience wrapper: record fill + create open lot
    if outcome_side is None:
        raise ValueError('outcome_side (YES/NO) required')
    insert_fill(market_id=market_id, token_id=token_id, outcome_side=outcome_side, qty=qty, price=avg_price, ts=ts, tx_hash=tx_hash, kind='buy')
    create_open_lot(market_id=market_id, token_id=token_id, outcome_side=outcome_side, qty=qty, avg_price=avg_price, ts=ts, tx_hash=tx_hash)


def get_open_lots(token_id: Optional[str] = None, market_id: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    q = 'SELECT id, market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash FROM open_lots'
    params = []
    where = []
    if token_id is not None:
        where.append('token_id = ?')
        params.append(token_id)
    if market_id is not None:
        where.append('market_id = ?')
        params.append(market_id)
    if where:
        q = q + ' WHERE ' + ' AND '.join(where)
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {'id': r[0], 'market_id': r[1], 'token_id': r[2], 'outcome_side': r[3], 'qty': r[4], 'avg_price': r[5], 'ts': r[6], 'tx_hash': r[7]}
        for r in rows
    ]


def get_open_lots_for_market(market_id: str) -> List[Dict]:
    return get_open_lots(market_id=market_id)


def _infer_linked_order_id(market_id: str, token_id: str, outcome_side: str, tx_hash: Optional[str] = None) -> Optional[int]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    if tx_hash:
        cur.execute(
            f'''SELECT id FROM {ORDER_TABLE}
                WHERE market_id = ? AND token_id = ? AND outcome_side = ? AND tx_hash = ?
                ORDER BY updated_ts DESC, id DESC LIMIT 1''',
            (market_id, token_id, outcome_side, tx_hash),
        )
        row = cur.fetchone()
        if row is not None:
            conn.close()
            return int(row[0])
    cur.execute(
        f'''SELECT id FROM {ORDER_TABLE}
            WHERE market_id = ? AND token_id = ? AND outcome_side = ? AND side = 'buy'
            ORDER BY updated_ts DESC, id DESC LIMIT 1''',
        (market_id, token_id, outcome_side),
    )
    row = cur.fetchone()
    conn.close()
    return None if row is None else int(row[0])


def create_dormant_lot(
    market_id: str,
    token_id: str,
    outcome_side: str,
    qty: float,
    avg_price: Optional[float],
    ts: Optional[str],
    *,
    tx_hash: Optional[str] = None,
    source_open_lot_id: Optional[int] = None,
    linked_order_id: Optional[int] = None,
    dormant_status: str = 'dust_ignored',
    dormant_reason: str = 'dust',
    created_ts: Optional[str] = None,
) -> Dict:
    if dormant_status not in DORMANT_LOT_ACTIVE_STATUSES:
        raise ValueError(f'unsupported dormant_status: {dormant_status}')
    created_ts = created_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''INSERT INTO {DORMANT_LOT_TABLE}(
            market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash, source_open_lot_id, linked_order_id,
            dormant_status, dormant_reason, created_ts, updated_ts, restored_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            market_id,
            token_id,
            outcome_side,
            float(qty),
            avg_price,
            ts,
            tx_hash,
            source_open_lot_id,
            linked_order_id,
            dormant_status,
            dormant_reason,
            created_ts,
            created_ts,
            None,
        ),
    )
    dormant_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return get_dormant_lot(dormant_id)


def get_dormant_lot(dormant_lot_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash, source_open_lot_id, linked_order_id,
                  dormant_status, dormant_reason, created_ts, updated_ts, restored_ts
            FROM {DORMANT_LOT_TABLE} WHERE id = ?''',
        (dormant_lot_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        'id': row[0],
        'market_id': row[1],
        'token_id': row[2],
        'outcome_side': row[3],
        'qty': float(row[4]),
        'avg_price': None if row[5] is None else float(row[5]),
        'ts': row[6],
        'tx_hash': row[7],
        'source_open_lot_id': row[8],
        'linked_order_id': row[9],
        'dormant_status': row[10],
        'dormant_reason': row[11],
        'created_ts': row[12],
        'updated_ts': row[13],
        'restored_ts': row[14],
    }


def list_dormant_lots(
    *,
    market_id: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    include_restored: bool = False,
) -> List[Dict]:
    statuses = list(statuses or DORMANT_LOT_ACTIVE_STATUSES)
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    params: List = []
    where = []
    if statuses:
        placeholders = ','.join('?' for _ in statuses)
        where.append(f'dormant_status IN ({placeholders})')
        params.extend(statuses)
    if market_id is not None:
        where.append('market_id = ?')
        params.append(market_id)
    if not include_restored:
        where.append('restored_ts IS NULL')
    q = f'''SELECT id, market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash, source_open_lot_id, linked_order_id,
                   dormant_status, dormant_reason, created_ts, updated_ts, restored_ts
            FROM {DORMANT_LOT_TABLE}'''
    if where:
        q += ' WHERE ' + ' AND '.join(where)
    q += ' ORDER BY created_ts ASC, id ASC'
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'market_id': row[1],
            'token_id': row[2],
            'outcome_side': row[3],
            'qty': float(row[4]),
            'avg_price': None if row[5] is None else float(row[5]),
            'ts': row[6],
            'tx_hash': row[7],
            'source_open_lot_id': row[8],
            'linked_order_id': row[9],
            'dormant_status': row[10],
            'dormant_reason': row[11],
            'created_ts': row[12],
            'updated_ts': row[13],
            'restored_ts': row[14],
        }
        for row in rows
    ]


def move_open_lot_to_dormant(
    lot_id: int,
    *,
    dormant_status: str = 'dust_ignored',
    dormant_reason: str = 'dust',
    linked_order_id: Optional[int] = None,
    updated_ts: Optional[str] = None,
) -> Dict:
    lot = get_open_lot(lot_id)
    if lot is None:
        raise RuntimeError(f'open lot {lot_id} not found')
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    if linked_order_id is None:
        linked_order_id = _infer_linked_order_id(lot['market_id'], lot['token_id'], lot['outcome_side'], tx_hash=lot.get('tx_hash'))
    dormant = create_dormant_lot(
        lot['market_id'],
        lot['token_id'],
        lot['outcome_side'],
        lot['qty'],
        lot.get('avg_price'),
        lot.get('ts'),
        tx_hash=lot.get('tx_hash'),
        source_open_lot_id=lot['id'],
        linked_order_id=linked_order_id,
        dormant_status=dormant_status,
        dormant_reason=dormant_reason,
        created_ts=ts,
    )
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('DELETE FROM open_lots WHERE id = ?', (lot_id,))
    conn.commit()
    conn.close()
    return dormant


def restore_dormant_lot(dormant_lot_id: int, *, restored_ts: Optional[str] = None) -> Dict:
    dormant = get_dormant_lot(dormant_lot_id)
    if dormant is None:
        raise RuntimeError(f'dormant lot {dormant_lot_id} not found')
    if dormant.get('restored_ts'):
        return dormant
    ts = restored_ts or datetime.now(timezone.utc).isoformat()
    create_open_lot(
        dormant['market_id'],
        dormant['token_id'],
        dormant['outcome_side'],
        dormant['qty'],
        dormant.get('avg_price') or 0.0,
        dormant.get('ts') or ts,
        tx_hash=dormant.get('tx_hash'),
    )
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'UPDATE {DORMANT_LOT_TABLE} SET dormant_status = ?, updated_ts = ?, restored_ts = ? WHERE id = ?',
        ('restored', ts, ts, dormant_lot_id),
    )
    conn.commit()
    conn.close()
    return get_dormant_lot(dormant_lot_id)


def set_dormant_lot_status(dormant_lot_id: int, dormant_status: str, *, updated_ts: Optional[str] = None) -> Dict:
    if dormant_status not in DORMANT_LOT_ACTIVE_STATUSES:
        raise ValueError(f'unsupported dormant_status: {dormant_status}')
    dormant = get_dormant_lot(dormant_lot_id)
    if dormant is None:
        raise RuntimeError(f'dormant lot {dormant_lot_id} not found')
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'UPDATE {DORMANT_LOT_TABLE} SET dormant_status = ?, updated_ts = ? WHERE id = ?',
        (dormant_status, ts, dormant_lot_id),
    )
    conn.commit()
    conn.close()
    return get_dormant_lot(dormant_lot_id)


def get_total_qty_by_token(token_id: str, market_id: Optional[str] = None) -> float:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    if market_id is None:
        cur.execute('SELECT SUM(qty) FROM open_lots WHERE token_id = ?', (token_id,))
    else:
        cur.execute('SELECT SUM(qty) FROM open_lots WHERE token_id = ? AND market_id = ?', (token_id, market_id))
    row = cur.fetchone()
    conn.close()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])


def get_total_qty_by_market_and_side(market_id: str, outcome_side: str) -> float:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT SUM(qty) FROM open_lots WHERE market_id = ? AND outcome_side = ?', (market_id, outcome_side))
    row = cur.fetchone()
    conn.close()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])


def get_pair_balance(market_id: str) -> Dict[str, float]:
    yes = get_total_qty_by_market_and_side(market_id, 'YES')
    no = get_total_qty_by_market_and_side(market_id, 'NO')
    return {'YES': yes, 'NO': no}


def record_lot_from_receipt(tx_hash: str, token_id: str, market_id: str, qty: float, avg_price: float, ts: str, outcome_side: str):
    # record a lot derived from an on-chain receipt
    if outcome_side not in ('YES', 'NO'):
        raise ValueError('outcome_side must be YES or NO for receipt-derived lots')
    insert_lot_with_ids(token_id, market_id, qty, avg_price, ts, tx_hash, outcome_side=outcome_side)

def _coalesce_market_value(existing, new_value):
    return existing if new_value is None else new_value


def _coalesce_market_status(existing_status: Optional[str], new_status: Optional[str]) -> Optional[str]:
    if new_status is None:
        return existing_status
    if existing_status is None:
        return new_status
    existing_rank = MARKET_STATUS_RANK.get(existing_status, 0)
    new_rank = MARKET_STATUS_RANK.get(new_status, 0)
    return new_status if new_rank >= existing_rank else existing_status


def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _derive_btc_hourly_resolution(market: Dict, *, checked_ts: str) -> Optional[Dict]:
    diagnostics = {
        'has_slug': bool((market or {}).get('slug')),
        'has_title': bool((market or {}).get('title')),
        'has_start_time': bool((market or {}).get('start_time')),
        'has_end_time': bool((market or {}).get('end_time')),
        'binance_open_found': False,
        'binance_close_found': False,
    }
    if not isinstance(market, dict):
        return {'diagnostics': diagnostics}
    if str(market.get('status') or '').lower() in {'resolved', 'redeemed', 'archived'} and market.get('winning_outcome') in ('YES', 'NO'):
        return {
            'status': market.get('status'),
            'winning_outcome': market.get('winning_outcome'),
            'diagnostics': diagnostics,
        }

    end_dt = _parse_iso_ts(market.get('end_time'))
    now_dt = _parse_iso_ts(checked_ts) or datetime.now(timezone.utc)
    if end_dt is None or end_dt > now_dt:
        return {'diagnostics': diagnostics}

    from . import polymarket_feed
    from .binance_feed import get_1h_close_for_timestamp, get_1h_open_for_timestamp

    slug = market.get('slug')
    title = market.get('title')
    derived_start = polymarket_feed._parse_btc_hourly_start_from_slug_or_title(slug, title)
    start_dt = _parse_iso_ts(market.get('start_time')) or (_parse_iso_ts(derived_start.isoformat()) if derived_start is not None else None)
    if start_dt is None or derived_start is None and not str(slug or '').lower().startswith('bitcoin-up-or-down-'):
        return {'diagnostics': diagnostics}

    strike_price = get_1h_open_for_timestamp(start_dt)
    final_price = get_1h_close_for_timestamp(start_dt)
    diagnostics['binance_open_found'] = strike_price is not None
    diagnostics['binance_close_found'] = final_price is not None
    if strike_price is None or final_price is None:
        return {'diagnostics': diagnostics}
    return {
        'status': 'resolved',
        'winning_outcome': 'YES' if float(final_price) > float(strike_price) else 'NO',
        'resolution_source': 'binance_1h_open_close',
        'resolution_prices': {
            'strike_price': float(strike_price),
            'final_price': float(final_price),
        },
        'diagnostics': diagnostics,
    }


def _normalize_gamma_market_status(raw_status: Optional[str]) -> Optional[str]:
    value = str(raw_status or '').strip().lower()
    if value in {'open', 'active', 'tradable', 'trading'}:
        return 'open'
    if value in {'closed', 'paused'}:
        return 'closed'
    if value in {'resolved', 'settled', 'finalized'}:
        return 'resolved'
    if value == 'redeemed':
        return 'redeemed'
    if value == 'archived':
        return 'archived'
    return None


def hydrate_market_metadata_by_id(market_id: str) -> Optional[Dict]:
    if not market_id:
        return None
    from . import polymarket_feed

    payload = polymarket_feed.fetch_market_by_id(market_id)
    if not isinstance(payload, dict):
        return None

    candidates = polymarket_feed._iter_market_candidates(payload)
    selected = None
    for candidate in candidates:
        candidate_market_id = (
            candidate.get('id')
            or candidate.get('marketId')
            or candidate.get('market_id')
            or candidate.get('questionID')
        )
        if str(candidate_market_id) == str(market_id):
            selected = candidate
            break
    if selected is None:
        selected = payload

    start_dt = polymarket_feed._parse_dt(
        selected.get('startDate') or selected.get('start') or payload.get('startDate') or payload.get('start')
    )
    end_dt = polymarket_feed._parse_dt(
        selected.get('endDate') or selected.get('end') or payload.get('endDate') or payload.get('end')
    )
    slug = selected.get('slug') or payload.get('slug') or payload.get('eventSlug')
    title = selected.get('title') or selected.get('question') or selected.get('name') or payload.get('title') or payload.get('name')
    status = _normalize_gamma_market_status(
        selected.get('status') or selected.get('state') or selected.get('marketStatus') or payload.get('status') or payload.get('state')
    )
    upsert_market(
        market_id=market_id,
        condition_id=selected.get('conditionId') or selected.get('condition_id') or selected.get('condition') or payload.get('conditionId') or payload.get('condition_id'),
        slug=slug,
        title=title,
        start_time=start_dt.isoformat() if start_dt is not None else None,
        end_time=end_dt.isoformat() if end_dt is not None else None,
        status=status,
    )
    hydrated = get_market(market_id)
    if hydrated is None:
        return None
    hydrated['hydration_source'] = 'gamma_market_id'
    return hydrated


def upsert_market(market_id: str, condition_id: Optional[str] = None, slug: Optional[str] = None, title: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, status: Optional[str] = 'open', winning_outcome: Optional[str] = None, last_checked_ts: Optional[str] = None, last_redeem_ts: Optional[str] = None):
    if not market_id:
        raise ValueError('market_id is required')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT market_id, condition_id, slug, title, start_time, end_time, status, winning_outcome, last_checked_ts, last_redeem_ts FROM markets WHERE market_id = ?', (market_id,))
    existing = cur.fetchone()
    if existing is None:
        cur.execute(
            'INSERT INTO markets(market_id, condition_id, slug, title, start_time, end_time, status, winning_outcome, last_checked_ts, last_redeem_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (market_id, condition_id, slug, title, start_time, end_time, status or 'open', winning_outcome, last_checked_ts, last_redeem_ts),
        )
    else:
        merged = {
            'condition_id': _coalesce_market_value(existing[1], condition_id),
            'slug': _coalesce_market_value(existing[2], slug),
            'title': _coalesce_market_value(existing[3], title),
            'start_time': _coalesce_market_value(existing[4], start_time),
            'end_time': _coalesce_market_value(existing[5], end_time),
            'status': _coalesce_market_status(existing[6], status),
            'winning_outcome': _coalesce_market_value(existing[7], winning_outcome),
            'last_checked_ts': _coalesce_market_value(existing[8], last_checked_ts),
            'last_redeem_ts': _coalesce_market_value(existing[9], last_redeem_ts),
        }
        cur.execute(
            'UPDATE markets SET condition_id = ?, slug = ?, title = ?, start_time = ?, end_time = ?, status = ?, winning_outcome = ?, last_checked_ts = ?, last_redeem_ts = ? WHERE market_id = ?',
            (
                merged['condition_id'],
                merged['slug'],
                merged['title'],
                merged['start_time'],
                merged['end_time'],
                merged['status'],
                merged['winning_outcome'],
                merged['last_checked_ts'],
                merged['last_redeem_ts'],
                market_id,
            ),
        )
    conn.commit()
    conn.close()


def create_market(market_id: str, condition_id: Optional[str] = None, slug: Optional[str] = None, title: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, status: str = 'open'):
    upsert_market(
        market_id=market_id,
        condition_id=condition_id,
        slug=slug,
        title=title,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


def update_market_status(market_id: str, status: str, winning_outcome: Optional[str] = None, last_checked_ts: Optional[str] = None):
    upsert_market(market_id, status=status, winning_outcome=winning_outcome, last_checked_ts=last_checked_ts)


def get_market(market_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT market_id, condition_id, slug, title, start_time, end_time, status, winning_outcome, last_checked_ts, last_redeem_ts FROM markets WHERE market_id = ?', (market_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {'market_id': row[0], 'condition_id': row[1], 'slug': row[2], 'title': row[3], 'start_time': row[4], 'end_time': row[5], 'status': row[6], 'winning_outcome': row[7], 'last_checked_ts': row[8], 'last_redeem_ts': row[9]}


def upsert_market_tokens(market_id: str, condition_id: Optional[str] = None, token_yes: Optional[str] = None, token_no: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, discovered_ts: Optional[str] = None):
    if not market_id:
        raise ValueError('market_id is required')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        'SELECT market_id, condition_id, token_yes, token_no, start_time, end_time, discovered_ts FROM market_tokens WHERE market_id = ?',
        (market_id,),
    )
    existing = cur.fetchone()
    if existing is None:
        cur.execute(
            'INSERT INTO market_tokens(market_id, condition_id, token_yes, token_no, start_time, end_time, discovered_ts) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (market_id, condition_id, token_yes, token_no, start_time, end_time, discovered_ts),
        )
    else:
        merged = {
            'condition_id': _coalesce_market_value(existing[1], condition_id),
            'token_yes': _coalesce_market_value(existing[2], token_yes),
            'token_no': _coalesce_market_value(existing[3], token_no),
            'start_time': _coalesce_market_value(existing[4], start_time),
            'end_time': _coalesce_market_value(existing[5], end_time),
            'discovered_ts': _coalesce_market_value(existing[6], discovered_ts),
        }
        cur.execute(
            'UPDATE market_tokens SET condition_id = ?, token_yes = ?, token_no = ?, start_time = ?, end_time = ?, discovered_ts = ? WHERE market_id = ?',
            (
                merged['condition_id'],
                merged['token_yes'],
                merged['token_no'],
                merged['start_time'],
                merged['end_time'],
                merged['discovered_ts'],
                market_id,
            ),
        )
    conn.commit()
    conn.close()


def get_market_tokens(market_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        'SELECT market_id, condition_id, token_yes, token_no, start_time, end_time, discovered_ts FROM market_tokens WHERE market_id = ?',
        (market_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'market_id': row[0],
        'condition_id': row[1],
        'token_yes': row[2],
        'token_no': row[3],
        'start_time': row[4],
        'end_time': row[5],
        'discovered_ts': row[6],
    }


def get_series_runtime_state(series_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''SELECT
            series_id,
            active_market_id,
            active_token_yes,
            active_token_no,
            active_start_time,
            active_end_time,
            strike_price,
            strike_source,
            strike_fixed_ts,
            last_switch_ts,
            status,
            previous_market_id
        FROM series_runtime_state
        WHERE series_id = ?''',
        (series_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'series_id': row[0],
        'active_market_id': row[1],
        'active_token_yes': row[2],
        'active_token_no': row[3],
        'active_start_time': row[4],
        'active_end_time': row[5],
        'strike_price': float(row[6]) if row[6] is not None else None,
        'strike_source': row[7],
        'strike_fixed_ts': row[8],
        'last_switch_ts': row[9],
        'status': row[10],
        'previous_market_id': row[11],
    }


def set_series_runtime_state(series_id: str, *, active_market_id: Optional[str] = None, active_token_yes: Optional[str] = None, active_token_no: Optional[str] = None, active_start_time: Optional[str] = None, active_end_time: Optional[str] = None, strike_price: Optional[float] = None, strike_source: Optional[str] = None, strike_fixed_ts: Optional[str] = None, last_switch_ts: Optional[str] = None, status: Optional[str] = None, previous_market_id: Optional[str] = None):
    if not series_id:
        raise ValueError('series_id is required')
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''SELECT
            active_market_id,
            active_token_yes,
            active_token_no,
            active_start_time,
            active_end_time,
            strike_price,
            strike_source,
            strike_fixed_ts,
            last_switch_ts,
            status,
            previous_market_id
        FROM series_runtime_state
        WHERE series_id = ?''',
        (series_id,),
    )
    existing = cur.fetchone()
    if existing is None:
        cur.execute(
            '''INSERT INTO series_runtime_state(
                series_id,
                active_market_id,
                active_token_yes,
                active_token_no,
                active_start_time,
                active_end_time,
                strike_price,
                strike_source,
                strike_fixed_ts,
                last_switch_ts,
                status,
                previous_market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                series_id,
                active_market_id,
                active_token_yes,
                active_token_no,
                active_start_time,
                active_end_time,
                strike_price,
                strike_source,
                strike_fixed_ts,
                last_switch_ts,
                status,
                previous_market_id,
            ),
        )
    else:
        merged = {
            'active_market_id': _coalesce_market_value(existing[0], active_market_id),
            'active_token_yes': _coalesce_market_value(existing[1], active_token_yes),
            'active_token_no': _coalesce_market_value(existing[2], active_token_no),
            'active_start_time': _coalesce_market_value(existing[3], active_start_time),
            'active_end_time': _coalesce_market_value(existing[4], active_end_time),
            'strike_price': existing[5] if strike_price is None else strike_price,
            'strike_source': _coalesce_market_value(existing[6], strike_source),
            'strike_fixed_ts': _coalesce_market_value(existing[7], strike_fixed_ts),
            'last_switch_ts': _coalesce_market_value(existing[8], last_switch_ts),
            'status': _coalesce_market_value(existing[9], status),
            'previous_market_id': _coalesce_market_value(existing[10], previous_market_id),
        }
        cur.execute(
            '''UPDATE series_runtime_state SET
                active_market_id = ?,
                active_token_yes = ?,
                active_token_no = ?,
                active_start_time = ?,
                active_end_time = ?,
                strike_price = ?,
                strike_source = ?,
                strike_fixed_ts = ?,
                last_switch_ts = ?,
                status = ?,
                previous_market_id = ?
            WHERE series_id = ?''',
            (
                merged['active_market_id'],
                merged['active_token_yes'],
                merged['active_token_no'],
                merged['active_start_time'],
                merged['active_end_time'],
                merged['strike_price'],
                merged['strike_source'],
                merged['strike_fixed_ts'],
                merged['last_switch_ts'],
                merged['status'],
                merged['previous_market_id'],
                series_id,
            ),
        )
    conn.commit()
    conn.close()
    return get_series_runtime_state(series_id)


def mark_series_market_switch(series_id: str, *, active_market_id: str, active_token_yes: str, active_token_no: str, active_start_time: Optional[str], active_end_time: Optional[str], strike_price: Optional[float], strike_source: Optional[str], strike_fixed_ts: Optional[str], switch_ts: str, status: str, previous_market_id: Optional[str] = None):
    return set_series_runtime_state(
        series_id,
        active_market_id=active_market_id,
        active_token_yes=active_token_yes,
        active_token_no=active_token_no,
        active_start_time=active_start_time,
        active_end_time=active_end_time,
        strike_price=strike_price,
        strike_source=strike_source,
        strike_fixed_ts=strike_fixed_ts,
        last_switch_ts=switch_ts,
        status=status,
        previous_market_id=previous_market_id,
    )


def refresh_market_lifecycle(market_id: str, source_data: Optional[Dict] = None, checked_ts: Optional[str] = None) -> Optional[Dict]:
    market = get_market(market_id)
    if market is None:
        return None
    ts = checked_ts or datetime.now(timezone.utc).isoformat()
    source_data = source_data or {}
    status = source_data.get('status')
    winning_outcome = source_data.get('winning_outcome')
    end_time = source_data.get('end_time') or source_data.get('endDate') or market.get('end_time')
    now_dt = _parse_iso_ts(ts) or datetime.now(timezone.utc)
    end_dt = _parse_iso_ts(end_time)
    resolution_diagnostics = None

    if status is None and market.get('status') == 'open' and end_dt is not None and end_dt <= now_dt:
        status = 'closed'
    if winning_outcome is None and status not in ('resolved', 'redeemed', 'archived'):
        derived_resolution = _derive_btc_hourly_resolution(
            {
                **market,
                'slug': source_data.get('slug') or market.get('slug'),
                'title': source_data.get('title') or market.get('title'),
                'start_time': source_data.get('start_time') or source_data.get('startDate') or market.get('start_time'),
                'end_time': end_time,
                'status': status or market.get('status'),
            },
            checked_ts=ts,
        )
        if derived_resolution is not None:
            resolution_diagnostics = derived_resolution.get('diagnostics')
            status = derived_resolution.get('status') or status
            winning_outcome = derived_resolution.get('winning_outcome') or winning_outcome

    upsert_market(
        market_id=market_id,
        condition_id=source_data.get('condition_id') or source_data.get('conditionId'),
        slug=source_data.get('slug'),
        title=source_data.get('title'),
        start_time=source_data.get('start_time') or source_data.get('startDate'),
        end_time=end_time,
        status=status,
        winning_outcome=winning_outcome,
        last_checked_ts=ts,
    )
    refreshed = get_market(market_id)
    if refreshed is None:
        return None
    if resolution_diagnostics is not None:
        refreshed['resolution_diagnostics'] = resolution_diagnostics
    return refreshed


def sweep_expired_open_markets(checked_ts: Optional[str] = None) -> List[Dict]:
    ts = checked_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT market_id FROM markets WHERE status = 'open'")
    market_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    refreshed = []
    for market_id in market_ids:
        market = get_market(market_id)
        if market is None:
            continue
        end_dt = _parse_iso_ts(market.get('end_time'))
        now_dt = _parse_iso_ts(ts) or datetime.now(timezone.utc)
        if end_dt is not None and end_dt <= now_dt:
            refreshed_market = refresh_market_lifecycle(market_id, checked_ts=ts)
            if refreshed_market is not None:
                refreshed.append(refreshed_market)
    return refreshed


def _consume_open_lots_fifo_rows(market_id: str, token_id: str, outcome_side: str, qty: float) -> List[Dict]:
    """Consume qty from open_lots FIFO and return the consumed lot fragments."""
    if qty <= 0:
        return []
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        'SELECT id, qty, avg_price, tx_hash, ts FROM open_lots WHERE market_id = ? AND token_id = ? AND outcome_side = ? ORDER BY ts ASC, id ASC',
        (market_id, token_id, outcome_side),
    )
    rows = cur.fetchall()
    remaining = qty
    consumed_rows = []
    for lot_id, lot_qty, avg_price, lot_tx_hash, lot_ts in rows:
        if remaining <= 0:
            break
        lot_qty = float(lot_qty)
        lot_price = float(avg_price) if avg_price is not None else None
        if lot_qty <= remaining + 1e-12:
            cur.execute('DELETE FROM open_lots WHERE id = ?', (lot_id,))
            take = lot_qty
        else:
            take = remaining
            cur.execute('UPDATE open_lots SET qty = ? WHERE id = ?', (lot_qty - take, lot_id))
        consumed_rows.append({'open_lot_id': int(lot_id), 'qty': float(take), 'avg_price': lot_price, 'source_tx_hash': lot_tx_hash, 'source_ts': lot_ts})
        remaining -= take

    conn.commit()
    conn.close()
    total_consumed = sum(float(row['qty']) for row in consumed_rows)
    if total_consumed + 1e-12 < qty:
        raise RuntimeError(f'Not enough inventory to consume: requested {qty}, consumed {total_consumed}')
    return consumed_rows


def _coerce_regime_scores(regime_state: Optional[Dict]) -> Dict:
    regime_state = regime_state or {}
    return {
        'trend_score': regime_state.get('trend_score'),
        'tail_score': regime_state.get('tail_score'),
        'reversal_score': regime_state.get('reversal_score'),
        'market_polarization': regime_state.get('market_polarization'),
    }


def _extract_microstructure_fields(*sources: Optional[Dict]) -> Dict:
    fields = (
        'microstructure_regime',
        'spectral_entropy',
        'low_freq_power_ratio',
        'high_freq_power_ratio',
        'smoothness_score',
        'spectral_observation_count',
        'spectral_window_minutes',
        'spectral_ready',
        'spectral_reason',
    )
    resolved: Dict = {}
    for field in fields:
        value = None
        for source in sources:
            if isinstance(source, dict) and source.get(field) is not None:
                value = source.get(field)
                break
        resolved[field] = value
    return resolved


def _resolve_exit_regime_state(market_id: str, extra: Optional[Dict] = None) -> Dict:
    extra = extra or {}
    decision_context = extra.get('decision_context') or {}
    regime_state = decision_context.get('regime_state')
    if isinstance(regime_state, dict) and regime_state:
        return regime_state
    latest = get_latest_regime_observation(market_id)
    if latest is None:
        return {}
    source = latest.get('source') or {}
    return {
        'regime_label': latest.get('regime_label'),
        'trend_score': latest.get('trend_score'),
        'tail_score': latest.get('tail_score'),
        'reversal_score': latest.get('reversal_score'),
        'market_polarization': source.get('market_polarization'),
    }


def consume_open_lots_fifo(
    market_id: str,
    token_id: str,
    outcome_side: str,
    qty: float,
    consume_tx: Optional[str] = None,
    ts: Optional[str] = None,
    execution_price: Optional[float] = None,
    extra: Optional[Dict] = None,
):
    """Consume qty from open_lots FIFO for the given market/token/side. Moves consumed quantities to fills(kind='sell') and adjusts lots.
    Returns total consumed quantity.
    """
    consumed_rows = _consume_open_lots_fifo_rows(market_id, token_id, outcome_side, qty)
    exit_regime_state = _resolve_exit_regime_state(market_id, extra=extra)
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    for row in consumed_rows:
        lot_attr = get_lot_regime_attribution(int(row['open_lot_id']))
        entry_price = float(row['avg_price'] or 0.0)
        exit_price = float(execution_price if execution_price is not None else entry_price)
        gross_pnl = (exit_price - entry_price) * float(row['qty'])
        fee_slippage_buffer = None
        if extra is not None and extra.get('fee_slippage_buffer') is not None:
            fee_slippage_buffer = float(extra.get('fee_slippage_buffer') or 0.0)
        net_pnl = gross_pnl if fee_slippage_buffer is None else gross_pnl - (fee_slippage_buffer * float(row['qty']))
        fill_extra = {
            **(extra or {}),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_per_share': exit_price - entry_price,
            'profit_total': gross_pnl,
            'open_lot_id': row['open_lot_id'],
            'entry_regime_label': None if lot_attr is None else lot_attr.get('entry_regime_label'),
            'exit_regime_label': exit_regime_state.get('regime_label'),
        }
        cur.execute(
            'INSERT INTO fills(market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                market_id,
                token_id,
                outcome_side,
                consume_tx,
                -float(row['qty']),
                exit_price,
                ts,
                'sell',
                json.dumps(fill_extra),
            ),
        )
        _insert_realized_pnl_event_cur(
            cur,
            ts=ts or datetime.now(timezone.utc).isoformat(),
            market_id=market_id,
            token_id=token_id,
            outcome_side=outcome_side,
            qty=float(row['qty']),
            disposition_type='sell',
            entry_price=entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fee_slippage_buffer=fee_slippage_buffer,
            source_fill_id=None if extra is None else extra.get('source_fill_id'),
            source_disposal_id=None if extra is None else extra.get('source_disposal_id'),
            source_tx_hash=consume_tx,
            open_lot_id=row['open_lot_id'],
            entry_regime_label=None if lot_attr is None else lot_attr.get('entry_regime_label'),
            entry_regime_scores=None if lot_attr is None else lot_attr.get('entry_regime_scores'),
            exit_regime_label=exit_regime_state.get('regime_label'),
            exit_regime_scores=_coerce_regime_scores(exit_regime_state),
            extra=fill_extra,
        )
    conn.commit()
    conn.close()
    return sum(float(row['qty']) for row in consumed_rows)


def get_unreconciled_fills() -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT id, market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind FROM fills WHERE tx_hash IS NOT NULL AND tx_hash != '' AND receipt_processed = 0")
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'market_id': r[1], 'token_id': r[2], 'outcome_side': r[3], 'tx_hash': r[4], 'qty': r[5], 'price': r[6], 'ts': r[7], 'kind': r[8]} for r in rows]


def get_reconciliation_issues() -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id, tx_hash, fill_id, observed_json, expected_json, reason, ts FROM reconciliation_issues ORDER BY ts DESC')
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'tx_hash': r[1], 'fill_id': r[2], 'observed_json': r[3], 'expected_json': r[4], 'reason': r[5], 'ts': r[6]} for r in rows]


def get_pending_receipts() -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id, tx_hash, raw_json, parsed, ts FROM receipts WHERE parsed = 0')
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'tx_hash': r[1], 'raw_json': r[2], 'parsed': r[3], 'ts': r[4]} for r in rows]


def get_order_tx_hashes() -> List[str]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT tx_hash FROM {ORDER_TABLE} WHERE tx_hash IS NOT NULL AND tx_hash != ''")
    rows = [tx_hash for (tx_hash,) in cur.fetchall() if tx_hash]
    conn.close()
    return sorted(set(rows))


def get_orders_pending_tx_recovery() -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id, client_order_id, venue_order_id, market_id, token_id, outcome_side, side, requested_qty, limit_price, filled_qty, remaining_qty, status, tx_hash, created_ts, updated_ts, raw_response_json
            FROM {ORDER_TABLE}
            WHERE status IN ('unknown', 'not_found_on_venue')
            ORDER BY created_ts ASC, id ASC'''
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        tx_hash = r[12]
        if not tx_hash:
            continue
        out.append(
            {
                'id': r[0], 'client_order_id': r[1], 'venue_order_id': r[2], 'market_id': r[3], 'token_id': r[4],
                'outcome_side': r[5], 'side': r[6], 'requested_qty': float(r[7]), 'limit_price': None if r[8] is None else float(r[8]),
                'filled_qty': float(r[9]), 'remaining_qty': float(r[10]), 'status': r[11], 'tx_hash': tx_hash, 'created_ts': r[13], 'updated_ts': r[14], 'raw_response_json': r[15]
            }
        )
    return out


def _recover_missing_fills_from_receipt(tx_hash: str, wallet_effects: Dict[Tuple[str, str], float], ts: str) -> List[Dict]:
    recovered = []
    for order in get_orders_by_tx_hash(tx_hash):
        direction = 'in' if order['side'] == 'buy' else 'out'
        observed_qty = float(wallet_effects.get((str(order['token_id']), direction), 0.0) or 0.0)
        if observed_qty <= 1e-12:
            continue
        requested_qty = float(order['requested_qty'] or 0.0)
        if observed_qty > requested_qty + 1e-9:
            continue
        if order['side'] == 'sell':
            available_qty = get_total_qty_by_token(str(order['token_id']), market_id=order['market_id'])
            if available_qty + 1e-12 < observed_qty:
                continue
        old_status = order['status']
        apply_res = apply_incremental_order_fill(
            order['id'],
            observed_qty,
            fill_ts=ts,
            tx_hash=tx_hash,
            price=order['limit_price'],
            raw={'recovered_from_receipt': True, 'tx_hash': tx_hash},
        )
        if apply_res.get('applied_qty', 0.0) <= 0:
            continue
        updated = get_order(order_id=order['id'])
        target_state = 'filled' if observed_qty >= requested_qty - 1e-12 else 'partially_filled'
        if updated and updated['status'] != target_state and can_transition_order_state(updated['status'], target_state):
            updated = transition_order_state(
                order['id'],
                target_state,
                reason='receipt_recovered_fill',
                raw={'recovered_from_receipt': True, 'tx_hash': tx_hash, 'recovered_qty': observed_qty},
                ts=ts,
            )
        repair_order_reservations(order['id'], updated_ts=ts)
        append_order_event(
            order['id'],
            'receipt_recovery',
            old_status=old_status,
            new_status=(updated or get_order(order_id=order['id']))['status'],
            response={'tx_hash': tx_hash, 'recovered_qty': observed_qty, 'recovered_from_receipt': True},
            ts=ts,
        )
        recovered.append(
            {
                'order_id': order['id'],
                'client_order_id': order['client_order_id'],
                'token_id': order['token_id'],
                'side': order['side'],
                'recovered_qty': observed_qty,
                'status': (updated or get_order(order_id=order['id']))['status'],
            }
        )
    return recovered


def repair_not_found_on_venue_fill_evidence(updated_ts: Optional[str] = None) -> Dict:
    ts = updated_ts or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute(
        f'''SELECT id FROM {ORDER_TABLE}
            WHERE status = 'not_found_on_venue'
            ORDER BY created_ts ASC, id ASC'''
    )
    order_ids = [int(row[0]) for row in cur.fetchall()]
    conn.close()

    repaired = []
    skipped = []
    for order_id in order_ids:
        order = get_order(order_id=order_id)
        if order is None:
            continue
        fill_events = get_order_fill_events(order_id)
        event_cumulative = max((float(event['cumulative_filled_qty']) for event in fill_events), default=0.0)
        evidence_qty = max(float(order.get('filled_qty') or 0.0), event_cumulative)
        if evidence_qty <= ORDER_QTY_TOLERANCE:
            skipped.append({'order_id': order_id, 'reason': 'no_fill_evidence'})
            continue
        requested_qty = max(0.0, float(order.get('requested_qty') or 0.0))
        evidence_qty = min(requested_qty, evidence_qty) if requested_qty > 0 else evidence_qty
        remaining_qty = max(0.0, requested_qty - evidence_qty)
        if remaining_qty <= ORDER_QTY_TOLERANCE:
            target_status = 'filled'
            remaining_qty = 0.0
        elif 0.0 < remaining_qty < requested_qty:
            target_status = 'partially_filled'
        else:
            skipped.append({'order_id': order_id, 'reason': 'zero_remaining_delta'})
            continue

        update_order(order_id, filled_qty=evidence_qty, remaining_qty=remaining_qty, updated_ts=ts)
        current = get_order(order_id=order_id)
        if current is not None and current['status'] != target_status and can_transition_order_state(current['status'], target_status):
            current = transition_order_state(
                order_id,
                target_status,
                reason='repair_not_found_on_venue_fill_evidence',
                raw={
                    'repair_source': 'fill_evidence',
                    'filled_qty': evidence_qty,
                    'remaining_qty': remaining_qty,
                    'fill_event_count': len(fill_events),
                },
                ts=ts,
            )
        repair_order_reservations(order_id, updated_ts=ts)
        repaired.append(
            {
                'order_id': order_id,
                'from_status': order['status'],
                'to_status': (current or get_order(order_id=order_id))['status'],
                'filled_qty': evidence_qty,
                'remaining_qty': remaining_qty,
                'fill_event_count': len(fill_events),
            }
        )

    return {
        'examined': len(order_ids),
        'repaired': len(repaired),
        'skipped': skipped,
        'orders': repaired,
    }


def get_active_order_diagnostics(now_ts: Optional[str] = None) -> Dict:
    now_dt = _parse_iso_ts(now_ts) or datetime.now(timezone.utc)
    diagnostics = []
    summary = {}
    for order in get_open_orders():
        reservations = get_order_reservations(order['id'], active_only=True)
        reservation_amount = sum(r['qty'] for r in reservations)
        reservation_type = reservations[0]['reservation_type'] if reservations else None
        updated_dt = _parse_iso_ts(order['updated_ts']) or now_dt
        conn = sqlite3.connect(get_db_path())
        cur = conn.cursor()
        cur.execute('SELECT reason FROM reconciliation_issues WHERE tx_hash = ? ORDER BY ts DESC, id DESC LIMIT 1', (order.get('tx_hash'),))
        issue_row = cur.fetchone()
        conn.close()
        state = order['status']
        age_sec = max(0.0, (now_dt - updated_dt).total_seconds())
        cancel_recommended = state in ('pending_submit', 'submitted', 'open', 'partially_filled') and age_sec > 300
        diagnostics.append({
            'order_id': order['id'],
            'client_order_id': order['client_order_id'],
            'venue_order_id': order['venue_order_id'],
            'market_id': order['market_id'],
            'token_id': order['token_id'],
            'side': order['side'],
            'outcome_side': order['outcome_side'],
            'limit_price': order['limit_price'],
            'requested_qty': order['requested_qty'],
            'filled_qty': order['filled_qty'],
            'remaining_qty': order['remaining_qty'],
            'state': state,
            'reservation_type': reservation_type,
            'reservation_amount': reservation_amount,
            'age_sec': age_sec,
            'tx_hash': order.get('tx_hash'),
            'last_reconciliation_reason': issue_row[0] if issue_row else None,
            'cancel_recommended': cancel_recommended,
        })
        summary[state] = summary.get(state, 0) + 1
    return {'summary': summary, 'orders': diagnostics}


def get_clean_start_status() -> Dict:
    """Return whether the ledger is in a clean zero-inventory startup state."""
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    counts = {}
    for table in ('open_lots', 'merged_lots', 'redeemed_lots', 'reconciliation_issues'):
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        counts[table] = int(cur.fetchone()[0])
    conn.close()
    return {
        'open_lots': counts['open_lots'],
        'merged_lots': counts['merged_lots'],
        'redeemed_lots': counts['redeemed_lots'],
        'reconciliation_issues': counts['reconciliation_issues'],
        'clean_start': all(counts[name] == 0 for name in counts),
    }


def _get_unique_token_for_side(cur, market_id: str, outcome_side: str) -> Optional[str]:
    cur.execute('SELECT DISTINCT token_id FROM open_lots WHERE market_id = ? AND outcome_side = ?', (market_id, outcome_side))
    rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeError(f'Market {market_id} has multiple token_ids for side {outcome_side}: {rows}')
    return rows[0]


def merge_position(market_id: str, yes_token_id: str, no_token_id: str, qty: float, merge_tx_hash: Optional[str] = None, ts: Optional[str] = None, collateral_returned: Optional[Dict] = None, exit_regime_state: Optional[Dict] = None) -> float:
    """Consume equal YES/NO inventory for a merge without marking the market redeemed."""
    if qty <= 0:
        raise ValueError('merge qty must be > 0')
    ts = ts or datetime.now(timezone.utc).isoformat()
    yes_available = get_available_qty(market_id, yes_token_id, 'YES')
    no_available = get_available_qty(market_id, no_token_id, 'NO')
    if yes_available + 1e-12 < qty or no_available + 1e-12 < qty:
        raise RuntimeError(f'Not enough paired inventory to merge: requested {qty}, have YES={yes_available}, NO={no_available}')

    yes_rows = _consume_open_lots_fifo_rows(market_id, yes_token_id, 'YES', qty)
    no_rows = _consume_open_lots_fifo_rows(market_id, no_token_id, 'NO', qty)
    extra = {'collateral_returned': collateral_returned} if collateral_returned is not None else None
    exit_regime_state = exit_regime_state or _resolve_exit_regime_state(market_id)

    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    for side, token_id, consumed_rows in [('YES', yes_token_id, yes_rows), ('NO', no_token_id, no_rows)]:
        for row in consumed_rows:
            lot_attr = get_lot_regime_attribution(int(row['open_lot_id']))
            cur.execute(
                'INSERT INTO merged_lots(market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash, merge_tx_hash, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (market_id, token_id, side, row['qty'], row['avg_price'], row['source_ts'], row['source_tx_hash'], merge_tx_hash, json.dumps(extra) if extra is not None else None),
            )
            cur.execute(
                'INSERT INTO fills(market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (market_id, token_id, side, merge_tx_hash, -float(row['qty']), row['avg_price'] or 0.0, ts, 'merge', json.dumps(extra) if extra is not None else None),
            )
            _insert_realized_pnl_event_cur(
                cur,
                ts=ts,
                market_id=market_id,
                token_id=token_id,
                outcome_side=side,
                qty=float(row['qty']),
                disposition_type='merge',
                entry_price=row['avg_price'],
                exit_price=1.0,
                gross_pnl=(1.0 - float(row['avg_price'] or 0.0)) * float(row['qty']),
                net_pnl=(1.0 - float(row['avg_price'] or 0.0)) * float(row['qty']),
                source_tx_hash=merge_tx_hash,
                open_lot_id=row['open_lot_id'],
                entry_regime_label=None if lot_attr is None else lot_attr.get('entry_regime_label'),
                entry_regime_scores=None if lot_attr is None else lot_attr.get('entry_regime_scores'),
                exit_regime_label=exit_regime_state.get('regime_label'),
                exit_regime_scores=_coerce_regime_scores(exit_regime_state),
                extra={'collateral_returned': collateral_returned, 'kind': 'merge'},
            )
    conn.commit()
    conn.close()
    return float(qty)


def merge_market_pair(market_id: str, qty: float, merge_tx_hash: Optional[str] = None, ts: Optional[str] = None, collateral_returned: Optional[Dict] = None, exit_regime_state: Optional[Dict] = None) -> float:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    yes_token_id = _get_unique_token_for_side(cur, market_id, 'YES')
    no_token_id = _get_unique_token_for_side(cur, market_id, 'NO')
    conn.close()
    if yes_token_id is None or no_token_id is None:
        raise RuntimeError(f'Market {market_id} requires both YES and NO inventory for merge')
    return merge_position(market_id, yes_token_id, no_token_id, qty, merge_tx_hash=merge_tx_hash, ts=ts, collateral_returned=collateral_returned, exit_regime_state=exit_regime_state)


def redeem_market(market_id: str, winning_outcome: str, redeem_tx_hash: Optional[str] = None, ts: Optional[str] = None, exit_regime_state: Optional[Dict] = None):
    """Move winning open_lots into redeemed_lots and record redemption fills.

    This function is a synchronous, best-effort redeemer for resolved markets.
    """
    ts = ts or datetime.now(timezone.utc).isoformat()
    # move all open_lots for the market into redeemed_lots (both winning and losing sides)
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id, token_id, outcome_side, qty, avg_price, ts, tx_hash FROM open_lots WHERE market_id = ? ORDER BY outcome_side ASC, ts ASC', (market_id,))
    rows = cur.fetchall()
    total_redeemed = 0.0
    exit_regime_state = exit_regime_state or _resolve_exit_regime_state(market_id)
    for r in rows:
        lid, token_id, outcome_side, qty, avg_price, lot_ts, lot_tx = r
        lot_attr = get_lot_regime_attribution(int(lid))
        # record redeemed history for all sides; only winning side will have redeem_tx_hash populated
        redeem_hash = redeem_tx_hash if outcome_side == winning_outcome else None
        cur.execute('INSERT INTO redeemed_lots(market_id, token_id, outcome_side, qty, avg_price, ts, tx_hash, redeem_tx_hash, redeem_receipt_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (market_id, token_id, outcome_side, qty, avg_price, lot_ts, lot_tx, redeem_hash, None))
        # delete from open_lots
        cur.execute('DELETE FROM open_lots WHERE id = ?', (lid,))
        # record a fill for audit: 'redeem' for winning side, 'settle' for losing side
        kind = 'redeem' if outcome_side == winning_outcome else 'settle'
        fill_qty = float(qty) if outcome_side == winning_outcome else 0.0
        cur.execute('INSERT INTO fills(market_id, token_id, outcome_side, tx_hash, qty, price, ts, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (market_id, token_id, outcome_side, redeem_hash, fill_qty, avg_price or 0.0, ts, kind))
        if outcome_side == winning_outcome:
            total_redeemed += float(qty)
            _insert_realized_pnl_event_cur(
                cur,
                ts=ts,
                market_id=market_id,
                token_id=token_id,
                outcome_side=outcome_side,
                qty=float(qty),
                disposition_type='redeem',
                entry_price=avg_price,
                exit_price=1.0,
                gross_pnl=(1.0 - float(avg_price or 0.0)) * float(qty),
                net_pnl=(1.0 - float(avg_price or 0.0)) * float(qty),
                source_tx_hash=redeem_hash,
                open_lot_id=lid,
                entry_regime_label=None if lot_attr is None else lot_attr.get('entry_regime_label'),
                entry_regime_scores=None if lot_attr is None else lot_attr.get('entry_regime_scores'),
                exit_regime_label=exit_regime_state.get('regime_label'),
                exit_regime_scores=_coerce_regime_scores(exit_regime_state),
                extra={'kind': 'redeem'},
            )

    # update market status to redeemed
    cur.execute('UPDATE markets SET status = ?, last_redeem_ts = ? WHERE market_id = ?', ('redeemed', ts, market_id))
    conn.commit()
    conn.close()
    return total_redeemed


def _normalize_addr(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    addr = str(addr).lower()
    return addr if addr.startswith('0x') else f'0x{addr}'


def _decode_addr_topic(topic: Optional[str]) -> Optional[str]:
    if not topic:
        return None
    topic = str(topic).lower()
    if topic.startswith('0x'):
        topic = topic[2:]
    return '0x' + topic[-40:]


def _decode_uint256(data: Optional[str]) -> Optional[int]:
    if not data:
        return None
    data = str(data)
    if data.startswith('0x'):
        data = data[2:]
    if not data:
        return None
    return int(data, 16)


def _direction_for_wallet(from_addr: Optional[str], to_addr: Optional[str], wallet: Optional[str]) -> str:
    wallet = _normalize_addr(wallet)
    from_addr = _normalize_addr(from_addr)
    to_addr = _normalize_addr(to_addr)
    if wallet and to_addr == wallet:
        return 'in'
    if wallet and from_addr == wallet:
        return 'out'
    return 'other'


def _decode_erc20_transfer_log(log: Dict) -> Optional[Dict]:
    topics = [str(t).lower() for t in (log.get('topics') or [])]
    if len(topics) < 3 or topics[0] != ERC20_TRANSFER_SIG:
        return None
    return {
        'token_contract': _normalize_addr(log.get('address')),
        'from': _decode_addr_topic(topics[1]),
        'to': _decode_addr_topic(topics[2]),
        'value': _decode_uint256(log.get('data')),
    }


def classify_merge_receipt(receipt: Dict, wallet_address: Optional[str], market_tokens: Optional[Dict[str, str]] = None) -> Dict:
    """Classify whether a receipt looks like a merge of paired ERC-1155 outcome tokens."""
    from . import polymarket_client

    wallet = _normalize_addr(wallet_address)
    erc1155_out = {}
    erc20_in = []
    decoded_transfers = []
    logs = receipt.get('logs') or []
    for log in logs:
        decoded = None
        try:
            decoded = polymarket_client._try_decode_erc1155_log(log)
        except Exception:
            decoded = None
        if decoded and decoded.get('event') and decoded.get('args'):
            args = decoded['args']
            if decoded['event'] == 'TransferSingle':
                entries = [(str(args.get('id')), float(args.get('value') or 0.0))]
            else:
                entries = [(str(token_id), float(value or 0.0)) for token_id, value in zip(args.get('ids') or [], args.get('values') or [])]
            from_addr = _normalize_addr(args.get('from'))
            to_addr = _normalize_addr(args.get('to'))
            direction = _direction_for_wallet(from_addr, to_addr, wallet)
            for token_id, qty in entries:
                decoded_transfers.append({'token_id': token_id, 'from': from_addr, 'to': to_addr, 'qty': qty, 'direction': direction})
                if direction == 'out':
                    erc1155_out[token_id] = erc1155_out.get(token_id, 0.0) + qty
        decoded_erc20 = _decode_erc20_transfer_log(log)
        if decoded_erc20 is not None:
            direction = _direction_for_wallet(decoded_erc20.get('from'), decoded_erc20.get('to'), wallet)
            if direction == 'in':
                erc20_in.append({**decoded_erc20, 'direction': direction})

    out_effects = [{'token_id': token_id, 'qty': qty} for token_id, qty in sorted(erc1155_out.items())]
    equal_pair = len(out_effects) == 2 and abs(out_effects[0]['qty'] - out_effects[1]['qty']) <= 1e-9
    tokens_match_market = True
    if market_tokens:
        expected_tokens = {str(v) for v in market_tokens.values() if v is not None}
        tokens_match_market = {item['token_id'] for item in out_effects} == expected_tokens

    if equal_pair and erc20_in and tokens_match_market:
        status = 'merge_confirmed'
    elif equal_pair:
        status = 'merge_candidate'
    else:
        status = 'not_merge'

    return {
        'status': status,
        'wallet_address': wallet,
        'erc1155_outflows': out_effects,
        'erc20_inflows': erc20_in,
        'decoded_transfers': decoded_transfers,
        'tokens_match_market': tokens_match_market,
    }


def reconcile_tx(tx_hash: str, wallet_address: Optional[str] = None):
    """Fetch receipt via polymarket client and reconcile fills associated with tx_hash.

    Returns a dict with keys including status:
    'ok' | 'pending' | 'mismatch' | 'unexpected_observed' |
    'decoded_no_wallet_effect' | 'decode_failed' | 'missing'.
    """
    def _normalize_effects(effects: Dict) -> List[Dict]:
        rows = []
        for (token_id, direction), qty in sorted(effects.items(), key=lambda item: (item[0][0], item[0][1])):
            rows.append({'token_id': str(token_id), 'direction': direction, 'qty': float(qty)})
        return rows

    def _record_issue(cur, tx_hash_val: str, reason: str, observed_payload, expected_payload, ts_val: str):
        cur.execute(
            'INSERT INTO reconciliation_issues(tx_hash, fill_id, observed_json, expected_json, reason, ts) VALUES (?, ?, ?, ?, ?, ?)',
            (
                tx_hash_val,
                None,
                json.dumps(observed_payload),
                json.dumps(expected_payload),
                reason,
                ts_val,
            ),
        )

    def _load_expected_fills(cur) -> Tuple[List[Tuple], Dict[Tuple[str, str], float], Optional[Dict[str, str]], Dict[Tuple[str, str], List[float]]]:
        cur.execute('SELECT id, market_id, token_id, outcome_side, qty, kind FROM fills WHERE tx_hash = ?', (tx_hash,))
        fill_rows = cur.fetchall()
        expected_effects = {}
        refs: Dict[Tuple[str, str], List[float]] = {}
        for fill_row in fill_rows:
            _, mid, token_id, outcome_side, qty, kind = fill_row
            if kind == 'buy':
                direction = 'in'
            elif kind in ('sell', 'redeem', 'merge'):
                direction = 'out'
            else:
                direction = 'in' if qty > 0 else 'out'
            key = (str(token_id), direction)
            expected_effects[key] = expected_effects.get(key, 0.0) + float(abs(qty))
            refs.setdefault(key, []).append(float(abs(qty)))
        token_map = None
        if fill_rows:
            market_id = fill_rows[0][1]
            per_market = {}
            for _, mid, token_id, outcome_side, _, _ in fill_rows:
                if mid == market_id and outcome_side in ('YES', 'NO') and outcome_side not in per_market:
                    per_market[outcome_side] = str(token_id)
            if per_market:
                token_map = per_market
        return fill_rows, expected_effects, token_map, refs

    from . import polymarket_client
    if not tx_hash:
        return {'status': 'missing', 'reason': 'no-tx-hash'}

    receipt = polymarket_client.get_tx_receipt(tx_hash)
    if not receipt:
        return {'status': 'pending', 'reason': 'no-receipt'}

    # store raw receipt
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id FROM receipts WHERE tx_hash = ?', (tx_hash,))
    existing_receipt = cur.fetchone()
    if existing_receipt is None:
        cur.execute('INSERT INTO receipts(tx_hash, raw_json, parsed, ts) VALUES (?, ?, ?, ?)', (tx_hash, json.dumps(receipt), 0, datetime.now(timezone.utc).isoformat()))
    else:
        cur.execute('UPDATE receipts SET raw_json = ?, ts = ? WHERE id = ?', (json.dumps(receipt), datetime.now(timezone.utc).isoformat(), existing_receipt[0]))
    conn.commit()
    conn.close()

    # attempt to parse logs for ERC-1155 TransferSingle / TransferBatch
    observed = []
    decode_failures = []
    collateral_transfers = []
    logs = receipt.get('logs') or []
    for log in logs:
        # try to decode raw logs using polymarket_client helper when available
        decoded = None
        try:
            decoded = polymarket_client._try_decode_erc1155_log(log)
        except Exception as exc:
            decoded = {'decode_failed': True, 'raw': log, 'error': str(exc)}

        if decoded and decoded.get('event') and decoded.get('args'):
            ev = decoded['event']
            args = decoded['args']
            if ev == 'TransferSingle':
                operator = args.get('operator')
                from_addr = args.get('from')
                to_addr = args.get('to')
                tid = args.get('id')
                val = args.get('value')
                observed.append({'type': 'TransferSingle', 'operator': operator, 'from': from_addr, 'to': to_addr, 'token_id': str(tid), 'value': float(val)})
            else:
                operator = args.get('operator')
                from_addr = args.get('from')
                to_addr = args.get('to')
                ids = args.get('ids') or []
                values = args.get('values') or []
                for tid, val in zip(ids, values):
                    observed.append({'type': 'TransferBatch', 'operator': operator, 'from': from_addr, 'to': to_addr, 'token_id': str(tid), 'value': float(val)})
        elif decoded and decoded.get('decode_failed'):
            decode_failures.append({'log': decoded.get('raw', log), 'error': decoded.get('error')})
        else:
            # keep unrelated/unknown logs available for operator review but do not
            # treat them as wallet effects.
            observed.append({'type': 'raw', 'log': log})

        decoded_erc20 = _decode_erc20_transfer_log(log)
        if decoded_erc20 is not None:
            decoded_erc20['direction'] = _direction_for_wallet(decoded_erc20.get('from'), decoded_erc20.get('to'), wallet_address or polymarket_client.WALLET_ADDRESS)
            collateral_transfers.append(decoded_erc20)

    ts = datetime.now(timezone.utc).isoformat()
    wallet = wallet_address or polymarket_client.WALLET_ADDRESS
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT id FROM receipts WHERE tx_hash = ?', (tx_hash,))
    receipt_id = cur.fetchone()[0]

    orders_by_tx = get_orders_by_tx_hash(tx_hash)
    fills, expected, market_tokens, reference_qtys = _load_expected_fills(cur)

    for order in orders_by_tx:
        direction = 'in' if order['side'] == 'buy' else 'out'
        key = (str(order['token_id']), direction)
        reference_qtys.setdefault(key, []).append(float(abs(order.get('requested_qty') or 0.0)))
        if float(order.get('filled_qty') or 0.0) > ORDER_QTY_TOLERANCE:
            reference_qtys.setdefault(key, []).append(float(abs(order['filled_qty'])))

    merge_classification = classify_merge_receipt(receipt, wallet, market_tokens=market_tokens)

    agg_observed = {}
    wallet_effects = {}
    decoded_transfers = []
    for o in observed:
        if o.get('type') not in ('TransferSingle', 'TransferBatch'):
            continue
        tid = str(o.get('token_id'))
        direction = _direction_for_wallet(o.get('from'), o.get('to'), wallet)
        key = (tid, direction)
        raw_value = float(o.get('value') or 0.0)
        normalized_value = _normalize_erc1155_transfer_qty(raw_value, reference_qtys.get(key))
        decoded_transfers.append(
            {
                **o,
                'direction': direction,
                'value': normalized_value,
                'raw_value': raw_value,
            }
        )
        agg_observed[key] = agg_observed.get(key, 0.0) + normalized_value
        if direction in ('in', 'out'):
            wallet_effects[key] = wallet_effects.get(key, 0.0) + normalized_value

    receipt_recoveries = _recover_missing_fills_from_receipt(tx_hash, wallet_effects, ts)
    if receipt_recoveries:
        fills, expected, market_tokens, refreshed_refs = _load_expected_fills(cur)
        for key, values in refreshed_refs.items():
            reference_qtys.setdefault(key, []).extend(values)

    observed_payload = {
        'wallet_address': wallet,
        'wallet_effects': _normalize_effects(wallet_effects),
        'all_effects': _normalize_effects(agg_observed),
        'decoded_transfers': decoded_transfers,
        'collateral_transfers': collateral_transfers,
        'merge_classification': merge_classification,
        'receipt_recoveries': receipt_recoveries,
        'decode_failures': decode_failures,
        'raw_log_count': len([o for o in observed if o.get('type') == 'raw']),
    }
    expected_payload = {
        'expected_effects': _normalize_effects(expected),
        'fill_count': len(fills),
        'orders': orders_by_tx,
    }

    if decode_failures:
        _record_issue(cur, tx_hash, 'decode_failed', observed_payload, expected_payload, ts)
        conn.commit()
        conn.close()
        return {
            'status': 'decode_failed',
            'observed': observed_payload,
            'expected': expected_payload,
            'reason': 'relevant_logs_not_decoded',
        }

    if not expected and wallet_effects:
        _record_issue(cur, tx_hash, 'unexpected_observed', observed_payload, expected_payload, ts)
        cur.execute('UPDATE receipts SET parsed = 1 WHERE id = ?', (receipt_id,))
        conn.commit()
        conn.close()
        return {
            'status': 'unexpected_observed',
            'observed': observed_payload,
            'expected': expected_payload,
        }

    if not expected and not wallet_effects:
        cur.execute('UPDATE receipts SET parsed = 1 WHERE id = ?', (receipt_id,))
        conn.commit()
        conn.close()
        return {
            'status': 'decoded_no_wallet_effect',
            'observed': observed_payload,
            'expected': expected_payload,
        }

    # compare expected vs observed
    mismatches = []
    all_keys = set(expected.keys()) | set(wallet_effects.keys())
    for k in sorted(all_keys):
        exp_qty = expected.get(k, 0.0)
        obs_qty = wallet_effects.get(k, 0.0)
        if abs(obs_qty - exp_qty) > 1e-9:
            mismatches.append({'token_id': k[0], 'direction': k[1], 'expected': exp_qty, 'observed': obs_qty})

    if mismatches:
        _record_issue(
            cur,
            tx_hash,
            'mismatch',
            {**observed_payload, 'mismatches': mismatches},
            expected_payload,
            ts,
        )
        conn.commit()
        conn.close()
        return {'status': 'mismatch', 'mismatches': mismatches, 'observed': observed_payload, 'expected': expected_payload}

    # no mismatches -> mark fills as reconciled and receipts parsed
    cur.execute('UPDATE fills SET receipt_processed = 1 WHERE tx_hash = ?', (tx_hash,))
    cur.execute('UPDATE receipts SET parsed = 1 WHERE id = ?', (receipt_id,))
    conn.commit()
    conn.close()
    return {'status': 'ok', 'observed': observed_payload, 'expected': expected_payload}


def run_reconciliation_sweep(wallet_address: Optional[str] = None) -> Dict:
    tx_hashes = set()
    for fill in get_unreconciled_fills():
        if fill.get('tx_hash'):
            tx_hashes.add(fill['tx_hash'])
    for tx_hash in get_order_tx_hashes():
        tx_hashes.add(tx_hash)

    summary = {'ok': 0, 'pending': 0, 'mismatch': 0, 'decode_failed': 0, 'no_receipt': 0, 'skipped': 0}
    results = []
    for tx_hash in sorted(tx_hashes):
        if not tx_hash:
            summary['skipped'] += 1
            results.append({'tx_hash': tx_hash, 'status': 'skipped'})
            continue
        result = reconcile_tx(tx_hash, wallet_address=wallet_address)
        status = result.get('status') or 'skipped'
        if status == 'pending':
            summary['pending'] += 1
            summary['no_receipt'] += 1
        elif status in summary:
            summary[status] += 1
        else:
            summary['skipped'] += 1
        results.append({'tx_hash': tx_hash, 'status': status, 'result': result})
    return {'summary': summary, 'results': results}


def get_position_snapshot() -> List[Dict]:
    """Return a snapshot of positions grouped by market with inventory buckets.

    Each item includes market metadata and quantities per bucket and per side.
    """
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT market_id, condition_id, slug, title, start_time, end_time, status, winning_outcome, last_checked_ts, last_redeem_ts FROM markets')
    markets = cur.fetchall()
    snapshot = []
    for m in markets:
        market_id = m[0]
        status = m[6]
        winning = m[7]

        def sum_side(table, side):
            cur.execute(f"SELECT SUM(qty) FROM {table} WHERE market_id = ? AND outcome_side = ?", (market_id, side))
            r = cur.fetchone()
            return float(r[0]) if r and r[0] is not None else 0.0

        open_yes = sum_side('open_lots', 'YES')
        open_no = sum_side('open_lots', 'NO')
        reserved_yes = get_reserved_qty(market_id, _get_unique_token_for_side(cur, market_id, 'YES') or '', 'YES') if _get_unique_token_for_side(cur, market_id, 'YES') else 0.0
        reserved_no = get_reserved_qty(market_id, _get_unique_token_for_side(cur, market_id, 'NO') or '', 'NO') if _get_unique_token_for_side(cur, market_id, 'NO') else 0.0
        redeemed_yes = 0.0
        redeemed_no = 0.0
        # redeemed_lots store historical redeemed
        cur.execute('SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ? AND outcome_side = ?', (market_id, 'YES'))
        r = cur.fetchone()
        if r and r[0] is not None:
            redeemed_yes = float(r[0])
        cur.execute('SELECT SUM(qty) FROM redeemed_lots WHERE market_id = ? AND outcome_side = ?', (market_id, 'NO'))
        r = cur.fetchone()
        if r and r[0] is not None:
            redeemed_no = float(r[0])

        # classify buckets
        tradable_open = {'YES': 0.0, 'NO': 0.0}
        closed_unresolved = {'YES': 0.0, 'NO': 0.0}
        resolved_unredeemed = {'YES': 0.0, 'NO': 0.0}
        redeemed_inventory = {'YES': redeemed_yes, 'NO': redeemed_no}

        if status == 'open':
            tradable_open['YES'] = open_yes
            tradable_open['NO'] = open_no
        elif status == 'closed':
            closed_unresolved['YES'] = open_yes
            closed_unresolved['NO'] = open_no
        elif status == 'resolved':
            resolved_unredeemed['YES'] = open_yes
            resolved_unredeemed['NO'] = open_no
        elif status == 'redeemed' or status == 'archived':
            # no open lots considered tradable/resolved
            pass

        # compute weighted avg price per side for open lots
        def avg_price_for_side(side):
            cur.execute('SELECT SUM(qty * avg_price), SUM(qty) FROM open_lots WHERE market_id = ? AND outcome_side = ?', (market_id, side))
            s = cur.fetchone()
            if s and s[1]:
                return float(s[0]) / float(s[1])
            return None

        avg_yes = avg_price_for_side('YES')
        avg_no = avg_price_for_side('NO')
        available_yes = max(0.0, open_yes - reserved_yes)
        available_no = max(0.0, open_no - reserved_no)
        mergeable_pair_qty = min(available_yes, available_no)
        resolved_redeemable_qty = 0.0
        if status == 'resolved' and winning in ('YES', 'NO'):
            resolved_redeemable_qty = open_yes if winning == 'YES' else open_no

        open_orders = get_open_orders(market_id=market_id)
        partially_filled_orders = [o for o in open_orders if o['status'] == 'partially_filled']
        stale_pending_orders = [o for o in open_orders if o['status'] == 'pending_submit']

        snapshot.append({
            'market_id': market_id,
            'condition_id': m[1],
            'slug': m[2],
            'title': m[3],
            'start_time': m[4],
            'end_time': m[5],
            'status': status,
            'winning_outcome': winning,
            'tradable_open_inventory': tradable_open,
            'reserved_inventory': {'YES': reserved_yes, 'NO': reserved_no},
            'available_inventory': {'YES': available_yes, 'NO': available_no},
            'closed_unresolved_inventory': closed_unresolved,
            'resolved_unredeemed_inventory': resolved_unredeemed,
            'mergeable_pair_qty': mergeable_pair_qty,
            'resolved_redeemable_qty': resolved_redeemable_qty,
            'redeemed_inventory': redeemed_inventory,
            'open_orders': open_orders,
            'partially_filled_orders': partially_filled_orders,
            'stale_pending_orders': stale_pending_orders,
            'avg_entry_price_yes': avg_yes,
            'avg_entry_price_no': avg_no,
            'last_checked_ts': m[8],
            'last_redeem_ts': m[9],
        })

    conn.close()
    return snapshot


def print_position_snapshot():
    snap = get_position_snapshot()
    for s in snap:
        print('Market:', s['market_id'], '|', s.get('slug') or s.get('title'))
        print('  Status:', s['status'], 'Winning:', s['winning_outcome'])
        print('  Tradable open YES/NO:', s['tradable_open_inventory']['YES'], '/', s['tradable_open_inventory']['NO'])
        print('  Reserved YES/NO:', s['reserved_inventory']['YES'], '/', s['reserved_inventory']['NO'])
        print('  Available YES/NO:', s['available_inventory']['YES'], '/', s['available_inventory']['NO'])
        print('  Mergeable pair qty:', s['mergeable_pair_qty'], 'Resolved redeemable qty:', s['resolved_redeemable_qty'])
        print('  Closed unresolved YES/NO:', s['closed_unresolved_inventory']['YES'], '/', s['closed_unresolved_inventory']['NO'])
        print('  Resolved unredeemed YES/NO:', s['resolved_unredeemed_inventory']['YES'], '/', s['resolved_unredeemed_inventory']['NO'])
        print('  Redeemed YES/NO:', s['redeemed_inventory']['YES'], '/', s['redeemed_inventory']['NO'])
        print('  Open orders:', len(s['open_orders']), 'Partial:', len(s['partially_filled_orders']), 'Stale pending:', len(s['stale_pending_orders']))
        print('  Avg entry YES/NO:', s['avg_entry_price_yes'], '/', s['avg_entry_price_no'])
        print('  last_checked:', s['last_checked_ts'], 'last_redeem:', s['last_redeem_ts'])
        print('')


def apply_order_9493_dust_migration(ts: Optional[str] = None) -> Dict:
    ts = ts or datetime.now(timezone.utc).isoformat()
    order = get_order(order_id=9493)
    if order is None:
        return {'status': 'missing', 'order_id': 9493}
    reservations_before = get_order_reservations(9493, active_only=True)
    moved = move_order_to_dust_status(
        9493,
        dust_status='dust_finalized',
        reason='one_time_dust_migration_9493',
        raw={'migration': 'order_9493_dust_finalized'},
        ts=ts,
    )
    reservations_after = get_order_reservations(9493, active_only=True)
    return {
        'status': 'ok',
        'order_id': 9493,
        'from_status': order['status'],
        'to_status': moved['status'],
        'released_reservation_count': len(reservations_before) - len(reservations_after),
    }
