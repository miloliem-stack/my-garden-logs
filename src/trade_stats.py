from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import storage


def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except Exception:
        return None


def _compact_json(value) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _decision_log_default_path() -> Path:
    return Path(os.getenv('DECISION_LOG_PATH', 'decision_state.jsonl')).expanduser()


def load_decision_log(path: Optional[Path] = None) -> List[Dict]:
    log_path = Path(path).expanduser() if path is not None else _decision_log_default_path()
    if not log_path.exists():
        return []
    decisions: List[Dict] = []
    with log_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                decisions.append(item)
    return decisions


def index_decision_log_by_market_and_time(decisions: List[Dict]) -> dict:
    index = defaultdict(list)
    for decision in decisions:
        market_id = decision.get('market_id')
        ts = decision.get('timestamp') or decision.get('ts') or decision.get('decision_ts')
        dt = _parse_iso_ts(ts)
        if not market_id or dt is None:
            continue
        index[str(market_id)].append((dt, decision))
    ordered = {}
    for market_id, rows in index.items():
        rows.sort(key=lambda item: item[0])
        ordered[market_id] = {
            'timestamps': [item[0] for item in rows],
            'decisions': [item[1] for item in rows],
        }
    return ordered


def match_fill_to_decision(fill: dict, decisions_index: dict, max_lookback_minutes: int = 15) -> Optional[dict]:
    market_id = fill.get('market_id')
    fill_dt = _parse_iso_ts(fill.get('ts'))
    market_bucket = decisions_index.get(str(market_id))
    if not market_id or fill_dt is None or not market_bucket:
        return None
    timestamps = market_bucket['timestamps']
    decisions = market_bucket['decisions']
    pos = bisect_right(timestamps, fill_dt) - 1
    best = None
    best_delta_sec = None
    fill_side = _normalize_fill_side(fill)
    fill_outcome = str(fill.get('outcome_side') or '').upper() or None
    for idx in range(pos, -1, -1):
        decision_dt = timestamps[idx]
        delta_sec = (fill_dt - decision_dt).total_seconds()
        if delta_sec < 0:
            continue
        if delta_sec > max_lookback_minutes * 60:
            break
        decision = decisions[idx]
        chosen_side = str(decision.get('chosen_side') or '').upper() or None
        action = str(decision.get('action') or '').lower() or None
        if fill_side == 'buy' and chosen_side and fill_outcome and chosen_side != fill_outcome:
            continue
        if fill_side == 'sell' and action and not action.startswith('sell'):
            pass
        best = decision
        best_delta_sec = delta_sec
        if delta_sec == 0:
            break
    return best if best_delta_sec is not None else None


def _normalize_fill_side(fill: Dict) -> Optional[str]:
    kind = str(fill.get('kind') or '').lower()
    qty = float(fill.get('qty') or 0.0)
    if kind == 'redeem':
        return None
    if kind == 'sell' or qty < 0:
        return 'sell'
    if kind in {'buy', 'trade'} or qty > 0:
        return 'buy'
    return None


def _extract_realized_pnl(fill: Dict) -> Optional[float]:
    extra = fill.get('extra') if isinstance(fill.get('extra'), dict) else {}
    if 'profit_total' not in extra:
        return None
    try:
        return float(extra.get('profit_total'))
    except Exception:
        return None


def _decision_field(decision: Optional[Dict], *names):
    if not isinstance(decision, dict):
        return None
    for name in names:
        if name in decision and decision.get(name) is not None:
            return decision.get(name)
    return None


def _decision_policy_bucket(decision: Optional[Dict]) -> Optional[str]:
    if not isinstance(decision, dict):
        return None
    policy = decision.get('policy') or {}
    return decision.get('policy_bucket') or policy.get('policy_bucket')


def _decision_tau(decision: Optional[Dict]) -> Optional[int]:
    value = _decision_field(decision, 'tau_minutes')
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _decision_bool(decision: Optional[Dict], *names) -> Optional[bool]:
    value = _decision_field(decision, *names)
    if value is None:
        return None
    return bool(value)


def _order_match_score(fill: Dict, order: Dict, fill_event: Optional[Dict] = None) -> tuple:
    fill_dt = _parse_iso_ts(fill.get('ts'))
    order_dt = _parse_iso_ts(order.get('created_ts')) or _parse_iso_ts(order.get('updated_ts'))
    event_dt = _parse_iso_ts((fill_event or {}).get('fill_ts'))
    reference_dt = event_dt or order_dt
    delta_sec = abs((fill_dt - reference_dt).total_seconds()) if fill_dt and reference_dt else 10**9
    qty_gap = abs(abs(float(fill.get('qty') or 0.0)) - abs(float((fill_event or {}).get('fill_qty') or order.get('filled_qty') or 0.0)))
    price_gap = abs(float(fill.get('price') or 0.0) - float((fill_event or {}).get('price') if (fill_event or {}).get('price') is not None else order.get('limit_price') or 0.0))
    return (qty_gap, price_gap, delta_sec, int(order.get('id') or 0))


def _match_fill_to_order(fill: Dict, orders: List[Dict], order_fill_events: List[Dict]) -> Optional[Dict]:
    exact_candidates = []
    fill_tx_hash = fill.get('tx_hash')
    fill_dt = _parse_iso_ts(fill.get('ts'))
    fill_qty = abs(float(fill.get('qty') or 0.0))
    fill_price = fill.get('price')
    fill_side = _normalize_fill_side(fill)
    for order in orders:
        if order.get('market_id') != fill.get('market_id'):
            continue
        if order.get('token_id') != fill.get('token_id'):
            continue
        if str(order.get('outcome_side') or '').upper() != str(fill.get('outcome_side') or '').upper():
            continue
        if fill_side is not None and order.get('side') != fill_side:
            continue
        fill_events_for_order = [event for event in order_fill_events if int(event.get('order_id') or 0) == int(order['id'])]
        if fill_tx_hash and fill_tx_hash == order.get('tx_hash'):
            if fill_events_for_order:
                for event in fill_events_for_order:
                    if fill_tx_hash and event.get('tx_hash') and event.get('tx_hash') != fill_tx_hash:
                        continue
                    exact_candidates.append((order, event))
            else:
                exact_candidates.append((order, None))
            continue
        for event in fill_events_for_order:
            if fill_tx_hash and event.get('tx_hash') == fill_tx_hash:
                exact_candidates.append((order, event))
                continue
            event_dt = _parse_iso_ts(event.get('fill_ts'))
            event_qty = abs(float(event.get('fill_qty') or 0.0))
            if fill_dt and event_dt and abs((fill_dt - event_dt).total_seconds()) <= 120 and abs(event_qty - fill_qty) <= 1e-9:
                if fill_price is None or event.get('price') is None or abs(float(event['price']) - float(fill_price)) <= 1e-9:
                    exact_candidates.append((order, event))
    if exact_candidates:
        best_order, best_event = min(exact_candidates, key=lambda item: _order_match_score(fill, item[0], item[1]))
        return {'order': best_order, 'order_fill_event': best_event}

    proximity_candidates = []
    for order in orders:
        if order.get('market_id') != fill.get('market_id'):
            continue
        if fill_side is not None and order.get('side') != fill_side:
            continue
        if order.get('token_id') and fill.get('token_id') and order.get('token_id') != fill.get('token_id'):
            continue
        if order.get('outcome_side') and fill.get('outcome_side') and str(order.get('outcome_side')).upper() != str(fill.get('outcome_side')).upper():
            continue
        order_dt = _parse_iso_ts(order.get('created_ts')) or _parse_iso_ts(order.get('updated_ts'))
        if fill_dt and order_dt and abs((fill_dt - order_dt).total_seconds()) <= 15 * 60:
            proximity_candidates.append((order, None))
    if not proximity_candidates:
        return None
    best_order, best_event = min(proximity_candidates, key=lambda item: _order_match_score(fill, item[0], item[1]))
    return {'order': best_order, 'order_fill_event': best_event}


def build_trade_journal_row(fill: dict, matched_order: Optional[dict], matched_decision: Optional[dict]) -> dict:
    order = (matched_order or {}).get('order') if isinstance(matched_order, dict) else matched_order
    extra = fill.get('extra') if isinstance(fill.get('extra'), dict) else {}
    kind = str(fill.get('kind') or 'trade').lower()
    side = _normalize_fill_side(fill)
    qty = abs(float(fill.get('qty') or 0.0))
    price = fill.get('price')
    notional = None if price is None else qty * float(price)
    decision_ts = _decision_field(matched_decision, 'timestamp', 'ts', 'decision_ts')
    raw_p_yes = _decision_field(matched_decision, 'raw_p_yes')
    raw_p_no = _decision_field(matched_decision, 'raw_p_no')
    adjusted_p_yes = _decision_field(matched_decision, 'p_yes', 'adjusted_p_yes', 'calibrated_p_yes')
    adjusted_p_no = _decision_field(matched_decision, 'p_no', 'adjusted_p_no')
    raw_edge_yes = _decision_field(matched_decision, 'raw_edge_yes')
    raw_edge_no = _decision_field(matched_decision, 'raw_edge_no')
    adjusted_edge_yes = _decision_field(matched_decision, 'edge_yes', 'adjusted_edge_yes')
    adjusted_edge_no = _decision_field(matched_decision, 'edge_no', 'adjusted_edge_no')
    tail_penalty = _decision_field(matched_decision, 'tail_penalty_score', 'polarized_tail_penalty')
    if tail_penalty is None:
        tail_penalty = _decision_field(matched_decision, 'q_tail')
    row = {
        'ts': fill.get('ts'),
        'market_id': fill.get('market_id'),
        'token_id': fill.get('token_id'),
        'outcome_side': fill.get('outcome_side'),
        'side': side,
        'kind': 'redeem' if kind == 'redeem' else kind,
        'qty': qty,
        'price': price,
        'notional': notional,
        'tx_hash': fill.get('tx_hash'),
        'order_id': order.get('id') if isinstance(order, dict) else None,
        'client_order_id': (order or {}).get('client_order_id') or extra.get('client_order_id'),
        'venue_order_id': (order or {}).get('venue_order_id'),
        'decision_ts': decision_ts,
        'decision_reason': _decision_field(matched_decision, 'reason'),
        'policy_bucket': _decision_policy_bucket(matched_decision),
        'tau_minutes': _decision_tau(matched_decision),
        'raw_p_yes': None if raw_p_yes is None else float(raw_p_yes),
        'raw_p_no': None if raw_p_no is None else float(raw_p_no),
        'adjusted_p_yes': None if adjusted_p_yes is None else float(adjusted_p_yes),
        'adjusted_p_no': None if adjusted_p_no is None else float(adjusted_p_no),
        'raw_edge_yes': None if raw_edge_yes is None else float(raw_edge_yes),
        'raw_edge_no': None if raw_edge_no is None else float(raw_edge_no),
        'adjusted_edge_yes': None if adjusted_edge_yes is None else float(adjusted_edge_yes),
        'adjusted_edge_no': None if adjusted_edge_no is None else float(adjusted_edge_no),
        'tail_penalty_score': None if tail_penalty is None else float(tail_penalty),
        'tail_hard_block': _decision_bool(matched_decision, 'tail_hard_block', 'polarized_tail_blocked'),
        'reeval_action': _decision_field(matched_decision, 'position_reeval_action'),
        'reeval_reason': _decision_field(matched_decision, 'position_reeval_reason'),
        'realized_pnl': _extract_realized_pnl(fill),
        'extra_json': _compact_json(extra) if extra else fill.get('extra_json'),
    }
    return row


def rebuild_trade_journal(clear_existing: bool = False, decision_log_path: Optional[str] = None) -> Dict:
    storage.ensure_db()
    if clear_existing:
        storage.clear_trade_journal()
    decisions = load_decision_log(Path(decision_log_path)) if decision_log_path else load_decision_log()
    decisions_index = index_decision_log_by_market_and_time(decisions)
    fills = storage.list_fills()
    orders = storage.list_orders()
    order_fill_events = storage.list_order_fill_events()
    inserted = 0
    if not clear_existing:
        storage.clear_trade_journal()
    for fill in fills:
        matched_order = _match_fill_to_order(fill, orders, order_fill_events)
        matched_decision = match_fill_to_decision(fill, decisions_index)
        row = build_trade_journal_row(fill, matched_order, matched_decision)
        storage.insert_trade_journal_row(**row)
        inserted += 1
    summary = storage.get_trade_journal_summary()
    return {
        'fills_seen': len(fills),
        'journal_rows_inserted': inserted,
        'decision_log_rows': len(decisions),
        'summary': summary,
    }


def build_cumulative_realized_pnl_series(rows: Optional[List[Dict]] = None) -> List[Dict]:
    journal_rows = rows if rows is not None else storage.list_trade_journal()
    running = 0.0
    series = []
    for row in journal_rows:
        realized_pnl = float(row.get('realized_pnl') or 0.0)
        running += realized_pnl
        series.append(
            {
                'ts': row.get('ts'),
                'realized_pnl': realized_pnl,
                'cumulative_realized_pnl': running,
            }
        )
    return series


def build_trade_stats_report(rows: Optional[List[Dict]] = None) -> Dict:
    journal_rows = rows if rows is not None else storage.list_trade_journal()
    summary = storage.get_trade_journal_summary()
    realized_by_policy_bucket: Dict[str, float] = defaultdict(float)
    realized_by_tail_regime: Dict[str, float] = defaultdict(float)
    realized_by_reeval_action: Dict[str, float] = defaultdict(float)
    for row in journal_rows:
        realized = float(row.get('realized_pnl') or 0.0)
        realized_by_policy_bucket[row.get('policy_bucket') or 'unknown'] += realized
        tail_key = 'tail_penalty' if float(row.get('tail_penalty_score') or 0.0) > 0.0 else 'no_tail_penalty'
        realized_by_tail_regime[tail_key] += realized
        realized_by_reeval_action[row.get('reeval_action') or 'unknown'] += realized
    by_market = []
    for market_id in sorted(summary['counts_by_market_id']):
        market_rows = [row for row in journal_rows if row.get('market_id') == market_id]
        by_market.append(
            {
                'market_id': market_id,
                'journal_rows': len(market_rows),
                'buy_notional': sum(float(row.get('notional') or 0.0) for row in market_rows if row.get('side') == 'buy'),
                'sell_notional': sum(float(row.get('notional') or 0.0) for row in market_rows if row.get('side') == 'sell'),
                'realized_pnl': sum(float(row.get('realized_pnl') or 0.0) for row in market_rows),
                'first_trade_ts': market_rows[0].get('ts') if market_rows else None,
                'last_trade_ts': market_rows[-1].get('ts') if market_rows else None,
            }
        )
    return {
        'summary': summary,
        'by_market': by_market,
        'cumulative_realized_pnl': build_cumulative_realized_pnl_series(journal_rows),
        'realized_pnl_by_policy_bucket': dict(realized_by_policy_bucket),
        'realized_pnl_by_tail_regime': dict(realized_by_tail_regime),
        'realized_pnl_by_reeval_action': dict(realized_by_reeval_action),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
