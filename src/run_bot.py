"""Minimal runner demonstrating model fit and probability call.

Usage (quick):
  python -m src.run_bot

This script backfills recent 1m klines (Binance) and runs the model.
"""
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import asyncio
import json
import websockets

from .binance_feed import backfill_klines
from .polymarket_feed import get_quote_snapshot, classify_quote_snapshot
from . import execution
from .storage import ensure_db
from . import storage
from .polymarket_client import LIVE as POLY_LIVE
from .strategy_manager import build_trade_action, decide_and_execute
from .strategy_sizing import fractional_kelly
from .market_router import resolve_active_market_bundle, resolve_active_market_bundle_with_debug
from .time_policy import apply_schedule_overlay, build_time_policy, policy_schedule_mode, policy_schedule_name
from .probability_engine_factory import (
    PROBABILITY_ENGINE_ENV_FLAGS,
    build_probability_engine,
    get_default_probability_engine_name,
)
from .wallet_state import fetch_wallet_state
from .redeemer import settle_inventory_candidates
from .decision_overlay import apply_polarized_tail_overlay
from .growth_optimizer import (
    entry_growth_mode,
    evaluate_entry_shadow,
    expected_growth_shadow_enabled,
)
from .polarization_credibility import (
    classify_polarization_zone,
    compute_credibility_discount,
    evaluate_same_side_reentry_veto,
    polarization_credibility_mode,
    same_side_existing_exposure_stats,
)
from .reversal_evidence import compute_reversal_evidence
from .live_heartbeat import (
    RollingEventBuffer,
    format_heartbeat,
    format_console_action_line,
    format_console_status_line,
    get_log_mode,
    get_log_policy,
    is_debug_mode,
    should_print_console_event,
)
from .regime_detector import MICROSTRUCTURE_FIELD_NAMES, compute_microstructure_regime, microstructure_spectral_mode

STALE_ORDER_MAINTENANCE_SEC = int(os.getenv('STALE_ORDER_MAINTENANCE_SEC', '15'))
DECISION_LOG_PATH = Path(os.getenv('DECISION_LOG_PATH', 'decision_state.jsonl'))
SHADOW_PROBABILITY_ENGINES_ENV = 'SHADOW_PROBABILITY_ENGINES'
SHADOW_PROBABILITY_ENGINE_ENV_FLAGS = {
    engine_name: f'SHADOW_{env_name}'
    for engine_name, env_name in PROBABILITY_ENGINE_ENV_FLAGS.items()
}
def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).lower() in ('1', 'true', 'yes', 'on')


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_nested(mapping: dict, *keys: str):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


EXECUTION_ENABLED = _env_flag("EXECUTION_ENABLED", POLY_LIVE)
INVENTORY_SETTLEMENT_CADENCE = max(1, int(os.getenv('INVENTORY_SETTLEMENT_CADENCE', '1')))
BINANCE_WS_OPEN_TIMEOUT_SEC = max(1.0, float(os.getenv('BINANCE_WS_OPEN_TIMEOUT_SEC', '10')))
BINANCE_WS_RETRY_DELAY_SEC = max(1.0, float(os.getenv('BINANCE_WS_RETRY_DELAY_SEC', '5')))


def _microstructure_price_history_limit() -> int:
    try:
        window = max(2, int(os.getenv('MICROSTRUCTURE_SPECTRAL_WINDOW', '32')))
    except (TypeError, ValueError):
        window = 32
    try:
        min_obs = max(2, int(os.getenv('MICROSTRUCTURE_SPECTRAL_MIN_OBS', '24')))
    except (TypeError, ValueError):
        min_obs = 24
    return max(window, min_obs) + 1


def _microstructure_fields_from_state(state: Optional[dict]) -> dict:
    state = state or {}
    return {field: state.get(field) for field in MICROSTRUCTURE_FIELD_NAMES}


def get_live_console_mode() -> str:
    return get_log_mode()


def get_live_heartbeat_sec(console_mode: Optional[str] = None) -> int:
    policy = get_log_policy(console_mode)
    raw = os.getenv('LIVE_HEARTBEAT_SEC')
    if raw is None or not str(raw).strip():
        return max(1, int(policy.default_heartbeat_sec))
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(1, int(policy.default_heartbeat_sec))


def should_emit_console_heartbeat(last_ts: Optional[pd.Timestamp], now: pd.Timestamp, console_mode: Optional[str] = None) -> bool:
    if last_ts is None:
        return True
    interval_sec = get_live_heartbeat_sec(console_mode)
    return (now - last_ts).total_seconds() >= interval_sec


def get_console_inventory_summary() -> dict:
    conn = storage.sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM open_lots')
    open_lot_count = int(cur.fetchone()[0] or 0)
    conn.close()
    realized_summary = storage.get_realized_pnl_summary()
    resolved_pending_count = sum(1 for item in storage.get_position_snapshot() if float(item.get('resolved_redeemable_qty') or 0.0) > 0)
    return {
        'open_lot_count': open_lot_count,
        'resolved_pending_count': resolved_pending_count,
        'realized_pnl_total': realized_summary.get('realized_pnl_total', 0.0),
        'realized_pnl_by_entry_regime': storage.get_realized_pnl_by_entry_regime(),
    }


def print_console_event_line(message: str) -> None:
    print(message)


def emit_console_action(action: Optional[dict], *, console_mode: str) -> None:
    if not isinstance(action, dict):
        return
    tone = 'info'
    status = str(((action.get('resp') or {}).get('status')) or action.get('status') or '').lower()
    critical = status in {'error', 'failed', 'rejected'}
    if not should_print_console_event(mode=console_mode, tone=tone, critical=critical):
        return
    print_console_event_line(format_console_action_line(action, mode=console_mode))


def emit_console_status(label: str, *, console_mode: str, tone: str = 'info', **fields) -> None:
    critical = tone in {'error', 'warning'} or any(fields.get(key) is not None for key in ('reason', 'error_message', 'skip_reason', 'response_status'))
    if not should_print_console_event(mode=console_mode, tone=tone, critical=critical):
        return
    print_console_event_line(format_console_status_line(label, mode=console_mode, tone=tone, **fields))


def _is_live_loop_active(end_time: float) -> bool:
    return asyncio.get_event_loop().time() < end_time


async def consume_binance_klines(
    *,
    live_engine,
    shadow_engines=None,
    state_lock: asyncio.Lock,
    duration: int,
    console_mode: str,
) -> None:
    url = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
    end_time = asyncio.get_event_loop().time() + duration
    last_failure_reason = None
    while _is_live_loop_active(end_time):
        try:
            async with websockets.connect(url, open_timeout=BINANCE_WS_OPEN_TIMEOUT_SEC) as ws:
                if last_failure_reason is not None:
                    emit_console_status('feed', console_mode=console_mode, tone='success', status='reconnected')
                    last_failure_reason = None
                while _is_live_loop_active(end_time):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    data = json.loads(msg)
                    async with state_lock:
                        apply_completed_kline_to_engine(live_engine, data)
                    apply_completed_kline_to_engines(shadow_engines or [], data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = type(exc).__name__
            if str(exc):
                reason = f'{reason}: {exc}'
            if reason != last_failure_reason:
                emit_console_status('feed', console_mode=console_mode, tone='warning', status='disconnected', reason=reason)
                last_failure_reason = reason
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(BINANCE_WS_RETRY_DELAY_SEC, max(0.0, remaining)))


def format_startup_recovery_summary(report: list[dict]) -> str:
    total = len(report or [])
    recovered = 0
    canceled = 0
    not_found = 0
    still_unknown = 0
    for item in report or []:
        status = str(item.get('status') or '').lower()
        if status in {'submitted', 'open', 'partially_filled', 'filled'}:
            recovered += 1
        elif status in {'canceled', 'expired'}:
            canceled += 1
        elif status == 'not_found_on_venue':
            not_found += 1
        elif status == 'unknown':
            still_unknown += 1
    return (
        f'Startup order recovery | total={total} recovered={recovered} '
        f'canceled={canceled} not_found={not_found} still_unknown={still_unknown}'
    )


def _dust_winning_tag(market: Optional[dict], outcome_side: Optional[str]) -> str:
    status = str((market or {}).get('status') or '').lower()
    winning_outcome = (market or {}).get('winning_outcome')
    if status not in {'resolved', 'redeemed', 'archived'} or winning_outcome not in {'YES', 'NO'}:
        return 'unknown'
    return 'winning' if str(outcome_side or '').upper() == winning_outcome else 'losing'


def collect_startup_dust_review(checked_ts: Optional[str] = None, refresh_markets: bool = True) -> dict:
    ts = checked_ts or datetime.now(timezone.utc).isoformat()
    threshold = storage.get_dust_qty_threshold()
    order_candidates = storage.get_order_dust_candidates(threshold=threshold)
    inventory_candidates = storage.get_inventory_dust_candidates(threshold=threshold)
    dormant_orders = []
    for status in ('dust_ignored', 'dust_finalized'):
        dormant_orders.extend(storage.get_open_orders(statuses=[status]))
    dormant_inventory = storage.list_dormant_lots()
    market_ids = sorted(
        {item['market_id'] for item in order_candidates}
        | {item['market_id'] for item in inventory_candidates}
        | {item['market_id'] for item in dormant_orders}
        | {item['market_id'] for item in dormant_inventory}
    )
    refreshed_markets = {}
    for market_id in market_ids:
        market = storage.get_market(market_id)
        if refresh_markets and market is not None:
            try:
                market = storage.refresh_market_lifecycle(market_id, checked_ts=ts) or market
            except Exception:
                pass
        refreshed_markets[market_id] = market

    items = []
    for order in order_candidates:
        market = refreshed_markets.get(order['market_id']) or storage.get_market(order['market_id']) or {}
        items.append(
            {
                'item_type': 'order',
                'item_key': f"order:{order['id']}",
                'market_id': order['market_id'],
                'title': market.get('title') or market.get('slug') or order['market_id'],
                'outcome_side': order['outcome_side'],
                'qty': float(order['remaining_qty']),
                'linked_order_id': order['id'],
                'market_status': market.get('status'),
                'winning_tag': _dust_winning_tag(market, order['outcome_side']),
                'status': order['status'],
                'source_state': 'candidate',
            }
        )
    for order in dormant_orders:
        market = refreshed_markets.get(order['market_id']) or storage.get_market(order['market_id']) or {}
        items.append(
            {
                'item_type': 'order',
                'item_key': f"order:{order['id']}",
                'market_id': order['market_id'],
                'title': market.get('title') or market.get('slug') or order['market_id'],
                'outcome_side': order['outcome_side'],
                'qty': float(order['remaining_qty']),
                'linked_order_id': order['id'],
                'market_status': market.get('status'),
                'winning_tag': _dust_winning_tag(market, order['outcome_side']),
                'status': order['status'],
                'source_state': 'dormant',
            }
        )
    for lot in inventory_candidates:
        market = refreshed_markets.get(lot['market_id']) or storage.get_market(lot['market_id']) or {}
        items.append(
            {
                'item_type': 'inventory',
                'item_key': f"inventory:{lot['market_id']}:{lot['token_id']}:{lot['outcome_side']}",
                'market_id': lot['market_id'],
                'title': market.get('title') or market.get('slug') or lot['market_id'],
                'outcome_side': lot['outcome_side'],
                'qty': float(lot['net_qty']),
                'linked_order_id': lot.get('linked_order_id'),
                'market_status': market.get('status'),
                'winning_tag': _dust_winning_tag(market, lot['outcome_side']),
                'lot_ids': list(lot.get('lot_ids') or []),
                'source_state': 'candidate',
            }
        )
    for lot in dormant_inventory:
        market = refreshed_markets.get(lot['market_id']) or storage.get_market(lot['market_id']) or {}
        items.append(
            {
                'item_type': 'inventory',
                'item_key': f"dormant_inventory:{lot['id']}",
                'market_id': lot['market_id'],
                'title': market.get('title') or market.get('slug') or lot['market_id'],
                'outcome_side': lot['outcome_side'],
                'qty': float(lot['qty']),
                'linked_order_id': lot.get('linked_order_id'),
                'market_status': market.get('status'),
                'winning_tag': _dust_winning_tag(market, lot['outcome_side']),
                'dormant_lot_id': lot['id'],
                'status': lot['dormant_status'],
                'source_state': 'dormant',
            }
        )
    return {
        'threshold': threshold,
        'checked_ts': ts,
        'orders': order_candidates,
        'inventory': inventory_candidates,
        'items': items,
    }


def format_startup_dust_report(review: dict) -> str:
    items = list((review or {}).get('items') or [])
    threshold = float((review or {}).get('threshold') or storage.get_dust_qty_threshold())
    if not items:
        return f'Startup dust review | threshold={threshold:.4f} items=0'
    lines = [f'Startup dust review | threshold={threshold:.4f} items={len(items)}']
    for item in items:
        linked = item.get('linked_order_id')
        linked_text = '-' if linked is None else str(linked)
        lines.append(
            f"{item['item_type']} | market={item['market_id']} | title={item['title']} | side={item['outcome_side']} | "
            f"qty={float(item['qty']):.6f} | order_id={linked_text} | market_status={item.get('market_status') or 'unknown'} | "
            f"winning={item.get('winning_tag') or 'unknown'}"
        )
    return '\n'.join(lines)


def _dust_item_matches_target(item: dict, target) -> bool:
    if target is None:
        return False
    return str(target) in {
        str(item.get('item_key')),
        str(item.get('market_id')),
        str(item.get('linked_order_id')),
        str(item.get('dormant_lot_id')),
    }


def apply_startup_dust_action(
    review: dict,
    *,
    action: str,
    targets: Optional[list] = None,
    ts: Optional[str] = None,
) -> list[dict]:
    ts = ts or datetime.now(timezone.utc).isoformat()
    items = list((review or {}).get('items') or [])
    if action not in {'finalize_all', 'keep_all_dormant', 'restore_selected', 'finalize_selected'}:
        raise ValueError(f'unsupported dust action: {action}')
    selected = items
    if action in {'restore_selected', 'finalize_selected'}:
        selected = [item for item in items if any(_dust_item_matches_target(item, target) for target in (targets or []))]
    results = []
    target_status = 'dust_finalized' if action in {'finalize_all', 'finalize_selected'} else 'dust_ignored'
    for item in selected:
        if item['item_type'] == 'order':
            order_id = int(item['linked_order_id'])
            if action == 'restore_selected':
                updated = storage.restore_dust_order(order_id, ts=ts)
                results.append({'item_key': item['item_key'], 'action': action, 'status': updated['status']})
            else:
                updated = storage.move_order_to_dust_status(order_id, dust_status=target_status, reason='startup_dust_action', raw={'startup_action': action}, ts=ts)
                results.append({'item_key': item['item_key'], 'action': action, 'status': updated['status']})
            continue

        if action == 'restore_selected':
            restored_ids = []
            dormant_lot_ids = [item['dormant_lot_id']] if item.get('dormant_lot_id') is not None else []
            for dormant_lot_id in dormant_lot_ids:
                restored = storage.restore_dormant_lot(int(dormant_lot_id), restored_ts=ts)
                restored_ids.append(restored['id'])
            results.append({'item_key': item['item_key'], 'action': action, 'restored_ids': restored_ids})
            continue

        if item.get('dormant_lot_id') is not None:
            updated = storage.set_dormant_lot_status(int(item['dormant_lot_id']), target_status, updated_ts=ts)
            results.append({'item_key': item['item_key'], 'action': action, 'dormant_ids': [updated['id']], 'status': updated['dormant_status']})
        else:
            moved_ids = []
            for lot_id in item.get('lot_ids') or []:
                moved = storage.move_open_lot_to_dormant(
                    int(lot_id),
                    dormant_status=target_status,
                    dormant_reason='startup_dust_action',
                    linked_order_id=item.get('linked_order_id'),
                    updated_ts=ts,
                )
                moved_ids.append(moved['id'])
            results.append({'item_key': item['item_key'], 'action': action, 'dormant_ids': moved_ids, 'status': target_status})
    return results


def _iso_or_none(value):
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _is_quote_valid(value) -> bool:
    return value is not None and 0 < float(value) < 1


def _parse_ts(value):
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _market_from_bundle(bundle: Optional[dict]) -> dict:
    if not bundle:
        return {}
    return {
        'market_id': bundle.get('market_id'),
        'condition_id': bundle.get('condition_id'),
        'token_yes': bundle.get('token_yes'),
        'token_no': bundle.get('token_no'),
        'startDate': _parse_ts(bundle.get('start_time')),
        'endDate': _parse_ts(bundle.get('end_time')),
        'status': bundle.get('status'),
        'detection_source': bundle.get('detection_source'),
    }


def _empty_probability_state(reason: str, bundle: Optional[dict], now=None) -> dict:
    return {
        'blocked': True,
        'reason': reason,
        'timestamp': _iso_or_none(now or datetime.now(timezone.utc)),
        'series_id': (bundle or {}).get('series_id'),
        'market_id': (bundle or {}).get('market_id'),
        'spot_now': None,
        'strike_price': (bundle or {}).get('strike_price'),
        'tau_minutes': None,
        'p_yes': None,
        'p_no': None,
    }


def parse_shadow_probability_engine_names(raw: Optional[str] = None) -> list[str]:
    names = []
    if raw is not None:
        values = str(raw or '').split(',')
    else:
        values = []
        for engine_name, env_name in SHADOW_PROBABILITY_ENGINE_ENV_FLAGS.items():
            if _env_flag(env_name, False):
                values.append(engine_name)
        legacy = os.getenv(SHADOW_PROBABILITY_ENGINES_ENV, '')
        if str(legacy or '').strip():
            values.extend(str(legacy).split(','))
    for item in values:
        normalized = str(item).strip().lower()
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def build_shadow_probability_engines(
    engine_names: Optional[list[str]] = None,
    *,
    primary_engine_name: Optional[str] = None,
    fit_prices: Optional[pd.Series] = None,
    engine_kwargs: Optional[dict] = None,
) -> dict[str, object]:
    names = engine_names if engine_names is not None else parse_shadow_probability_engine_names()
    primary = None if primary_engine_name is None else str(primary_engine_name).strip().lower()
    engines: dict[str, object] = {}
    for engine_name in names:
        normalized = str(engine_name).strip().lower()
        if not normalized or normalized == primary or normalized in engines:
            continue
        engine = build_probability_engine(normalized, **(engine_kwargs or {}))
        if fit_prices is not None:
            engine.fit_history(fit_prices)
        engines[normalized] = engine
    return engines


def compute_market_probabilities(bundle, engine, now=None, n_sims=None, seed=None) -> dict:
    now = _parse_ts(now) or pd.Timestamp.now(tz='UTC')
    spot_now = engine.current_spot() if hasattr(engine, 'current_spot') else getattr(engine, 'last_price', None)
    if not bundle:
        return _empty_probability_state('missing_active_bundle', bundle, now=now)
    strike_price = bundle.get('strike_price')
    if strike_price is None:
        return _empty_probability_state('missing_strike_price', bundle, now=now)
    end_time = _parse_ts(bundle.get('end_time'))
    if end_time is None:
        return _empty_probability_state('missing_end_time', bundle, now=now)
    tau_minutes = int((end_time - now).total_seconds() // 60)
    if tau_minutes <= 0:
        return {
            **_empty_probability_state('market_expired', bundle, now=now),
            'tau_minutes': tau_minutes,
            'spot_now': spot_now,
        }
    try:
        if hasattr(engine, 'predict'):
            result = engine.predict(float(strike_price), tau_minutes=tau_minutes, n_sims=n_sims or 500, seed=seed)
        else:
            result = engine.simulate_probability(float(strike_price), tau_minutes=tau_minutes, n_sims=n_sims or 500, seed=seed)
    except Exception as exc:
        return {
            **_empty_probability_state('probability_compute_failed', bundle, now=now),
            'tau_minutes': tau_minutes,
            'spot_now': spot_now,
            'error': str(exc),
        }
    if result.get('failed'):
        return {
            **_empty_probability_state(result.get('reason') or 'probability_compute_failed', bundle, now=now),
            'tau_minutes': tau_minutes,
            'spot_now': spot_now,
            'raw_model_output': result,
        }
    p_yes = float(result['p_yes'] if result.get('p_yes') is not None else result['p_hat'])
    return {
        'blocked': False,
        'reason': None,
        'timestamp': _iso_or_none(now),
        'series_id': bundle.get('series_id'),
        'market_id': bundle.get('market_id'),
        'spot_now': spot_now,
        'strike_price': float(strike_price),
        'tau_minutes': tau_minutes,
        'p_yes': p_yes,
        'p_no': 1.0 - p_yes,
        'sigma_per_sqrt_min': result.get('sigma_per_sqrt_min'),
        'raw_model_output': result,
    }


def build_recent_price_history(engine, limit: int = 10) -> pd.Series:
    series = getattr(engine, 'prices', None)
    if isinstance(series, pd.Series):
        return series.dropna().astype(float).iloc[-max(1, int(limit)):]
    nested = getattr(getattr(engine, 'model', None), 'prices', None)
    if isinstance(nested, pd.Series):
        return nested.dropna().astype(float).iloc[-max(1, int(limit)):]
    return pd.Series(dtype=float)


def _probability_lineage(probability_state: Optional[dict]) -> dict:
    probability_state = probability_state or {}
    raw_model_output = probability_state.get('raw_model_output') or {}
    raw_p_yes = raw_model_output.get('raw_p_yes')
    if raw_p_yes is None:
        raw_p_yes = probability_state.get('p_yes')
    raw_p_no = raw_model_output.get('raw_p_no')
    if raw_p_no is None and raw_p_yes is not None:
        raw_p_no = 1.0 - float(raw_p_yes)
    calibrated_p_yes = raw_model_output.get('calibrated_p_yes')
    if calibrated_p_yes is None:
        calibrated_p_yes = probability_state.get('p_yes')
    calibrated_p_no = raw_model_output.get('calibrated_p_no')
    if calibrated_p_no is None and calibrated_p_yes is not None:
        calibrated_p_no = 1.0 - float(calibrated_p_yes)
    return {
        'raw_p_yes': raw_p_yes,
        'raw_p_no': raw_p_no,
        'calibrated_p_yes': calibrated_p_yes,
        'calibrated_p_no': calibrated_p_no,
    }


def _entry_reversal_evidence_by_side(
    *,
    price_history: Optional[pd.Series],
    strike_price: Optional[float],
    spot_now: Optional[float],
) -> dict:
    series = price_history if isinstance(price_history, pd.Series) else pd.Series(dtype=float)
    if series.empty:
        return {}
    evidence_by_side = {}
    for side in ('YES', 'NO'):
        try:
            evidence_by_side[side] = compute_reversal_evidence(
                series,
                side=side,
                strike_price=strike_price,
                spot_now=spot_now,
            )
        except Exception:
            evidence_by_side[side] = {
                'side': side,
                'passes_min_score': False,
                'score': 0,
                'score_ratio': 0.0,
                'reason': 'reversal_evidence_compute_failed',
            }
    return evidence_by_side


def apply_completed_kline_to_engine(engine, payload) -> bool:
    kline = payload.get('k') or {}
    if not kline or not kline.get('x'):
        return False
    price = float(kline['c'])
    ts = pd.to_datetime(int(kline.get('T')), unit='ms', utc=True)
    engine.observe_bar(price, ts=ts, finalized=True)
    return True


def apply_completed_kline_to_engines(engines, payload) -> bool:
    kline = payload.get('k') or {}
    if not kline or not kline.get('x'):
        return False
    price = float(kline['c'])
    ts = pd.to_datetime(int(kline.get('T')), unit='ms', utc=True)
    observed = False
    for engine in engines or []:
        try:
            engine.observe_bar(price, ts=ts, finalized=True)
            observed = True
        except Exception:
            continue
    return observed


def compute_effective_decision_trade_state(ctx: dict, decision_state: Optional[dict]) -> tuple[bool, str]:
    trading_allowed, disabled_reason = can_trade_context(ctx)
    if trading_allowed and decision_state is not None and not decision_state.get('trade_allowed'):
        trading_allowed = False
        disabled_reason = decision_state.get('reason') or 'decision_state_blocked'
    return bool(trading_allowed), str(disabled_reason or 'ok')


def _entry_side_from_decision(decision_state: Optional[dict], *, trade_allowed: bool) -> Optional[str]:
    if not trade_allowed or not isinstance(decision_state, dict):
        return None
    action = str(decision_state.get('action') or '').lower()
    if action == 'buy_yes':
        return 'YES'
    if action == 'buy_no':
        return 'NO'
    return None


def _shadow_probability_model_payload(
    *,
    engine_name: str,
    decision_state: Optional[dict],
    trade_allowed: bool,
    disabled_reason: Optional[str],
    error: Optional[str] = None,
) -> dict:
    state = decision_state or {}
    policy = state.get('policy') or {}
    if error is not None:
        return {
            'engine_name': engine_name,
            'error': error,
            'p_yes': None,
            'p_no': None,
            'q_yes': None,
            'q_no': None,
            'edge_yes': None,
            'edge_no': None,
            'chosen_side': None,
            'chosen_action': None,
            'trade_allowed': False,
            'disabled_reason': 'shadow_engine_failed',
            'policy_bucket': None,
            'polarization_zone': None,
            'regime_label': None,
            'tail_penalty_score': None,
            'tail_hard_block': None,
            'same_side_existing_qty': None,
            'same_side_existing_filled_entry_count': None,
            'agrees_with_live_entry': None,
            'would_veto_live_entry': None,
            'would_flip_live_side': None,
            'shadow_only_entry': None,
            'live_entry_side': None,
            'shadow_entry_side': None,
        }
    return {
        'engine_name': engine_name,
        'error': None,
        'p_yes': state.get('p_yes'),
        'p_no': state.get('p_no'),
        'q_yes': state.get('q_yes'),
        'q_no': state.get('q_no'),
        'edge_yes': state.get('edge_yes'),
        'edge_no': state.get('edge_no'),
        'chosen_side': state.get('chosen_side'),
        'chosen_action': state.get('action') if trade_allowed else None,
        'trade_allowed': trade_allowed,
        'disabled_reason': None if trade_allowed else (disabled_reason or state.get('reason')),
        'policy_bucket': policy.get('policy_bucket'),
        'polarization_zone': state.get('polarization_zone'),
        'regime_label': _get_nested(state, 'regime_state', 'regime_label'),
        'tail_penalty_score': state.get('tail_penalty_score'),
        'tail_hard_block': state.get('tail_hard_block'),
        'same_side_existing_qty': state.get('same_side_existing_qty'),
        'same_side_existing_filled_entry_count': state.get('same_side_existing_filled_entry_count'),
    }


def compare_shadow_decision_to_live(
    *,
    live_decision_state: Optional[dict],
    live_trade_allowed: bool,
    shadow_decision_state: Optional[dict],
    shadow_trade_allowed: bool,
) -> dict:
    # Deterministic definitions:
    # live entry exists when live chosen_action is buy_yes/buy_no and live trade_allowed is true.
    # agrees_with_live_entry: live entry exists and shadow also allows an entry on the same side.
    # would_veto_live_entry: live entry exists and shadow does not allow any entry on that cycle.
    # would_flip_live_side: live entry exists and shadow would enter the opposite side.
    # shadow_only_entry: live does not enter but shadow would enter.
    live_entry_side = _entry_side_from_decision(live_decision_state, trade_allowed=live_trade_allowed)
    shadow_entry_side = _entry_side_from_decision(shadow_decision_state, trade_allowed=shadow_trade_allowed)
    return {
        'agrees_with_live_entry': live_entry_side is not None and shadow_entry_side == live_entry_side,
        'would_veto_live_entry': live_entry_side is not None and shadow_entry_side is None,
        'would_flip_live_side': (
            live_entry_side is not None
            and shadow_entry_side is not None
            and shadow_entry_side != live_entry_side
        ),
        'shadow_only_entry': live_entry_side is None and shadow_entry_side is not None,
        'live_entry_side': live_entry_side,
        'shadow_entry_side': shadow_entry_side,
    }


def evaluate_shadow_probability_models(
    *,
    shadow_engines: dict[str, object],
    bundle: Optional[dict],
    now,
    n_sims: Optional[int],
    wallet_state: Optional[dict],
    trade_context: dict,
    entry_context: Optional[dict],
    microstructure_state: Optional[dict],
    live_decision_state: Optional[dict],
    live_trade_allowed: bool,
) -> dict[str, dict]:
    if not shadow_engines or not bundle:
        return {}
    results: dict[str, dict] = {}
    for engine_name, engine in shadow_engines.items():
        try:
            probability_state = compute_market_probabilities(bundle, engine, now=now, n_sims=n_sims)
            if probability_state.get('error'):
                results[engine_name] = _shadow_probability_model_payload(
                    engine_name=engine_name,
                    decision_state=None,
                    trade_allowed=False,
                    disabled_reason='shadow_engine_failed',
                    error=str(probability_state.get('error')),
                )
                continue
            decision_state = build_market_decision_state(
                bundle,
                probability_state,
                wallet_state=wallet_state,
                entry_context=entry_context,
                microstructure_state=microstructure_state,
            )
            shadow_ctx = dict(trade_context)
            shadow_ctx['probability_state'] = probability_state
            shadow_ctx['decision_state'] = decision_state
            shadow_ctx['policy'] = decision_state.get('policy') or {}
            shadow_trade_allowed, shadow_disabled_reason = compute_effective_decision_trade_state(shadow_ctx, decision_state)
            payload = _shadow_probability_model_payload(
                engine_name=engine_name,
                decision_state=decision_state,
                trade_allowed=shadow_trade_allowed,
                disabled_reason=shadow_disabled_reason,
            )
            payload.update(
                compare_shadow_decision_to_live(
                    live_decision_state=live_decision_state,
                    live_trade_allowed=live_trade_allowed,
                    shadow_decision_state=decision_state,
                    shadow_trade_allowed=shadow_trade_allowed,
                )
            )
            results[engine_name] = payload
        except Exception as exc:
            results[engine_name] = _shadow_probability_model_payload(
                engine_name=engine_name,
                decision_state=None,
                trade_allowed=False,
                disabled_reason='shadow_engine_failed',
                error=f'{type(exc).__name__}: {exc}',
            )
    return results


def build_market_decision_state(
    bundle,
    probability_state,
    wallet_state: Optional[dict] = None,
    entry_context: Optional[dict] = None,
    microstructure_state: Optional[dict] = None,
) -> dict:
    yes_quote = (bundle or {}).get('yes_quote') or {}
    no_quote = (bundle or {}).get('no_quote') or {}
    q_yes = yes_quote.get('mid')
    q_no = no_quote.get('mid')
    lineage = _probability_lineage(probability_state)
    raw_p_yes = lineage.get('raw_p_yes')
    raw_p_no = lineage.get('raw_p_no')
    calibrated_p_yes = lineage.get('calibrated_p_yes')
    calibrated_p_no = lineage.get('calibrated_p_no')
    state = {
        'timestamp': probability_state.get('timestamp'),
        'ts': probability_state.get('timestamp'),
        'series_id': (bundle or {}).get('series_id'),
        'market_id': (bundle or {}).get('market_id'),
        'token_yes': (bundle or {}).get('token_yes'),
        'token_no': (bundle or {}).get('token_no'),
        'strike_price': probability_state.get('strike_price'),
        'spot_now': probability_state.get('spot_now'),
        'tau_minutes': probability_state.get('tau_minutes'),
        'p_yes': calibrated_p_yes,
        'p_no': calibrated_p_no,
        'raw_p_yes': raw_p_yes,
        'raw_p_no': raw_p_no,
        'calibrated_p_yes': calibrated_p_yes,
        'calibrated_p_no': calibrated_p_no,
        'sigma_per_sqrt_min': probability_state.get('sigma_per_sqrt_min'),
        'q_yes': q_yes,
        'q_no': q_no,
        'fair_yes': calibrated_p_yes,
        'fair_no': calibrated_p_no,
        'edge_yes': None,
        'edge_no': None,
        'raw_edge_yes': None,
        'raw_edge_no': None,
        'adjusted_edge_yes': None,
        'adjusted_edge_no': None,
        'credibility_weight_yes': 1.0,
        'credibility_weight_no': 1.0,
        'discounted_p_yes': calibrated_p_yes,
        'discounted_p_no': calibrated_p_no,
        'discounted_edge_yes': None,
        'discounted_edge_no': None,
        'admission_edge_yes': None,
        'admission_edge_no': None,
        'admission_probability_source': 'adjusted_probability',
        'edge_credibility_reason': None,
        'polarization_zone': None,
        'polarization_zone_yes': None,
        'polarization_zone_no': None,
        'chosen_side_quote': None,
        'minority_side_quote': None,
        'polarization_credibility_mode': polarization_credibility_mode(),
        'credibility_shadow_trade_allowed': None,
        'credibility_shadow_action': None,
        'credibility_shadow_reason': None,
        'credibility_block_reason': None,
        'same_side_reentry_shadow_blocked': False,
        'same_side_reentry_live_blocked': False,
        'same_side_reentry_reason': None,
        'reversal_evidence_by_side': (entry_context or {}).get('reversal_evidence_by_side') or probability_state.get('reversal_evidence_by_side') or {},
        'trade_allowed': False,
        'action': None,
        'reason': probability_state.get('reason'),
        'policy': None,
        'tail_overlay': None,
        'tail_guard_enabled': False,
        'tail_penalty_score': 0.0,
        'tail_q_tail': None,
        'tail_z_signed': None,
        'tail_z_abs': None,
        'tail_hard_block': False,
        'favored_side': None,
        'contrarian_side': None,
        'q_tail': None,
        'tail_side': None,
        'chosen_side': None,
        'polarized_tail_penalty': 1.0,
        'polarized_tail_blocked': False,
        'z_distance_to_strike': None,
        'expected_log_growth_entry': None,
        'expected_log_growth_entry_conservative': None,
        'expected_log_growth_entry_conservative_old': None,
        'expected_log_growth_entry_conservative_discounted': None,
        'expected_log_growth_pass_shadow': None,
        'expected_log_growth_reason_shadow': None,
        'growth_gate_pass_shadow': None,
        'growth_gate_reason_shadow': None,
        'expected_terminal_wealth_if_yes': None,
        'expected_terminal_wealth_if_no': None,
        'entry_growth_eval_mode': 'off',
        'entry_growth_candidate_side': None,
        'entry_growth_qty': None,
        'entry_growth_trade_notional': None,
        'entry_growth_probability_conservative': None,
        'entry_growth_fragility_score': None,
        'entry_growth_candidates': [],
        'position_reeval_enabled': False,
        'position_reeval_action': 'hold',
        'position_reeval_reason': 'position_reeval_hold_default',
        'position_reeval_hold_ev_per_share': None,
        'position_reeval_candidate_entry_ev_per_share': None,
        'position_reeval_flip_advantage_per_share': None,
        'position_reeval_reversal_score': None,
        'position_reeval_reversal_passed': None,
        'position_reeval_shadow_best_action': None,
        'position_reeval_shadow_best_delta_qty': None,
        'position_reeval_shadow_best_growth_gain': None,
        'position_reeval_shadow_best_executable': None,
        'position_reeval_shadow_keep_current_position': None,
        'position_reeval': None,
        'action_origin': 'first_entry',
        'candidate_stage': 'pre_candidate',
        'terminal_reason': probability_state.get('reason'),
        'terminal_reason_family': _reason_family(probability_state.get('reason')),
        'blocked_by': None,
        'blocked_by_stage': None,
        'first_blocking_guard': None,
        'all_triggered_blockers': [],
    }
    state.update(_microstructure_fields_from_state(microstructure_state or compute_microstructure_regime(None)))
    state['policy'] = apply_schedule_overlay(
        build_time_policy(state),
        mode=policy_schedule_mode(),
        schedule_name=policy_schedule_name(),
    )
    _ensure_blocker_telemetry(state, action_origin='first_entry')
    if probability_state.get('blocked'):
        _finalize_non_trading_state(
            state,
            blocker_name=str(probability_state.get('reason') or 'probability_blocked'),
            reason=str(probability_state.get('reason') or 'probability_blocked'),
            stage='pre_candidate',
            inputs={
                'blocked': True,
                'probability_state_reason': probability_state.get('reason'),
                'strike_price': state.get('strike_price'),
                'spot_now': state.get('spot_now'),
                'tau_minutes': state.get('tau_minutes'),
            },
        )
        return state
    if q_yes is None:
        state['reason'] = 'missing_yes_quote'
        _finalize_non_trading_state(
            state,
            blocker_name='quote_invalid',
            reason='missing_yes_quote',
            stage='pre_candidate',
            inputs={'q_yes': q_yes, 'q_no': q_no},
        )
        return state
    if q_no is None:
        state['reason'] = 'missing_no_quote'
        _finalize_non_trading_state(
            state,
            blocker_name='quote_invalid',
            reason='missing_no_quote',
            stage='pre_candidate',
            inputs={'q_yes': q_yes, 'q_no': q_no},
        )
        return state
    policy = state['policy'] or {}
    raw_edge_yes = None if _safe_float(raw_p_yes) is None or _safe_float(q_yes) is None else float(raw_p_yes) - float(q_yes)
    raw_edge_no = None if _safe_float(raw_p_no) is None or _safe_float(q_no) is None else float(raw_p_no) - float(q_no)
    overlay = apply_polarized_tail_overlay(bundle, probability_state, policy=policy)
    adj_p_yes = overlay.get('adj_p_yes') if overlay.get('adj_p_yes') is not None else calibrated_p_yes
    adj_p_no = overlay.get('adj_p_no') if overlay.get('adj_p_no') is not None else calibrated_p_no
    adj_edge_yes = overlay.get('adj_edge_yes') if overlay.get('adj_edge_yes') is not None else raw_edge_yes
    adj_edge_no = overlay.get('adj_edge_no') if overlay.get('adj_edge_no') is not None else raw_edge_no
    same_side_yes = same_side_existing_exposure_stats(state.get('market_id'), 'YES')
    same_side_no = same_side_existing_exposure_stats(state.get('market_id'), 'NO')
    regime_state = state.get('regime_state') or {}
    credibility_yes = compute_credibility_discount(
        outcome_side='YES',
        chosen_side_quote=q_yes,
        raw_probability=raw_p_yes,
        adjusted_probability=adj_p_yes,
        same_side_stats=same_side_yes,
        reversal_evidence=(state.get('reversal_evidence_by_side') or {}).get('YES'),
        regime_state=regime_state,
        tail_penalty_score=overlay.get('tail_penalty_score'),
        tail_hard_block=bool(overlay.get('tail_hard_block')),
    )
    credibility_no = compute_credibility_discount(
        outcome_side='NO',
        chosen_side_quote=q_no,
        raw_probability=raw_p_no,
        adjusted_probability=adj_p_no,
        same_side_stats=same_side_no,
        reversal_evidence=(state.get('reversal_evidence_by_side') or {}).get('NO'),
        regime_state=regime_state,
        tail_penalty_score=overlay.get('tail_penalty_score'),
        tail_hard_block=bool(overlay.get('tail_hard_block')),
    )
    discounted_p_yes = credibility_yes.get('discounted_probability')
    discounted_p_no = credibility_no.get('discounted_probability')
    discounted_edge_yes = credibility_yes.get('discounted_edge')
    discounted_edge_no = credibility_no.get('discounted_edge')
    credibility_mode = polarization_credibility_mode()
    admission_edge_yes = adj_edge_yes if credibility_mode == 'off' else discounted_edge_yes
    admission_edge_no = adj_edge_no if credibility_mode == 'off' else discounted_edge_no
    state.update({
        'p_yes': adj_p_yes,
        'p_no': adj_p_no,
        'fair_yes': adj_p_yes,
        'fair_no': adj_p_no,
        'edge_yes': adj_edge_yes,
        'edge_no': adj_edge_no,
        'raw_edge_yes': raw_edge_yes,
        'raw_edge_no': raw_edge_no,
        'adjusted_edge_yes': adj_edge_yes,
        'adjusted_edge_no': adj_edge_no,
        'credibility_weight_yes': credibility_yes.get('credibility_weight'),
        'credibility_weight_no': credibility_no.get('credibility_weight'),
        'discounted_p_yes': discounted_p_yes,
        'discounted_p_no': discounted_p_no,
        'discounted_edge_yes': discounted_edge_yes,
        'discounted_edge_no': discounted_edge_no,
        'admission_edge_yes': admission_edge_yes,
        'admission_edge_no': admission_edge_no,
        'admission_probability_source': 'adjusted_probability' if credibility_mode == 'off' else 'discounted_probability',
        'polarization_zone_yes': credibility_yes.get('polarization_zone'),
        'polarization_zone_no': credibility_no.get('polarization_zone'),
        'tail_overlay': overlay,
        'tail_guard_enabled': bool(overlay.get('enabled')),
        'tail_penalty_score': float(overlay.get('tail_penalty_score') or 0.0),
        'tail_q_tail': overlay.get('q_tail'),
        'tail_z_signed': overlay.get('z_signed'),
        'tail_z_abs': overlay.get('z_abs'),
        'tail_hard_block': bool(overlay.get('tail_hard_block')),
        'favored_side': overlay.get('favored_side'),
        'contrarian_side': overlay.get('contrarian_side'),
        'q_tail': overlay.get('q_tail'),
        'tail_side': overlay.get('contrarian_side'),
        'polarized_tail_penalty': max(0.0, 1.0 - float(overlay.get('tail_penalty_score') or 0.0)),
        'polarized_tail_blocked': bool(overlay.get('tail_hard_block')),
        'z_distance_to_strike': overlay.get('z_abs'),
    })
    _set_candidate_stage(state, 'candidate_built', candidate_available=True)
    _capture_candidate_snapshot(state)
    if not policy.get('allow_new_entries', True):
        state['reason'] = 'policy_blocks_new_entries'
        _finalize_non_trading_state(
            state,
            blocker_name='policy_blocks_new_entries',
            reason='policy_blocks_new_entries',
            stage='post_candidate_blocked',
            inputs={'policy_bucket': policy.get('policy_bucket'), 'allow_new_entries': policy.get('allow_new_entries')},
        )
        return state

    edge_threshold_yes = float(policy.get('edge_threshold_yes', state['edge_yes'] if state['edge_yes'] is not None else 0.0))
    edge_threshold_no = float(policy.get('edge_threshold_no', state['edge_no'] if state['edge_no'] is not None else 0.0))
    eligible_yes = adj_edge_yes is not None and adj_edge_yes >= edge_threshold_yes
    eligible_no = adj_edge_no is not None and adj_edge_no >= edge_threshold_no
    credibility_eligible_yes = (
        admission_edge_yes is not None
        and admission_edge_yes >= edge_threshold_yes
        and not bool(credibility_yes.get('hard_block'))
    )
    credibility_eligible_no = (
        admission_edge_no is not None
        and admission_edge_no >= edge_threshold_no
        and not bool(credibility_no.get('hard_block'))
    )

    choice = None
    credibility_choice = None
    growth_candidates = []
    if eligible_yes or eligible_no:
        candidates = []
        if eligible_yes:
            candidates.append({'side': 'buy_yes', 'outcome_side': 'YES', 'edge': adj_edge_yes})
        if eligible_no:
            candidates.append({'side': 'buy_no', 'outcome_side': 'NO', 'edge': adj_edge_no})
        candidates.sort(key=lambda item: item['edge'], reverse=True)
        choice = candidates[0]
    if credibility_eligible_yes or credibility_eligible_no:
        credibility_candidates = []
        if credibility_eligible_yes:
            credibility_candidates.append({'side': 'buy_yes', 'outcome_side': 'YES', 'edge': admission_edge_yes})
        if credibility_eligible_no:
            credibility_candidates.append({'side': 'buy_no', 'outcome_side': 'NO', 'edge': admission_edge_no})
        credibility_candidates.sort(key=lambda item: item['edge'], reverse=True)
        credibility_choice = credibility_candidates[0]
    if eligible_yes or eligible_no:
        if expected_growth_shadow_enabled():
            effective_bankroll = (wallet_state or {}).get('effective_bankroll')
            if effective_bankroll is None:
                effective_bankroll = _env_float('BOT_BANKROLL', 1000.0)
            free_bankroll = (wallet_state or {}).get('free_usdc')
            if free_bankroll is None:
                free_bankroll = effective_bankroll
            kelly_k = _env_float('KELLY_K', 0.1)
            per_trade_cap_pct = _env_float('PER_TRADE_CAP_PCT', 0.01)
            kelly_multiplier = float(policy.get('kelly_multiplier', 1.0))
            max_trade_notional_multiplier = float(policy.get('max_trade_notional_multiplier', 1.0))
            for candidate in candidates:
                outcome_side = candidate['outcome_side']
                probability = adj_p_yes if outcome_side == 'YES' else adj_p_no
                quote = q_yes if outcome_side == 'YES' else q_no
                kelly_fraction = 0.0 if probability is None or quote is None else fractional_kelly(float(probability), float(quote), k=kelly_k)
                growth_eval = evaluate_entry_shadow(
                    decision_state=state,
                    outcome_side=outcome_side,
                    effective_bankroll=effective_bankroll,
                    free_bankroll=free_bankroll,
                    kelly_fraction=kelly_fraction,
                    kelly_multiplier=kelly_multiplier,
                    max_trade_notional_multiplier=max_trade_notional_multiplier,
                    per_trade_cap_pct=per_trade_cap_pct,
                    conservative_probability_override=discounted_p_yes if outcome_side == 'YES' else discounted_p_no,
                    polarization_zone_override=credibility_yes.get('polarization_zone') if outcome_side == 'YES' else credibility_no.get('polarization_zone'),
                )
                growth_candidates.append({
                    'side': candidate['side'],
                    'outcome_side': outcome_side,
                    'edge': candidate['edge'],
                    **growth_eval,
                })
            state['entry_growth_candidates'] = growth_candidates

    state['chosen_side'] = None if choice is None else choice['outcome_side']
    state['credibility_shadow_trade_allowed'] = credibility_choice is not None
    state['credibility_shadow_action'] = None if credibility_choice is None else credibility_choice.get('side')
    if credibility_choice is None:
        if credibility_yes.get('hard_block') or credibility_no.get('hard_block'):
            state['credibility_shadow_reason'] = 'veto_polarization_hard_block'
        else:
            state['credibility_shadow_reason'] = 'no_admission_edge_above_threshold'
    else:
        state['credibility_shadow_reason'] = 'ok'
    if choice is None:
        state['entry_growth_eval_mode'] = 'shadow' if expected_growth_shadow_enabled() else 'off'
        state['trade_allowed'] = False
        state['action'] = None
        state['reason'] = 'no_edge_above_threshold'
        _capture_candidate_snapshot(state, choice=credibility_choice)
        _record_blocker(
            state,
            blocker_name='no_edge_above_threshold',
            evaluated=True,
            mode='live',
            would_block=True,
            blocked=True,
            reason='no_edge_above_threshold',
            inputs={
                'eligible_yes': eligible_yes,
                'eligible_no': eligible_no,
                'edge_yes': adj_edge_yes,
                'edge_no': adj_edge_no,
                'edge_threshold_yes': edge_threshold_yes,
                'edge_threshold_no': edge_threshold_no,
            },
            stage='candidate_built',
            terminal=True,
        )
        return state

    if growth_candidates:
        selected_growth = next((item for item in growth_candidates if item.get('outcome_side') == choice['outcome_side']), growth_candidates[0])
        state.update({
            'expected_log_growth_entry': selected_growth.get('expected_log_growth_entry'),
            'expected_log_growth_entry_conservative': selected_growth.get('expected_log_growth_entry_conservative'),
            'expected_log_growth_entry_conservative_old': selected_growth.get('expected_log_growth_entry_conservative_old'),
            'expected_log_growth_entry_conservative_discounted': selected_growth.get('expected_log_growth_entry_conservative_discounted'),
            'expected_log_growth_pass_shadow': selected_growth.get('expected_log_growth_pass_shadow'),
            'expected_log_growth_reason_shadow': selected_growth.get('expected_log_growth_reason_shadow'),
            'growth_gate_pass_shadow': selected_growth.get('growth_gate_pass_shadow'),
            'growth_gate_reason_shadow': selected_growth.get('growth_gate_reason_shadow'),
            'expected_terminal_wealth_if_yes': selected_growth.get('expected_terminal_wealth_if_yes'),
            'expected_terminal_wealth_if_no': selected_growth.get('expected_terminal_wealth_if_no'),
            'entry_growth_eval_mode': selected_growth.get('entry_growth_eval_mode'),
            'entry_growth_candidate_side': selected_growth.get('entry_growth_candidate_side'),
            'entry_growth_qty': selected_growth.get('entry_growth_qty'),
            'entry_growth_trade_notional': selected_growth.get('entry_growth_trade_notional'),
            'entry_growth_probability_conservative': selected_growth.get('entry_growth_probability_conservative'),
            'entry_growth_fragility_score': selected_growth.get('entry_growth_fragility_score'),
        })
    else:
        state['entry_growth_eval_mode'] = 'shadow' if expected_growth_shadow_enabled() else 'off'

    current_growth_mode = entry_growth_mode()
    if current_growth_mode == 'live' and growth_candidates:
        selected = next((item for item in growth_candidates if item.get('outcome_side') == choice['outcome_side']), growth_candidates[0])
        if not selected.get('growth_gate_pass_shadow'):
            state['trade_allowed'] = False
            state['action'] = None
            state['reason'] = 'entry_growth_optimizer_veto'
            state['entry_growth_eval_mode'] = 'live'
            _finalize_non_trading_state(
                state,
                blocker_name='entry_growth_optimizer_veto',
                reason='entry_growth_optimizer_veto',
                stage='post_candidate_blocked',
                inputs={
                    'chosen_side': choice.get('outcome_side'),
                    'growth_gate_pass_shadow': selected.get('growth_gate_pass_shadow'),
                    'growth_gate_reason_shadow': selected.get('growth_gate_reason_shadow'),
                    'expected_log_growth_entry': selected.get('expected_log_growth_entry'),
                    'expected_log_growth_entry_conservative': selected.get('expected_log_growth_entry_conservative'),
                },
                choice=choice,
            )
            return state

    if choice['outcome_side'] == state.get('contrarian_side') and state.get('tail_hard_block'):
        state['trade_allowed'] = False
        state['action'] = None
        state['reason'] = 'tail_contrarian_hard_block'
        _finalize_non_trading_state(
            state,
            blocker_name='veto_polarization_hard_block',
            reason='tail_contrarian_hard_block',
            stage='post_candidate_blocked',
            inputs={
                'chosen_side': choice.get('outcome_side'),
                'contrarian_side': state.get('contrarian_side'),
                'tail_hard_block': state.get('tail_hard_block'),
                'tail_penalty_score': state.get('tail_penalty_score'),
            },
            choice=choice,
        )
        return state

    effective_choice = credibility_choice if credibility_mode == 'live' and credibility_choice is not None else choice
    chosen_outcome_side = effective_choice['outcome_side']
    chosen_quote = q_yes if chosen_outcome_side == 'YES' else q_no
    chosen_credibility = credibility_yes if chosen_outcome_side == 'YES' else credibility_no
    chosen_same_side_stats = same_side_yes if chosen_outcome_side == 'YES' else same_side_no
    state['chosen_side_quote'] = chosen_quote
    state['minority_side_quote'] = min(q_yes, q_no)
    state['polarization_zone'] = classify_polarization_zone(chosen_quote)
    state['edge_credibility_reason'] = chosen_credibility.get('reason')
    state['same_side_existing_qty'] = chosen_same_side_stats.get('same_side_existing_qty')
    state['same_side_existing_filled_entry_count'] = chosen_same_side_stats.get('same_side_existing_filled_entry_count')

    same_side_veto = evaluate_same_side_reentry_veto(
        market_id=state.get('market_id'),
        outcome_side=chosen_outcome_side,
        chosen_side_quote=chosen_quote,
        same_side_stats=chosen_same_side_stats,
    )
    state['same_side_reentry_shadow_blocked'] = bool(same_side_veto.get('would_block')) and same_side_veto.get('mode') == 'shadow'
    state['same_side_reentry_live_blocked'] = bool(same_side_veto.get('blocked'))
    state['same_side_reentry_reason'] = same_side_veto.get('reason')
    _capture_candidate_snapshot(state, choice=effective_choice)
    _record_blocker(
        state,
        blocker_name=str(same_side_veto.get('reason') or 'same_side_reentry_guard'),
        evaluated=True,
        mode=same_side_veto.get('mode'),
        would_block=bool(same_side_veto.get('would_block')),
        blocked=bool(same_side_veto.get('blocked')),
        reason=same_side_veto.get('reason'),
        inputs={
            'chosen_side': chosen_outcome_side,
            'chosen_side_quote': chosen_quote,
            'same_side_existing_qty': chosen_same_side_stats.get('same_side_existing_qty'),
            'same_side_existing_filled_entry_count': chosen_same_side_stats.get('same_side_existing_filled_entry_count'),
        },
    )
    _record_blocker(
        state,
        blocker_name='veto_polarization_hard_block',
        evaluated=True,
        mode=credibility_mode,
        would_block=bool(chosen_credibility.get('hard_block')),
        blocked=bool(credibility_mode == 'live' and chosen_credibility.get('hard_block')),
        reason='veto_polarization_hard_block' if chosen_credibility.get('hard_block') else None,
        inputs={
            'chosen_side': chosen_outcome_side,
            'chosen_side_quote': chosen_quote,
            'polarization_zone': chosen_credibility.get('polarization_zone'),
            'credibility_reason': chosen_credibility.get('reason'),
        },
    )

    if credibility_mode == 'live':
        if chosen_credibility.get('hard_block'):
            state['trade_allowed'] = False
            state['action'] = None
            state['reason'] = 'veto_polarization_hard_block'
            state['credibility_block_reason'] = state['reason']
            _finalize_non_trading_state(
                state,
                blocker_name='veto_polarization_hard_block',
                reason='veto_polarization_hard_block',
                stage='post_candidate_blocked',
                inputs={
                    'chosen_side': chosen_outcome_side,
                    'chosen_side_quote': chosen_quote,
                    'polarization_zone': chosen_credibility.get('polarization_zone'),
                    'credibility_reason': chosen_credibility.get('reason'),
                },
                choice=effective_choice,
            )
            return state
        if same_side_veto.get('blocked'):
            state['trade_allowed'] = False
            state['action'] = None
            state['reason'] = same_side_veto.get('reason')
            state['credibility_block_reason'] = state['reason']
            _finalize_non_trading_state(
                state,
                blocker_name=str(same_side_veto.get('reason') or 'same_side_reentry_guard'),
                reason=str(same_side_veto.get('reason') or 'same_side_reentry_guard'),
                stage='post_candidate_blocked',
                inputs={
                    'chosen_side': chosen_outcome_side,
                    'chosen_side_quote': chosen_quote,
                    'same_side_existing_qty': chosen_same_side_stats.get('same_side_existing_qty'),
                    'same_side_existing_filled_entry_count': chosen_same_side_stats.get('same_side_existing_filled_entry_count'),
                },
                choice=effective_choice,
            )
            return state
        if credibility_choice is None:
            state['trade_allowed'] = False
            state['action'] = None
            state['reason'] = state.get('credibility_shadow_reason') or 'no_admission_edge_above_threshold'
            state['credibility_block_reason'] = state['reason']
            _finalize_non_trading_state(
                state,
                blocker_name='no_admission_edge_above_threshold',
                reason=state['reason'],
                stage='post_candidate_blocked',
                inputs={
                    'admission_edge_yes': admission_edge_yes,
                    'admission_edge_no': admission_edge_no,
                    'edge_threshold_yes': edge_threshold_yes,
                    'edge_threshold_no': edge_threshold_no,
                    'credibility_hard_block_yes': bool(credibility_yes.get('hard_block')),
                    'credibility_hard_block_no': bool(credibility_no.get('hard_block')),
                },
                choice=choice,
            )
            return state
        state['trade_allowed'] = True
        state['action'] = effective_choice['side']
        state['chosen_side'] = effective_choice['outcome_side']
        state['reason'] = 'ok'
        _set_candidate_stage(state, 'candidate_built', candidate_available=True)
        _capture_candidate_snapshot(state, choice=effective_choice)
        return state

    if credibility_mode == 'shadow' and same_side_veto.get('would_block'):
        state['credibility_block_reason'] = same_side_veto.get('reason')
        state['credibility_shadow_reason'] = same_side_veto.get('reason')
        state['credibility_shadow_trade_allowed'] = False
        state['credibility_shadow_action'] = None
    elif credibility_mode == 'shadow' and chosen_credibility.get('hard_block'):
        state['credibility_block_reason'] = 'veto_polarization_hard_block'
        state['credibility_shadow_reason'] = 'veto_polarization_hard_block'
        state['credibility_shadow_trade_allowed'] = False
        state['credibility_shadow_action'] = None

    state['trade_allowed'] = True
    state['action'] = choice['side']
    state['reason'] = 'ok'
    _set_candidate_stage(state, 'candidate_built', candidate_available=True)
    _capture_candidate_snapshot(state, choice=choice)
    return state


def _policy_quote_reason(snapshot: dict, policy: Optional[dict]) -> Optional[str]:
    if not policy:
        return None
    if snapshot.get('fetch_failed'):
        return 'quote_fetch_failed'
    if snapshot.get('is_empty'):
        return 'quote_empty'
    if snapshot.get('is_crossed'):
        return 'quote_crossed'
    age_seconds = snapshot.get('age_seconds')
    if age_seconds is not None and age_seconds > float(policy.get('quote_max_age_sec', age_seconds)):
        return 'quote_stale'
    spread = snapshot.get('spread')
    if spread is not None and spread > float(policy.get('quote_max_spread', spread)):
        return 'quote_too_wide'
    return None


def _normalize_blocker_action_origin(value: Optional[str]) -> str:
    origin = str(value or 'unknown').strip().lower()
    if origin in {'first_entry', 'reeval_add', 'reeval_reduce', 'reeval_flip', 'inventory_exit', 'other_buy', 'unknown'}:
        return origin
    return 'unknown'


def _reason_family(reason: Optional[str]) -> str:
    text = str(reason or 'unknown').strip().lower()
    if text in {'missing_yes_quote', 'missing_no_quote', 'quote_stale', 'quote_too_wide', 'quote_empty', 'quote_crossed', 'quote_fetch_failed'}:
        return 'quote'
    if text in {'missing_strike_price', 'missing_probability_state'}:
        return 'probability'
    if text in {'market_not_open', 'market_expired'}:
        return 'market'
    if text in {'policy_blocks_new_entries', 'position_reeval_disabled_final_bucket'}:
        return 'policy'
    if text in {'no_edge_above_threshold', 'no_admission_edge_above_threshold'}:
        return 'edge'
    if text in {'entry_growth_optimizer_veto'}:
        return 'growth'
    if text in {'microstructure_noisy_block'}:
        return 'microstructure'
    if text in {'polarized_tail_block', 'tail_contrarian_hard_block', 'veto_polarization_hard_block', 'veto_same_side_reentry_polarized_zone'}:
        return 'polarization'
    if text in {'veto_same_side_reentry_cap'}:
        return 'same_side'
    if text in {'existing_active_buy_order'}:
        return 'active_order'
    if text in {'exposure_cap'}:
        return 'exposure'
    if text.startswith('veto_'):
        return 'regime'
    return 'other'


def _ensure_blocker_telemetry(decision_state: dict, *, action_origin: Optional[str] = None) -> dict:
    origin = _normalize_blocker_action_origin(action_origin or decision_state.get('action_origin'))
    decision_state['action_origin'] = origin
    telemetry = decision_state.get('blocker_telemetry')
    if not isinstance(telemetry, dict):
        telemetry = {}
        decision_state['blocker_telemetry'] = telemetry
    telemetry.setdefault('candidate_stage', 'pre_candidate')
    telemetry.setdefault('candidate_available', False)
    telemetry['action_origin'] = origin
    telemetry.setdefault('terminal_reason', decision_state.get('reason'))
    telemetry.setdefault('terminal_reason_family', _reason_family(decision_state.get('reason')))
    telemetry.setdefault('blocked_by', None)
    telemetry.setdefault('blocked_by_stage', None)
    telemetry.setdefault('first_blocking_guard', None)
    telemetry.setdefault('all_triggered_blockers', [])
    telemetry.setdefault('chosen_side_snapshot', {
        'chosen_side': decision_state.get('chosen_side'),
        'chosen_action_candidate': decision_state.get('action'),
        'chosen_side_quote': decision_state.get('chosen_side_quote'),
    })
    telemetry.setdefault('candidate_snapshot', {})
    telemetry.setdefault('blockers', {})
    decision_state.setdefault('candidate_stage', telemetry.get('candidate_stage'))
    decision_state.setdefault('terminal_reason', telemetry.get('terminal_reason'))
    decision_state.setdefault('terminal_reason_family', telemetry.get('terminal_reason_family'))
    decision_state.setdefault('blocked_by', telemetry.get('blocked_by'))
    decision_state.setdefault('blocked_by_stage', telemetry.get('blocked_by_stage'))
    decision_state.setdefault('first_blocking_guard', telemetry.get('first_blocking_guard'))
    decision_state.setdefault('all_triggered_blockers', list(telemetry.get('all_triggered_blockers') or []))
    return telemetry


def _set_candidate_stage(decision_state: dict, stage: str, *, candidate_available: Optional[bool] = None) -> None:
    telemetry = _ensure_blocker_telemetry(decision_state)
    telemetry['candidate_stage'] = stage
    if candidate_available is not None:
        telemetry['candidate_available'] = bool(candidate_available)
    decision_state['candidate_stage'] = telemetry['candidate_stage']


def _capture_candidate_snapshot(decision_state: dict, *, choice: Optional[dict] = None) -> None:
    telemetry = _ensure_blocker_telemetry(decision_state)
    chosen_side_candidate = None
    chosen_action_candidate = None
    token_id_candidate = None
    chosen_side_quote = decision_state.get('chosen_side_quote')
    if isinstance(choice, dict):
        chosen_side_candidate = choice.get('outcome_side')
        chosen_action_candidate = choice.get('side')
        token_id_candidate = choice.get('token_id')
        if choice.get('quote') is not None:
            chosen_side_quote = choice.get('quote')
    else:
        chosen_side_candidate = decision_state.get('chosen_side')
        chosen_action_candidate = decision_state.get('action')
        if chosen_side_candidate == 'YES':
            token_id_candidate = decision_state.get('token_yes')
        elif chosen_side_candidate == 'NO':
            token_id_candidate = decision_state.get('token_no')

    snapshot = {
        'ts': decision_state.get('ts') or decision_state.get('timestamp'),
        'now_ts': decision_state.get('timestamp'),
        'market_id': decision_state.get('market_id'),
        'action_origin': decision_state.get('action_origin'),
        'position_reeval_action': decision_state.get('position_reeval_action'),
        'chosen_action_candidate': chosen_action_candidate,
        'chosen_side_candidate': chosen_side_candidate,
        'token_id_candidate': token_id_candidate,
        'q_yes': decision_state.get('q_yes'),
        'q_no': decision_state.get('q_no'),
        'chosen_side_quote': chosen_side_quote,
        'raw_p_yes': decision_state.get('raw_p_yes'),
        'raw_p_no': decision_state.get('raw_p_no'),
        'adjusted_p_yes': decision_state.get('p_yes'),
        'adjusted_p_no': decision_state.get('p_no'),
        'discounted_p_yes': decision_state.get('discounted_p_yes'),
        'discounted_p_no': decision_state.get('discounted_p_no'),
        'raw_edge_yes': decision_state.get('raw_edge_yes'),
        'raw_edge_no': decision_state.get('raw_edge_no'),
        'adjusted_edge_yes': decision_state.get('adjusted_edge_yes', decision_state.get('edge_yes')),
        'adjusted_edge_no': decision_state.get('adjusted_edge_no', decision_state.get('edge_no')),
        'admission_edge_yes': decision_state.get('admission_edge_yes'),
        'admission_edge_no': decision_state.get('admission_edge_no'),
        'edge_threshold_yes': (decision_state.get('policy') or {}).get('edge_threshold_yes'),
        'edge_threshold_no': (decision_state.get('policy') or {}).get('edge_threshold_no'),
        'policy_bucket': (decision_state.get('policy') or {}).get('policy_bucket'),
        'allow_new_entries': (decision_state.get('policy') or {}).get('allow_new_entries'),
        'tau_minutes': decision_state.get('tau_minutes'),
        'strike_price': decision_state.get('strike_price'),
        'spot_now': decision_state.get('spot_now'),
        'favored_side': decision_state.get('favored_side'),
        'contrarian_side': decision_state.get('contrarian_side'),
        'tail_penalty_score': decision_state.get('tail_penalty_score'),
        'tail_hard_block': decision_state.get('tail_hard_block'),
        'regime_state': decision_state.get('regime_state'),
        'microstructure_regime': decision_state.get('microstructure_regime'),
        'smoothness_score': decision_state.get('smoothness_score'),
        'spectral_entropy': decision_state.get('spectral_entropy'),
        'low_freq_power_ratio': decision_state.get('low_freq_power_ratio'),
        'high_freq_power_ratio': decision_state.get('high_freq_power_ratio'),
        'spectral_ready': decision_state.get('spectral_ready'),
        'spectral_reason': decision_state.get('spectral_reason'),
        'expected_log_growth_entry': decision_state.get('expected_log_growth_entry'),
        'expected_log_growth_entry_conservative': decision_state.get('expected_log_growth_entry_conservative'),
        'expected_log_growth_pass_shadow': decision_state.get('expected_log_growth_pass_shadow'),
        'growth_gate_pass_shadow': decision_state.get('growth_gate_pass_shadow'),
        'growth_gate_reason_shadow': decision_state.get('growth_gate_reason_shadow'),
    }
    telemetry['candidate_snapshot'] = snapshot
    telemetry['chosen_side_snapshot'] = {
        'chosen_side': chosen_side_candidate,
        'chosen_action_candidate': chosen_action_candidate,
        'chosen_side_quote': chosen_side_quote,
    }
    telemetry['candidate_available'] = any(
        snapshot.get(key) is not None
        for key in ('q_yes', 'q_no', 'raw_p_yes', 'raw_p_no', 'admission_edge_yes', 'admission_edge_no', 'chosen_side_candidate')
    )
    decision_state['candidate_stage'] = telemetry.get('candidate_stage')


def _record_blocker(
    decision_state: dict,
    *,
    blocker_name: str,
    evaluated: bool,
    mode: Optional[str],
    would_block: bool,
    blocked: bool,
    reason: Optional[str],
    inputs: Optional[dict] = None,
    stage: Optional[str] = None,
    terminal: bool = False,
) -> None:
    telemetry = _ensure_blocker_telemetry(decision_state)
    blockers = telemetry.setdefault('blockers', {})
    blockers[blocker_name] = {
        'evaluated': bool(evaluated),
        'mode': mode,
        'would_block': bool(would_block),
        'blocked': bool(blocked),
        'reason': reason,
        'inputs': inputs or {},
    }
    if would_block or blocked:
        ordered = telemetry.setdefault('all_triggered_blockers', [])
        if blocker_name not in ordered:
            ordered.append(blocker_name)
            decision_state['all_triggered_blockers'] = list(ordered)
    if terminal:
        if stage is not None:
            _set_candidate_stage(decision_state, stage, candidate_available=telemetry.get('candidate_available'))
        telemetry['terminal_reason'] = reason
        telemetry['terminal_reason_family'] = _reason_family(reason)
        telemetry['blocked_by'] = blocker_name
        telemetry['blocked_by_stage'] = telemetry.get('candidate_stage')
        if telemetry.get('first_blocking_guard') is None:
            telemetry['first_blocking_guard'] = blocker_name
        decision_state['terminal_reason'] = telemetry['terminal_reason']
        decision_state['terminal_reason_family'] = telemetry['terminal_reason_family']
        decision_state['blocked_by'] = telemetry['blocked_by']
        decision_state['blocked_by_stage'] = telemetry['blocked_by_stage']
        decision_state['first_blocking_guard'] = telemetry['first_blocking_guard']


def _finalize_non_trading_state(
    decision_state: dict,
    *,
    blocker_name: str,
    reason: str,
    stage: str,
    inputs: Optional[dict] = None,
    mode: Optional[str] = 'live',
    choice: Optional[dict] = None,
) -> None:
    _capture_candidate_snapshot(decision_state, choice=choice)
    _record_blocker(
        decision_state,
        blocker_name=blocker_name,
        evaluated=True,
        mode=mode,
        would_block=True,
        blocked=True,
        reason=reason,
        inputs=inputs,
        stage=stage,
        terminal=True,
    )


def append_decision_log(decision_state: dict):
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG_PATH.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(decision_state, sort_keys=True) + '\n')


def build_decision_log_record(
    *,
    decision_state: dict,
    trading_allowed: bool,
    disabled_reason: str,
    wallet_snapshot: dict,
    routing_debug: dict,
    engine_diagnostics: Optional[dict] = None,
    shadow_probability_models: Optional[dict] = None,
) -> dict:
    decision_log_record = {
        'timestamp': decision_state.get('timestamp'),
        'series_id': decision_state.get('series_id'),
        'market_id': decision_state.get('market_id'),
        'token_yes': decision_state.get('token_yes'),
        'token_no': decision_state.get('token_no'),
        'strike_price': decision_state.get('strike_price'),
        'spot_now': decision_state.get('spot_now'),
        'tau_minutes': decision_state.get('tau_minutes'),
        'p_yes': decision_state.get('p_yes'),
        'p_no': decision_state.get('p_no'),
        'raw_p_yes_decision': decision_state.get('raw_p_yes'),
        'raw_p_no_decision': decision_state.get('raw_p_no'),
        'calibrated_p_yes': decision_state.get('calibrated_p_yes'),
        'calibrated_p_no': decision_state.get('calibrated_p_no'),
        'adjusted_p_yes': decision_state.get('p_yes'),
        'adjusted_p_no': decision_state.get('p_no'),
        'discounted_p_yes': decision_state.get('discounted_p_yes'),
        'discounted_p_no': decision_state.get('discounted_p_no'),
        'q_yes': decision_state.get('q_yes'),
        'q_no': decision_state.get('q_no'),
        'chosen_side_quote': decision_state.get('chosen_side_quote'),
        'minority_side_quote': decision_state.get('minority_side_quote'),
        'q_tail': decision_state.get('q_tail'),
        'tail_side': decision_state.get('tail_side'),
        'chosen_side': decision_state.get('chosen_side'),
        'favored_side': decision_state.get('favored_side'),
        'contrarian_side': decision_state.get('contrarian_side'),
        'polarization_zone': decision_state.get('polarization_zone'),
        'polarization_zone_yes': decision_state.get('polarization_zone_yes'),
        'polarization_zone_no': decision_state.get('polarization_zone_no'),
        'polarization_credibility_mode': decision_state.get('polarization_credibility_mode'),
        'polarized_tail_penalty': decision_state.get('polarized_tail_penalty'),
        'polarized_tail_blocked': decision_state.get('polarized_tail_blocked'),
        'z_distance_to_strike': decision_state.get('z_distance_to_strike'),
        'edge_yes': decision_state.get('edge_yes'),
        'edge_no': decision_state.get('edge_no'),
        'expected_log_growth_entry': decision_state.get('expected_log_growth_entry'),
        'expected_log_growth_entry_conservative': decision_state.get('expected_log_growth_entry_conservative'),
        'expected_log_growth_entry_conservative_old': decision_state.get('expected_log_growth_entry_conservative_old'),
        'expected_log_growth_entry_conservative_discounted': decision_state.get('expected_log_growth_entry_conservative_discounted'),
        'expected_log_growth_pass_shadow': decision_state.get('expected_log_growth_pass_shadow'),
        'expected_log_growth_reason_shadow': decision_state.get('expected_log_growth_reason_shadow'),
        'growth_gate_pass_shadow': decision_state.get('growth_gate_pass_shadow'),
        'growth_gate_reason_shadow': decision_state.get('growth_gate_reason_shadow'),
        'expected_terminal_wealth_if_yes': decision_state.get('expected_terminal_wealth_if_yes'),
        'expected_terminal_wealth_if_no': decision_state.get('expected_terminal_wealth_if_no'),
        'entry_growth_eval_mode': decision_state.get('entry_growth_eval_mode'),
        'entry_growth_candidate_side': decision_state.get('entry_growth_candidate_side'),
        'entry_growth_qty': decision_state.get('entry_growth_qty'),
        'entry_growth_trade_notional': decision_state.get('entry_growth_trade_notional'),
        'entry_growth_probability_conservative': decision_state.get('entry_growth_probability_conservative'),
        'entry_growth_fragility_score': decision_state.get('entry_growth_fragility_score'),
        'entry_growth_candidates': decision_state.get('entry_growth_candidates'),
        'raw_edge_yes_decision': decision_state.get('raw_edge_yes'),
        'raw_edge_no_decision': decision_state.get('raw_edge_no'),
        'adjusted_edge_yes': decision_state.get('edge_yes'),
        'adjusted_edge_no': decision_state.get('edge_no'),
        'credibility_weight_yes': decision_state.get('credibility_weight_yes'),
        'credibility_weight_no': decision_state.get('credibility_weight_no'),
        'discounted_edge_yes': decision_state.get('discounted_edge_yes'),
        'discounted_edge_no': decision_state.get('discounted_edge_no'),
        'admission_edge_yes': decision_state.get('admission_edge_yes'),
        'admission_edge_no': decision_state.get('admission_edge_no'),
        'admission_probability_source': decision_state.get('admission_probability_source'),
        'edge_credibility_reason': decision_state.get('edge_credibility_reason'),
        'credibility_shadow_trade_allowed': decision_state.get('credibility_shadow_trade_allowed'),
        'credibility_shadow_action': decision_state.get('credibility_shadow_action'),
        'credibility_shadow_reason': decision_state.get('credibility_shadow_reason'),
        'credibility_block_reason': decision_state.get('credibility_block_reason'),
        'tail_penalty_score': decision_state.get('tail_penalty_score'),
        'tail_q_tail': decision_state.get('tail_q_tail'),
        'tail_z_signed': decision_state.get('tail_z_signed'),
        'tail_z_abs': decision_state.get('tail_z_abs'),
        'tail_hard_block': decision_state.get('tail_hard_block'),
        'tail_overlay_version': ((decision_state.get('tail_overlay') or {}).get('version')),
        'microstructure_regime': decision_state.get('microstructure_regime'),
        'spectral_entropy': decision_state.get('spectral_entropy'),
        'low_freq_power_ratio': decision_state.get('low_freq_power_ratio'),
        'high_freq_power_ratio': decision_state.get('high_freq_power_ratio'),
        'smoothness_score': decision_state.get('smoothness_score'),
        'spectral_observation_count': decision_state.get('spectral_observation_count'),
        'spectral_window_minutes': decision_state.get('spectral_window_minutes'),
        'spectral_ready': decision_state.get('spectral_ready'),
        'spectral_reason': decision_state.get('spectral_reason'),
        'microstructure_mode': decision_state.get('microstructure_mode'),
        'microstructure_would_block': decision_state.get('microstructure_would_block'),
        'regime_state': decision_state.get('regime_state'),
        'regime_guard_mode': decision_state.get('regime_guard_mode'),
        'regime_guard_evaluated': decision_state.get('regime_guard_evaluated'),
        'regime_guard_blocked': decision_state.get('regime_guard_blocked'),
        'regime_guard_reason': decision_state.get('regime_guard_reason'),
        'regime_guard_details': decision_state.get('regime_guard_details'),
        'action_origin': decision_state.get('action_origin'),
        'candidate_stage': decision_state.get('candidate_stage'),
        'terminal_reason': decision_state.get('terminal_reason'),
        'terminal_reason_family': decision_state.get('terminal_reason_family'),
        'blocked_by': decision_state.get('blocked_by'),
        'blocked_by_stage': decision_state.get('blocked_by_stage'),
        'first_blocking_guard': decision_state.get('first_blocking_guard'),
        'all_triggered_blockers': decision_state.get('all_triggered_blockers'),
        'blocker_telemetry': decision_state.get('blocker_telemetry'),
        'shared_buy_gate_evaluated': decision_state.get('shared_buy_gate_evaluated'),
        'shared_buy_gate_passed': decision_state.get('shared_buy_gate_passed'),
        'shared_buy_gate_reason': decision_state.get('shared_buy_gate_reason'),
        'shared_guard_trade_allowed_checked': decision_state.get('shared_guard_trade_allowed_checked'),
        'shared_guard_quote_checked': decision_state.get('shared_guard_quote_checked'),
        'shared_guard_market_open_checked': decision_state.get('shared_guard_market_open_checked'),
        'shared_guard_policy_checked': decision_state.get('shared_guard_policy_checked'),
        'shared_guard_tail_checked': decision_state.get('shared_guard_tail_checked'),
        'shared_guard_microstructure_checked': decision_state.get('shared_guard_microstructure_checked'),
        'shared_guard_regime_checked': decision_state.get('shared_guard_regime_checked'),
        'shared_guard_exposure_checked': decision_state.get('shared_guard_exposure_checked'),
        'shared_guard_active_order_checked': decision_state.get('shared_guard_active_order_checked'),
        'shared_guard_result_json': decision_state.get('shared_guard_result_json'),
        'same_side_existing_qty': decision_state.get('same_side_existing_qty'),
        'same_side_existing_filled_entry_count': decision_state.get('same_side_existing_filled_entry_count'),
        'would_block_in_shadow': decision_state.get('would_block_in_shadow'),
        'same_side_reentry_shadow_blocked': decision_state.get('same_side_reentry_shadow_blocked'),
        'same_side_reentry_live_blocked': decision_state.get('same_side_reentry_live_blocked'),
        'same_side_reentry_reason': decision_state.get('same_side_reentry_reason'),
        'reversal_evidence_by_side': decision_state.get('reversal_evidence_by_side'),
        'policy_bucket': (decision_state.get('policy') or {}).get('policy_bucket'),
        'edge_threshold_yes': (decision_state.get('policy') or {}).get('edge_threshold_yes'),
        'edge_threshold_no': (decision_state.get('policy') or {}).get('edge_threshold_no'),
        'kelly_multiplier': (decision_state.get('policy') or {}).get('kelly_multiplier'),
        'max_trade_notional_multiplier': (decision_state.get('policy') or {}).get('max_trade_notional_multiplier'),
        'allow_new_entries': (decision_state.get('policy') or {}).get('allow_new_entries'),
        'position_reeval_enabled': decision_state.get('position_reeval_enabled'),
        'position_reeval_action': decision_state.get('position_reeval_action'),
        'position_reeval_reason': decision_state.get('position_reeval_reason'),
        'position_reeval_hold_ev_per_share': decision_state.get('position_reeval_hold_ev_per_share'),
        'position_reeval_candidate_entry_ev_per_share': decision_state.get('position_reeval_candidate_entry_ev_per_share'),
        'position_reeval_flip_advantage_per_share': decision_state.get('position_reeval_flip_advantage_per_share'),
        'position_reeval_reversal_score': decision_state.get('position_reeval_reversal_score'),
        'position_reeval_reversal_passed': decision_state.get('position_reeval_reversal_passed'),
        'position_reeval_shadow_best_action': decision_state.get('position_reeval_shadow_best_action'),
        'position_reeval_shadow_best_delta_qty': decision_state.get('position_reeval_shadow_best_delta_qty'),
        'position_reeval_shadow_best_growth_gain': decision_state.get('position_reeval_shadow_best_growth_gain'),
        'position_reeval_shadow_best_executable': decision_state.get('position_reeval_shadow_best_executable'),
        'position_reeval_shadow_keep_current_position': decision_state.get('position_reeval_shadow_keep_current_position'),
        'position_reeval': decision_state.get('position_reeval'),
        'trade_allowed': trading_allowed,
        'action': decision_state.get('action') if trading_allowed else None,
        'reason': disabled_reason if not trading_allowed else decision_state.get('reason'),
        'wallet_address': wallet_snapshot.get('wallet_address'),
        'wallet_usdc_e': wallet_snapshot.get('usdc_e_balance'),
        'wallet_pol': wallet_snapshot.get('pol_balance'),
        'wallet_reserved_exposure': wallet_snapshot.get('reserved_exposure_usdc'),
        'wallet_free_usdc': wallet_snapshot.get('free_usdc'),
        'effective_bankroll': wallet_snapshot.get('effective_bankroll'),
        'bankroll_source': wallet_snapshot.get('bankroll_source'),
        'wallet_fetch_failed': wallet_snapshot.get('fetch_failed'),
        'routing_series_arg': routing_debug.get('series_arg'),
        'routing_bundle_present': routing_debug.get('routing_bundle_present'),
        'routing_runtime_status': routing_debug.get('runtime_status'),
        'routing_strike_price': routing_debug.get('strike_price'),
        'routing_epoch_key': routing_debug.get('epoch_key'),
        'routing_detection_source': routing_debug.get('detection_source'),
        'routing_reason': routing_debug.get('routing_reason'),
        'routing_candidate_slugs': routing_debug.get('candidate_slugs'),
        'routing_attempt_count': routing_debug.get('attempt_count'),
        'routing_attempts': routing_debug.get('attempts'),
        'routing_switched': routing_debug.get('switched'),
        'routing_missing_market_id': routing_debug.get('missing_market_id'),
        'routing_missing_token_yes': routing_debug.get('missing_token_yes'),
        'routing_missing_token_no': routing_debug.get('missing_token_no'),
        'shadow_probability_models': shadow_probability_models or {},
    }
    decision_log_record.update(extract_engine_shadow_diagnostics(engine_diagnostics))
    return decision_log_record


def extract_engine_shadow_diagnostics(engine_diagnostics: Optional[dict]) -> dict:
    diagnostics = engine_diagnostics or {}
    last_prediction = diagnostics.get('last_prediction') or {}
    return {
        'engine_name': diagnostics.get('engine_name'),
        'engine_version': diagnostics.get('engine_version'),
        'history_len': diagnostics.get('history_len'),
        'sigma_per_sqrt_min': diagnostics.get('sigma_per_sqrt_min'),
        'horizon_sigma': last_prediction.get('horizon_sigma'),
        'z_score': last_prediction.get('z_score'),
        'raw_p_yes': last_prediction.get('raw_p_yes'),
        'calibrated_p_yes': last_prediction.get('calibrated_p_yes'),
        'fallback_sigma_used': diagnostics.get('fallback_sigma_used'),
        'vol_window': diagnostics.get('vol_window'),
        'min_periods': diagnostics.get('min_periods'),
        'calibration_mode': diagnostics.get('calibration_mode'),
    }


def build_routing_debug(series_arg: Optional[str], routing_bundle: Optional[dict]) -> dict:
    bundle = routing_bundle or {}
    return {
        'series_arg': series_arg,
        'routing_bundle_present': routing_bundle is not None,
        'routing_market_id': bundle.get('market_id'),
        'routing_token_yes': bundle.get('token_yes'),
        'routing_token_no': bundle.get('token_no'),
        'runtime_status': bundle.get('runtime_status'),
        'strike_price': bundle.get('strike_price'),
        'epoch_key': bundle.get('epoch_key'),
        'detection_source': bundle.get('detection_source'),
        'routing_reason': bundle.get('routing_reason'),
        'candidate_slugs': bundle.get('routing_candidate_slugs') or [],
        'attempt_count': bundle.get('routing_attempt_count'),
        'attempts': bundle.get('routing_attempts') or [],
        'switched': bundle.get('switched'),
        'missing_market_id': bundle.get('market_id') is None,
        'missing_token_yes': bundle.get('token_yes') is None,
        'missing_token_no': bundle.get('token_no') is None,
    }


def resolve_live_market_context(args, now=None) -> dict:
    now = now or pd.Timestamp.now(tz='UTC')
    explicit_series = getattr(args, 'series', None)
    manual_args_present = any(getattr(args, field, None) for field in ('token_yes', 'token_no', 'market_id'))
    auto_series = explicit_series or (None if manual_args_present else 'bitcoin-up-or-down')
    routing_result = resolve_active_market_bundle_with_debug(auto_series, now=now) if auto_series else {
        'bundle': None,
        'routing_reason': None,
        'candidate_slugs': [],
        'attempt_count': 0,
        'detection_source': None,
        'attempts': [],
    }
    routing_bundle = routing_result.get('bundle')
    if auto_series:
        routing_debug = build_routing_debug(auto_series, routing_bundle)
        if routing_bundle is None:
            routing_debug.update({
                'routing_reason': routing_result.get('routing_reason'),
                'candidate_slugs': routing_result.get('candidate_slugs') or [],
                'attempt_count': routing_result.get('attempt_count'),
                'attempts': routing_result.get('attempts') or [],
                'detection_source': routing_result.get('detection_source'),
            })
        return {
            'routing_bundle': routing_bundle,
            'routing_debug': routing_debug,
            'market_meta': _market_from_bundle(routing_bundle),
            'market_id': (routing_bundle or {}).get('market_id'),
            'token_yes': (routing_bundle or {}).get('token_yes'),
            'token_no': (routing_bundle or {}).get('token_no'),
            'yes_quote': (routing_bundle or {}).get('yes_quote') or {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None},
            'no_quote': (routing_bundle or {}).get('no_quote') or {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None},
        }
    token_yes = getattr(args, 'token_yes', None)
    token_no = getattr(args, 'token_no', None)
    market_id = getattr(args, 'market_id', None)
    return {
        'routing_bundle': None,
        'routing_debug': {},
        'market_meta': {
            'market_id': market_id,
            'token_yes': token_yes,
            'token_no': token_no,
            'status': (storage.get_market(market_id) or {}).get('status') if market_id else None,
        },
        'market_id': market_id,
        'token_yes': token_yes,
        'token_no': token_no,
        'yes_quote': get_quote_snapshot(token_yes) if token_yes else {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None},
        'no_quote': get_quote_snapshot(token_no) if token_no else {'fetch_failed': True, 'is_empty': True, 'mid': None, 'age_seconds': None},
    }


def build_trade_context(
    market_meta: dict,
    yes_quote: dict,
    no_quote: dict,
    now=None,
    routing_bundle: Optional[dict] = None,
    wallet_state: Optional[dict] = None,
) -> dict:
    now = now or pd.Timestamp.now(tz='UTC')
    market_id = market_meta.get('market_id') if market_meta else None
    if market_id:
        storage.refresh_market_lifecycle(
            market_id,
            source_data={
                'status': market_meta.get('status'),
                'winning_outcome': market_meta.get('winning_outcome'),
                'endDate': _iso_or_none(market_meta.get('endDate')),
                'startDate': _iso_or_none(market_meta.get('startDate')),
                'condition_id': market_meta.get('condition_id'),
                'slug': market_meta.get('slug'),
                'title': market_meta.get('title'),
            },
            checked_ts=now.isoformat(),
        )
    stored_market = storage.get_market(market_id) if market_id else None
    snapshot_by_market = {item['market_id']: item for item in storage.get_position_snapshot()}
    position = snapshot_by_market.get(market_id, {})
    open_orders = storage.get_open_orders(market_id=market_id) if market_id else []
    return {
        'now': now.isoformat(),
        'market': market_meta or {},
        'stored_market': stored_market or {},
        'wallet_state': wallet_state or {},
        'quotes': {'yes': yes_quote or {}, 'no': no_quote or {}},
        'quote_state': {'yes': classify_quote_snapshot(yes_quote or {}), 'no': classify_quote_snapshot(no_quote or {})},
        'routing': {
            'series_id': (routing_bundle or {}).get('series_id'),
            'active_market_id': (routing_bundle or {}).get('market_id'),
            'active_token_yes': (routing_bundle or {}).get('token_yes'),
            'active_token_no': (routing_bundle or {}).get('token_no'),
            'strike_price': (routing_bundle or {}).get('strike_price'),
            'strike_source': (routing_bundle or {}).get('strike_source'),
            'runtime_status': (routing_bundle or {}).get('runtime_status'),
            'switched_this_iteration': bool((routing_bundle or {}).get('switched')),
            'epoch_key': (routing_bundle or {}).get('epoch_key'),
        },
        'market_window': {
            'start': _iso_or_none((market_meta or {}).get('startDate')),
            'end': _iso_or_none((market_meta or {}).get('endDate')),
        },
        'position_summary': {
            'inflight_exposure': storage.get_inflight_exposure(market_id=market_id) if market_id else 0.0,
            'redeemable_qty': position.get('resolved_redeemable_qty', 0.0),
            'available_inventory': position.get('available_inventory', {'YES': 0.0, 'NO': 0.0}),
            'avg_entry_price_yes': position.get('avg_entry_price_yes'),
            'avg_entry_price_no': position.get('avg_entry_price_no'),
        },
        'open_orders': open_orders,
        'position_management_state': storage.get_position_management_state(market_id) if market_id else None,
    }


def can_trade_context(ctx: dict) -> tuple[bool, str]:
    market = ctx.get('market') or {}
    stored_market = ctx.get('stored_market') or {}
    routing = ctx.get('routing') or {}
    policy = ctx.get('policy') or {}
    now = _parse_ts(ctx.get('now')) or pd.Timestamp.now(tz='UTC')
    start = _parse_ts((ctx.get('market_window') or {}).get('start'))
    end = _parse_ts((ctx.get('market_window') or {}).get('end'))
    if not market:
        return False, 'market_detection_failed'
    if not market.get('market_id') or not market.get('token_yes') or not market.get('token_no'):
        return False, 'incomplete_market_metadata'
    if (stored_market.get('status') or market.get('status')) != 'open':
        return False, 'market_not_open'
    if start is None or end is None or not (start <= now < end):
        return False, 'market_window_expired'
    if routing.get('series_id') and routing.get('strike_price') is None:
        return False, 'missing_strike_price'
    if policy and not policy.get('allow_new_entries', True):
        return False, 'policy_blocks_new_entries'
    yes_reason = ((ctx.get('quote_state') or {}).get('yes') or {}).get('reason')
    if yes_reason:
        return False, yes_reason
    policy_yes_reason = _policy_quote_reason((ctx.get('quotes') or {}).get('yes') or {}, policy)
    if policy_yes_reason:
        return False, policy_yes_reason
    return True, 'ok'


def build_diagnostic_heartbeat(ctx: dict, trading_allowed: bool, reason: str) -> dict:
    quotes = ctx.get('quotes') or {}
    stored_market = ctx.get('stored_market') or {}
    position_summary = ctx.get('position_summary') or {}
    routing = ctx.get('routing') or {}
    routing_debug = ctx.get('routing_debug') or {}
    policy = ctx.get('policy') or {}
    wallet_state = ctx.get('wallet_state') or {}
    decision_state = ctx.get('decision_state') or {}
    heartbeat = {
        'series_id': routing.get('series_id'),
        'active_market_id': routing.get('active_market_id') or (ctx.get('market') or {}).get('market_id'),
        'active_token_yes': routing.get('active_token_yes') or (ctx.get('market') or {}).get('token_yes'),
        'active_token_no': routing.get('active_token_no') or (ctx.get('market') or {}).get('token_no'),
        'strike_price': routing.get('strike_price'),
        'switched_this_iteration': routing.get('switched_this_iteration', False),
        'market_id': (ctx.get('market') or {}).get('market_id'),
        'market_window': ctx.get('market_window'),
        'market_status': stored_market.get('status') or (ctx.get('market') or {}).get('status'),
        'yes_mid': (quotes.get('yes') or {}).get('mid'),
        'no_mid': (quotes.get('no') or {}).get('mid'),
        'quote_ages': {
            'yes': (quotes.get('yes') or {}).get('age_seconds'),
            'no': (quotes.get('no') or {}).get('age_seconds'),
        },
        'tau_minutes': (ctx.get('probability_state') or {}).get('tau_minutes'),
        'p_yes': (ctx.get('probability_state') or {}).get('p_yes'),
        'p_no': (ctx.get('probability_state') or {}).get('p_no'),
        'q_yes': decision_state.get('q_yes'),
        'q_no': decision_state.get('q_no'),
        'chosen_side_quote': decision_state.get('chosen_side_quote'),
        'minority_side_quote': decision_state.get('minority_side_quote'),
        'q_tail': decision_state.get('q_tail'),
        'tail_side': decision_state.get('tail_side'),
        'chosen_side': decision_state.get('chosen_side'),
        'raw_p_yes_decision': decision_state.get('raw_p_yes'),
        'raw_p_no_decision': decision_state.get('raw_p_no'),
        'calibrated_p_yes': decision_state.get('calibrated_p_yes'),
        'calibrated_p_no': decision_state.get('calibrated_p_no'),
        'adjusted_p_yes': decision_state.get('p_yes'),
        'adjusted_p_no': decision_state.get('p_no'),
        'discounted_p_yes': decision_state.get('discounted_p_yes'),
        'discounted_p_no': decision_state.get('discounted_p_no'),
        'polarized_tail_penalty': decision_state.get('polarized_tail_penalty'),
        'polarized_tail_blocked': decision_state.get('polarized_tail_blocked'),
        'z_distance_to_strike': decision_state.get('z_distance_to_strike'),
        'tail_penalty_score': decision_state.get('tail_penalty_score'),
        'tail_q_tail': decision_state.get('tail_q_tail'),
        'tail_z_abs': decision_state.get('tail_z_abs'),
        'tail_hard_block': decision_state.get('tail_hard_block'),
        'edge_yes': decision_state.get('edge_yes'),
        'edge_no': decision_state.get('edge_no'),
        'raw_edge_yes': (ctx.get('decision_state') or {}).get('raw_edge_yes'),
        'raw_edge_no': (ctx.get('decision_state') or {}).get('raw_edge_no'),
        'adjusted_edge_yes': (ctx.get('decision_state') or {}).get('adjusted_edge_yes'),
        'adjusted_edge_no': (ctx.get('decision_state') or {}).get('adjusted_edge_no'),
        'credibility_weight_yes': (ctx.get('decision_state') or {}).get('credibility_weight_yes'),
        'credibility_weight_no': (ctx.get('decision_state') or {}).get('credibility_weight_no'),
        'discounted_edge_yes': (ctx.get('decision_state') or {}).get('discounted_edge_yes'),
        'discounted_edge_no': (ctx.get('decision_state') or {}).get('discounted_edge_no'),
        'admission_edge_yes': (ctx.get('decision_state') or {}).get('admission_edge_yes'),
        'admission_edge_no': (ctx.get('decision_state') or {}).get('admission_edge_no'),
        'admission_probability_source': (ctx.get('decision_state') or {}).get('admission_probability_source'),
        'edge_credibility_reason': (ctx.get('decision_state') or {}).get('edge_credibility_reason'),
        'polarization_zone': (ctx.get('decision_state') or {}).get('polarization_zone'),
        'polarization_zone_yes': (ctx.get('decision_state') or {}).get('polarization_zone_yes'),
        'polarization_zone_no': (ctx.get('decision_state') or {}).get('polarization_zone_no'),
        'polarization_credibility_mode': (ctx.get('decision_state') or {}).get('polarization_credibility_mode'),
        'credibility_shadow_trade_allowed': (ctx.get('decision_state') or {}).get('credibility_shadow_trade_allowed'),
        'credibility_shadow_action': (ctx.get('decision_state') or {}).get('credibility_shadow_action'),
        'credibility_shadow_reason': (ctx.get('decision_state') or {}).get('credibility_shadow_reason'),
        'credibility_block_reason': (ctx.get('decision_state') or {}).get('credibility_block_reason'),
        'chosen_action': (ctx.get('decision_state') or {}).get('action'),
        'expected_log_growth_entry': (ctx.get('decision_state') or {}).get('expected_log_growth_entry'),
        'expected_log_growth_entry_conservative': (ctx.get('decision_state') or {}).get('expected_log_growth_entry_conservative'),
        'expected_log_growth_entry_conservative_old': (ctx.get('decision_state') or {}).get('expected_log_growth_entry_conservative_old'),
        'expected_log_growth_entry_conservative_discounted': (ctx.get('decision_state') or {}).get('expected_log_growth_entry_conservative_discounted'),
        'expected_log_growth_pass_shadow': (ctx.get('decision_state') or {}).get('expected_log_growth_pass_shadow'),
        'expected_log_growth_reason_shadow': (ctx.get('decision_state') or {}).get('expected_log_growth_reason_shadow'),
        'expected_terminal_wealth_if_yes': (ctx.get('decision_state') or {}).get('expected_terminal_wealth_if_yes'),
        'expected_terminal_wealth_if_no': (ctx.get('decision_state') or {}).get('expected_terminal_wealth_if_no'),
        'entry_growth_eval_mode': (ctx.get('decision_state') or {}).get('entry_growth_eval_mode'),
        'regime_label': _get_nested(decision_state, 'regime_state', 'regime_label'),
        'regime_trend_score': _get_nested(decision_state, 'regime_state', 'trend_score'),
        'regime_tail_score': _get_nested(decision_state, 'regime_state', 'tail_score'),
        'regime_reversal_score': _get_nested(decision_state, 'regime_state', 'reversal_score'),
        'microstructure_regime': decision_state.get('microstructure_regime') or _get_nested(decision_state, 'regime_state', 'microstructure_regime'),
        'spectral_entropy': decision_state.get('spectral_entropy') if decision_state.get('spectral_entropy') is not None else _get_nested(decision_state, 'regime_state', 'spectral_entropy'),
        'low_freq_power_ratio': decision_state.get('low_freq_power_ratio') if decision_state.get('low_freq_power_ratio') is not None else _get_nested(decision_state, 'regime_state', 'low_freq_power_ratio'),
        'high_freq_power_ratio': decision_state.get('high_freq_power_ratio') if decision_state.get('high_freq_power_ratio') is not None else _get_nested(decision_state, 'regime_state', 'high_freq_power_ratio'),
        'smoothness_score': decision_state.get('smoothness_score') if decision_state.get('smoothness_score') is not None else _get_nested(decision_state, 'regime_state', 'smoothness_score'),
        'spectral_observation_count': decision_state.get('spectral_observation_count') if decision_state.get('spectral_observation_count') is not None else _get_nested(decision_state, 'regime_state', 'spectral_observation_count'),
        'spectral_window_minutes': decision_state.get('spectral_window_minutes') if decision_state.get('spectral_window_minutes') is not None else _get_nested(decision_state, 'regime_state', 'spectral_window_minutes'),
        'spectral_ready': decision_state.get('spectral_ready') if decision_state.get('spectral_ready') is not None else _get_nested(decision_state, 'regime_state', 'spectral_ready'),
        'spectral_reason': decision_state.get('spectral_reason') or _get_nested(decision_state, 'regime_state', 'spectral_reason'),
        'microstructure_mode': decision_state.get('microstructure_mode'),
        'microstructure_would_block': decision_state.get('microstructure_would_block'),
        'regime_state': decision_state.get('regime_state'),
        'regime_guard_mode': decision_state.get('regime_guard_mode'),
        'regime_guard_evaluated': decision_state.get('regime_guard_evaluated'),
        'regime_guard_blocked': decision_state.get('regime_guard_blocked'),
        'regime_guard_reason': decision_state.get('regime_guard_reason'),
        'regime_guard_details': decision_state.get('regime_guard_details'),
        'action_origin': decision_state.get('action_origin'),
        'candidate_stage': decision_state.get('candidate_stage'),
        'terminal_reason': decision_state.get('terminal_reason'),
        'terminal_reason_family': decision_state.get('terminal_reason_family'),
        'blocked_by': decision_state.get('blocked_by'),
        'blocked_by_stage': decision_state.get('blocked_by_stage'),
        'first_blocking_guard': decision_state.get('first_blocking_guard'),
        'all_triggered_blockers': decision_state.get('all_triggered_blockers'),
        'blocker_telemetry': decision_state.get('blocker_telemetry'),
        'shared_buy_gate_evaluated': decision_state.get('shared_buy_gate_evaluated'),
        'shared_buy_gate_passed': decision_state.get('shared_buy_gate_passed'),
        'shared_buy_gate_reason': decision_state.get('shared_buy_gate_reason'),
        'shared_guard_trade_allowed_checked': decision_state.get('shared_guard_trade_allowed_checked'),
        'shared_guard_quote_checked': decision_state.get('shared_guard_quote_checked'),
        'shared_guard_market_open_checked': decision_state.get('shared_guard_market_open_checked'),
        'shared_guard_policy_checked': decision_state.get('shared_guard_policy_checked'),
        'shared_guard_tail_checked': decision_state.get('shared_guard_tail_checked'),
        'shared_guard_microstructure_checked': decision_state.get('shared_guard_microstructure_checked'),
        'shared_guard_regime_checked': decision_state.get('shared_guard_regime_checked'),
        'shared_guard_exposure_checked': decision_state.get('shared_guard_exposure_checked'),
        'shared_guard_active_order_checked': decision_state.get('shared_guard_active_order_checked'),
        'shared_guard_result_json': decision_state.get('shared_guard_result_json'),
        'same_side_existing_qty': decision_state.get('same_side_existing_qty'),
        'same_side_existing_filled_entry_count': decision_state.get('same_side_existing_filled_entry_count'),
        'would_block_in_shadow': decision_state.get('would_block_in_shadow'),
        'same_side_reentry_shadow_blocked': decision_state.get('same_side_reentry_shadow_blocked'),
        'same_side_reentry_live_blocked': decision_state.get('same_side_reentry_live_blocked'),
        'same_side_reentry_reason': decision_state.get('same_side_reentry_reason'),
        'reversal_evidence_by_side': decision_state.get('reversal_evidence_by_side'),
        'policy_bucket': policy.get('policy_bucket'),
        'edge_threshold_yes': policy.get('edge_threshold_yes'),
        'edge_threshold_no': policy.get('edge_threshold_no'),
        'kelly_multiplier': policy.get('kelly_multiplier'),
        'max_trade_notional_multiplier': policy.get('max_trade_notional_multiplier'),
        'allow_new_entries': policy.get('allow_new_entries'),
        'position_reeval_shadow_best_action': decision_state.get('position_reeval_shadow_best_action'),
        'position_reeval_shadow_best_delta_qty': decision_state.get('position_reeval_shadow_best_delta_qty'),
        'position_reeval_shadow_best_growth_gain': decision_state.get('position_reeval_shadow_best_growth_gain'),
        'position_reeval_shadow_best_executable': decision_state.get('position_reeval_shadow_best_executable'),
        'position_reeval_shadow_keep_current_position': decision_state.get('position_reeval_shadow_keep_current_position'),
        'trading_allowed': trading_allowed,
        'disabled_reason': None if trading_allowed else reason,
        'exposure': position_summary.get('inflight_exposure'),
        'redeemable_qty': position_summary.get('redeemable_qty'),
        'wallet_address': wallet_state.get('wallet_address'),
        'wallet_usdc_e': wallet_state.get('usdc_e_balance'),
        'wallet_pol': wallet_state.get('pol_balance'),
        'wallet_reserved_exposure': wallet_state.get('reserved_exposure_usdc'),
        'wallet_free_usdc': wallet_state.get('free_usdc'),
        'effective_bankroll': wallet_state.get('effective_bankroll'),
        'bankroll_source': wallet_state.get('bankroll_source'),
        'wallet_fetch_failed': wallet_state.get('fetch_failed'),
        'realized_pnl_by_entry_regime': storage.get_realized_pnl_by_entry_regime(),
        'routing_series_arg': routing_debug.get('series_arg'),
        'routing_bundle_present': routing_debug.get('routing_bundle_present'),
        'routing_runtime_status': routing_debug.get('runtime_status'),
        'routing_strike_price': routing_debug.get('strike_price'),
        'routing_epoch_key': routing_debug.get('epoch_key'),
        'routing_detection_source': routing_debug.get('detection_source'),
        'routing_reason': routing_debug.get('routing_reason'),
        'routing_candidate_slugs': routing_debug.get('candidate_slugs'),
        'routing_attempt_count': routing_debug.get('attempt_count'),
        'routing_attempts': routing_debug.get('attempts'),
        'routing_switched': routing_debug.get('switched'),
        'routing_missing_market_id': routing_debug.get('missing_market_id'),
        'routing_missing_token_yes': routing_debug.get('missing_token_yes'),
        'routing_missing_token_no': routing_debug.get('missing_token_no'),
        'shadow_probability_models': ctx.get('shadow_probability_models') or {},
    }
    heartbeat.update(extract_engine_shadow_diagnostics(ctx.get('engine_diagnostics')))
    return heartbeat


def build_market_switch_event(bundle: Optional[dict]) -> Optional[dict]:
    if not bundle or not bundle.get('switched'):
        return None
    return {
        'series_id': bundle.get('series_id'),
        'previous_market_id': bundle.get('previous_market_id'),
        'new_market_id': bundle.get('market_id'),
        'previous_token_yes': bundle.get('previous_token_yes'),
        'previous_token_no': bundle.get('previous_token_no'),
        'new_token_yes': bundle.get('token_yes'),
        'new_token_no': bundle.get('token_no'),
        'previous_strike_price': bundle.get('previous_strike_price'),
        'new_strike_price': bundle.get('strike_price'),
        'switch_ts': bundle.get('switch_ts'),
    }


def _record_market_switch(buffer: RollingEventBuffer, switch_event: Optional[dict], now) -> None:
    if switch_event is None:
        return
    buffer.record(
        'market_switch',
        ts=switch_event.get('switch_ts') or now,
        market_id=switch_event.get('new_market_id'),
        new_market_id=switch_event.get('new_market_id'),
        previous_market_id=switch_event.get('previous_market_id'),
    )


def _record_stale_actions(buffer: RollingEventBuffer, stale_actions: list[dict], now) -> None:
    for stale_action in stale_actions:
        action = stale_action.get('action')
        status = stale_action.get('status')
        if action == 'cancel_requested' or status == 'canceled':
            buffer.record('cancel', ts=now, status=status, reason=action)
        elif status in {'failed', 'rejected', 'unknown'}:
            buffer.record('error', ts=now, status=status, reason=action)


def _record_order_flow(buffer: RollingEventBuffer, payload: Optional[dict], now, *, live_execution: bool) -> None:
    if not isinstance(payload, dict):
        return
    status = str(payload.get('status') or '').lower()
    side = payload.get('side') or payload.get('action')
    qty = payload.get('filled_qty')
    if qty is None:
        qty = payload.get('filledQuantity')
    if qty is None:
        qty = payload.get('qty')
    price = payload.get('price') or payload.get('limit_price')
    if status in {'filled', 'partially_filled'}:
        buffer.record('fill', ts=now, status=status, side=side, qty=qty, price=price, real_execution=live_execution)
    elif status in {'accepted', 'submitted', 'open', 'ok'}:
        buffer.record('order_accept', ts=now, status=status, side=side, qty=qty, price=price, real_execution=live_execution)
    elif status in {'canceled', 'cancel_requested'}:
        buffer.record('cancel', ts=now, status=status, side=side, qty=qty, price=price, real_execution=live_execution)
    elif status in {'error', 'failed', 'rejected', 'unknown'}:
        buffer.record('error', ts=now, status=status, side=side, qty=qty, price=price)


def _record_strategy_action(buffer: RollingEventBuffer, action: Optional[dict], now, *, live_execution: bool) -> None:
    if not isinstance(action, dict):
        return
    action_name = action.get('side') or action.get('action')
    resp = action.get('resp')
    if action.get('side') in {'buy_yes', 'buy_no', 'sell_yes', 'sell_no'}:
        buffer.record('order_attempt', ts=now, action=action_name, side=action_name, qty=action.get('qty'), price=action.get('price'))
        _record_order_flow(buffer, resp, now, live_execution=live_execution)
    elif action_name == 'recycle_pair':
        for leg in action.get('legs') or []:
            leg_action = leg.get('action')
            buffer.record(
                'order_attempt',
                ts=now,
                action=leg_action,
                side=leg_action,
                qty=leg.get('submitted_qty') or leg.get('target_exit_qty'),
                price=leg.get('executable_exit_price'),
            )
            _record_order_flow(buffer, leg.get('resp'), now, live_execution=live_execution)
    elif str(action_name).startswith('skipped') or action_name is None:
        return
    else:
        status = action.get('status') or action_name
        if status in {'error', 'failed', 'rejected'}:
            buffer.record('error', ts=now, action=action_name, reason=action.get('reason') or status)


def maybe_manage_stale_orders(last_run_ts: Optional[pd.Timestamp], now: Optional[pd.Timestamp] = None, dry_run: bool = False, thresholds: Optional[dict] = None):
    now = now or pd.Timestamp.now(tz='UTC')
    if last_run_ts is not None and (now - last_run_ts).total_seconds() < STALE_ORDER_MAINTENANCE_SEC:
        return last_run_ts, []
    kwargs = {'now_ts': now.isoformat(), 'dry_run': dry_run}
    if thresholds is not None:
        kwargs['thresholds'] = thresholds
    actions = execution.manage_stale_orders(**kwargs)
    return now, actions


def recover_dirty_start() -> dict:
    report = {
        'db_path': str(storage.get_db_path()),
        'snapshot': storage.get_position_snapshot(),
        'open_orders': storage.get_open_orders(),
        'unreconciled_fills': storage.get_unreconciled_fills(),
        'pending_receipts': storage.get_pending_receipts(),
        'reservation_repairs': [],
        'tx_reconciliations': [],
        'stale_orders_for_review': [],
    }
    tx_hashes = set()
    for order in report['open_orders']:
        if order.get('tx_hash'):
            tx_hashes.add(order['tx_hash'])
    for fill in report['unreconciled_fills']:
        if fill.get('tx_hash'):
            tx_hashes.add(fill['tx_hash'])
    for tx_hash in sorted(tx_hashes):
        try:
            report['tx_reconciliations'].append({'tx_hash': tx_hash, 'result': storage.reconcile_tx(tx_hash)})
        except Exception as exc:
            report['tx_reconciliations'].append({'tx_hash': tx_hash, 'result': {'status': 'error', 'reason': str(exc)}})
    report['reservation_repairs'] = storage.repair_all_active_order_reservations()
    report['stale_orders_for_review'] = [
        {
            'order_id': order['id'],
            'client_order_id': order['client_order_id'],
            'status': order['status'],
            'remaining_qty': order['remaining_qty'],
            'tx_hash': order['tx_hash'],
            'updated_ts': order['updated_ts'],
        }
        for order in report['open_orders']
        if order['status'] in ('pending_submit', 'submitted', 'open', 'partially_filled', 'cancel_requested', 'unknown')
    ]
    return report


def enforce_startup_gate(allow_dirty_start: bool = False):
    ensure_db()
    require_clean_start = _env_flag('BOT_REQUIRE_CLEAN_START', False)
    allow_dirty = allow_dirty_start or _env_flag('BOT_ALLOW_DIRTY_START', False)
    status = storage.get_clean_start_status()
    if require_clean_start and not allow_dirty and not status['clean_start']:
        raise SystemExit(
            'Refusing startup: BOT_REQUIRE_CLEAN_START=true but DB is not clean '
            f'at {storage.get_db_path()}. '
            f'Status={status}'
        )
    if allow_dirty and not status['clean_start']:
        status['recovery_report'] = recover_dirty_start()
    return status


def main(backfill_limit: int = 1500, tau: int = 30, sims: int = 500, allow_dirty_start: bool = False, probability_engine: str = 'ar_egarch'):
    enforce_startup_gate(allow_dirty_start=allow_dirty_start)
    print('Backfilling recent klines...')
    prices = backfill_klines(limit=backfill_limit)
    print('Sample points:', len(prices))

    engine = build_probability_engine(probability_engine)
    engine.fit_history(prices)
    target = engine.current_spot() * 1.0005
    out = engine.predict(target, tau_minutes=tau, n_sims=sims)
    ts = datetime.now(timezone.utc).isoformat()
    print(f'[{ts}] p_yes={out["p_yes"]:.6f} target={target:.2f} engine={probability_engine}')

    # Print diagnostics for inspection
    diag = engine.get_diagnostics()
    print('\nModel diagnostics:')
    print('  engine_name:', diag.get('engine_name'))
    for key, value in diag.items():
        if key != 'engine_name':
            print(f'  {key}:', value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1500)
    parser.add_argument('--tau', type=int, default=30)
    parser.add_argument('--sims', type=int, default=500)
    parser.add_argument('--live', action='store_true', help='Run live WebSocket repricing loop')
    parser.add_argument('--duration', type=int, default=20, help='Live run duration in seconds')
    parser.add_argument('--series', type=str, default=None, help='Polymarket series id/slug for event discovery')
    parser.add_argument('--token-yes', type=str, default=None, help='YES token id for the current market')
    parser.add_argument('--token-no', type=str, default=None, help='NO token id for the current market')
    parser.add_argument('--market-id', type=str, default=None, help='Market id (used for lot tracking)')
    parser.add_argument('--allow-dirty-start', action='store_true', help='Override clean-start guard for dev/test use only')
    parser.add_argument(
        '--probability-engine',
        type=str,
        default=get_default_probability_engine_name(),
        choices=[
            'ar_egarch',
            'gaussian_vol',
            'kalman_blended_sigma_v1_cfg1',
            'gaussian_pde_diffusion_kalman_v1_cfg1',
            'lgbm',
        ],
    )
    args = parser.parse_args()

    if not args.live:
        main(
            backfill_limit=args.limit,
            tau=args.tau,
            sims=args.sims,
            allow_dirty_start=args.allow_dirty_start,
            probability_engine=args.probability_engine,
        )
    else:
        async def live_run():
            enforce_startup_gate(allow_dirty_start=args.allow_dirty_start)
            console_mode = get_live_console_mode()
            heartbeat_sec = get_live_heartbeat_sec(console_mode)
            print('Backfilling recent klines...')
            prices = backfill_klines(limit=args.limit)
            print('Sample points:', len(prices))

            engine = build_probability_engine(args.probability_engine)
            engine.fit_history(prices)
            shadow_engines = build_shadow_probability_engines(
                primary_engine_name=args.probability_engine,
                fit_prices=prices,
            )
            ensure_db()
            print('Polymarket LIVE mode:', POLY_LIVE)
            print('Execution enabled:', EXECUTION_ENABLED)
            startup_recovery = execution.recover_active_orders_on_startup(dry_run=(not EXECUTION_ENABLED))
            print(format_startup_recovery_summary(startup_recovery))
            dust_review = collect_startup_dust_review()
            print(format_startup_dust_report(dust_review))
            dust_action = str(os.getenv('BOT_STARTUP_DUST_ACTION', 'keep_all_dormant') or 'keep_all_dormant').strip().lower()
            if dust_action in {'keep_all_dormant', 'finalize_all'} and dust_review.get('items'):
                apply_startup_dust_action(dust_review, action=dust_action)
                dust_review = collect_startup_dust_review()
                print(format_startup_dust_report(dust_review))

            # shared state
            state_lock = asyncio.Lock()
            last_stale_maintenance = {'ts': None}
            heartbeat_events = RollingEventBuffer(window_seconds=max(60, heartbeat_sec))
            last_heartbeat_ts = {'ts': None}
            last_critical_reason = {'value': None}
            settlement_cycle = {'count': 0}

            async def repricer():
                end_time = asyncio.get_event_loop().time() + args.duration
                while asyncio.get_event_loop().time() < end_time:
                    skip_iteration = False
                    shadow_probability_models = {}
                    heartbeat = None
                    action = None
                    should_emit_heartbeat = False
                    heartbeat_now = None
                    heartbeat_event_summary = None
                    decision_log_record = None
                    shadow_eval_args = None
                    async with state_lock:
                        storage.sweep_expired_open_markets()
                        settlement_cycle['count'] += 1
                        if settlement_cycle['count'] % INVENTORY_SETTLEMENT_CADENCE == 0:
                            settlement_results = settle_inventory_candidates(
                                dry_run=(not EXECUTION_ENABLED),
                                checked_ts=datetime.now(timezone.utc).isoformat(),
                            )
                            debug_mode = is_debug_mode(console_mode)
                            for settlement in settlement_results:
                                settlement_status = settlement.get('status')
                                tone = 'info'
                                if settlement_status in {'redeemed', 'finalized_loss'}:
                                    tone = 'success'
                                elif settlement_status == 'failed':
                                    tone = 'error'
                                elif settlement.get('skip_reason'):
                                    tone = 'warning'
                                emit_console_status(
                                    'settle',
                                    console_mode=console_mode,
                                    tone=tone,
                                    market_id=settlement.get('market_id'),
                                    qty=settlement.get('redeemable_qty'),
                                    status=settlement_status,
                                    reason=settlement.get('reason') or settlement.get('skip_reason'),
                                    resp=settlement if debug_mode else None,
                                )
                        now = pd.Timestamp.now(tz='UTC')
                        live_market_ctx = resolve_live_market_context(args, now=now)
                        routing_bundle = live_market_ctx['routing_bundle']
                        routing_debug = live_market_ctx.get('routing_debug') or {}
                        debug_mode = is_debug_mode(console_mode)
                        switch_event = build_market_switch_event(routing_bundle)
                        if switch_event is not None:
                            if debug_mode:
                                emit_console_status('switch', console_mode=console_mode, tone='info', market=switch_event.get('new_market_id'), prev=switch_event.get('previous_market_id'), resp=switch_event)
                            else:
                                emit_console_status('switch', console_mode=console_mode, tone='info', market=switch_event.get('new_market_id'), prev=switch_event.get('previous_market_id'))
                        _record_market_switch(heartbeat_events, switch_event, now)

                        market_meta = live_market_ctx['market_meta']
                        token_yes = live_market_ctx['token_yes']
                        token_no = live_market_ctx['token_no']
                        market_id = live_market_ctx['market_id']
                        yes_quote = live_market_ctx['yes_quote']
                        no_quote = live_market_ctx['no_quote']
                        if (routing_debug.get('series_arg') is not None) and routing_bundle is None:
                            emit_console_status(
                                'error',
                                console_mode=console_mode,
                                tone='error',
                                market=market_id,
                                reason='active_market_routing_unavailable',
                                series=routing_debug.get('series_arg'),
                                resp=routing_debug if debug_mode else {'routing_reason': routing_debug.get('routing_reason'), 'runtime_status': routing_debug.get('runtime_status')},
                            )

                        wallet_snapshot = fetch_wallet_state(storage=storage)
                        ctx = build_trade_context(
                            market_meta,
                            yes_quote,
                            no_quote,
                            now=now,
                            routing_bundle=routing_bundle,
                            wallet_state=wallet_snapshot,
                        )
                        ctx['routing_debug'] = routing_debug
                        series_mode_active = routing_debug.get('series_arg') is not None
                        probability_state = compute_market_probabilities(routing_bundle, engine, now=now, n_sims=args.sims) if series_mode_active else None
                        engine_diagnostics = engine.get_diagnostics() if series_mode_active and hasattr(engine, 'get_diagnostics') else None
                        price_history = build_recent_price_history(engine) if series_mode_active else pd.Series(dtype=float)
                        microstructure_price_history = (
                            build_recent_price_history(engine, limit=_microstructure_price_history_limit())
                            if series_mode_active
                            else pd.Series(dtype=float)
                        )
                        microstructure_state = compute_microstructure_regime(microstructure_price_history) if series_mode_active else None
                        entry_context = (
                            {
                                'reversal_evidence_by_side': _entry_reversal_evidence_by_side(
                                    price_history=price_history,
                                    strike_price=None if probability_state is None else probability_state.get('strike_price'),
                                    spot_now=None if probability_state is None else probability_state.get('spot_now'),
                                ),
                            }
                            if series_mode_active
                            else None
                        )
                        decision_state = build_market_decision_state(
                            routing_bundle,
                            probability_state,
                            wallet_state=wallet_snapshot,
                            entry_context=entry_context,
                            microstructure_state=microstructure_state,
                        ) if series_mode_active else None
                        if probability_state is not None:
                            ctx['probability_state'] = probability_state
                        if engine_diagnostics is not None:
                            ctx['engine_diagnostics'] = engine_diagnostics
                        if decision_state is not None:
                            decision_state['wallet_state'] = wallet_snapshot
                            ctx['position_management_state'] = storage.get_position_management_state(market_id) if market_id else None
                            decision_state.update({
                                'position_reeval_enabled': False,
                                'position_reeval_action': 'hold',
                                'position_reeval_reason': 'position_reeval_disabled',
                                'position_reeval': None,
                            })
                            ctx['decision_state'] = decision_state
                            ctx['policy'] = decision_state.get('policy') or {}
                        live_trading_allowed_preview = False
                        live_disabled_reason_preview = 'ok'
                        if decision_state is not None:
                            live_trading_allowed_preview, live_disabled_reason_preview = compute_effective_decision_trade_state(ctx, decision_state)
                        if shadow_engines and series_mode_active and decision_state is not None:
                            shadow_eval_args = {
                                'shadow_engines': dict(shadow_engines),
                                'bundle': routing_bundle,
                                'now': now,
                                'n_sims': args.sims,
                                'wallet_state': wallet_snapshot,
                                'trade_context': dict(ctx),
                                'entry_context': entry_context,
                                'microstructure_state': microstructure_state,
                                'live_decision_state': decision_state,
                                'live_trade_allowed': live_trading_allowed_preview,
                            }
                        maintenance_thresholds = None
                        if ctx.get('policy'):
                            cancel_open_after = int(ctx['policy'].get('cancel_open_orders_after_sec', STALE_ORDER_MAINTENANCE_SEC))
                            maintenance_thresholds = {
                                'max_open_age_sec': cancel_open_after,
                                'max_pending_submit_age_sec': max(15, cancel_open_after // 3),
                                'cancel_retry_sec': max(10, cancel_open_after // 4),
                            }
                        maintenance_ts, stale_actions = maybe_manage_stale_orders(
                            last_stale_maintenance['ts'],
                            now=now,
                            dry_run=(not EXECUTION_ENABLED),
                            thresholds=maintenance_thresholds,
                        )
                        last_stale_maintenance['ts'] = maintenance_ts
                        if stale_actions:
                            for stale_action in stale_actions:
                                tone = 'error' if stale_action.get('status') in {'failed', 'rejected', 'unknown'} else 'warning'
                                emit_console_status(
                                    'stale',
                                    console_mode=console_mode,
                                    tone=tone,
                                    order_id=stale_action.get('order_id'),
                                    action=stale_action.get('action'),
                                    status=stale_action.get('status'),
                                    resp=stale_action if debug_mode else None,
                                )
                        _record_stale_actions(heartbeat_events, stale_actions, now)
                        trading_allowed, disabled_reason = compute_effective_decision_trade_state(ctx, decision_state)
                        severe_disabled_reasons = {'quote_fetch_failed', 'quote_stale', 'quote_crossed', 'quote_empty', 'market_detection_failed', 'incomplete_market_metadata'}
                        if (
                            not trading_allowed
                            and disabled_reason in severe_disabled_reasons
                            and last_critical_reason['value'] != disabled_reason
                        ):
                            emit_console_status('error', console_mode=console_mode, tone='error', market=market_id, reason=disabled_reason)
                            last_critical_reason['value'] = disabled_reason
                        elif trading_allowed:
                            last_critical_reason['value'] = None
                        if decision_state is not None:
                            decision_log_record = {
                                'decision_state': decision_state,
                                'trading_allowed': trading_allowed,
                                'disabled_reason': disabled_reason,
                                'wallet_snapshot': wallet_snapshot,
                                'routing_debug': routing_debug,
                                'engine_diagnostics': engine_diagnostics,
                            }
                        if not trading_allowed:
                            skip_iteration = True

                        if skip_iteration:
                            action = None
                        else:
                            ts = datetime.now(timezone.utc).isoformat()
                            if probability_state is not None and debug_mode:
                                emit_console_status(
                                    'model',
                                    console_mode=console_mode,
                                    tone='info',
                                    p_yes=probability_state["p_yes"],
                                    strike=probability_state["strike_price"],
                                    spot=probability_state["spot_now"],
                                    tau=probability_state["tau_minutes"],
                                    ts=ts,
                                )
                            # decide and execute (dry-run when POLY_LIVE is False)
                            try:
                                if decision_state is not None:
                                    action = build_trade_action(
                                        decision_state,
                                        token_yes,
                                        token_no,
                                        market_id=market_id,
                                        dry_run=(not EXECUTION_ENABLED),
                                        wallet_state=wallet_snapshot,
                                    )
                                else:
                                    action = decide_and_execute(
                                        0.5,
                                        yes_quote.get('mid'),
                                        token_yes,
                                        token_no,
                                        market_id=market_id,
                                        dry_run=(not EXECUTION_ENABLED),
                                        wallet_state=wallet_snapshot,
                                    )
                            except Exception as e:
                                emit_console_status('error', console_mode=console_mode, tone='error', market=market_id, reason='strategy_error', detail=str(e))
                                heartbeat_events.record('error', ts=now, reason='strategy_error')
                                action = None
                        should_emit_heartbeat = should_emit_console_heartbeat(last_heartbeat_ts['ts'], now, console_mode)
                        heartbeat_now = now
                        heartbeat = build_diagnostic_heartbeat(ctx, trading_allowed, disabled_reason)
                    if shadow_eval_args is not None:
                        shadow_probability_models = evaluate_shadow_probability_models(**shadow_eval_args)
                    if shadow_probability_models and heartbeat is not None:
                        heartbeat['shadow_probability_models'] = shadow_probability_models
                    if decision_log_record is not None:
                        append_decision_log(
                            build_decision_log_record(
                                decision_state=decision_log_record['decision_state'],
                                trading_allowed=decision_log_record['trading_allowed'],
                                disabled_reason=decision_log_record['disabled_reason'],
                                wallet_snapshot=decision_log_record['wallet_snapshot'],
                                routing_debug=decision_log_record['routing_debug'],
                                engine_diagnostics=decision_log_record['engine_diagnostics'],
                                shadow_probability_models=shadow_probability_models,
                            )
                        )
                    if action is not None:
                        emit_console_action(action, console_mode=console_mode)
                    _record_strategy_action(heartbeat_events, action, heartbeat_now, live_execution=EXECUTION_ENABLED)
                    if should_emit_heartbeat and heartbeat is not None:
                        heartbeat_event_summary = heartbeat_events.summarize(now=heartbeat_now, since=last_heartbeat_ts['ts'])
                    if should_emit_heartbeat and heartbeat is not None and heartbeat_event_summary is not None:
                        rendered_heartbeat = format_heartbeat(
                            heartbeat,
                            heartbeat_event_summary,
                            inventory_summary=get_console_inventory_summary(),
                            now=heartbeat_now,
                            mode=console_mode,
                        )
                        print(rendered_heartbeat.styled_text())
                        last_heartbeat_ts['ts'] = heartbeat_now
                    await asyncio.sleep(1.0)

            # run consumer and repricer concurrently
            await asyncio.gather(
                consume_binance_klines(
                    live_engine=engine,
                    shadow_engines=list(shadow_engines.values()),
                    state_lock=state_lock,
                    duration=args.duration,
                    console_mode=console_mode,
                ),
                repricer(),
            )

        asyncio.run(live_run())
