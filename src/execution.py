"""Paper-trade executor for Polymarket CLOB using storage for persistence.
This is a simplified executor that records orders and instantly fills them for simulation.
"""
from . import storage
import json
from datetime import datetime, timezone
from . import polymarket_client
from typing import Optional
import os
from datetime import datetime, timedelta, timezone


ACTIVE_ORDER_STATUSES = storage.ACTIVE_ORDER_STATUSES
ORDER_MAX_OPEN_AGE_SEC = int(os.getenv('ORDER_MAX_OPEN_AGE_SEC', '300'))
ORDER_MAX_PENDING_SUBMIT_AGE_SEC = int(os.getenv('ORDER_MAX_PENDING_SUBMIT_AGE_SEC', '60'))
ORDER_CANCEL_RETRY_SEC = int(os.getenv('ORDER_CANCEL_RETRY_SEC', '30'))
ORDER_UNKNOWN_REVIEW_AGE_SEC = int(os.getenv('ORDER_UNKNOWN_REVIEW_AGE_SEC', '1800'))
SUPPORTED_MARKETABLE_ORDER_TYPES = {'FAK', 'FOK'}
_CLIENT_ORDER_SEQ = 0
place_limit_order = polymarket_client.place_limit_order
place_marketable_order = polymarket_client.place_marketable_order
get_order_status = polymarket_client.get_order_status
cancel_order = polymarket_client.cancel_order


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_client_order_id(prefix: str = 'ord', market_id: Optional[str] = None, side: Optional[str] = None, outcome_side: Optional[str] = None) -> str:
    global _CLIENT_ORDER_SEQ
    _CLIENT_ORDER_SEQ += 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    parts = [prefix, stamp, f'{_CLIENT_ORDER_SEQ:04d}']
    if market_id:
        parts.append(str(market_id))
    if side:
        parts.append(str(side).lower())
    if outcome_side:
        parts.append(str(outcome_side).upper())
    return '-'.join(parts)


def get_marketable_order_type() -> str:
    order_type = str(os.getenv('MARKETABLE_ORDER_TYPE', 'FAK') or 'FAK').upper()
    if order_type not in SUPPORTED_MARKETABLE_ORDER_TYPES:
        raise ValueError(
            f"unsupported MARKETABLE_ORDER_TYPE: {order_type} "
            f"(expected one of {sorted(SUPPORTED_MARKETABLE_ORDER_TYPES)})"
        )
    return order_type


def _extract_cumulative_filled(resp: dict, fallback_qty: float = 0.0) -> float:
    if 'filled_qty' in resp and resp.get('filled_qty') is not None:
        return float(resp['filled_qty'] or 0.0)
    if 'filledQuantity' in resp:
        return float(resp['filledQuantity'] or 0.0)
    if 'fills' in resp and isinstance(resp['fills'], list):
        return sum(
            float(
                f.get('qty')
                if f.get('qty') is not None
                else f.get('size')
                if f.get('size') is not None
                else f.get('quantity')
                if f.get('quantity') is not None
                else 0.0
            )
            for f in resp['fills']
        )
    if resp.get('status') == 'filled':
        return float(fallback_qty)
    return 0.0


def _extract_remaining_qty(resp: dict) -> Optional[float]:
    for key in ('remaining_qty', 'remainingQuantity', 'leavesQty'):
        if key in resp and resp.get(key) is not None:
            return float(resp[key])
    return None


def _clamp_cumulative_fill(requested_qty: float, cumulative_filled: float):
    requested = float(requested_qty or 0.0)
    cumulative = float(cumulative_filled or 0.0)
    if requested <= 0 or cumulative <= requested:
        return cumulative, None
    return requested, {
        'fill_clamped': True,
        'fill_clamp_from': cumulative,
        'fill_clamp_to': requested,
        'fill_clamp_excess': cumulative - requested,
    }


def _get_order_fill_evidence(order_id: int) -> float:
    fill_events = storage.get_order_fill_events(order_id)
    return max((float(event['cumulative_filled_qty']) for event in fill_events), default=0.0)


def _is_empty_not_found_response(normalized: dict, status: str, remaining_qty: Optional[float], cumulative_filled: float) -> bool:
    if status != 'not_found_on_venue':
        return False
    if remaining_qty is not None:
        return False
    if cumulative_filled > 1e-12:
        return False
    if normalized.get('filled_qty') is not None or normalized.get('filledQuantity') is not None:
        return False
    fills = normalized.get('fills')
    if isinstance(fills, list) and fills:
        return False
    return True


def _extract_tx_hash(resp: dict) -> Optional[str]:
    tx_hash = resp.get('tx_hash') or resp.get('txHash') or resp.get('transactionHash')
    if tx_hash:
        return tx_hash
    return None


def _extract_venue_order_id(resp: dict) -> Optional[str]:
    return resp.get('order_id') or resp.get('orderId') or resp.get('id')


def _call_submit_helper(helper, *args, client_order_id: Optional[str] = None, **kwargs):
    call_kwargs = dict(kwargs)
    if client_order_id is not None:
        call_kwargs['client_order_id'] = client_order_id
    try:
        return helper(*args, **call_kwargs)
    except TypeError as exc:
        message = str(exc)
        if 'compiled_intent' in message and 'compiled_intent' in call_kwargs:
            retry_kwargs = dict(call_kwargs)
            retry_kwargs.pop('compiled_intent', None)
            return _call_submit_helper(helper, *args, **retry_kwargs)
        if 'client_order_id' not in message or 'client_order_id' not in call_kwargs:
            raise
        retry_kwargs = dict(call_kwargs)
        retry_kwargs.pop('client_order_id', None)
        return helper(*args, **retry_kwargs)


def _is_ambiguous_submit_response(resp: dict) -> bool:
    if not isinstance(resp, dict):
        return True
    confirmed_id = _extract_venue_order_id(resp) or _extract_tx_hash(resp)
    status = str(resp.get('status') or '').lower()
    http_status = resp.get('http_status')
    cumulative_filled = _extract_cumulative_filled(resp, fallback_qty=0.0)
    if confirmed_id:
        return False
    if cumulative_filled > 0:
        return False
    if status in ('unknown', 'invalid_response', 'no_response'):
        return True
    if http_status is None:
        return True
    if status == 'error' and int(http_status) >= 500:
        return True
    return False


def _parse_trade_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        normalized = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _extract_trade_qty(trade: dict) -> float:
    for key in ('size', 'qty', 'quantity'):
        if trade.get(key) is not None:
            try:
                return float(trade.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _extract_trade_price(trade: dict) -> Optional[float]:
    for key in ('price', 'avg_price', 'averagePrice'):
        if trade.get(key) is not None:
            try:
                return float(trade.get(key))
            except (TypeError, ValueError):
                return None
    return None


def _recover_buy_from_trade_history(order: dict, *, dry_run: bool = False) -> Optional[dict]:
    if dry_run or order.get('side') != 'buy' or not order.get('client_order_id'):
        return None
    if str(order.get('status') or '').lower() not in ('unknown', 'not_found_on_venue'):
        return None
    if storage.get_order_fill_events(order['id']):
        return None
    trades_resp = polymarket_client.get_user_trades(order.get('market_id'), order.get('token_id'), dry_run=dry_run)
    trades = trades_resp.get('trades') or []
    if not isinstance(trades, list) or not trades:
        return None
    created_ts = _parse_trade_ts(order.get('created_ts'))
    lower_bound = created_ts - timedelta(minutes=10) if created_ts else None
    upper_bound = created_ts + timedelta(minutes=10) if created_ts else None
    candidate_trades = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        trade_ts = _parse_trade_ts(
            trade.get('timestamp')
            or trade.get('match_time')
            or trade.get('created_at')
            or trade.get('createdAt')
            or trade.get('traded_at')
            or trade.get('time')
        )
        if lower_bound and trade_ts and trade_ts < lower_bound:
            continue
        if upper_bound and trade_ts and trade_ts > upper_bound:
            continue
        qty = _extract_trade_qty(trade)
        if qty <= 0:
            continue
        candidate_trades.append((trade_ts or created_ts, trade))
    if not candidate_trades:
        return None
    candidate_trades.sort(key=lambda item: item[0] or datetime.now(timezone.utc))
    requested_qty = float(order.get('requested_qty') or 0.0)
    cumulative = 0.0
    weighted_notional = 0.0
    matched = []
    for _, trade in candidate_trades:
        qty = min(_extract_trade_qty(trade), max(0.0, requested_qty - cumulative))
        if qty <= 0:
            continue
        price = _extract_trade_price(trade) or order.get('limit_price') or 0.0
        cumulative += qty
        weighted_notional += qty * float(price)
        matched.append(trade)
        if cumulative >= requested_qty - 1e-12:
            break
    if cumulative <= 0:
        return None
    avg_price = weighted_notional / cumulative if cumulative > 0 else order.get('limit_price')
    return {
        'status': 'filled' if cumulative >= requested_qty - 1e-12 else 'partially_filled',
        'filled_qty': cumulative,
        'avg_price': avg_price,
        'price': avg_price,
        'client_order_id': order.get('client_order_id'),
        'trades': matched,
        'recovered_via': 'user_trade_history',
    }


def normalize_order_status(raw_status, raw_payload: Optional[dict] = None, *, requested_qty: float = 0.0, cumulative_filled: float = 0.0, dry_run: bool = False) -> str:
    raw_payload = raw_payload or {}
    if dry_run or raw_payload.get('dry_run'):
        return 'filled'
    status_raw = str(raw_status or raw_payload.get('status') or '').lower()
    order_type = str(raw_payload.get('order_type') or raw_payload.get('orderType') or '').upper()
    marketable = bool(raw_payload.get('marketable'))
    if cumulative_filled >= requested_qty - 1e-12 and requested_qty > 0:
        return 'filled'
    if marketable and order_type in SUPPORTED_MARKETABLE_ORDER_TYPES and cumulative_filled > 0:
        return 'canceled'
    if status_raw in ('error', 'failed'):
        return 'failed'
    if status_raw == 'rejected':
        return 'rejected'
    if status_raw in ('canceled', 'cancelled'):
        return 'canceled'
    if status_raw == 'expired':
        return 'expired'
    if status_raw in ('cancel_requested', 'cancelrequest', 'pending_cancel'):
        return 'cancel_requested'
    if status_raw in ('not_found_on_venue', 'missing_on_venue', 'unknown_order_id'):
        return 'not_found_on_venue'
    if status_raw in ('submitted', 'accepted', 'received'):
        return 'submitted'
    if status_raw == 'matched':
        if cumulative_filled > 0:
            return 'partially_filled'
        return 'submitted'
    if status_raw in ('open', 'resting', 'live'):
        return 'open'
    if status_raw in ('partial', 'partially_filled'):
        return 'partially_filled'
    if status_raw in ('filled', 'completed'):
        return 'filled'
    if cumulative_filled > 0:
        return 'partially_filled'
    if status_raw == 'ok' and (raw_payload.get('order_id') or raw_payload.get('orderId') or raw_payload.get('id')):
        return 'submitted'
    if status_raw in ('unknown', ''):
        return 'unknown'
    return 'unknown'


def _legalize_transition(order: dict, target_state: str) -> str:
    current_state = order['status']
    if storage.can_transition_order_state(current_state, target_state):
        return target_state
    if target_state in storage.TERMINAL_ORDER_STATUSES:
        return current_state
    if storage.can_transition_order_state(current_state, 'unknown'):
        return 'unknown'
    return current_state


def _should_escalate_unknown_order(order: dict, refreshed: Optional[dict], *, age_sec: float, review_age_sec: int) -> bool:
    if str(order.get('status') or '').lower() != 'unknown':
        return False
    if age_sec < float(review_age_sec):
        return False
    if order.get('venue_order_id'):
        return False
    if float(order.get('filled_qty') or 0.0) > 1e-12:
        return False
    if storage.get_order_fill_events(order['id']):
        return False
    response = (refreshed or {}).get('response') or {}
    if response.get('recovered_via') in {'tx_receipt_reconciliation', 'user_trade_history'}:
        return False
    return True


def _parse_iso_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        normalized = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _order_raw_response(order: dict) -> dict:
    try:
        return json.loads(order.get('raw_response_json') or '{}')
    except Exception:
        return {}


def _payload_order_type(payload: Optional[dict]) -> str:
    if not isinstance(payload, dict):
        return ''
    return str(payload.get('order_type') or payload.get('orderType') or '').upper()


def _payload_policy(payload: Optional[dict]) -> str:
    if not isinstance(payload, dict):
        return ''
    return str(payload.get('policy') or '').strip().lower()


def _raw_status(payload: Optional[dict]) -> str:
    if not isinstance(payload, dict):
        return ''
    return str(payload.get('status') or '').strip().lower()


def _is_marketable_buy_order(order: dict, normalized: dict) -> bool:
    if str(order.get('side') or '').lower() != 'buy':
        return False
    existing = _order_raw_response(order)
    payloads = [normalized, existing]
    if any(bool(payload.get('marketable')) for payload in payloads if isinstance(payload, dict)):
        return True
    if any(_payload_order_type(payload) in SUPPORTED_MARKETABLE_ORDER_TYPES for payload in payloads):
        return True
    if any(_payload_policy(payload).startswith('marketable_') for payload in payloads):
        return True
    if any(payload.get('venue_intent_mode') for payload in payloads if isinstance(payload, dict)):
        return True
    return False


def _residual_terminal_state(order: dict, normalized: dict, status: str, cumulative_filled: float) -> Optional[str]:
    if float(order.get('requested_qty') or 0.0) > 0 and cumulative_filled >= float(order.get('requested_qty') or 0.0) - 1e-12:
        return None
    if not _is_marketable_buy_order(order, normalized):
        return None
    if str(order.get('side') or '').lower() != 'buy' or cumulative_filled <= 1e-12:
        return None
    raw_status = _raw_status(normalized)
    if raw_status in ('not_found_on_venue', 'missing_on_venue', 'unknown_order_id'):
        return 'not_found_on_venue'
    if raw_status in ('canceled', 'cancelled'):
        return 'canceled'
    if raw_status == 'expired':
        return 'expired'
    if status == 'not_found_on_venue':
        return 'not_found_on_venue'
    if status == 'canceled':
        return 'canceled'
    if status == 'expired':
        return 'expired'
    if raw_status == 'matched':
        return 'canceled'
    return None


def _should_terminalize_buy_residual(order: dict, normalized: dict, cumulative_filled: float, status: str) -> bool:
    return _residual_terminal_state(order, normalized, status, cumulative_filled) is not None


def _create_order_and_reservation(side: str, qty: float, price: float, token_id: str, market_id: str, outcome_side: str, reservation_qty: float, reservation_type: str, client_order_id: Optional[str] = None, decision_context: Optional[dict] = None) -> dict:
    ts = _now_ts()
    client_order_id = client_order_id or _new_client_order_id('paper' if reservation_type == 'paper' else 'live', market_id=market_id, side=side, outcome_side=outcome_side)
    existing = storage.get_order(client_order_id=client_order_id)
    if existing is not None:
        return existing
    order = storage.create_order(
        client_order_id=client_order_id,
        market_id=market_id,
        token_id=token_id,
        outcome_side=outcome_side,
        side=side.lower(),
        requested_qty=qty,
        limit_price=price,
        status='pending_submit',
        created_ts=ts,
        decision_context=decision_context,
    )
    if reservation_qty > 0:
        storage.create_reservation(order['id'], market_id, token_id if reservation_type == 'inventory' else token_id, outcome_side, reservation_type, reservation_qty, ts)
    return order


def _merge_submission_intent(resp: dict, intent: Optional[dict]) -> dict:
    if not intent:
        return resp
    merged = dict(resp or {})
    merged['quantization'] = intent.get('quantization')
    merged.update(intent.get('quantization') or {})
    merged['venue_intent_mode'] = intent.get('venue_intent_mode')
    merged['submitted_qty'] = intent.get('submitted_qty')
    merged['submitted_notional'] = intent.get('submitted_notional')
    merged['submitted_amount'] = intent.get('submitted_notional') if intent.get('venue_intent_mode') == 'amount' else None
    return merged


def _process_order_response(order_id: int, requested_qty: float, resp: dict, dry_run: bool = False):
    ts = _now_ts()
    normalized = polymarket_client.normalize_client_response(resp, default_status='unknown')
    cumulative_filled = _extract_cumulative_filled(normalized, fallback_qty=requested_qty)
    if dry_run and cumulative_filled <= 0:
        cumulative_filled = float(requested_qty)
    cumulative_filled, clamp_meta = _clamp_cumulative_fill(requested_qty, cumulative_filled)
    if clamp_meta is not None:
        normalized = {**normalized, **clamp_meta}
    tx_hash = _extract_tx_hash(normalized)
    venue_order_id = _extract_venue_order_id(normalized)
    order = storage.get_order(order_id=order_id)
    if tx_hash is None and order.get('tx_hash'):
        tx_hash = order['tx_hash']
    status = normalize_order_status(normalized.get('status'), normalized, requested_qty=requested_qty, cumulative_filled=cumulative_filled, dry_run=dry_run)
    remaining_qty = _extract_remaining_qty(normalized)
    ledger_cumulative = max(float(order.get('filled_qty') or 0.0), _get_order_fill_evidence(order_id))
    empty_not_found = _is_empty_not_found_response(normalized, status, remaining_qty, cumulative_filled)
    if empty_not_found and ledger_cumulative > cumulative_filled + 1e-12:
        cumulative_filled = ledger_cumulative
        if cumulative_filled >= float(requested_qty or 0.0) - 1e-12 and float(requested_qty or 0.0) > 0:
            status = 'filled'
        elif cumulative_filled > 1e-12:
            status = 'partially_filled'
        remaining_qty = max(0.0, float(requested_qty or 0.0) - cumulative_filled)
        normalized = {
            **normalized,
            'ledger_fill_preserved': True,
            'ledger_fill_preserved_qty': cumulative_filled,
        }
    elif remaining_qty is None:
        remaining_qty = max(0.0, requested_qty - cumulative_filled)
    residual_terminal_state = _residual_terminal_state(order, normalized, status, cumulative_filled)
    if residual_terminal_state is not None:
        status = residual_terminal_state
        normalized = {
            **normalized,
            'residual_terminalized': True,
            'residual_terminalization_reason': 'marketable_buy_dead_residual',
            'ledger_fill_preserved': True,
            'ledger_fill_preserved_qty': cumulative_filled,
        }
    status = _legalize_transition(order, status)

    storage.update_order(order_id, venue_order_id=venue_order_id, tx_hash=tx_hash, raw_response=normalized, updated_ts=ts)
    apply_res = storage.apply_incremental_order_fill(order_id, cumulative_filled, fill_ts=ts, tx_hash=tx_hash, price=order['limit_price'], raw=normalized)
    if empty_not_found and ledger_cumulative > float(order.get('filled_qty') or 0.0) + 1e-12 and apply_res.get('applied_qty', 0.0) <= 0:
        storage.update_order(order_id, filled_qty=ledger_cumulative, remaining_qty=remaining_qty, updated_ts=ts)
        apply_res = {**apply_res, 'order': storage.get_order(order_id=order_id)}
    if apply_res.get('applied_qty', 0.0) > 0:
        storage.append_order_event(
            order_id,
            'fill_applied',
            old_status=order['status'],
            new_status=order['status'],
            response=normalized,
            ts=ts,
        )
    storage.update_order(order_id, remaining_qty=remaining_qty, updated_ts=ts)
    if status != order['status']:
        storage.transition_order_state(order_id, status, raw=normalized, ts=ts)
    storage.repair_order_reservations(order_id, updated_ts=ts)
    return {'order': storage.get_order(order_id=order_id), 'applied': apply_res, 'response': normalized}


def process_order_update(order_id: int, resp: dict):
    order = storage.get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    return _process_order_response(order_id, order['requested_qty'], resp, dry_run=False)


def refresh_order_status(order_id: int, dry_run: bool = False):
    order = storage.get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    if not order.get('venue_order_id') and not order.get('client_order_id'):
        storage.append_order_event(order_id, 'refresh_before', old_status=order['status'], request={'dry_run': dry_run}, ts=_now_ts())
        storage.transition_order_state(order_id, _legalize_transition(order, 'unknown'), reason='missing_order_identifier_for_refresh', ts=_now_ts())
        storage.repair_order_reservations(order_id, updated_ts=_now_ts())
        storage.append_order_event(order_id, 'refresh_after', old_status=order['status'], new_status='unknown', error_text='missing_order_identifier_for_refresh', ts=_now_ts())
        return {'order': storage.get_order(order_id=order_id), 'response': None}
    storage.append_order_event(
        order_id,
        'refresh_before',
        old_status=order['status'],
        request={'dry_run': dry_run, 'venue_order_id': order.get('venue_order_id'), 'client_order_id': order.get('client_order_id')},
        ts=_now_ts(),
    )
    if not dry_run and order.get('tx_hash'):
        reconciliation = storage.reconcile_tx(order['tx_hash'])
        current = storage.get_order(order_id=order_id)
        if current and current['filled_qty'] > order['filled_qty']:
            response = {
                'status': current['status'],
                'tx_hash': order['tx_hash'],
                'filled_qty': current['filled_qty'],
                'recovered_via': 'tx_receipt_reconciliation',
                'reconciliation': reconciliation,
            }
            out = {'order': current, 'applied': {'applied_qty': current['filled_qty'] - order['filled_qty'], 'order': current, 'duplicate': False}, 'response': response}
            storage.append_order_event(order_id, 'refresh_after', old_status=order['status'], new_status=current['status'], response=response, ts=_now_ts())
            return out
    recovered = _recover_buy_from_trade_history(order, dry_run=dry_run)
    if recovered is not None:
        out = _process_order_response(order_id, order['requested_qty'], recovered, dry_run=dry_run)
        storage.append_order_event(order_id, 'refresh_after', old_status=order['status'], new_status=out['order']['status'], response=out['response'], ts=_now_ts())
        return out
    if order.get('venue_order_id'):
        resp = get_order_status(order_id=order.get('venue_order_id'), dry_run=dry_run)
    else:
        resp = get_order_status(client_order_id=order.get('client_order_id'), dry_run=dry_run)
    out = _process_order_response(order_id, order['requested_qty'], resp, dry_run=dry_run)
    storage.append_order_event(order_id, 'refresh_after', old_status=order['status'], new_status=out['order']['status'], response=out['response'], ts=_now_ts())
    return out


def cancel_and_reconcile_order(order_id: int, dry_run: bool = False):
    order = storage.get_order(order_id=order_id)
    if order is None:
        raise RuntimeError(f'Order {order_id} not found')
    storage.append_order_event(
        order_id,
        'cancel_before',
        old_status=order['status'],
        request={'dry_run': dry_run, 'venue_order_id': order.get('venue_order_id'), 'client_order_id': order.get('client_order_id')},
        ts=_now_ts(),
    )
    target_state = _legalize_transition(order, 'cancel_requested')
    if target_state != order['status']:
        storage.transition_order_state(order_id, target_state, reason='cancel_requested', ts=_now_ts())
    resp = cancel_order(order_id=order.get('venue_order_id'), client_order_id=order.get('client_order_id'), dry_run=dry_run)
    out = _process_order_response(order_id, order['requested_qty'], resp, dry_run=dry_run)
    refreshed = storage.get_order(order_id=order_id)
    storage.append_order_event(order_id, 'cancel_after', old_status=order['status'], new_status=refreshed['status'], response=out['response'], ts=_now_ts())
    if refreshed['status'] not in storage.TERMINAL_ORDER_STATUSES:
        return refresh_order_status(order_id, dry_run=dry_run)
    return out


def find_stale_active_orders(now_ts: Optional[str] = None, thresholds: Optional[dict] = None):
    now = datetime.fromisoformat((now_ts or _now_ts()).replace('Z', '+00:00'))
    thresholds = thresholds or {}
    max_pending_submit_age_sec = int(thresholds.get('max_pending_submit_age_sec', ORDER_MAX_PENDING_SUBMIT_AGE_SEC))
    cancel_retry_sec = int(thresholds.get('cancel_retry_sec', ORDER_CANCEL_RETRY_SEC))
    max_open_age_sec = int(thresholds.get('max_open_age_sec', ORDER_MAX_OPEN_AGE_SEC))
    stale = []
    for order in storage.get_open_orders():
        updated = datetime.fromisoformat(order['updated_ts'].replace('Z', '+00:00'))
        age_sec = max(0.0, (now - updated).total_seconds())
        if order['status'] in ('pending_submit', 'submitted'):
            threshold = max_pending_submit_age_sec
        elif order['status'] == 'cancel_requested':
            threshold = cancel_retry_sec
        else:
            threshold = max_open_age_sec
        if age_sec >= threshold:
            stale.append({**order, 'age_sec': age_sec})
    return stale


def manage_stale_orders(now_ts: Optional[str] = None, dry_run: bool = False, thresholds: Optional[dict] = None):
    results = []
    thresholds = thresholds or {}
    max_open_age_sec = int(thresholds.get('max_open_age_sec', ORDER_MAX_OPEN_AGE_SEC))
    unknown_review_age_sec = int(thresholds.get('unknown_review_age_sec', ORDER_UNKNOWN_REVIEW_AGE_SEC))
    now_dt = _parse_iso_ts(now_ts or _now_ts()) or datetime.now(timezone.utc)
    for order in find_stale_active_orders(now_ts=now_ts, thresholds=thresholds):
        refreshed = refresh_order_status(order['id'], dry_run=dry_run)
        current = storage.get_order(order_id=order['id'])
        action = 'refreshed'
        if current['status'] in ('open', 'partially_filled') and order['age_sec'] >= max_open_age_sec:
            cancel_and_reconcile_order(order['id'], dry_run=dry_run)
            action = 'cancel_requested'
        elif current['status'] == 'submitted' and order['age_sec'] >= max_open_age_sec and current.get('venue_order_id'):
            cancel_and_reconcile_order(order['id'], dry_run=dry_run)
            action = 'cancel_requested'
        elif current['status'] == 'not_found_on_venue':
            action = 'released_not_found_on_venue'
        elif current['status'] == 'unknown':
            unknown_since_ts = storage.get_unknown_since_ts(order['id'])
            unknown_since_dt = _parse_iso_ts(unknown_since_ts)
            unknown_age_sec = max(0.0, (now_dt - unknown_since_dt).total_seconds()) if unknown_since_dt else order['age_sec']
            if _should_escalate_unknown_order(current, refreshed, age_sec=unknown_age_sec, review_age_sec=unknown_review_age_sec):
                raw = {
                    'escalation_reason': 'stale_unknown_order_requires_manual_review',
                    'age_sec': order['age_sec'],
                    'stale_refresh_age_sec': order['age_sec'],
                    'unknown_since_ts': unknown_since_ts,
                    'unknown_age_sec': unknown_age_sec,
                    'review_age_sec': unknown_review_age_sec,
                    'venue_order_id': current.get('venue_order_id'),
                    'filled_qty': current.get('filled_qty'),
                    'tx_hash': current.get('tx_hash'),
                    'order_fill_event_count': len(storage.get_order_fill_events(order['id'])),
                    'last_refresh_response': refreshed.get('response') if isinstance(refreshed, dict) else None,
                }
                storage.transition_order_state(order['id'], 'manual_review', reason='stale_unknown_order_requires_manual_review', raw=raw, ts=_now_ts())
                action = 'manual_review'
            else:
                action = 'reported_unknown_pending_review_window'
        results.append({'order_id': order['id'], 'action': action, 'status': storage.get_order(order_id=order['id'])['status'], 'refresh': refreshed})
    return results


def recover_active_orders_on_startup(dry_run: bool = False):
    report = []
    for order in storage.get_open_orders():
        try:
            refreshed = refresh_order_status(order['id'], dry_run=dry_run)
            storage.repair_order_reservations(order['id'], updated_ts=_now_ts())
            report.append({'order_id': order['id'], 'status': storage.get_order(order_id=order['id'])['status'], 'refreshed': True, 'result': refreshed})
        except Exception as exc:
            storage.repair_order_reservations(order['id'], updated_ts=_now_ts())
            report.append({'order_id': order['id'], 'status': storage.get_order(order_id=order['id'])['status'], 'refreshed': False, 'error': str(exc)})
    for order in storage.get_orders_pending_tx_recovery():
        try:
            tx_hash = order.get('tx_hash')
            if not tx_hash:
                continue
            reconciliation = storage.reconcile_tx(tx_hash)
            updated = storage.get_order(order_id=order['id'])
            report.append(
                {
                    'order_id': order['id'],
                    'status': updated['status'] if updated else order['status'],
                    'refreshed': False,
                    'reconciled_tx': tx_hash,
                    'reconciliation': reconciliation,
                }
            )
        except Exception as exc:
            report.append({'order_id': order['id'], 'status': storage.get_order(order_id=order['id'])['status'], 'refreshed': False, 'reconciliation_error': str(exc)})
    return report


def place_paper_limit(side: str, qty: float, price: float, token_id: str, market_id: str, outcome_side: str, client_order_id: Optional[str] = None):
    if not market_id or not token_id or not outcome_side:
        raise ValueError('market_id, token_id, and outcome_side are required')
    if side.lower() == 'sell' and storage.get_available_qty(market_id, token_id, outcome_side) + 1e-12 < qty:
        raise RuntimeError(f'Not enough available inventory to sell: requested {qty}')
    reservation_type = 'inventory' if side.lower() == 'sell' else 'exposure'
    reservation_qty = qty if reservation_type == 'inventory' else qty * max(0.0, price)
    order = _create_order_and_reservation(side, qty, price, token_id, market_id, outcome_side, reservation_qty, reservation_type, client_order_id=client_order_id)
    resp = {'status': 'filled', 'filledQuantity': qty, 'txHash': None, 'dry_run': True}
    out = _process_order_response(order['id'], qty, resp, dry_run=True)
    return {"status": "filled", "side": side, "qty": qty, "price": price, "ts": _now_ts(), "order_id": out['order']['id'], "client_order_id": out['order']['client_order_id']}


def place_live_limit(side: str, qty: float, price: float, token_id: str, market_id: str, outcome_side: str, client_order_id: Optional[str] = None):
    if not market_id or not token_id or not outcome_side:
        raise ValueError('market_id, token_id, and outcome_side are required')
    side_lc = (side or '').lower()
    if side_lc not in ('buy', 'sell'):
        return {'status': 'error', 'reason': 'invalid side'}
    if side_lc == 'sell':
        available_qty = storage.get_available_qty(market_id, token_id, outcome_side)
        if available_qty + 1e-12 < qty:
            raise RuntimeError(f'Not enough available inventory to sell: requested {qty}, available {available_qty}')
        reservation_type = 'inventory'
        reservation_qty = qty
    else:
        reservation_type = 'exposure'
        reservation_qty = qty * max(0.0, price)
    order = _create_order_and_reservation(side_lc, qty, price, token_id, market_id, outcome_side, reservation_qty, reservation_type, client_order_id=client_order_id)
    if client_order_id is not None and (order['created_ts'] != order['updated_ts'] or order['status'] != 'pending_submit'):
        return {'status': 'deduped_existing_order', 'client_order_id': order['client_order_id'], 'order_id': order['id']}
    if (order['status'] in ('pending_submit', 'submitted', 'open', 'partially_filled', 'unknown')) or (order['status'] in ACTIVE_ORDER_STATUSES and order['filled_qty'] > 0):
        if order['created_ts'] != order['updated_ts'] or order['status'] != 'pending_submit':
            return {'status': 'deduped_active_order', 'client_order_id': order['client_order_id'], 'order_id': order['id']}
    storage.append_order_event(order['id'], 'submit_before', old_status=order['status'], request={'side': side_lc, 'qty': qty, 'price': price, 'token_id': token_id, 'market_id': market_id, 'outcome_side': outcome_side}, ts=_now_ts())
    # call polymarket_client; by default polymarket_client runs dry-run unless LIVE=true
    resp = _call_submit_helper(place_limit_order, token_id, side_lc, qty, price, post_only=True, dry_run=False, client_order_id=order['client_order_id'])
    normalized = polymarket_client.normalize_client_response(resp, default_status='unknown')
    storage.append_order_event(order['id'], 'submit_after', old_status=order['status'], response=normalized, ts=_now_ts())
    if _is_ambiguous_submit_response(normalized):
        current = storage.get_order(order_id=order['id'])
        target_state = _legalize_transition(current, 'unknown')
        storage.update_order(order['id'], venue_order_id=_extract_venue_order_id(normalized), tx_hash=_extract_tx_hash(normalized), raw_response=normalized, updated_ts=_now_ts())
        if target_state != current['status']:
            storage.transition_order_state(order['id'], target_state, reason='ambiguous_submit_response', raw=normalized, ts=_now_ts())
        storage.repair_order_reservations(order['id'], updated_ts=_now_ts())
        storage.append_order_event(order['id'], 'submit_ambiguous', old_status=current['status'], new_status=storage.get_order(order_id=order['id'])['status'], response=normalized, ts=_now_ts())
        normalized['client_order_id'] = order['client_order_id']
        normalized['order_id'] = order['id']
        return normalized
    _process_order_response(order['id'], qty, normalized, dry_run=False)
    normalized['client_order_id'] = order['client_order_id']
    normalized['order_id'] = order['id']
    return normalized


def place_marketable_buy(token_id: str, size: float, limit_price: Optional[float] = None, dry_run: bool = True, market_id: str = None, outcome_side: str = 'YES', client_order_id: Optional[str] = None, decision_context: Optional[dict] = None):
    """Helper to place a marketable BUY order (FAK/IOC semantics).

    Uses `polymarket_client.place_marketable_order` for live send; records orders/lots on success.
    """
    sz = float(size)
    if sz <= 0:
        return {'status': 'error', 'reason': 'size<=0'}

    if market_id is None:
        raise ValueError('market_id required for buy orders')
    px = 0.99 if limit_price is None else float(limit_price)
    order_type = get_marketable_order_type()
    intent = polymarket_client.build_marketable_order_intent(
        token_id=token_id,
        side='buy',
        quantity=sz,
        limit_price=px,
        order_type=order_type,
        client_order_id=client_order_id,
    )
    if intent.get('skipped'):
        reason = intent.get('skip_reason')
        status = reason if reason == 'skipped_below_min_market_buy_notional' else 'skipped_invalid_quantized_order'
        quantization = intent.get('quantization') or {}
        payload = {
            'status': status,
            'reason': reason,
            'client_order_id': client_order_id,
            'token_id': token_id,
            'side': 'buy',
            'min_notional': float(polymarket_client.MIN_MARKET_BUY_SPEND),
            'submitted_notional': quantization.get('raw_requested_notional'),
            'quantization': quantization,
            **quantization,
        }
        return polymarket_client.normalize_client_response(payload, default_status=status, ok_statuses=set())

    submitted_qty = float(intent['submitted_qty'])
    submitted_price = float(intent['request_body']['price'])
    submitted_notional = float(intent['submitted_notional'])
    order = _create_order_and_reservation('buy', submitted_qty, submitted_price, token_id, market_id, outcome_side, submitted_notional, 'exposure', client_order_id=client_order_id, decision_context=decision_context)
    if client_order_id is not None and (order['created_ts'] != order['updated_ts'] or order['status'] != 'pending_submit'):
        return {'status': 'deduped_existing_order', 'client_order_id': order['client_order_id'], 'order_id': order['id']}
    storage.append_order_event(
        order['id'],
        'submit_before',
        old_status=order['status'],
        request={
            'side': 'buy',
            'amount': submitted_notional,
            'expected_fill_qty': submitted_qty,
            'price': submitted_price,
            'token_id': token_id,
            'market_id': market_id,
            'outcome_side': outcome_side,
            'dry_run': dry_run,
            'venue_intent_mode': intent['venue_intent_mode'],
            'venue_request_body': intent['request_body'],
            'raw_requested_qty': sz,
            'raw_requested_notional': intent['quantization']['raw_requested_notional'],
            'submitted_qty': submitted_qty,
            'submitted_notional': submitted_notional,
            'quantization': intent['quantization'],
            'order_type': order_type,
        },
        ts=_now_ts(),
    )
    resp = _call_submit_helper(place_marketable_order, token_id, 'buy', submitted_qty, limit_price=submitted_price, order_type=order_type, dry_run=dry_run, client_order_id=order['client_order_id'], compiled_intent=intent)
    normalized = _merge_submission_intent(polymarket_client.normalize_client_response(resp, default_status='unknown'), intent)
    storage.append_order_event(order['id'], 'submit_after', old_status=order['status'], response=normalized, ts=_now_ts())
    if _is_ambiguous_submit_response(normalized):
        current = storage.get_order(order_id=order['id'])
        target_state = _legalize_transition(current, 'unknown')
        storage.update_order(order['id'], venue_order_id=_extract_venue_order_id(normalized), tx_hash=_extract_tx_hash(normalized), raw_response=normalized, updated_ts=_now_ts())
        if target_state != current['status']:
            storage.transition_order_state(order['id'], target_state, reason='ambiguous_submit_response', raw=normalized, ts=_now_ts())
        storage.repair_order_reservations(order['id'], updated_ts=_now_ts())
        storage.append_order_event(order['id'], 'submit_ambiguous', old_status=current['status'], new_status=storage.get_order(order_id=order['id'])['status'], response=normalized, ts=_now_ts())
        normalized['client_order_id'] = order['client_order_id']
        normalized['order_id'] = order['id']
        return normalized
    _process_order_response(order['id'], submitted_qty, normalized, dry_run=dry_run)
    normalized['client_order_id'] = order['client_order_id']
    normalized['order_id'] = order['id']
    return normalized


def place_marketable_sell(token_id: str, size: float, limit_price: Optional[float] = None, dry_run: bool = True, market_id: str = None, outcome_side: str = 'YES', client_order_id: Optional[str] = None, decision_context: Optional[dict] = None):
    """Place a marketable SELL by calling the same client but with side 'sell'."""
    sz = float(size)
    if sz <= 0:
        return {'status': 'error', 'reason': 'size<=0'}

    if market_id is None:
        raise ValueError('market_id required for sell orders')
    px = 0.01 if limit_price is None else float(limit_price)
    order_type = get_marketable_order_type()
    intent = polymarket_client.build_marketable_order_intent(
        token_id=token_id,
        side='sell',
        quantity=sz,
        limit_price=px,
        order_type=order_type,
        client_order_id=client_order_id,
    )
    if intent.get('skipped'):
        return polymarket_client.normalize_client_response(
            {
                'status': 'skipped_invalid_quantized_order',
                'reason': intent.get('skip_reason'),
                'client_order_id': client_order_id,
                'quantization': intent.get('quantization'),
                **(intent.get('quantization') or {}),
            },
            default_status='skipped_invalid_quantized_order',
            ok_statuses=set(),
        )
    submitted_qty = float(intent['submitted_qty'])
    submitted_price = float(intent['request_body']['price'])
    available_qty = storage.get_available_qty(market_id, token_id, outcome_side)
    if available_qty + 1e-12 < submitted_qty:
        raise RuntimeError(f'Not enough available inventory to sell: requested {submitted_qty}, available {available_qty}')
    order = _create_order_and_reservation('sell', submitted_qty, submitted_price, token_id, market_id, outcome_side, submitted_qty, 'inventory', client_order_id=client_order_id, decision_context=decision_context)
    if client_order_id is not None and (order['created_ts'] != order['updated_ts'] or order['status'] != 'pending_submit'):
        return {'status': 'deduped_existing_order', 'client_order_id': order['client_order_id'], 'order_id': order['id']}
    storage.append_order_event(
        order['id'],
        'submit_before',
        old_status=order['status'],
        request={
            'side': 'sell',
            'qty': submitted_qty,
            'price': submitted_price,
            'token_id': token_id,
            'market_id': market_id,
            'outcome_side': outcome_side,
            'dry_run': dry_run,
            'venue_intent_mode': intent['venue_intent_mode'],
            'venue_request_body': intent['request_body'],
            'raw_requested_qty': sz,
            'raw_requested_notional': intent['quantization']['raw_requested_notional'],
            'submitted_qty': submitted_qty,
            'submitted_notional': intent['submitted_notional'],
            'quantization': intent['quantization'],
            'order_type': order_type,
        },
        ts=_now_ts(),
    )
    resp = _call_submit_helper(place_marketable_order, token_id, 'sell', submitted_qty, limit_price=submitted_price, order_type=order_type, dry_run=dry_run, client_order_id=order['client_order_id'], compiled_intent=intent)
    normalized = _merge_submission_intent(polymarket_client.normalize_client_response(resp, default_status='unknown'), intent)
    storage.append_order_event(order['id'], 'submit_after', old_status=order['status'], response=normalized, ts=_now_ts())
    if _is_ambiguous_submit_response(normalized):
        current = storage.get_order(order_id=order['id'])
        target_state = _legalize_transition(current, 'unknown')
        storage.update_order(order['id'], venue_order_id=_extract_venue_order_id(normalized), tx_hash=_extract_tx_hash(normalized), raw_response=normalized, updated_ts=_now_ts())
        if target_state != current['status']:
            storage.transition_order_state(order['id'], target_state, reason='ambiguous_submit_response', raw=normalized, ts=_now_ts())
        storage.repair_order_reservations(order['id'], updated_ts=_now_ts())
        storage.append_order_event(order['id'], 'submit_ambiguous', old_status=current['status'], new_status=storage.get_order(order_id=order['id'])['status'], response=normalized, ts=_now_ts())
        normalized['client_order_id'] = order['client_order_id']
        normalized['order_id'] = order['id']
        return normalized
    _process_order_response(order['id'], submitted_qty, normalized, dry_run=dry_run)
    normalized['client_order_id'] = order['client_order_id']
    normalized['order_id'] = order['id']
    return normalized
