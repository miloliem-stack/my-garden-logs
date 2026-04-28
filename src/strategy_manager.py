"""Strategy manager: compute edge, size, and execute trades.

This module ties together model output, market quotes, sizing, and execution.
It is conservative by default (paper/dry-run unless LIVE=true).
"""
import math
import os
from typing import Optional
from .strategy_sizing import fractional_kelly
from .execution import place_marketable_buy
from . import storage
from .regime_detector import detect_regime, microstructure_spectral_mode
from .storage import get_available_qty, get_market, get_inflight_exposure, get_open_lots, get_open_orders
from .wallet_state import get_effective_bankroll


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


# Config via env with sensible defaults
BOT_BANKROLL = _env_float('BOT_BANKROLL', 1000.0)
PER_TRADE_CAP_PCT = _env_float('PER_TRADE_CAP_PCT', 0.01)  # 1% default
TOTAL_EXPOSURE_CAP = _env_float('TOTAL_EXPOSURE_CAP', 0.3)
KELLY_K = _env_float('KELLY_K', 0.1)
EDGE_THRESHOLD = _env_float('EDGE_THRESHOLD', 0.02)
ALLOW_SAME_SIDE_ENTRY = _env_flag('ALLOW_SAME_SIDE_ENTRY', True)
ALLOW_OPPOSITE_SIDE_ENTRY = _env_flag('ALLOW_OPPOSITE_SIDE_ENTRY', True)
REGIME_SHADOW_MODE = _env_flag('REGIME_SHADOW_MODE', True)
REGIME_DECISION_ACTIVE = _env_flag('REGIME_DECISION_ACTIVE', False)


def _regime_entry_guard_mode() -> str:
    raw = str(os.getenv('REGIME_ENTRY_GUARD_MODE', 'shadow')).strip().lower()
    if raw not in {'off', 'shadow', 'live'}:
        return 'shadow'
    return raw


def _regime_extreme_minority_quote_max() -> float:
    return _env_float('REGIME_EXTREME_MINORITY_QUOTE_MAX', 0.05)


def _regime_polarized_tail_minority_quote_max() -> float:
    return _env_float('REGIME_POLARIZED_TAIL_MINORITY_QUOTE_MAX', 0.10)


def _regime_same_side_max_entries() -> int:
    raw = os.getenv('REGIME_SAME_SIDE_MAX_ENTRIES_PER_MARKET_SIDE')
    try:
        return max(0, int(raw)) if raw is not None else 1
    except (TypeError, ValueError):
        return 1


def _regime_same_side_min_filled_qty() -> float:
    return _env_float('REGIME_SAME_SIDE_MIN_FILLED_QTY', 0.01)


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _decision_regime_label(decision_state: dict) -> Optional[str]:
    regime_state = decision_state.get('regime_state') or {}
    if isinstance(regime_state, dict):
        label = regime_state.get('regime_label')
        return None if label in (None, '') else str(label)
    return None


def _entry_guard_side_quote(decision_state: dict, outcome_side: str) -> Optional[float]:
    if str(outcome_side).upper() == 'YES':
        return _safe_float(decision_state.get('q_yes'))
    if str(outcome_side).upper() == 'NO':
        return _safe_float(decision_state.get('q_no'))
    return None


def _same_side_existing_exposure_stats(
    market_id: str,
    desired_outcome_side: str,
    *,
    min_filled_qty: float,
) -> dict:
    same_side_lots = [
        lot
        for lot in get_open_lots(market_id=market_id)
        if lot.get('outcome_side') == desired_outcome_side and float(lot.get('qty') or 0.0) > float(min_filled_qty)
    ]
    live_inventory_qty = sum(float(lot.get('qty') or 0.0) for lot in same_side_lots)
    materially_filled_orders = [
        order
        for order in storage.list_orders(market_id=market_id)
        if order.get('side') == 'buy'
        and order.get('outcome_side') == desired_outcome_side
        and float(order.get('filled_qty') or 0.0) > float(min_filled_qty)
    ]
    materially_filled_qty = sum(float(order.get('filled_qty') or 0.0) for order in materially_filled_orders)
    return {
        'same_side_existing_qty': max(live_inventory_qty, materially_filled_qty),
        'same_side_existing_live_qty': live_inventory_qty,
        'same_side_existing_filled_entry_count': len(materially_filled_orders),
        'same_side_existing_order_ids': [order['id'] for order in materially_filled_orders],
    }


def evaluate_regime_entry_guard(
    *,
    decision_state: dict,
    market_id: str,
    outcome_side: str,
) -> dict:
    mode = _regime_entry_guard_mode()
    if mode == 'off':
        return {
            'allowed': True,
            'veto_reason': None,
            'veto_details': {},
            'guard_mode': mode,
            'regime_guard_evaluated': False,
            'regime_guard_blocked': False,
            'minority_side_quote': _entry_guard_side_quote(decision_state, outcome_side),
            'same_side_existing_qty': 0.0,
            'same_side_existing_filled_entry_count': 0,
            'would_block_in_shadow': False,
        }

    chosen_quote = _entry_guard_side_quote(decision_state, outcome_side)
    regime_label = _decision_regime_label(decision_state) or 'normal'
    extreme_quote_max = _regime_extreme_minority_quote_max()
    polarized_tail_quote_max = _regime_polarized_tail_minority_quote_max()
    same_side_max_entries = _regime_same_side_max_entries()
    min_filled_qty = _regime_same_side_min_filled_qty()
    same_side_stats = _same_side_existing_exposure_stats(
        market_id,
        outcome_side,
        min_filled_qty=min_filled_qty,
    )

    veto_reason = None
    veto_details = {
        'market_id': market_id,
        'outcome_side': outcome_side,
        'regime_label': regime_label,
        'minority_side_quote': chosen_quote,
        'extreme_minority_quote_max': extreme_quote_max,
        'polarized_tail_minority_quote_max': polarized_tail_quote_max,
        'same_side_max_entries_per_market_side': same_side_max_entries,
        'same_side_min_filled_qty': min_filled_qty,
        'same_side_existing_qty': same_side_stats['same_side_existing_qty'],
        'same_side_existing_live_qty': same_side_stats['same_side_existing_live_qty'],
        'same_side_existing_filled_entry_count': same_side_stats['same_side_existing_filled_entry_count'],
        'same_side_existing_order_ids': same_side_stats['same_side_existing_order_ids'],
    }

    if chosen_quote is not None and chosen_quote <= extreme_quote_max:
        veto_reason = 'veto_extreme_minority_side'
    elif regime_label == 'polarized_tail' and chosen_quote is not None and chosen_quote <= polarized_tail_quote_max:
        veto_reason = 'veto_regime_polarized_tail_minority_side'
    elif (
        same_side_stats['same_side_existing_live_qty'] > float(min_filled_qty)
        or (
            same_side_max_entries >= 0
            and same_side_stats['same_side_existing_filled_entry_count'] >= same_side_max_entries
        )
    ):
        veto_reason = 'veto_same_side_reentry_cap'

    would_block = veto_reason is not None
    blocked = mode == 'live' and would_block
    return {
        'allowed': not blocked,
        'veto_reason': veto_reason,
        'veto_details': veto_details,
        'guard_mode': mode,
        'regime_guard_evaluated': True,
        'regime_guard_blocked': blocked,
        'minority_side_quote': chosen_quote,
        'same_side_existing_qty': same_side_stats['same_side_existing_qty'],
        'same_side_existing_filled_entry_count': same_side_stats['same_side_existing_filled_entry_count'],
        'would_block_in_shadow': mode == 'shadow' and would_block,
    }


def _apply_entry_guard_diagnostics(decision_state: dict, guard_result: dict) -> None:
    decision_state['regime_guard_mode'] = guard_result.get('guard_mode')
    decision_state['regime_guard_evaluated'] = bool(guard_result.get('regime_guard_evaluated'))
    decision_state['regime_guard_blocked'] = bool(guard_result.get('regime_guard_blocked'))
    decision_state['regime_guard_reason'] = guard_result.get('veto_reason')
    decision_state['regime_guard_details'] = guard_result.get('veto_details') or {}
    decision_state['minority_side_quote'] = guard_result.get('minority_side_quote')
    decision_state['same_side_existing_qty'] = guard_result.get('same_side_existing_qty')
    decision_state['same_side_existing_filled_entry_count'] = guard_result.get('same_side_existing_filled_entry_count')
    decision_state['would_block_in_shadow'] = bool(guard_result.get('would_block_in_shadow'))


def _normalize_action_origin(action_origin: Optional[str]) -> str:
    value = str(action_origin or 'unknown').strip().lower()
    if value not in {'first_entry', 'reeval_add', 'reeval_reduce', 'reeval_flip', 'inventory_exit', 'other_buy', 'unknown'}:
        return 'unknown'
    return value


def _expected_buy_side_action(outcome_side: str) -> str:
    return 'buy_yes' if str(outcome_side).upper() == 'YES' else 'buy_no'


def _reason_family(reason: Optional[str]) -> str:
    text = str(reason or 'unknown').strip().lower()
    if text in {'missing_or_invalid_yes_quote', 'missing_or_invalid_no_quote', 'missing_or_invalid_entry_quote'}:
        return 'quote'
    if text in {'market_not_open', 'market_expired'}:
        return 'market'
    if text in {'policy_blocks_new_entries'}:
        return 'policy'
    if text in {'polarized_tail_block'}:
        return 'polarization'
    if text in {'microstructure_noisy_block'}:
        return 'microstructure'
    if text in {'existing_active_buy_order'}:
        return 'active_order'
    if text in {'exposure_cap'}:
        return 'exposure'
    if text in {'invalid_buy_action', 'decision_state_blocked'}:
        return 'decision_state'
    if text.startswith('veto_'):
        return 'regime'
    return 'other'


def _ensure_blocker_telemetry(decision_state: dict, *, action_origin: str) -> dict:
    origin = _normalize_action_origin(action_origin or decision_state.get('action_origin'))
    decision_state['action_origin'] = origin
    telemetry = decision_state.get('blocker_telemetry')
    if not isinstance(telemetry, dict):
        telemetry = {}
        decision_state['blocker_telemetry'] = telemetry
    telemetry.setdefault('candidate_stage', decision_state.get('candidate_stage') or 'candidate_built')
    telemetry.setdefault('candidate_available', True)
    telemetry['action_origin'] = origin
    telemetry.setdefault('terminal_reason', decision_state.get('reason'))
    telemetry.setdefault('terminal_reason_family', _reason_family(decision_state.get('reason')))
    telemetry.setdefault('blocked_by', decision_state.get('blocked_by'))
    telemetry.setdefault('blocked_by_stage', decision_state.get('blocked_by_stage'))
    telemetry.setdefault('first_blocking_guard', decision_state.get('first_blocking_guard'))
    telemetry.setdefault('all_triggered_blockers', list(decision_state.get('all_triggered_blockers') or []))
    telemetry.setdefault('chosen_side_snapshot', {})
    telemetry.setdefault('candidate_snapshot', {})
    telemetry.setdefault('blockers', {})
    return telemetry


def _capture_buy_candidate_snapshot(decision_state: dict, choice: dict) -> None:
    telemetry = _ensure_blocker_telemetry(decision_state, action_origin=str(decision_state.get('action_origin') or 'other_buy'))
    outcome_side = choice.get('outcome_side')
    snapshot = telemetry.get('candidate_snapshot') or {}
    snapshot.update({
        'ts': decision_state.get('ts') or decision_state.get('timestamp'),
        'now_ts': decision_state.get('timestamp'),
        'market_id': decision_state.get('market_id'),
        'action_origin': decision_state.get('action_origin'),
        'position_reeval_action': decision_state.get('position_reeval_action'),
        'chosen_action_candidate': choice.get('side'),
        'chosen_side_candidate': outcome_side,
        'token_id_candidate': choice.get('token_id'),
        'q_yes': decision_state.get('q_yes'),
        'q_no': decision_state.get('q_no'),
        'chosen_side_quote': choice.get('quote'),
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
    })
    telemetry['candidate_snapshot'] = snapshot
    telemetry['chosen_side_snapshot'] = {
        'chosen_side': outcome_side,
        'chosen_action_candidate': choice.get('side'),
        'chosen_side_quote': choice.get('quote'),
    }
    telemetry['candidate_available'] = True
    telemetry['candidate_stage'] = 'candidate_built'
    decision_state['candidate_stage'] = 'candidate_built'


def _record_buy_blocker(
    decision_state: dict,
    *,
    blocker_name: str,
    mode: Optional[str],
    would_block: bool,
    blocked: bool,
    reason: Optional[str],
    inputs: Optional[dict] = None,
    stage: Optional[str] = None,
    terminal: bool = False,
) -> None:
    telemetry = _ensure_blocker_telemetry(decision_state, action_origin=str(decision_state.get('action_origin') or 'other_buy'))
    blockers = telemetry.setdefault('blockers', {})
    blockers[blocker_name] = {
        'evaluated': True,
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
    if stage is not None:
        telemetry['candidate_stage'] = stage
        decision_state['candidate_stage'] = stage
    if terminal:
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


def _initialize_shared_buy_gate(decision_state: dict, *, action_origin: str) -> None:
    origin = _normalize_action_origin(action_origin)
    decision_state['action_origin'] = origin
    _ensure_blocker_telemetry(decision_state, action_origin=origin)
    decision_state['shared_buy_gate_evaluated'] = False
    decision_state['shared_buy_gate_passed'] = False
    decision_state['shared_buy_gate_reason'] = 'not_evaluated'
    decision_state['shared_guard_trade_allowed_checked'] = False
    decision_state['shared_guard_quote_checked'] = False
    decision_state['shared_guard_market_open_checked'] = False
    decision_state['shared_guard_policy_checked'] = False
    decision_state['shared_guard_tail_checked'] = False
    decision_state['shared_guard_microstructure_checked'] = False
    decision_state['shared_guard_regime_checked'] = False
    decision_state['shared_guard_exposure_checked'] = False
    decision_state['shared_guard_active_order_checked'] = False
    decision_state['shared_guard_result_json'] = {
        'action_origin': origin,
        'evaluated': False,
        'passed': False,
        'reason': 'not_evaluated',
        'checks': {
            'trade_allowed': {'checked': False, 'applicable': True},
            'quote': {'checked': False, 'applicable': True},
            'market_open': {'checked': False, 'applicable': True},
            'policy': {'checked': False, 'applicable': True},
            'tail': {'checked': False, 'applicable': True},
            'microstructure': {'checked': False, 'applicable': True},
            'regime': {'checked': False, 'applicable': True},
            'exposure': {'checked': False, 'applicable': True},
            'active_order': {'checked': False, 'applicable': True},
        },
        'details': {},
    }


def _mark_shared_buy_gate_check(decision_state: dict, check_name: str) -> None:
    key = f'shared_guard_{check_name}_checked'
    decision_state[key] = True
    result = decision_state.get('shared_guard_result_json')
    if isinstance(result, dict):
        checks = result.setdefault('checks', {})
        entry = checks.setdefault(check_name, {'checked': False, 'applicable': True})
        entry['checked'] = True
        entry['applicable'] = True


def _finalize_shared_buy_gate(
    decision_state: dict,
    *,
    passed: bool,
    reason: str,
    details: Optional[dict] = None,
) -> None:
    decision_state['shared_buy_gate_evaluated'] = True
    decision_state['shared_buy_gate_passed'] = bool(passed)
    decision_state['shared_buy_gate_reason'] = str(reason)
    result = decision_state.get('shared_guard_result_json')
    if isinstance(result, dict):
        result['action_origin'] = decision_state.get('action_origin')
        result['evaluated'] = True
        result['passed'] = bool(passed)
        result['reason'] = str(reason)
        result['details'] = details or {}
    _record_buy_blocker(
        decision_state,
        blocker_name='shared_buy_gate_result',
        mode='live',
        would_block=not bool(passed),
        blocked=not bool(passed),
        reason=str(reason),
        inputs=details or {},
        stage='submitted' if passed else 'post_candidate_blocked',
        terminal=False,
    )


def _shared_buy_skip(
    decision_state: dict,
    *,
    market_id: str,
    action: str,
    reason: str,
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        'action': action,
        'reason': reason,
        'market_id': market_id,
        'decision_state': decision_state,
    }
    if extra:
        payload.update(extra)
    return payload


def _validate_buy_quotes(decision_state: dict, choice: dict) -> Optional[str]:
    q_yes = _safe_float(decision_state.get('q_yes'))
    q_no = _safe_float(decision_state.get('q_no'))
    quote = _safe_float(choice.get('quote'))
    if q_yes is None or q_yes <= 0 or q_yes >= 1:
        return 'missing_or_invalid_yes_quote'
    if q_no is None or q_no <= 0 or q_no >= 1:
        return 'missing_or_invalid_no_quote'
    if quote is None or quote <= 0 or quote >= 1:
        return 'missing_or_invalid_entry_quote'
    return None


def evaluate_buy_admission(
    *,
    decision_state: dict,
    market_id: str,
    token_yes: str,
    token_no: str,
    choice: dict,
    wallet_state: Optional[dict],
    action_origin: str,
) -> dict:
    origin = _normalize_action_origin(action_origin)
    _initialize_shared_buy_gate(decision_state, action_origin=origin)
    decision_state['token_yes'] = decision_state.get('token_yes') or token_yes
    decision_state['token_no'] = decision_state.get('token_no') or token_no
    decision_state['chosen_side'] = choice.get('outcome_side')
    _capture_buy_candidate_snapshot(decision_state, choice)

    _mark_shared_buy_gate_check(decision_state, 'trade_allowed')
    expected_action = _expected_buy_side_action(choice.get('outcome_side'))
    current_action = decision_state.get('action')
    if current_action not in (None, expected_action):
        decision_state['action'] = None
        decision_state['trade_allowed'] = False
        decision_state['reason'] = 'invalid_buy_action'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='invalid_buy_action',
            details={'expected_action': expected_action, 'current_action': current_action},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='invalid_buy_action',
            mode='live',
            would_block=True,
            blocked=True,
            reason='invalid_buy_action',
            inputs={'expected_action': expected_action, 'current_action': current_action},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_shared_buy_gate',
                reason='invalid_buy_action',
                extra={'expected_action': expected_action, 'current_action': current_action},
            ),
        }
    if decision_state.get('trade_allowed') is False:
        reason = decision_state.get('reason') or 'decision_state_blocked'
        decision_state['action'] = None
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason=reason,
            details={'expected_action': expected_action},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name=str(reason),
            mode='live',
            would_block=True,
            blocked=True,
            reason=str(reason),
            inputs={'expected_action': expected_action},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_shared_buy_gate',
                reason=reason,
                extra={'expected_action': expected_action},
            ),
        }
    decision_state['action'] = expected_action

    _mark_shared_buy_gate_check(decision_state, 'quote')
    quote_error = _validate_buy_quotes(decision_state, choice)
    _record_buy_blocker(
        decision_state,
        blocker_name='quote_invalid',
        mode='live',
        would_block=quote_error is not None,
        blocked=quote_error is not None,
        reason=quote_error,
        inputs={
            'q_yes': decision_state.get('q_yes'),
            'q_no': decision_state.get('q_no'),
            'entry_quote': choice.get('quote'),
        },
    )
    if quote_error is not None:
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = quote_error
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason=quote_error,
            details={'outcome_side': choice.get('outcome_side')},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='quote_invalid',
            mode='live',
            would_block=True,
            blocked=True,
            reason=quote_error,
            inputs={
                'q_yes': decision_state.get('q_yes'),
                'q_no': decision_state.get('q_no'),
                'entry_quote': choice.get('quote'),
            },
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_invalid_quote',
                reason=quote_error,
            ),
        }
    if choice.get('fair') is None:
        _record_buy_blocker(
            decision_state,
            blocker_name='missing_probability_state',
            mode='live',
            would_block=True,
            blocked=True,
            reason='missing_probability_state',
            inputs={'outcome_side': choice.get('outcome_side')},
        )
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'missing_probability_state'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='missing_probability_state',
            details={'outcome_side': choice.get('outcome_side')},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='missing_probability_state',
            mode='live',
            would_block=True,
            blocked=True,
            reason='missing_probability_state',
            inputs={'outcome_side': choice.get('outcome_side')},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_missing_probability',
                reason='missing_probability_state',
            ),
        }

    _mark_shared_buy_gate_check(decision_state, 'market_open')
    minfo = get_market(market_id)
    market_status = None if not minfo else minfo.get('status')
    market_reason = 'market_expired' if market_status == 'expired' else 'market_not_open'
    _record_buy_blocker(
        decision_state,
        blocker_name=market_reason,
        mode='live',
        would_block=bool(not minfo or market_status != 'open'),
        blocked=bool(not minfo or market_status != 'open'),
        reason=market_reason if (not minfo or market_status != 'open') else None,
        inputs={'market_status': market_status},
    )
    if not minfo or minfo.get('status') != 'open':
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = market_reason
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason=market_reason,
            details={'market_status': market_status},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name=market_reason,
            mode='live',
            would_block=True,
            blocked=True,
            reason=market_reason,
            inputs={'market_status': market_status},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_market_not_open',
                reason=market_reason,
            ),
        }

    _mark_shared_buy_gate_check(decision_state, 'policy')
    policy = decision_state.get('policy') or {}
    _record_buy_blocker(
        decision_state,
        blocker_name='policy_blocks_new_entries',
        mode='live',
        would_block=bool(policy and not policy.get('allow_new_entries', True)),
        blocked=bool(policy and not policy.get('allow_new_entries', True)),
        reason='policy_blocks_new_entries' if policy and not policy.get('allow_new_entries', True) else None,
        inputs={'policy_bucket': policy.get('policy_bucket'), 'allow_new_entries': policy.get('allow_new_entries')},
    )
    if policy and not policy.get('allow_new_entries', True):
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'policy_blocks_new_entries'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='policy_blocks_new_entries',
            details={'policy_bucket': policy.get('policy_bucket')},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='policy_blocks_new_entries',
            mode='live',
            would_block=True,
            blocked=True,
            reason='policy_blocks_new_entries',
            inputs={'policy_bucket': policy.get('policy_bucket'), 'allow_new_entries': policy.get('allow_new_entries')},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_policy_blocked_entries',
                reason='policy_blocks_new_entries',
            ),
        }

    _mark_shared_buy_gate_check(decision_state, 'tail')
    tail_guard = _compute_polarized_tail_guard(decision_state, choice.get('outcome_side'))
    decision_state.update(tail_guard)
    _record_buy_blocker(
        decision_state,
        blocker_name='veto_polarization_hard_block',
        mode='live',
        would_block=bool(tail_guard.get('polarized_tail_blocked')),
        blocked=bool(tail_guard.get('polarized_tail_blocked')),
        reason='polarized_tail_block' if tail_guard.get('polarized_tail_blocked') else None,
        inputs={
            'chosen_side': choice.get('outcome_side'),
            'q_tail': tail_guard.get('q_tail'),
            'tail_side': tail_guard.get('tail_side'),
            'z_distance_to_strike': tail_guard.get('z_distance_to_strike'),
        },
    )
    if tail_guard.get('polarized_tail_blocked'):
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'polarized_tail_block'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='polarized_tail_block',
            details={
                'q_tail': tail_guard.get('q_tail'),
                'tail_side': tail_guard.get('tail_side'),
                'z_distance_to_strike': tail_guard.get('z_distance_to_strike'),
            },
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='veto_polarization_hard_block',
            mode='live',
            would_block=True,
            blocked=True,
            reason='polarized_tail_block',
            inputs={
                'chosen_side': choice.get('outcome_side'),
                'q_tail': tail_guard.get('q_tail'),
                'tail_side': tail_guard.get('tail_side'),
                'z_distance_to_strike': tail_guard.get('z_distance_to_strike'),
            },
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_polarized_tail_block',
                reason='polarized_tail_block',
                extra={
                    'q_tail': tail_guard.get('q_tail'),
                    'z': tail_guard.get('z_distance_to_strike'),
                    'chosen_side': choice.get('outcome_side'),
                    'spot_now': decision_state.get('spot_now'),
                    'strike_price': decision_state.get('strike_price'),
                },
            ),
        }

    _mark_shared_buy_gate_check(decision_state, 'microstructure')
    micro_mode = microstructure_spectral_mode()
    micro_regime = decision_state.get('microstructure_regime')
    micro_ready = decision_state.get('spectral_ready', False)
    _record_buy_blocker(
        decision_state,
        blocker_name='microstructure_noisy_block',
        mode=micro_mode,
        would_block=bool(micro_ready and micro_regime == 'noisy'),
        blocked=bool(micro_mode == 'live' and micro_ready and micro_regime == 'noisy'),
        reason='microstructure_noisy_block' if micro_ready and micro_regime == 'noisy' else None,
        inputs={
            'microstructure_regime': micro_regime,
            'smoothness_score': decision_state.get('smoothness_score'),
            'spectral_entropy': decision_state.get('spectral_entropy'),
            'spectral_ready': micro_ready,
        },
    )
    if micro_mode == 'live' and micro_ready and micro_regime == 'noisy':
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'microstructure_noisy_block'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='microstructure_noisy_block',
            details={
                'microstructure_regime': micro_regime,
                'smoothness_score': decision_state.get('smoothness_score'),
                'spectral_entropy': decision_state.get('spectral_entropy'),
                'microstructure_mode': micro_mode,
            },
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='microstructure_noisy_block',
            mode=micro_mode,
            would_block=True,
            blocked=True,
            reason='microstructure_noisy_block',
            inputs={
                'microstructure_regime': micro_regime,
                'smoothness_score': decision_state.get('smoothness_score'),
                'spectral_entropy': decision_state.get('spectral_entropy'),
                'spectral_ready': micro_ready,
            },
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_microstructure_noisy',
                reason='microstructure_noisy_block',
                extra={
                    'microstructure_regime': micro_regime,
                    'smoothness_score': decision_state.get('smoothness_score'),
                    'microstructure_mode': micro_mode,
                },
            ),
        }
    elif micro_mode == 'shadow' and micro_ready and micro_regime == 'noisy':
        decision_state['microstructure_would_block'] = True

    _mark_shared_buy_gate_check(decision_state, 'regime')
    guard_result = evaluate_regime_entry_guard(
        decision_state=decision_state,
        market_id=market_id,
        outcome_side=choice['outcome_side'],
    )
    _apply_entry_guard_diagnostics(decision_state, guard_result)
    _record_buy_blocker(
        decision_state,
        blocker_name=str(guard_result.get('veto_reason') or 'regime_entry_guard_result'),
        mode=guard_result.get('guard_mode'),
        would_block=bool(guard_result.get('veto_reason')),
        blocked=not bool(guard_result.get('allowed')),
        reason=guard_result.get('veto_reason'),
        inputs=guard_result.get('veto_details') or {},
    )
    if not guard_result['allowed']:
        decision_state['trade_allowed'] = False
        decision_state['reason'] = guard_result['veto_reason']
        decision_state['action'] = None
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason=str(guard_result['veto_reason']),
            details={'guard_details': guard_result.get('veto_details') or {}},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name=str(guard_result['veto_reason']),
            mode=guard_result.get('guard_mode'),
            would_block=True,
            blocked=True,
            reason=guard_result['veto_reason'],
            inputs=guard_result.get('veto_details') or {},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_regime_entry_guard',
                reason=guard_result['veto_reason'],
                extra={
                    'outcome_side': choice['outcome_side'],
                    'guard_mode': guard_result['guard_mode'],
                    'guard_details': guard_result['veto_details'],
                },
            ),
        }

    q = float(choice['quote'])
    p = float(choice['fair'])
    runtime_wallet_state = wallet_state or decision_state.get('wallet_state') or {}
    effective_bankroll = get_effective_bankroll(wallet_state=runtime_wallet_state, fallback_bankroll=BOT_BANKROLL)
    bankroll_source = runtime_wallet_state.get('bankroll_source', 'env_fallback')
    policy = decision_state.get('policy') or {}
    kelly_multiplier = float(policy.get('kelly_multiplier', 1.0))
    max_trade_notional_multiplier = float(policy.get('max_trade_notional_multiplier', 1.0))
    adjusted_kelly_multiplier = float(choice.get('kelly_multiplier', 1.0))
    f = fractional_kelly(p, q, k=KELLY_K) * kelly_multiplier * adjusted_kelly_multiplier
    trade_amount = min(f * effective_bankroll, PER_TRADE_CAP_PCT * effective_bankroll * max_trade_notional_multiplier)

    _mark_shared_buy_gate_check(decision_state, 'exposure')
    total_exposure_cap = TOTAL_EXPOSURE_CAP * effective_bankroll
    inflight_exposure = get_inflight_exposure()
    _record_buy_blocker(
        decision_state,
        blocker_name='exposure_cap',
        mode='live',
        would_block=bool(inflight_exposure + trade_amount > total_exposure_cap + 1e-12),
        blocked=bool(inflight_exposure + trade_amount > total_exposure_cap + 1e-12),
        reason='exposure_cap' if inflight_exposure + trade_amount > total_exposure_cap + 1e-12 else None,
        inputs={
            'requested_notional': trade_amount,
            'inflight_exposure': inflight_exposure,
            'total_exposure_cap': total_exposure_cap,
        },
    )
    if inflight_exposure + trade_amount > total_exposure_cap + 1e-12:
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'exposure_cap'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='exposure_cap',
            details={
                'requested_notional': trade_amount,
                'inflight_exposure': inflight_exposure,
                'total_exposure_cap': total_exposure_cap,
            },
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='exposure_cap',
            mode='live',
            would_block=True,
            blocked=True,
            reason='exposure_cap',
            inputs={
                'requested_notional': trade_amount,
                'inflight_exposure': inflight_exposure,
                'total_exposure_cap': total_exposure_cap,
            },
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_due_to_exposure_cap',
                reason='exposure_cap',
                extra={
                    'requested_notional': trade_amount,
                    'inflight_exposure': inflight_exposure,
                    'total_exposure_cap': total_exposure_cap,
                    'effective_bankroll': effective_bankroll,
                    'bankroll_source': bankroll_source,
                },
            ),
        }

    _mark_shared_buy_gate_check(decision_state, 'active_order')
    try:
        active_same_side = [
            order for order in get_open_orders(market_id=market_id)
            if order.get('side') == 'buy' and order.get('outcome_side') == choice['outcome_side']
        ]
    except Exception:
        active_same_side = []
    _record_buy_blocker(
        decision_state,
        blocker_name='existing_active_buy_order',
        mode='live',
        would_block=bool(active_same_side),
        blocked=bool(active_same_side),
        reason='existing_active_buy_order' if active_same_side else None,
        inputs={'active_order_ids': [order['id'] for order in active_same_side]},
    )
    if active_same_side:
        decision_state['trade_allowed'] = False
        decision_state['action'] = None
        decision_state['reason'] = 'existing_active_buy_order'
        _finalize_shared_buy_gate(
            decision_state,
            passed=False,
            reason='existing_active_buy_order',
            details={'active_order_ids': [order['id'] for order in active_same_side]},
        )
        _record_buy_blocker(
            decision_state,
            blocker_name='existing_active_buy_order',
            mode='live',
            would_block=True,
            blocked=True,
            reason='existing_active_buy_order',
            inputs={'active_order_ids': [order['id'] for order in active_same_side]},
            stage='post_candidate_blocked',
            terminal=True,
        )
        return {
            'allowed': False,
            'result': _shared_buy_skip(
                decision_state,
                market_id=market_id,
                action='skipped_existing_active_order',
                reason='existing_active_buy_order',
                extra={
                    'outcome_side': choice['outcome_side'],
                    'active_order_ids': [order['id'] for order in active_same_side],
                },
            ),
        }

    qty = max(1e-6, trade_amount / max(1e-6, q))
    decision_state['trade_allowed'] = True
    decision_state['reason'] = 'ok'
    decision_state['action'] = expected_action
    _finalize_shared_buy_gate(
        decision_state,
        passed=True,
        reason='ok',
        details={
            'outcome_side': choice['outcome_side'],
            'price': q,
            'requested_qty': qty,
            'requested_notional': trade_amount,
            'effective_bankroll': effective_bankroll,
            'bankroll_source': bankroll_source,
            'inflight_exposure': inflight_exposure,
            'total_exposure_cap': total_exposure_cap,
        },
    )
    return {
        'allowed': True,
        'choice': choice,
        'price': q,
        'probability': p,
        'qty': qty,
        'trade_amount': trade_amount,
        'effective_bankroll': effective_bankroll,
        'bankroll_source': bankroll_source,
        'inflight_exposure': inflight_exposure,
        'total_exposure_cap': total_exposure_cap,
    }


def _attach_regime_state(decision_state: dict, market_id: Optional[str], token_yes: Optional[str], token_no: Optional[str]) -> dict:
    if not REGIME_SHADOW_MODE or not isinstance(decision_state, dict) or not market_id:
        return decision_state
    existing = decision_state.get('regime_state')
    if isinstance(existing, dict) and existing.get('regime_label'):
        return decision_state

    recent_observations = list(reversed(storage.list_regime_observations(market_id=market_id, limit=6)))
    latest = recent_observations[-1] if recent_observations else None
    previous_state = dict(latest or {})
    previous_state['recent_observations'] = recent_observations
    regime_state = detect_regime(
        {
            **decision_state,
            'market_id': market_id,
            'token_yes': token_yes or decision_state.get('token_yes'),
            'token_no': token_no or decision_state.get('token_no'),
        },
        previous_state=previous_state,
    )
    decision_state['regime_state'] = regime_state
    if not REGIME_DECISION_ACTIVE:
        decision_state['regime_shadow_mode'] = True
    storage.record_regime_observation(
        ts=str(decision_state.get('ts') or decision_state.get('timestamp') or storage.datetime.now(storage.timezone.utc).isoformat()),
        market_id=market_id,
        token_yes=token_yes or decision_state.get('token_yes'),
        token_no=token_no or decision_state.get('token_no'),
        regime_label=regime_state.get('regime_label') or 'normal',
        trend_score=regime_state.get('trend_score'),
        tail_score=regime_state.get('tail_score'),
        reversal_score=regime_state.get('reversal_score'),
        regime_reason=regime_state.get('regime_reason'),
        persistence_count=int(regime_state.get('persistence_count') or 0),
        decision_state=decision_state,
        source=regime_state.get('source'),
    )
    return decision_state


def _compute_polarized_tail_guard(decision_state: dict, chosen_side: Optional[str]) -> dict:
    q_yes = _safe_float(decision_state.get('q_yes'))
    q_no = _safe_float(decision_state.get('q_no'))
    spot_now = _safe_float(decision_state.get('spot_now'))
    strike_price = _safe_float(decision_state.get('strike_price'))
    sigma_per_sqrt_min = _safe_float(decision_state.get('sigma_per_sqrt_min'))
    tau_minutes = _safe_float(decision_state.get('tau_minutes'))

    q_tail = None
    tail_side = None
    if q_yes is not None and q_no is not None:
        q_tail = min(q_yes, q_no)
        tail_side = 'YES' if q_yes < q_no else 'NO'

    z_distance = None
    if (
        sigma_per_sqrt_min is not None
        and spot_now is not None and spot_now > 0
        and strike_price is not None and strike_price > 0
    ):
        tau_floor = max(tau_minutes or 0.0, 1.0)
        denom = max(sigma_per_sqrt_min * math.sqrt(tau_floor), 1e-9)
        z_distance = abs(math.log(spot_now / strike_price)) / denom

    penalty = 1.0
    blocked = False
    if chosen_side == tail_side and q_tail is not None and z_distance is not None:
        if q_tail <= 0.01 and z_distance >= 2.5:
            blocked = True
        elif q_tail <= 0.05 and z_distance > 1.5:
            penalty = math.exp(-1.25 * (z_distance - 1.5))

    return {
        'q_tail': q_tail,
        'tail_side': tail_side,
        'chosen_side': chosen_side,
        'polarized_tail_penalty': penalty,
        'polarized_tail_blocked': blocked,
        'z_distance_to_strike': z_distance,
        'spot_now': spot_now,
        'strike_price': strike_price,
    }


def select_entry_choice(decision_state: dict) -> dict:
    policy = decision_state.get('policy') or {}
    p_yes = _safe_float(decision_state.get('p_yes'))
    p_no = _safe_float(decision_state.get('p_no'))
    q_yes = _safe_float(decision_state.get('q_yes'))
    q_no = _safe_float(decision_state.get('q_no'))
    edge_yes = _safe_float(decision_state.get('edge_yes'))
    edge_no = _safe_float(decision_state.get('edge_no'))
    edge_threshold_yes = float(policy.get('edge_threshold_yes', EDGE_THRESHOLD))
    edge_threshold_no = float(policy.get('edge_threshold_no', EDGE_THRESHOLD))

    base_guard = _compute_polarized_tail_guard(decision_state, None)
    raw_candidates = []
    candidates = []
    for outcome_side, side, fair, quote, edge, threshold, token_key in (
        ('YES', 'buy_yes', p_yes, q_yes, edge_yes, edge_threshold_yes, 'token_yes'),
        ('NO', 'buy_no', p_no, q_no, edge_no, edge_threshold_no, 'token_no'),
    ):
        if fair is None or quote is None or edge is None:
            continue
        if edge >= threshold:
            raw_candidates.append({
                'side': side,
                'fair': fair,
                'quote': quote,
                'edge': edge,
                'threshold': threshold,
                'token_id': decision_state.get(token_key),
                'outcome_side': outcome_side,
            })
        adjusted_edge = edge
        penalty = 1.0
        blocked = False
        if outcome_side == base_guard.get('tail_side'):
            tail_guard = _compute_polarized_tail_guard(decision_state, outcome_side)
            penalty = float(tail_guard['polarized_tail_penalty'])
            blocked = bool(tail_guard['polarized_tail_blocked'])
            adjusted_edge = edge * penalty
        if blocked:
            continue
        if adjusted_edge < threshold:
            continue
        candidates.append({
            'side': side,
            'fair': fair,
            'quote': quote,
            'edge': edge,
            'adjusted_edge': adjusted_edge,
            'threshold': threshold,
            'token_id': decision_state.get(token_key),
            'outcome_side': outcome_side,
            'kelly_multiplier': penalty,
        })

    choice = None
    if candidates:
        candidates.sort(key=lambda item: (item['adjusted_edge'], item['edge']), reverse=True)
        choice = candidates[0]

    raw_choice = None
    if raw_candidates:
        raw_candidates.sort(key=lambda item: item['edge'], reverse=True)
        raw_choice = raw_candidates[0]

    blocked_tail_choice = raw_choice if raw_choice is not None and raw_choice['outcome_side'] == base_guard.get('tail_side') else None
    guard_side = choice['outcome_side'] if choice is not None else (blocked_tail_choice['outcome_side'] if blocked_tail_choice is not None else None)
    guard = _compute_polarized_tail_guard(decision_state, guard_side)
    return {
        'choice': choice,
        'diagnostics': guard,
        'reason': 'polarized_tail_block' if choice is None and guard.get('polarized_tail_blocked') else ('ok' if choice is not None else 'no_edge_above_threshold'),
    }


def _opposite_outcome_side(outcome_side: str) -> str:
    return 'NO' if str(outcome_side).upper() == 'YES' else 'YES'


def _market_side_conflict(
    market_id: str,
    token_yes: Optional[str],
    token_no: Optional[str],
    desired_outcome_side: str,
    desired_price: float,
) -> Optional[dict]:
    opposite_side = _opposite_outcome_side(desired_outcome_side)
    opposite_token = token_yes if opposite_side == 'YES' else token_no
    opposite_orders = [
        order
        for order in get_open_orders(market_id=market_id)
        if order.get('outcome_side') == opposite_side and float(order.get('remaining_qty') or 0.0) > 1e-12
    ]
    opposite_lots = [
        lot
        for lot in get_open_lots(market_id=market_id)
        if lot.get('token_id') == opposite_token and lot.get('outcome_side') == opposite_side and float(lot.get('qty') or 0.0) > 1e-12
    ]
    held_qty = sum(float(lot.get('qty') or 0.0) for lot in opposite_lots)
    entry_notional = sum(float(lot.get('qty') or 0.0) * float(lot.get('avg_price') or 0.0) for lot in opposite_lots)
    avg_entry_price = None if held_qty <= 1e-12 else entry_notional / held_qty
    max_opposite_entry_price = None if avg_entry_price is None else max(0.0, 1.0 - avg_entry_price)
    blocking_order_ids = [order['id'] for order in opposite_orders]
    blocking_buy_order_ids = [order['id'] for order in opposite_orders if order.get('side') == 'buy']
    if held_qty <= 1e-12 and not blocking_order_ids:
        return None
    if blocking_buy_order_ids:
        return {
            'desired_outcome_side': desired_outcome_side,
            'blocking_outcome_side': opposite_side,
            'blocking_token_id': opposite_token,
            'blocking_order_ids': blocking_order_ids,
            'blocking_buy_order_ids': blocking_buy_order_ids,
            'blocking_qty': held_qty,
            'avg_entry_price': avg_entry_price,
            'max_opposite_entry_price': max_opposite_entry_price,
            'reason': 'opposite_side_buy_order_open',
        }
    if not ALLOW_OPPOSITE_SIDE_ENTRY:
        return {
            'desired_outcome_side': desired_outcome_side,
            'blocking_outcome_side': opposite_side,
            'blocking_token_id': opposite_token,
            'blocking_order_ids': blocking_order_ids,
            'blocking_buy_order_ids': blocking_buy_order_ids,
            'blocking_qty': held_qty,
            'avg_entry_price': avg_entry_price,
            'max_opposite_entry_price': max_opposite_entry_price,
            'reason': 'opposite_side_entry_disabled',
        }
    if max_opposite_entry_price is not None and desired_price <= max_opposite_entry_price + 1e-12:
        return None
    return {
        'desired_outcome_side': desired_outcome_side,
        'blocking_outcome_side': opposite_side,
        'blocking_token_id': opposite_token,
        'blocking_qty': held_qty,
        'avg_entry_price': avg_entry_price,
        'max_opposite_entry_price': max_opposite_entry_price,
        'blocking_order_ids': blocking_order_ids,
        'blocking_buy_order_ids': blocking_buy_order_ids,
        'reason': 'opposite_side_above_complement_price',
    }


def _same_side_entry_conflict(
    market_id: str,
    desired_token_id: str,
    desired_outcome_side: str,
) -> Optional[dict]:
    same_side_lots = [
        lot
        for lot in get_open_lots(market_id=market_id)
        if lot.get('token_id') == desired_token_id and lot.get('outcome_side') == desired_outcome_side and float(lot.get('qty') or 0.0) > 1e-12
    ]
    held_qty = sum(float(lot.get('qty') or 0.0) for lot in same_side_lots)
    entry_notional = sum(float(lot.get('qty') or 0.0) * float(lot.get('avg_price') or 0.0) for lot in same_side_lots)
    avg_entry_price = None if held_qty <= 1e-12 else entry_notional / held_qty
    if held_qty <= 1e-12 or ALLOW_SAME_SIDE_ENTRY:
        return None
    return {
        'desired_outcome_side': desired_outcome_side,
        'blocking_outcome_side': desired_outcome_side,
        'blocking_token_id': desired_token_id,
        'blocking_qty': held_qty,
        'avg_entry_price': avg_entry_price,
        'blocking_order_ids': [],
        'blocking_buy_order_ids': [],
        'reason': 'same_side_entry_disabled',
    }


def build_inventory_exit_action(
    market_id: Optional[str],
    token_yes: Optional[str],
    token_no: Optional[str],
    yes_quote: Optional[dict],
    no_quote: Optional[dict],
    *,
    dry_run: bool = True,
):
    """Deprecated no-op compatibility hook.

    Strategy-level open-market inventory exit was removed. Normal settlement,
    winner redemption, loser finalization, and historical storage audit tables
    remain handled by the settlement/redeemer/storage modules.
    """
    return {
        'policy_type': 'inventory_exit_disabled',
        'market_id': market_id,
        'action': 'hold',
        'skip_reason': 'strategy_inventory_exit_disabled',
        'submitted_qty': 0.0,
        'filled_qty': 0.0,
        'avg_fill_price': None,
        'residual_qty': 0.0,
    }


def build_trade_action(
    decision_state: dict,
    token_yes: Optional[str],
    token_no: Optional[str],
    market_id: Optional[str],
    dry_run: bool = True,
    wallet_state: Optional[dict] = None,
):
    if not market_id:
        raise ValueError('market_id is required for decide_and_execute')
    if not token_yes or not token_no:
        return {'action': 'skipped_incomplete_market_metadata', 'reason': 'missing_token_ids', 'market_id': market_id}
    if not decision_state:
        return {'action': 'skipped_missing_decision_state', 'reason': 'missing_decision_state', 'market_id': market_id}
    _attach_regime_state(decision_state, market_id, token_yes, token_no)

    q_yes = decision_state.get('q_yes')
    q_no = decision_state.get('q_no')
    p_yes = decision_state.get('p_yes')
    p_no = decision_state.get('p_no')
    edge_yes = decision_state.get('edge_yes')
    edge_no = decision_state.get('edge_no')
    policy = decision_state.get('policy') or {}
    edge_threshold_yes = float(policy.get('edge_threshold_yes', EDGE_THRESHOLD))
    edge_threshold_no = float(policy.get('edge_threshold_no', EDGE_THRESHOLD))
    decision_state['token_yes'] = decision_state.get('token_yes') or token_yes
    decision_state['token_no'] = decision_state.get('token_no') or token_no

    precomputed_choice = None
    action_name = decision_state.get('action')
    if action_name == 'buy_yes':
        precomputed_choice = {
            'side': 'buy_yes',
            'outcome_side': 'YES',
            'fair': p_yes,
            'quote': q_yes,
            'edge': edge_yes,
            'threshold': edge_threshold_yes,
            'token_id': decision_state.get('token_yes') or token_yes,
            'adjusted_edge': edge_yes,
            'kelly_multiplier': 1.0,
        }
    elif action_name == 'buy_no':
        precomputed_choice = {
            'side': 'buy_no',
            'outcome_side': 'NO',
            'fair': p_no,
            'quote': q_no,
            'edge': edge_no,
            'threshold': edge_threshold_no,
            'token_id': decision_state.get('token_no') or token_no,
            'adjusted_edge': edge_no,
            'kelly_multiplier': 1.0,
        }

    if precomputed_choice is not None:
        decision_state['action_origin'] = 'first_entry'
        choice = precomputed_choice
    else:
        selection = select_entry_choice(decision_state)
        decision_state.update(selection['diagnostics'])
        choice = selection['choice']
        decision_state['action'] = None if choice is None else choice['side']
        decision_state['trade_allowed'] = choice is not None
        decision_state['reason'] = selection['reason']
    if choice is None:
        if decision_state.get('chosen_side') == decision_state.get('tail_side') and decision_state.get('polarized_tail_blocked'):
            return {
                'action': 'skipped_polarized_tail_block',
                'reason': 'polarized_tail_block',
                'market_id': market_id,
                'q_tail': decision_state.get('q_tail'),
                'z': decision_state.get('z_distance_to_strike'),
                'chosen_side': decision_state.get('chosen_side'),
                'spot_now': decision_state.get('spot_now'),
                'strike_price': decision_state.get('strike_price'),
                'decision_state': decision_state,
            }
        return {
            'action': 'skipped_edge_below_threshold',
            'reason': 'no_edge_above_threshold',
            'market_id': market_id,
            'decision_state': decision_state,
        }
    buy_gate = evaluate_buy_admission(
        decision_state=decision_state,
        market_id=market_id,
        token_yes=token_yes,
        token_no=token_no,
        choice=choice,
        wallet_state=wallet_state,
        action_origin='first_entry',
    )
    if not buy_gate['allowed']:
        return buy_gate['result']
    decision_state['chosen_side'] = choice['outcome_side']
    decision_state['action'] = choice['side']
    q = buy_gate['price']
    side_conflict = _market_side_conflict(market_id, token_yes, token_no, choice['outcome_side'], q)
    if side_conflict is not None:
        return {
            'action': 'skipped_existing_market_side_exposure',
            'reason': side_conflict['reason'],
            'market_id': market_id,
            'outcome_side': choice['outcome_side'],
            'blocking_outcome_side': side_conflict['blocking_outcome_side'],
            'blocking_token_id': side_conflict['blocking_token_id'],
            'blocking_qty': side_conflict['blocking_qty'],
            'avg_entry_price': side_conflict['avg_entry_price'],
            'max_opposite_entry_price': side_conflict['max_opposite_entry_price'],
            'blocking_order_ids': side_conflict['blocking_order_ids'],
            'blocking_buy_order_ids': side_conflict['blocking_buy_order_ids'],
            'decision_state': decision_state,
        }
    same_side_conflict = _same_side_entry_conflict(market_id, choice['token_id'], choice['outcome_side'])
    if same_side_conflict is not None:
        return {
            'action': 'skipped_existing_same_side_exposure',
            'reason': same_side_conflict['reason'],
            'market_id': market_id,
            'outcome_side': choice['outcome_side'],
            'blocking_token_id': same_side_conflict['blocking_token_id'],
            'blocking_qty': same_side_conflict['blocking_qty'],
            'avg_entry_price': same_side_conflict['avg_entry_price'],
            'blocking_order_ids': same_side_conflict['blocking_order_ids'],
            'blocking_buy_order_ids': same_side_conflict['blocking_buy_order_ids'],
            'decision_state': decision_state,
        }

    effective_bankroll = buy_gate['effective_bankroll']
    bankroll_source = buy_gate['bankroll_source']
    trade_amount = buy_gate['trade_amount']
    qty = buy_gate['qty']
    resp = place_marketable_buy(
        choice['token_id'],
        qty,
        limit_price=q,
        dry_run=dry_run,
        market_id=market_id,
        outcome_side=choice['outcome_side'],
        decision_context=decision_state,
    )
    submitted_qty = resp.get('submitted_qty', resp.get('quantized_qty', qty))
    submitted_notional = resp.get('submitted_notional', resp.get('quantized_notional', submitted_qty * q))
    return {
        'side': choice['side'],
        'edge': choice['edge'],
        'qty': submitted_qty,
        'raw_requested_qty': qty,
        'raw_requested_notional': trade_amount,
        'quantized_qty': submitted_qty,
        'quantized_notional': submitted_notional,
        'price': q,
        'effective_bankroll': effective_bankroll,
        'bankroll_source': bankroll_source,
        'polarized_tail_penalty': decision_state.get('polarized_tail_penalty'),
        'resp': resp,
        'policy_bucket': policy.get('policy_bucket'),
        'decision_state': decision_state,
    }


def decide_and_execute(
    p_model: float,
    q_market: Optional[float],
    token_yes: Optional[str],
    token_no: Optional[str],
    market_id: Optional[str],
    dry_run: bool = True,
    wallet_state: Optional[dict] = None,
):
    """Decide whether to trade and execute paper/live orders.

    Returns an action dict or None.
    """
    decision_state = {
        'p_yes': p_model,
        'p_no': None if p_model is None else 1 - p_model,
        'q_yes': q_market,
        'q_no': None if q_market is None else 1 - q_market,
        'edge_yes': None if p_model is None or q_market is None else p_model - q_market,
        'edge_no': None if p_model is None or q_market is None else (1 - p_model) - (1 - q_market),
        'trade_allowed': q_market is not None and 0 < q_market < 1,
        'reason': None if q_market is not None and 0 < q_market < 1 else 'missing_or_invalid_yes_quote',
    }
    return build_trade_action(decision_state, token_yes, token_no, market_id, dry_run=dry_run, wallet_state=wallet_state)
