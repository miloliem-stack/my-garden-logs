from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .growth_optimizer import (
    candidate_avg_entry_price,
    candidate_cash_after_action,
    candidate_total_qty,
    conservative_probability,
    current_position_free_cash,
    evaluate_binary_terminal_wealth,
    evaluate_expected_log_growth,
    position_reeval_growth_optimizer_mode,
    reevaluation_candidate_qty_grid,
    target_add_qty_from_kelly,
)
from .polymarket_client import quantize_order_submission
from .reversal_evidence import compute_reversal_evidence


REEVAL_POLICY_VERSION = 'position_reeval_v1'


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _result_template(*, market_id: str | None, current_side: str | None, current_qty: float, chosen_side: str | None) -> dict:
    return {
        'enabled': False,
        'action': 'hold',
        'reason': 'position_reeval_hold_default',
        'market_id': market_id,
        'current_position_side': current_side,
        'current_position_qty': current_qty,
        'chosen_side': chosen_side,
        'same_side_as_position': bool(current_side and chosen_side and current_side == chosen_side),
        'contrarian_to_signed_state': False,
        'reversal_evidence': None,
        'current_avg_entry_price': None,
        'held_side_hold_ev_per_share': None,
        'candidate_side_entry_ev_per_share': None,
        'flip_advantage_per_share': None,
        'cooldown_active': False,
        'persistence_passed': False,
        'persistence_target_action': 'hold',
        'persistence_next_count': 0,
        'action_caps_ok': True,
        'executable_exit_price': None,
        'executable_entry_price': None,
        'growth_optimizer_mode': 'off',
        'reevaluation_growth_candidates': [],
        'reevaluation_shadow_best_action': 'hold',
        'reevaluation_shadow_best_delta_qty': 0.0,
        'reevaluation_shadow_best_growth_gain': 0.0,
        'reevaluation_shadow_best_executable': True,
        'reevaluation_shadow_keep_current_position': True,
        'reeval_policy_version': REEVAL_POLICY_VERSION,
    }


def _signed_state_sides(decision_state: dict) -> tuple[str | None, str | None]:
    favored = decision_state.get('favored_side')
    contrarian = decision_state.get('contrarian_side')
    if favored in {'YES', 'NO'} and contrarian in {'YES', 'NO'}:
        return favored, contrarian
    spot_now = _safe_float(decision_state.get('spot_now'))
    strike_price = _safe_float(decision_state.get('strike_price'))
    if spot_now is None or strike_price is None:
        return None, None
    if spot_now >= strike_price:
        return 'YES', 'NO'
    return 'NO', 'YES'


def _choose_side(decision_state: dict) -> str | None:
    chosen = decision_state.get('chosen_side')
    if chosen in {'YES', 'NO'}:
        return chosen
    action = decision_state.get('action')
    if action == 'buy_yes':
        return 'YES'
    if action == 'buy_no':
        return 'NO'
    return None


def _token_for_side(decision_state: dict, side: str | None) -> str | None:
    if side == 'YES':
        return decision_state.get('token_yes')
    if side == 'NO':
        return decision_state.get('token_no')
    return None


def _rank_growth_candidates(candidates: list[dict]) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item.get('reeval_candidate_expected_log_growth_conservative') or float('-inf')),
            float(item.get('reeval_candidate_growth_gain_vs_hold') or float('-inf')),
            1 if item.get('reeval_candidate_executable_now') else 0,
            -abs(float(item.get('reeval_candidate_delta_qty') or 0.0)),
        ),
        reverse=True,
    )
    ranked = []
    for index, item in enumerate(ordered, start=1):
        ranked.append({**item, 'reeval_candidate_rank': index})
    return ranked


def _build_growth_optimizer_candidates(
    *,
    decision_state: dict,
    trade_context: dict,
    current_side: str | None,
    current_qty: float,
    current_avg_entry_price: float | None,
    current_cash: float,
    entry_price: float | None,
    exit_price: float | None,
    effective_bankroll: float,
    free_bankroll: float,
    kelly_k: float,
    kelly_multiplier: float,
    max_trade_notional_multiplier: float,
    per_trade_cap_pct: float,
) -> list[dict]:
    if current_side not in {'YES', 'NO'} or current_qty <= 1e-12:
        return []
    side_probability = _safe_float(decision_state.get('p_yes') if current_side == 'YES' else decision_state.get('p_no'))
    entry_quote = _safe_float(entry_price)
    target_add_qty = target_add_qty_from_kelly(
        probability=side_probability,
        quote=entry_quote,
        effective_bankroll=effective_bankroll,
        free_bankroll=free_bankroll,
        kelly_k=kelly_k,
        kelly_multiplier=kelly_multiplier,
        max_trade_notional_multiplier=max_trade_notional_multiplier,
        per_trade_cap_pct=per_trade_cap_pct,
    )
    candidate_grid = reevaluation_candidate_qty_grid(current_qty=current_qty, target_add_qty=target_add_qty)
    token_id = _token_for_side(decision_state, current_side)
    p_yes = _safe_float(decision_state.get('p_yes'))
    quote_for_side = _safe_float(decision_state.get('q_yes') if current_side == 'YES' else decision_state.get('q_no'))
    fragility_score = max(0.0, _safe_float(decision_state.get('tail_penalty_score')) or 0.0)
    conservative_side_probability = conservative_probability(
        probability=side_probability,
        market_quote=quote_for_side,
        fragility_score=fragility_score,
    )
    p_yes_conservative = conservative_side_probability if current_side == 'YES' else 1.0 - conservative_side_probability
    baseline_wealth = max(1e-9, current_cash + max(0.0, current_qty) * max(0.0, float(current_avg_entry_price or 0.0)))

    candidates = []
    for action_name, delta_qty in candidate_grid.items():
        block_reason = None
        executable = True
        quantized = None
        delta_qty = float(delta_qty or 0.0)
        if action_name == 'hold':
            executable = True
        elif abs(delta_qty) <= 1e-12:
            executable = False
            block_reason = 'zero_candidate_qty'
        elif delta_qty > 0.0:
            if entry_quote is None or token_id is None:
                executable = False
                block_reason = 'missing_entry_quote'
            else:
                quantized = quantize_order_submission(
                    token_id=token_id,
                    side='buy',
                    quantity=delta_qty,
                    limit_price=entry_quote,
                    order_type='FAK',
                    marketable=True,
                )
                executable = quantized.get('status') != 'skipped_invalid_quantized_order'
                block_reason = None if executable else quantized.get('reason')
        elif delta_qty < 0.0:
            if exit_price is None or token_id is None:
                executable = False
                block_reason = 'missing_exit_quote'
            elif abs(delta_qty) > current_qty + 1e-12:
                executable = False
                block_reason = 'reduce_exceeds_current_qty'
            else:
                quantized = quantize_order_submission(
                    token_id=token_id,
                    side='sell',
                    quantity=abs(delta_qty),
                    limit_price=float(exit_price),
                    order_type='FAK',
                    marketable=True,
                )
                executable = quantized.get('status') != 'skipped_invalid_quantized_order'
                block_reason = None if executable else quantized.get('reason')

        free_cash_after = candidate_cash_after_action(
            free_cash_now=current_cash,
            delta_qty=delta_qty,
            entry_price=entry_quote,
            exit_price=exit_price,
        )
        new_total_qty = candidate_total_qty(current_qty, delta_qty)
        new_avg_entry_price = candidate_avg_entry_price(
            current_qty=current_qty,
            current_avg_entry_price=current_avg_entry_price,
            delta_qty=delta_qty,
            entry_price=entry_quote,
        )
        terminal = evaluate_binary_terminal_wealth(
            free_cash_now=free_cash_after,
            total_qty=new_total_qty,
            outcome_side=current_side,
        )
        expected_log_growth = evaluate_expected_log_growth(
            wealth_if_yes=terminal['wealth_if_yes'],
            wealth_if_no=terminal['wealth_if_no'],
            probability_yes=p_yes,
            baseline_wealth=baseline_wealth,
        )
        expected_log_growth_conservative = evaluate_expected_log_growth(
            wealth_if_yes=terminal['wealth_if_yes'],
            wealth_if_no=terminal['wealth_if_no'],
            probability_yes=p_yes_conservative,
            baseline_wealth=baseline_wealth,
        )
        candidates.append(
            {
                'reeval_candidate_action': action_name,
                'reeval_candidate_delta_qty': delta_qty,
                'reeval_candidate_new_total_qty': new_total_qty,
                'reeval_candidate_new_avg_entry_price': new_avg_entry_price,
                'reeval_candidate_cash_change': free_cash_after - current_cash,
                'reeval_candidate_expected_log_growth': expected_log_growth,
                'reeval_candidate_expected_log_growth_conservative': expected_log_growth_conservative,
                'reeval_candidate_growth_gain_vs_hold': None,
                'reeval_candidate_executable_now': executable,
                'reeval_candidate_block_reason': block_reason,
                'reeval_candidate_rank': None,
                'reeval_candidate_expected_terminal_wealth_if_yes': terminal['wealth_if_yes'],
                'reeval_candidate_expected_terminal_wealth_if_no': terminal['wealth_if_no'],
                'reeval_candidate_quantization': quantized,
            }
        )
    ranked = _rank_growth_candidates(candidates)
    hold_growth = next(
        (item['reeval_candidate_expected_log_growth_conservative'] for item in ranked if item['reeval_candidate_action'] == 'hold'),
        0.0,
    )
    enriched = []
    for item in ranked:
        enriched.append(
            {
                **item,
                'reeval_candidate_growth_gain_vs_hold': float(item.get('reeval_candidate_expected_log_growth_conservative') or 0.0) - float(hold_growth or 0.0),
            }
        )
    return _rank_growth_candidates(enriched)


def evaluate_position_reevaluation(*, decision_state: dict, trade_context: dict, price_history, now_ts: str | None = None) -> dict:
    enabled = _env_flag('POSITION_REEVAL_ENABLED', True)
    allow_add = _env_flag('POSITION_REEVAL_ALLOW_ADD', True)
    allow_reduce = _env_flag('POSITION_REEVAL_ALLOW_REDUCE', True)
    allow_flip = _env_flag('POSITION_REEVAL_ALLOW_FLIP', False)
    cooldown_sec = max(0, _env_int('POSITION_REEVAL_COOLDOWN_SEC', 180))
    failed_retry_sec = max(0, _env_int('POSITION_REEVAL_FAILED_RETRY_SEC', 5))
    persistence_required = max(1, _env_int('POSITION_REEVAL_PERSISTENCE_REQUIRED', 2))
    max_adds = max(0, _env_int('POSITION_REEVAL_MAX_ADDS_PER_MARKET', 1))
    max_reduces = max(0, _env_int('POSITION_REEVAL_MAX_REDUCES_PER_MARKET', 1))
    max_flips = max(0, _env_int('POSITION_REEVAL_MAX_FLIPS_PER_MARKET', 1))
    add_edge_threshold = _env_float('POSITION_REEVAL_ADD_EDGE_THRESHOLD', 0.04)
    reduce_threshold = _env_float('POSITION_REEVAL_REDUCE_NEGATIVE_EV_THRESHOLD', -0.03)
    flip_threshold = _env_float('POSITION_REEVAL_FLIP_ADVANTAGE_THRESHOLD', 0.08)
    disable_final_bucket = _env_flag('POSITION_REEVAL_DISABLE_IN_FINAL_BUCKET', True)
    require_reversal = _env_flag('POSITION_REEVAL_REQUIRE_REVERSAL_FOR_CONTRARIAN', True)
    friction_buffer = _env_float('POSITION_REEVAL_EXECUTION_FRICTION_BUFFER', 0.0)

    market_id = decision_state.get('market_id') or (trade_context.get('market') or {}).get('market_id')
    chosen_side = _choose_side(decision_state)
    position_summary = trade_context.get('position_summary') or {}
    available_inventory = position_summary.get('available_inventory') or {}
    available_yes = max(0.0, float(available_inventory.get('YES') or 0.0))
    available_no = max(0.0, float(available_inventory.get('NO') or 0.0))
    if available_yes > 1e-12 and available_no > 1e-12:
        current_side = 'YES' if available_yes >= available_no else 'NO'
        current_qty = max(available_yes, available_no)
    elif available_yes > 1e-12:
        current_side = 'YES'
        current_qty = available_yes
    elif available_no > 1e-12:
        current_side = 'NO'
        current_qty = available_no
    else:
        current_side = None
        current_qty = 0.0
    current_avg_entry_price = _safe_float(
        position_summary.get('avg_entry_price_yes') if current_side == 'YES' else position_summary.get('avg_entry_price_no')
    )

    result = _result_template(market_id=market_id, current_side=current_side, current_qty=current_qty, chosen_side=chosen_side)
    result['enabled'] = enabled
    result['current_avg_entry_price'] = current_avg_entry_price
    if not enabled:
        result['persistence_passed'] = True
        result['reason'] = 'position_reeval_disabled'
        return result
    if current_side is None or current_qty <= 1e-12:
        result['persistence_passed'] = True
        return result

    policy = decision_state.get('policy') or {}
    now = _parse_ts(now_ts) or pd.Timestamp.now(tz='UTC')
    quotes = trade_context.get('quotes') or {}
    yes_quote = quotes.get('yes') or {}
    no_quote = quotes.get('no') or {}
    entry_yes = _safe_float(yes_quote.get('best_ask'))
    entry_no = _safe_float(no_quote.get('best_ask'))
    exit_yes = _safe_float(yes_quote.get('best_bid'))
    exit_no = _safe_float(no_quote.get('best_bid'))
    p_yes = _safe_float(decision_state.get('p_yes'))
    p_no = _safe_float(decision_state.get('p_no'))
    edge_yes = _safe_float(decision_state.get('edge_yes'))
    edge_no = _safe_float(decision_state.get('edge_no'))
    current_state = trade_context.get('position_management_state') or {}
    last_action = str(current_state.get('last_action') or '').strip().lower()
    last_action_ts = _parse_ts(current_state.get('last_action_ts'))
    effective_cooldown_sec = failed_retry_sec if last_action == 'reeval_attempt_failed' else cooldown_sec
    cooldown_active = bool(last_action_ts is not None and effective_cooldown_sec > 0 and (now - last_action_ts).total_seconds() < effective_cooldown_sec)
    result['cooldown_active'] = cooldown_active

    held_exit = exit_yes if current_side == 'YES' else exit_no
    held_prob = p_yes if current_side == 'YES' else p_no
    held_hold_ev = None if held_prob is None or held_exit is None else held_prob - held_exit
    result['held_side_hold_ev_per_share'] = held_hold_ev

    favored_side, contrarian_side = _signed_state_sides(decision_state)
    chosen_is_contrarian = bool(chosen_side and contrarian_side and chosen_side == contrarian_side)
    result['contrarian_to_signed_state'] = bool(current_side and contrarian_side and current_side == contrarian_side)

    active_orders = [
        order for order in (trade_context.get('open_orders') or [])
        if order.get('market_id') == market_id and float(order.get('remaining_qty') or 0.0) > 1e-12
    ]
    if active_orders:
        result['persistence_passed'] = True
        result['reason'] = 'position_reeval_active_order_conflict'
        return result

    policy_bucket = str(policy.get('policy_bucket') or '')
    final_bucket_blocked = disable_final_bucket and policy_bucket == 'final'

    candidate_action = 'hold'
    candidate_reason = 'position_reeval_hold_default'
    entry_price = None
    candidate_entry_ev = None
    flip_advantage = None
    reversal_evidence = None
    action_caps_ok = True

    if chosen_side is not None and chosen_side == current_side and allow_add and not final_bucket_blocked:
        candidate_prob = p_yes if chosen_side == 'YES' else p_no
        candidate_edge = edge_yes if chosen_side == 'YES' else edge_no
        entry_price = entry_yes if chosen_side == 'YES' else entry_no
        candidate_entry_ev = None if candidate_prob is None or entry_price is None else candidate_prob - entry_price
        if (
            candidate_edge is not None and candidate_edge >= add_edge_threshold
            and candidate_entry_ev is not None and candidate_entry_ev > 0
        ):
            candidate_action = 'add_same_side'
            candidate_reason = 'position_reeval_add_same_side'
            action_caps_ok = int(current_state.get('add_count') or 0) < max_adds
    elif chosen_side is not None and chosen_side != current_side and allow_flip and not final_bucket_blocked:
        candidate_prob = p_yes if chosen_side == 'YES' else p_no
        entry_price = entry_yes if chosen_side == 'YES' else entry_no
        candidate_entry_ev = None if candidate_prob is None or entry_price is None else candidate_prob - entry_price
        flip_advantage = None if candidate_entry_ev is None or held_hold_ev is None else candidate_entry_ev - held_hold_ev - friction_buffer
        if candidate_entry_ev is not None and candidate_entry_ev > 0 and flip_advantage is not None and flip_advantage >= flip_threshold:
            candidate_action = 'flip_position'
            candidate_reason = 'position_reeval_flip_advantage'
            action_caps_ok = int(current_state.get('flip_count') or 0) < max_flips
    elif allow_reduce and held_hold_ev is not None and held_exit is not None and held_hold_ev <= reduce_threshold + 1e-12:
        candidate_action = 'reduce_position'
        candidate_reason = 'position_reeval_reduce_negative_hold_ev'
        entry_price = held_exit
        action_caps_ok = int(current_state.get('reduce_count') or 0) < max_reduces

    if candidate_action in {'add_same_side', 'flip_position'} and entry_price is None:
        result['persistence_passed'] = True
        result['reason'] = 'position_reeval_missing_executable_entry_price'
        return result
    if candidate_action == 'reduce_position' and held_exit is None:
        result['persistence_passed'] = True
        result['reason'] = 'position_reeval_missing_executable_exit_price'
        return result

    result['candidate_side_entry_ev_per_share'] = candidate_entry_ev
    result['flip_advantage_per_share'] = flip_advantage
    result['executable_exit_price'] = held_exit
    result['executable_entry_price'] = entry_price
    result['action_caps_ok'] = action_caps_ok

    growth_optimizer_mode = position_reeval_growth_optimizer_mode()
    result['growth_optimizer_mode'] = growth_optimizer_mode
    wallet_state = trade_context.get('wallet_state') or {}
    effective_bankroll = max(0.0, _safe_float(wallet_state.get('effective_bankroll')) or _safe_float(decision_state.get('wallet_state', {}).get('effective_bankroll')) or 0.0)
    if effective_bankroll <= 0.0:
        effective_bankroll = max(0.0, _env_float('BOT_BANKROLL', 1000.0))
    if effective_bankroll <= 0.0:
        free_cash_guess = current_position_free_cash(
            wallet_free_usdc=wallet_state.get('free_usdc'),
            effective_bankroll=trade_context.get('wallet_state', {}).get('effective_bankroll'),
            current_qty=current_qty,
            current_avg_entry_price=current_avg_entry_price,
        )
        effective_bankroll = max(free_cash_guess + max(0.0, current_qty) * max(0.0, float(current_avg_entry_price or 0.0)), 0.0)
    free_bankroll = current_position_free_cash(
        wallet_free_usdc=wallet_state.get('free_usdc'),
        effective_bankroll=effective_bankroll,
        current_qty=current_qty,
        current_avg_entry_price=current_avg_entry_price,
    )
    if growth_optimizer_mode != 'off':
        growth_candidates = _build_growth_optimizer_candidates(
            decision_state=decision_state,
            trade_context=trade_context,
            current_side=current_side,
            current_qty=current_qty,
            current_avg_entry_price=current_avg_entry_price,
            current_cash=free_bankroll,
            entry_price=entry_yes if current_side == 'YES' else entry_no,
            exit_price=held_exit,
            effective_bankroll=effective_bankroll,
            free_bankroll=free_bankroll,
            kelly_k=_env_float('KELLY_K', 0.1),
            kelly_multiplier=float(policy.get('kelly_multiplier', 1.0)),
            max_trade_notional_multiplier=float(policy.get('max_trade_notional_multiplier', 1.0)),
            per_trade_cap_pct=_env_float('PER_TRADE_CAP_PCT', 0.01),
        )
        result['reevaluation_growth_candidates'] = growth_candidates
        if growth_candidates:
            best = growth_candidates[0]
            result['reevaluation_shadow_best_action'] = best.get('reeval_candidate_action')
            result['reevaluation_shadow_best_delta_qty'] = best.get('reeval_candidate_delta_qty')
            result['reevaluation_shadow_best_growth_gain'] = best.get('reeval_candidate_growth_gain_vs_hold')
            result['reevaluation_shadow_best_executable'] = bool(best.get('reeval_candidate_executable_now'))
            result['reevaluation_shadow_keep_current_position'] = bool(
                best.get('reeval_candidate_action') == 'hold'
                or float(best.get('reeval_candidate_growth_gain_vs_hold') or 0.0) <= 0.0
            )
            if growth_optimizer_mode == 'live' and candidate_action in {'add_same_side', 'flip_position'}:
                growth_recommends_hold = (
                    best.get('reeval_candidate_action') == 'hold'
                    or float(best.get('reeval_candidate_growth_gain_vs_hold') or 0.0) <= 0.0
                )
                if growth_recommends_hold:
                    result['persistence_passed'] = False
                    result['reason'] = 'position_reeval_growth_optimizer_veto'
                    return result

    if candidate_action in {'add_same_side', 'flip_position'} and chosen_is_contrarian and require_reversal:
        reversal_evidence = compute_reversal_evidence(
            price_history,
            side=chosen_side,
            strike_price=_safe_float(decision_state.get('strike_price')),
            spot_now=_safe_float(decision_state.get('spot_now')),
        )
        if not reversal_evidence.get('passes_min_score'):
            result['reversal_evidence'] = reversal_evidence
            result['persistence_passed'] = False
            result['reason'] = 'position_reeval_reversal_evidence_failed'
            return result
    elif candidate_action == 'flip_position':
        reversal_evidence = compute_reversal_evidence(
            price_history,
            side=chosen_side,
            strike_price=_safe_float(decision_state.get('strike_price')),
            spot_now=_safe_float(decision_state.get('spot_now')),
        )
        if not reversal_evidence.get('passes_min_score'):
            result['reversal_evidence'] = reversal_evidence
            result['persistence_passed'] = False
            result['reason'] = 'position_reeval_reversal_evidence_failed'
            return result

    result['reversal_evidence'] = reversal_evidence

    persistence_target = current_state.get('persistence_target_action')
    persistence_count = int(current_state.get('persistence_count') or 0)
    next_count = 0 if candidate_action == 'hold' else (persistence_count + 1 if persistence_target == candidate_action else 1)
    result['persistence_target_action'] = candidate_action
    result['persistence_next_count'] = next_count

    if candidate_action == 'hold':
        result['persistence_passed'] = True
        if final_bucket_blocked and chosen_side and chosen_side != current_side and allow_flip:
            result['reason'] = 'position_reeval_disabled_final_bucket'
        elif final_bucket_blocked and chosen_side == current_side and allow_add:
            result['reason'] = 'position_reeval_disabled_final_bucket'
        return result

    if final_bucket_blocked and candidate_action in {'add_same_side', 'flip_position'}:
        result['persistence_passed'] = False
        result['reason'] = 'position_reeval_disabled_final_bucket'
        return result
    if cooldown_active:
        result['persistence_passed'] = False
        result['reason'] = 'position_reeval_failed_retry_cooldown' if last_action == 'reeval_attempt_failed' else 'position_reeval_cooldown_active'
        return result
    if not action_caps_ok:
        result['persistence_passed'] = False
        result['reason'] = 'position_reeval_action_cap_reached'
        return result

    persistence_passed = next_count >= persistence_required
    result['persistence_passed'] = persistence_passed
    if not persistence_passed:
        result['reason'] = 'position_reeval_persistence_not_met'
        return result

    result['action'] = candidate_action
    result['reason'] = candidate_reason
    return result
