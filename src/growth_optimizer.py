from __future__ import annotations

import math
import os
from typing import Any, Optional

from .strategy_sizing import fractional_kelly


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def expected_growth_shadow_enabled() -> bool:
    """Legacy compat: returns True unless entry growth mode is 'off'."""
    return entry_growth_mode() != 'off'


def entry_growth_mode() -> str:
    """Return the entry growth optimizer mode: 'off', 'shadow', or 'live'."""
    raw = str(os.getenv("ENTRY_GROWTH_MODE", "shadow")).strip().lower()
    if raw not in {"off", "shadow", "live"}:
        # Legacy compat: EXPECTED_GROWTH_SHADOW_ENABLED=true maps to 'shadow'
        legacy = _env_flag("EXPECTED_GROWTH_SHADOW_ENABLED", True)
        return "shadow" if legacy else "off"
    return raw


def expected_growth_epsilon() -> float:
    return max(1e-12, _env_float("EXPECTED_GROWTH_EPSILON", 1e-9))


def expected_growth_min_shadow_threshold() -> float:
    return _env_float("EXPECTED_GROWTH_MIN_SHADOW_THRESHOLD", 0.0)


def expected_growth_conservative_tail_penalty() -> float:
    return max(0.0, _env_float("EXPECTED_GROWTH_CONSERVATIVE_TAIL_PENALTY", 1.25))


def position_reeval_growth_optimizer_mode() -> str:
    raw = str(os.getenv("POSITION_REEVAL_GROWTH_OPTIMIZER_MODE", "shadow")).strip().lower()
    if raw not in {"off", "shadow", "live"}:
        return "shadow"
    return raw


def _clip_probability(value: Optional[float], *, epsilon: float) -> float:
    if value is None:
        return 0.5
    return min(1.0 - epsilon, max(epsilon, float(value)))


def _clip_positive(value: Optional[float], *, epsilon: float) -> float:
    if value is None:
        return epsilon
    return max(epsilon, float(value))


def _side_probability(decision_state: dict, outcome_side: str) -> tuple[Optional[float], Optional[float]]:
    normalized = str(outcome_side or "").upper()
    if normalized == "YES":
        return _safe_float(decision_state.get("p_yes")), _safe_float(decision_state.get("q_yes"))
    if normalized == "NO":
        return _safe_float(decision_state.get("p_no")), _safe_float(decision_state.get("q_no"))
    return None, None


def _fragility_score(decision_state: Optional[dict], *, include_regime: bool = True) -> float:
    state = decision_state or {}
    score = max(0.0, _safe_float(state.get("tail_penalty_score")) or 0.0)
    if state.get("tail_hard_block"):
        score += 1.0
    if include_regime and state.get("would_block_in_shadow"):
        score += 0.5
    if include_regime and state.get("regime_guard_blocked"):
        score += 0.75
    return max(0.0, score)


def conservative_probability(
    *,
    probability: Optional[float],
    market_quote: Optional[float],
    fragility_score: float,
    epsilon: Optional[float] = None,
) -> float:
    eps = expected_growth_epsilon() if epsilon is None else max(1e-12, float(epsilon))
    p = _clip_probability(probability, epsilon=eps)
    q = _clip_probability(market_quote, epsilon=eps)
    shrink = max(0.0, 1.0 - expected_growth_conservative_tail_penalty() * max(0.0, fragility_score))
    return _clip_probability(q + (p - q) * shrink, epsilon=eps)


def evaluate_binary_terminal_wealth(
    *,
    free_cash_now: float,
    total_qty: float,
    outcome_side: str,
    epsilon: Optional[float] = None,
) -> dict:
    eps = expected_growth_epsilon() if epsilon is None else max(1e-12, float(epsilon))
    cash = _clip_positive(free_cash_now, epsilon=eps)
    qty = max(0.0, float(total_qty or 0.0))
    normalized = str(outcome_side or "").upper()
    if normalized == "YES":
        wealth_if_yes = cash + qty
        wealth_if_no = cash
    elif normalized == "NO":
        wealth_if_yes = cash
        wealth_if_no = cash + qty
    else:
        wealth_if_yes = cash
        wealth_if_no = cash
    return {
        "wealth_if_yes": max(eps, wealth_if_yes),
        "wealth_if_no": max(eps, wealth_if_no),
    }


def evaluate_expected_log_growth(
    *,
    wealth_if_yes: float,
    wealth_if_no: float,
    probability_yes: Optional[float],
    baseline_wealth: float,
    epsilon: Optional[float] = None,
) -> float:
    eps = expected_growth_epsilon() if epsilon is None else max(1e-12, float(epsilon))
    p_yes = _clip_probability(probability_yes, epsilon=eps)
    w0 = _clip_positive(baseline_wealth, epsilon=eps)
    w_yes = _clip_positive(wealth_if_yes, epsilon=eps)
    w_no = _clip_positive(wealth_if_no, epsilon=eps)
    return p_yes * math.log(w_yes / w0) + (1.0 - p_yes) * math.log(w_no / w0)


def evaluate_entry_shadow(
    *,
    decision_state: dict,
    outcome_side: str,
    effective_bankroll: Optional[float],
    free_bankroll: Optional[float],
    kelly_fraction: float,
    kelly_multiplier: float,
    max_trade_notional_multiplier: float,
    per_trade_cap_pct: float,
    conservative_probability_override: Optional[float] = None,
    polarization_zone_override: Optional[str] = None,
    epsilon: Optional[float] = None,
) -> dict:
    eps = expected_growth_epsilon() if epsilon is None else max(1e-12, float(epsilon))
    probability, quote = _side_probability(decision_state, outcome_side)
    if probability is None or quote is None or quote <= 0.0 or quote >= 1.0:
        return {
            "entry_growth_eval_mode": "shadow_invalid",
            "expected_log_growth_entry": None,
            "expected_log_growth_entry_conservative": None,
            "expected_log_growth_entry_conservative_old": None,
            "expected_log_growth_entry_conservative_discounted": None,
            "expected_terminal_wealth_if_yes": None,
            "expected_terminal_wealth_if_no": None,
            "expected_log_growth_pass_shadow": False,
            "expected_log_growth_reason_shadow": "missing_probability_or_quote",
            "growth_gate_pass_shadow": False,
            "growth_gate_reason_shadow": "missing_probability_or_quote",
            "entry_growth_trade_notional": 0.0,
            "entry_growth_qty": 0.0,
            "entry_growth_candidate_side": outcome_side,
        }

    bankroll = _clip_positive(effective_bankroll, epsilon=eps)
    available_cash = bankroll if free_bankroll is None else max(eps, min(bankroll, float(free_bankroll)))
    per_trade_cap = per_trade_cap_pct * bankroll * max(0.0, float(max_trade_notional_multiplier))
    desired_notional = bankroll * max(0.0, float(kelly_fraction)) * max(0.0, float(kelly_multiplier))
    trade_notional = max(0.0, min(desired_notional, per_trade_cap, available_cash))
    qty = 0.0 if quote <= 0.0 else max(0.0, trade_notional / quote)
    cash_after = max(eps, available_cash - trade_notional)
    terminal = evaluate_binary_terminal_wealth(
        free_cash_now=cash_after,
        total_qty=qty,
        outcome_side=outcome_side,
        epsilon=eps,
    )
    expected_log_growth_entry = evaluate_expected_log_growth(
        wealth_if_yes=terminal["wealth_if_yes"],
        wealth_if_no=terminal["wealth_if_no"],
        probability_yes=_safe_float(decision_state.get("p_yes")),
        baseline_wealth=available_cash,
        epsilon=eps,
    )

    fragility_score = _fragility_score(decision_state, include_regime=False)
    p_side_conservative = conservative_probability(
        probability=probability,
        market_quote=quote,
        fragility_score=fragility_score,
        epsilon=eps,
    )
    p_yes_conservative_old = p_side_conservative if str(outcome_side).upper() == "YES" else 1.0 - p_side_conservative
    expected_log_growth_conservative_old = evaluate_expected_log_growth(
        wealth_if_yes=terminal["wealth_if_yes"],
        wealth_if_no=terminal["wealth_if_no"],
        probability_yes=p_yes_conservative_old,
        baseline_wealth=available_cash,
        epsilon=eps,
    )
    conservative_override = _safe_float(conservative_probability_override)
    if conservative_override is None:
        expected_log_growth_conservative_discounted = expected_log_growth_conservative_old
        p_yes_conservative_discounted = p_yes_conservative_old
    else:
        p_yes_conservative_discounted = conservative_override if str(outcome_side).upper() == "YES" else 1.0 - conservative_override
        expected_log_growth_conservative_discounted = evaluate_expected_log_growth(
            wealth_if_yes=terminal["wealth_if_yes"],
            wealth_if_no=terminal["wealth_if_no"],
            probability_yes=p_yes_conservative_discounted,
            baseline_wealth=available_cash,
            epsilon=eps,
        )

    zone = str(polarization_zone_override or decision_state.get("polarization_zone") or "normal")
    use_discounted_growth = zone in {"caution", "strict", "hard_block"} and conservative_override is not None
    expected_log_growth_conservative = (
        expected_log_growth_conservative_discounted if use_discounted_growth else expected_log_growth_conservative_old
    )
    passes = math.isfinite(expected_log_growth_conservative) and expected_log_growth_conservative >= expected_growth_min_shadow_threshold()
    reason = "ok" if passes else "below_shadow_threshold"
    if qty <= 0.0 or trade_notional <= 0.0:
        passes = False
        reason = "zero_sized_shadow_trade"
    return {
        "entry_growth_eval_mode": "shadow",
        "expected_log_growth_entry": expected_log_growth_entry,
        "expected_log_growth_entry_conservative": expected_log_growth_conservative,
        "expected_log_growth_entry_conservative_old": expected_log_growth_conservative_old,
        "expected_log_growth_entry_conservative_discounted": expected_log_growth_conservative_discounted,
        "expected_terminal_wealth_if_yes": terminal["wealth_if_yes"],
        "expected_terminal_wealth_if_no": terminal["wealth_if_no"],
        "expected_log_growth_pass_shadow": passes,
        "expected_log_growth_reason_shadow": reason,
        "growth_gate_pass_shadow": passes,
        "growth_gate_reason_shadow": reason,
        "entry_growth_trade_notional": trade_notional,
        "entry_growth_qty": qty,
        "entry_growth_fragility_score": fragility_score,
        "entry_growth_candidate_side": outcome_side,
        "entry_growth_probability_conservative": p_side_conservative,
    }


def current_position_free_cash(
    *,
    wallet_free_usdc: Optional[float],
    effective_bankroll: Optional[float],
    current_qty: float,
    current_avg_entry_price: Optional[float],
) -> float:
    free_cash = _safe_float(wallet_free_usdc)
    if free_cash is not None:
        return max(0.0, free_cash)
    bankroll = _safe_float(effective_bankroll)
    avg_entry = _safe_float(current_avg_entry_price)
    if bankroll is None:
        return 0.0
    if avg_entry is None:
        return max(0.0, bankroll)
    return max(0.0, bankroll - max(0.0, current_qty) * avg_entry)


def candidate_total_qty(current_qty: float, delta_qty: float) -> float:
    return max(0.0, float(current_qty or 0.0) + float(delta_qty or 0.0))


def candidate_cash_after_action(
    *,
    free_cash_now: float,
    delta_qty: float,
    entry_price: Optional[float],
    exit_price: Optional[float],
) -> float:
    cash = max(0.0, float(free_cash_now or 0.0))
    delta = float(delta_qty or 0.0)
    if delta > 0.0:
        return max(0.0, cash - delta * max(0.0, float(entry_price or 0.0)))
    if delta < 0.0:
        return max(0.0, cash + abs(delta) * max(0.0, float(exit_price or 0.0)))
    return cash


def candidate_avg_entry_price(
    *,
    current_qty: float,
    current_avg_entry_price: Optional[float],
    delta_qty: float,
    entry_price: Optional[float],
) -> Optional[float]:
    qty_now = max(0.0, float(current_qty or 0.0))
    avg_now = _safe_float(current_avg_entry_price)
    delta = float(delta_qty or 0.0)
    if qty_now <= 0.0 and delta <= 0.0:
        return avg_now
    if delta <= 0.0:
        return avg_now
    if avg_now is None:
        return _safe_float(entry_price)
    total_qty = qty_now + delta
    if total_qty <= 0.0:
        return avg_now
    return ((qty_now * avg_now) + (delta * max(0.0, float(entry_price or 0.0)))) / total_qty


def reevaluation_candidate_qty_grid(
    *,
    current_qty: float,
    target_add_qty: float,
) -> dict:
    qty_now = max(0.0, float(current_qty or 0.0))
    base_add = max(0.0, float(target_add_qty or 0.0))
    return {
        "hold": 0.0,
        "add_small": 0.5 * base_add,
        "add_medium": 1.0 * base_add,
        "reduce_small": -min(qty_now, max(qty_now * 0.25, 0.5 * base_add)),
        "reduce_medium": -min(qty_now, max(qty_now * 0.50, 1.0 * base_add)),
    }


def target_add_qty_from_kelly(
    *,
    probability: Optional[float],
    quote: Optional[float],
    effective_bankroll: Optional[float],
    free_bankroll: Optional[float],
    kelly_k: float,
    kelly_multiplier: float,
    max_trade_notional_multiplier: float,
    per_trade_cap_pct: float,
) -> float:
    p = _safe_float(probability)
    q = _safe_float(quote)
    if p is None or q is None or q <= 0.0 or q >= 1.0:
        return 0.0
    bankroll = max(0.0, _safe_float(effective_bankroll) or 0.0)
    available_cash = bankroll if free_bankroll is None else max(0.0, min(bankroll, _safe_float(free_bankroll) or 0.0))
    f = fractional_kelly(p, q, k=kelly_k) * max(0.0, float(kelly_multiplier))
    trade_notional = min(
        bankroll * f,
        per_trade_cap_pct * bankroll * max(0.0, float(max_trade_notional_multiplier)),
        available_cash,
    )
    return 0.0 if q <= 0.0 else max(0.0, trade_notional / q)
