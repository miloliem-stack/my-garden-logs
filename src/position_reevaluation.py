"""Deprecated position-reevaluation compatibility hook.

Live add/reduce/flip position reevaluation is intentionally disabled. Future
pre-resolution inventory management must be designed as an explicit replay-tested
sell-before-resolution or regime-switch policy.
"""

from __future__ import annotations


REEVAL_POLICY_VERSION = "position_reeval_disabled_v2"


def evaluate_position_reevaluation(*, decision_state: dict, trade_context: dict, price_history, now_ts: str | None = None) -> dict:
    market_id = decision_state.get("market_id") or (trade_context.get("market") or {}).get("market_id")
    chosen_side = decision_state.get("chosen_side")
    position_summary = trade_context.get("position_summary") or {}
    available_inventory = position_summary.get("available_inventory") or {}
    available_yes = max(0.0, float(available_inventory.get("YES") or 0.0))
    available_no = max(0.0, float(available_inventory.get("NO") or 0.0))
    if available_yes > 1e-12 and available_yes >= available_no:
        current_side = "YES"
        current_qty = available_yes
    elif available_no > 1e-12:
        current_side = "NO"
        current_qty = available_no
    else:
        current_side = None
        current_qty = 0.0

    return {
        "enabled": False,
        "action": "hold",
        "reason": "position_reeval_disabled",
        "market_id": market_id,
        "current_position_side": current_side,
        "current_position_qty": current_qty,
        "chosen_side": chosen_side,
        "same_side_as_position": bool(current_side and chosen_side and current_side == chosen_side),
        "contrarian_to_signed_state": False,
        "reversal_evidence": None,
        "current_avg_entry_price": None,
        "held_side_hold_ev_per_share": None,
        "candidate_side_entry_ev_per_share": None,
        "flip_advantage_per_share": None,
        "cooldown_active": False,
        "persistence_passed": True,
        "persistence_target_action": "hold",
        "persistence_next_count": 0,
        "action_caps_ok": True,
        "executable_exit_price": None,
        "executable_entry_price": None,
        "growth_optimizer_mode": "off",
        "reevaluation_growth_candidates": [],
        "reevaluation_shadow_best_action": "hold",
        "reevaluation_shadow_best_delta_qty": 0.0,
        "reevaluation_shadow_best_growth_gain": 0.0,
        "reevaluation_shadow_best_executable": False,
        "reevaluation_shadow_keep_current_position": True,
        "reeval_policy_version": REEVAL_POLICY_VERSION,
    }
