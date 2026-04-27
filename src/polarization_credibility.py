from __future__ import annotations

import os
from typing import Any, Optional

from . import storage


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _env_mode(name: str, default: str) -> str:
    raw = str(os.getenv(name, default)).strip().lower()
    if raw not in {"off", "shadow", "live"}:
        return str(default).strip().lower()
    return raw


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def polarization_credibility_mode() -> str:
    return _env_mode("POLARIZATION_CREDIBILITY_MODE", "shadow")


def polarization_same_side_reentry_mode() -> str:
    return _env_mode("POLARIZATION_SAME_SIDE_REENTRY_MODE", "shadow")


def classify_polarization_zone(chosen_side_quote: Optional[float]) -> str:
    q = _safe_float(chosen_side_quote)
    if q is None:
        return "unknown"
    caution_max = _env_float("POLARIZATION_ZONE_CAUTION_MAX", 0.20)
    strict_max = _env_float("POLARIZATION_ZONE_STRICT_MAX", 0.10)
    hard_block_max = _env_float("POLARIZATION_ZONE_HARD_BLOCK_MAX", 0.05)
    if q <= hard_block_max:
        return "hard_block"
    if q <= strict_max:
        return "strict"
    if q <= caution_max:
        return "caution"
    return "normal"


def same_side_existing_exposure_stats(
    market_id: Optional[str],
    outcome_side: str,
    *,
    min_filled_qty: Optional[float] = None,
) -> dict:
    if not market_id:
        return {
            "same_side_existing_qty": 0.0,
            "same_side_existing_live_qty": 0.0,
            "same_side_existing_filled_entry_count": 0,
            "same_side_existing_order_ids": [],
        }
    threshold = _env_float("POLARIZATION_SAME_SIDE_MIN_FILLED_QTY", 0.01) if min_filled_qty is None else float(min_filled_qty)
    same_side_lots = [
        lot
        for lot in storage.get_open_lots(market_id=market_id)
        if lot.get("outcome_side") == outcome_side and float(lot.get("qty") or 0.0) > threshold
    ]
    live_inventory_qty = sum(float(lot.get("qty") or 0.0) for lot in same_side_lots)
    materially_filled_orders = [
        order
        for order in storage.list_orders(market_id=market_id)
        if order.get("side") == "buy"
        and order.get("outcome_side") == outcome_side
        and float(order.get("filled_qty") or 0.0) > threshold
    ]
    materially_filled_qty = sum(float(order.get("filled_qty") or 0.0) for order in materially_filled_orders)
    return {
        "same_side_existing_qty": max(live_inventory_qty, materially_filled_qty),
        "same_side_existing_live_qty": live_inventory_qty,
        "same_side_existing_filled_entry_count": len(materially_filled_orders),
        "same_side_existing_order_ids": [order["id"] for order in materially_filled_orders],
    }


def _base_zone_weight(zone: str) -> float:
    if zone == "normal":
        return 1.0
    if zone == "caution":
        return _clip01(_env_float("POLARIZATION_CAUTION_BASE_WEIGHT", 0.60))
    if zone == "strict":
        return _clip01(_env_float("POLARIZATION_STRICT_BASE_WEIGHT", 0.25))
    if zone == "hard_block":
        return 0.0
    return 1.0


def _divergence_multiplier(raw_probability: Optional[float], adjusted_probability: Optional[float]) -> tuple[float, float]:
    raw_p = _safe_float(raw_probability)
    adjusted_p = _safe_float(adjusted_probability)
    divergence = None if raw_p is None or adjusted_p is None else abs(adjusted_p - raw_p)
    if not str(os.getenv("POLARIZATION_DIVERGENCE_DISCOUNT_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}:
        return 1.0, 0.0 if divergence is None else float(divergence)
    full_at = max(1e-9, _env_float("POLARIZATION_DIVERGENCE_FULL_DISCOUNT_AT", 0.20))
    if divergence is None:
        return 1.0, 0.0
    multiplier = max(0.0, 1.0 - min(float(divergence) / full_at, 1.0))
    return multiplier, float(divergence)


def _reversal_multiplier(reversal_evidence: Optional[dict]) -> tuple[float, bool, float]:
    no_evidence_multiplier = _clip01(_env_float("POLARIZATION_NO_REVERSAL_EVIDENCE_WEIGHT_MULTIPLIER", 0.60))
    evidence = reversal_evidence or {}
    score_ratio = _clip01(_safe_float(evidence.get("score_ratio")) or 0.0)
    passes = bool(evidence.get("passes_min_score"))
    present = bool(evidence) and evidence.get("reason") not in {None, "", "disabled", "insufficient_price_history"}
    if not present or not passes:
        return no_evidence_multiplier, present, score_ratio
    restored = no_evidence_multiplier + (1.0 - no_evidence_multiplier) * score_ratio
    return min(0.90, restored), True, score_ratio


def compute_credibility_discount(
    *,
    outcome_side: str,
    chosen_side_quote: Optional[float],
    raw_probability: Optional[float],
    adjusted_probability: Optional[float],
    same_side_stats: Optional[dict] = None,
    reversal_evidence: Optional[dict] = None,
    regime_state: Optional[dict] = None,
    tail_penalty_score: Optional[float] = None,
    tail_hard_block: bool = False,
) -> dict:
    quote = _safe_float(chosen_side_quote)
    raw_p = _safe_float(raw_probability)
    adjusted_p = _safe_float(adjusted_probability)
    zone = classify_polarization_zone(quote)
    same_side_stats = same_side_stats or {}
    same_side_present = (
        float(same_side_stats.get("same_side_existing_qty") or 0.0) > _env_float("POLARIZATION_SAME_SIDE_MIN_FILLED_QTY", 0.01)
        or int(same_side_stats.get("same_side_existing_filled_entry_count") or 0) > 0
    )

    if quote is None or adjusted_p is None:
        return {
            "outcome_side": outcome_side,
            "polarization_zone": zone,
            "credibility_weight": 1.0,
            "discounted_probability": adjusted_p,
            "discounted_edge": None if adjusted_p is None or quote is None else adjusted_p - quote,
            "hard_block": False,
            "same_side_exposure_present": same_side_present,
            "reversal_evidence_present": False,
            "reversal_evidence_score_ratio": 0.0,
            "divergence_abs": None if raw_p is None or adjusted_p is None else abs(adjusted_p - raw_p),
            "reason": "missing_quote_or_probability",
        }

    base_weight = _base_zone_weight(zone)
    divergence_multiplier, divergence_abs = _divergence_multiplier(raw_p, adjusted_p)
    same_side_multiplier = _clip01(_env_float("POLARIZATION_SAME_SIDE_REENTRY_WEIGHT_MULTIPLIER", 0.35)) if same_side_present else 1.0
    reversal_multiplier, reversal_present, reversal_score_ratio = _reversal_multiplier(reversal_evidence)
    regime_label = ((regime_state or {}).get("regime_label") or "").strip().lower()
    tail_like_state = regime_label == "polarized_tail" or bool(tail_hard_block) or float(tail_penalty_score or 0.0) > 0.0
    tail_state_multiplier = _clip01(_env_float("POLARIZATION_TAIL_STATE_WEIGHT_MULTIPLIER", 0.50)) if tail_like_state and zone != "normal" else 1.0
    hard_block = zone == "hard_block"
    if zone == "normal":
        weight = 1.0
    else:
        weight = 0.0 if hard_block else _clip01(base_weight * divergence_multiplier * same_side_multiplier * reversal_multiplier * tail_state_multiplier)
    discounted_probability = _clip01(quote + weight * (adjusted_p - quote))
    discounted_edge = discounted_probability - quote
    if zone == "normal":
        reason = "zone=normal;base=1.000;full_trust=yes"
    else:
        reason = (
            f"zone={zone};base={base_weight:.3f};div={0.0 if divergence_abs is None else divergence_abs:.3f};"
            f"div_mult={divergence_multiplier:.3f};same_side={'yes' if same_side_present else 'no'};"
            f"same_mult={same_side_multiplier:.3f};reversal={'yes' if reversal_present else 'no'};"
            f"rev_score={reversal_score_ratio:.3f};rev_mult={reversal_multiplier:.3f};"
            f"tail_state={'yes' if tail_like_state else 'no'};tail_mult={tail_state_multiplier:.3f};"
            f"hard_block={'yes' if hard_block else 'no'}"
        )
    return {
        "outcome_side": outcome_side,
        "polarization_zone": zone,
        "credibility_weight": weight,
        "discounted_probability": discounted_probability,
        "discounted_edge": discounted_edge,
        "hard_block": hard_block,
        "same_side_exposure_present": same_side_present,
        "reversal_evidence_present": reversal_present,
        "reversal_evidence_score_ratio": reversal_score_ratio,
        "divergence_abs": divergence_abs,
        "reason": reason,
    }


def evaluate_same_side_reentry_veto(
    *,
    market_id: Optional[str],
    outcome_side: Optional[str],
    chosen_side_quote: Optional[float],
    same_side_stats: Optional[dict] = None,
) -> dict:
    mode = polarization_same_side_reentry_mode()
    quote = _safe_float(chosen_side_quote)
    quote_max = _env_float("POLARIZATION_SAME_SIDE_REENTRY_QUOTE_MAX", 0.20)
    threshold = _env_float("POLARIZATION_SAME_SIDE_MIN_FILLED_QTY", 0.01)
    same_side_stats = same_side_stats or same_side_existing_exposure_stats(market_id, str(outcome_side or "").upper(), min_filled_qty=threshold)
    same_side_present = (
        float(same_side_stats.get("same_side_existing_qty") or 0.0) > threshold
        or int(same_side_stats.get("same_side_existing_filled_entry_count") or 0) > 0
    )
    would_block = bool(outcome_side) and quote is not None and quote <= quote_max and same_side_present
    blocked = mode == "live" and would_block
    return {
        "mode": mode,
        "quote_max": quote_max,
        "same_side_present": same_side_present,
        "would_block": would_block,
        "blocked": blocked,
        "reason": "veto_same_side_reentry_polarized_zone" if would_block else None,
        "same_side_existing_qty": same_side_stats.get("same_side_existing_qty"),
        "same_side_existing_filled_entry_count": same_side_stats.get("same_side_existing_filled_entry_count"),
    }
