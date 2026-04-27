import math
import os
from typing import Optional


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
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_tail_guard_config(policy: Optional[dict] = None) -> dict:
    policy = policy or {}
    return {
        "enabled": _env_flag("TAIL_GUARD_ENABLED", True),
        "q_soft": float(policy.get("tail_q_soft", _env_float("TAIL_Q_SOFT", 0.08))),
        "q_hard": float(policy.get("tail_q_hard", _env_float("TAIL_Q_HARD", 0.03))),
        "z_soft": float(policy.get("tail_z_soft", _env_float("TAIL_Z_SOFT", 1.75))),
        "z_hard": float(policy.get("tail_z_hard", _env_float("TAIL_Z_HARD", 2.50))),
        "penalty_lambda": float(policy.get("tail_penalty_lambda", _env_float("TAIL_PENALTY_LAMBDA", 0.75))),
        "hard_block_enabled": bool(policy.get("tail_hard_block_enabled", _env_flag("TAIL_HARD_BLOCK_ENABLED", True))),
    }


def apply_polarized_tail_overlay(bundle: dict, probability_state: dict, policy: dict | None = None) -> dict:
    bundle = bundle or {}
    probability_state = probability_state or {}
    config = get_tail_guard_config(policy)
    eps = 1e-12

    raw_p_yes = _safe_float(probability_state.get("p_yes"))
    raw_p_no = _safe_float(probability_state.get("p_no"))
    q_yes = _safe_float(((bundle.get("yes_quote") or {}).get("mid")))
    q_no = _safe_float(((bundle.get("no_quote") or {}).get("mid")))
    spot = _safe_float(probability_state.get("spot_now"))
    strike = _safe_float(probability_state.get("strike_price"))
    tau_minutes = _safe_float(probability_state.get("tau_minutes"))
    raw_model_output = probability_state.get("raw_model_output") or {}
    raw_output = raw_model_output.get("raw_output") or {}

    horizon_sigma = _safe_float(raw_model_output.get("horizon_sigma"))
    if horizon_sigma is None:
        horizon_sigma = _safe_float(raw_output.get("horizon_sigma"))
    z_signed = _safe_float(raw_model_output.get("z_score"))
    if z_signed is None:
        z_signed = _safe_float(raw_output.get("z_score"))
    if horizon_sigma is None and spot is not None and strike is not None and tau_minutes is not None:
        sigma_per_sqrt_min = _safe_float(probability_state.get("sigma_per_sqrt_min"))
        if sigma_per_sqrt_min is not None and tau_minutes >= 0:
            horizon_sigma = sigma_per_sqrt_min * math.sqrt(max(float(tau_minutes), 0.0))
    if z_signed is None and spot is not None and strike not in (None, 0.0) and horizon_sigma is not None:
        log_moneyness = math.log(max(spot, eps) / max(strike, eps))
        z_signed = log_moneyness / max(abs(horizon_sigma), eps)
    z_abs = None if z_signed is None else abs(float(z_signed))

    favored_side = None
    contrarian_side = None
    if z_signed is not None:
        favored_side = "YES" if z_signed >= 0 else "NO"
        contrarian_side = "NO" if favored_side == "YES" else "YES"

    q_tail = None
    if q_yes is not None and q_no is not None:
        q_tail = min(q_yes, q_no)

    raw_edge_yes = None if raw_p_yes is None or q_yes is None else raw_p_yes - q_yes
    raw_edge_no = None if raw_p_no is None or q_no is None else raw_p_no - q_no

    result = {
        "enabled": bool(config["enabled"]),
        "version": "polarized_tail_v1",
        "raw_p_yes": raw_p_yes,
        "raw_p_no": raw_p_no,
        "adj_p_yes": raw_p_yes,
        "adj_p_no": raw_p_no,
        "q_tail": q_tail,
        "z_signed": z_signed,
        "z_abs": z_abs,
        "horizon_sigma": horizon_sigma,
        "tail_penalty_score": 0.0,
        "favored_side": favored_side,
        "contrarian_side": contrarian_side,
        "tail_hard_block": False,
        "raw_edge_yes": raw_edge_yes,
        "raw_edge_no": raw_edge_no,
        "adj_edge_yes": raw_edge_yes,
        "adj_edge_no": raw_edge_no,
    }

    if not config["enabled"]:
        return result
    if None in (raw_p_yes, raw_p_no, q_yes, q_no, q_tail, z_abs) or contrarian_side is None:
        return result

    tail_score_q = _clip((config["q_soft"] - q_tail) / max(config["q_soft"] - config["q_hard"], eps))
    tail_score_z = _clip((z_abs - config["z_soft"]) / max(config["z_hard"] - config["z_soft"], eps))
    tail_penalty_score = tail_score_q * tail_score_z
    penalty_multiplier = 1.0 - float(config["penalty_lambda"]) * tail_penalty_score

    p_contrarian_raw = raw_p_no if contrarian_side == "NO" else raw_p_yes
    p_contrarian_adj = q_tail + (p_contrarian_raw - q_tail) * penalty_multiplier
    p_contrarian_adj = _clip(p_contrarian_adj)

    if contrarian_side == "NO":
        adj_p_no = p_contrarian_adj
        adj_p_yes = 1.0 - adj_p_no
    else:
        adj_p_yes = p_contrarian_adj
        adj_p_no = 1.0 - adj_p_yes

    tail_hard_block = bool(config["hard_block_enabled"] and q_tail <= config["q_hard"] and z_abs >= config["z_hard"])
    adj_edge_yes = adj_p_yes - q_yes
    adj_edge_no = adj_p_no - q_no

    return {
        **result,
        "adj_p_yes": adj_p_yes,
        "adj_p_no": adj_p_no,
        "tail_penalty_score": tail_penalty_score,
        "tail_hard_block": tail_hard_block,
        "adj_edge_yes": adj_edge_yes,
        "adj_edge_no": adj_edge_no,
    }
