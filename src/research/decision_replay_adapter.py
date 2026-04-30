from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .decision_contract import (
    DecisionInput,
    ExpectedGrowthSnapshot,
    HMMPolicyState,
    ProbabilitySnapshot,
    QuoteSnapshot,
    SafetyVetoSnapshot,
    TauPolicySnapshot,
    evaluate_replay_decision,
)


@dataclass(frozen=True)
class DecisionReplayConfig:
    yes_edge_threshold: float = 0.03
    no_edge_threshold: float = 0.03
    min_posterior_confidence: float = 0.70
    min_next_same_state_confidence: float = 0.65
    min_persistence: int = 2
    require_expected_growth_pass: bool = True
    default_block_tail_veto: bool = True
    default_block_reversal_veto: bool = True
    default_block_quote_quality_veto: bool = True
    allow_missing_hmm_state: bool = False
    allow_missing_expected_growth: bool = False
    allow_missing_safety_fields: bool = True
    output_diagnostics: bool = True


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _first_present(row: Mapping[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return None


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _to_timestamp_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _build_probability(row: Mapping[str, Any]) -> ProbabilitySnapshot:
    p_yes = _to_float(_first_present(row, ["p_yes"]))
    p_no = _to_float(_first_present(row, ["p_no"]))
    if p_yes is None:
        raise ValueError("replay row missing required probability field: p_yes")
    if p_no is None:
        p_no = 1.0 - p_yes
    return ProbabilitySnapshot(
        p_yes=p_yes,
        p_no=p_no,
        engine_name=str(_first_present(row, ["engine_name", "probability_engine", "model_name"]) or "unknown"),
        diagnostics={"p_yes_source": "p_yes", "p_no_inferred": "p_no" not in row or _is_missing(row.get("p_no"))},
    )


def _build_quote(row: Mapping[str, Any]) -> QuoteSnapshot:
    q_yes = _to_float(_first_present(row, ["q_yes", "q_yes_ask", "yes_price"]))
    q_no = _to_float(_first_present(row, ["q_no", "q_no_ask", "no_price"]))
    if q_yes is None:
        raise ValueError("replay row missing required quote field: q_yes/q_yes_ask")
    if q_no is None:
        raise ValueError("replay row missing required quote field: q_no/q_no_ask")
    return QuoteSnapshot(
        q_yes=q_yes,
        q_no=q_no,
        yes_bid=_to_float(_first_present(row, ["yes_bid", "q_yes_bid"])),
        yes_ask=_to_float(_first_present(row, ["yes_ask", "q_yes_ask"])),
        no_bid=_to_float(_first_present(row, ["no_bid", "q_no_bid"])),
        no_ask=_to_float(_first_present(row, ["no_ask", "q_no_ask"])),
        spread_yes=_to_float(_first_present(row, ["spread_yes"])),
        spread_no=_to_float(_first_present(row, ["spread_no"])),
        quote_age_sec=_to_float(_first_present(row, ["quote_age_sec", "quote_age_s"])),
        diagnostics={},
    )


def _build_tau_policy(row: Mapping[str, Any], *, config: DecisionReplayConfig) -> TauPolicySnapshot:
    return TauPolicySnapshot(
        tau_minutes=_to_float(_first_present(row, ["tau_minutes"])),
        tau_bucket=str(_first_present(row, ["tau_bucket"]) or "unknown"),
        edge_threshold_yes=_to_float(_first_present(row, ["yes_edge_threshold", "edge_threshold_yes"])) or config.yes_edge_threshold,
        edge_threshold_no=_to_float(_first_present(row, ["no_edge_threshold", "edge_threshold_no"])) or config.no_edge_threshold,
        allow_new_entries=bool(_to_bool(_first_present(row, ["allow_new_entries", "tau_allow_new_entries"])) is not False),
        diagnostics={},
    )


def _build_hmm_policy_state(row: Mapping[str, Any], *, config: DecisionReplayConfig) -> HMMPolicyState | None:
    map_state = _to_int(_first_present(row, ["hmm_map_state", "map_state"]))
    posterior = _to_float(_first_present(row, ["posterior_confidence", "hmm_map_confidence"]))
    next_same = _to_float(_first_present(row, ["next_same_state_confidence", "hmm_next_same_state_confidence"]))
    persistence = _to_int(_first_present(row, ["persistence_count", "hmm_map_state_persistence_count"]))
    policy_state = _first_present(row, ["policy_state", "regime_policy_state"])

    missing = any(value is None for value in [map_state, posterior, next_same, persistence])
    if missing and config.allow_missing_hmm_state:
        return None
    if missing:
        return HMMPolicyState(
            map_state=map_state,
            posterior_confidence=posterior,
            next_same_state_confidence=next_same,
            persistence_count=persistence,
            policy_state="transition_uncertain",
            diagnostics={"missing_hmm_state": True},
        )
    if policy_state is None:
        policy_state = f"state_{map_state}_confident"
    return HMMPolicyState(
        map_state=map_state,
        posterior_confidence=posterior,
        next_same_state_confidence=next_same,
        persistence_count=persistence,
        policy_state=str(policy_state),
        diagnostics={},
    )


def _build_expected_growth(row: Mapping[str, Any], *, config: DecisionReplayConfig) -> ExpectedGrowthSnapshot | None:
    expected = _to_float(_first_present(row, ["expected_log_growth"]))
    conservative = _to_float(_first_present(row, ["conservative_expected_log_growth"]))
    passes = _to_bool(_first_present(row, ["expected_growth_passes", "expected_growth_pass"]))

    missing = expected is None and conservative is None and passes is None
    if missing and not config.require_expected_growth_pass:
        return None
    if missing and config.allow_missing_expected_growth:
        return None
    if missing:
        return ExpectedGrowthSnapshot(
            expected_log_growth=None,
            conservative_expected_log_growth=0.0,
            passes=False,
            diagnostics={"missing_expected_growth": True},
        )
    if passes is None:
        if conservative is not None:
            passes = conservative > 0.0
        elif expected is not None:
            passes = expected > 0.0
    return ExpectedGrowthSnapshot(
        expected_log_growth=expected,
        conservative_expected_log_growth=conservative,
        passes=passes,
        diagnostics={},
    )


def _missing_safety_snapshot() -> SafetyVetoSnapshot:
    return SafetyVetoSnapshot(
        tail_veto=False,
        polarization_veto=False,
        reversal_veto=False,
        quote_quality_pass=True,
        tail_fields={"missing": True},
        polarization_fields={"missing": True},
        reversal_fields={"missing": True},
        quote_quality_fields={"missing": True},
    )


def _strict_missing_safety_snapshot(config: DecisionReplayConfig) -> SafetyVetoSnapshot:
    return SafetyVetoSnapshot(
        tail_veto=config.default_block_tail_veto,
        polarization_veto=False,
        reversal_veto=config.default_block_reversal_veto,
        quote_quality_pass=not config.default_block_quote_quality_veto,
        tail_fields={"missing": True},
        polarization_fields={"missing": True},
        reversal_fields={"missing": True},
        quote_quality_fields={"missing": True},
    )


def _build_safety_veto(row: Mapping[str, Any], *, config: DecisionReplayConfig) -> SafetyVetoSnapshot | None:
    tail = _to_bool(_first_present(row, ["tail_veto_flag", "tail_veto"]))
    polarization = _to_bool(_first_present(row, ["polarization_veto_flag", "polarization_veto"]))
    reversal = _to_bool(_first_present(row, ["reversal_veto_flag", "reversal_veto"]))
    quote_quality_pass = _to_bool(
        _first_present(row, ["quote_quality_pass", "quote_quality_ok", "quote_quality_allowed", "quote_quality_veto_pass"])
    )
    quote_quality_veto = _to_bool(_first_present(row, ["quote_quality_veto_flag", "quote_quality_veto"]))
    if quote_quality_pass is None and quote_quality_veto is not None:
        quote_quality_pass = not quote_quality_veto

    missing = all(value is None for value in [tail, polarization, reversal, quote_quality_pass])
    if missing and config.allow_missing_safety_fields:
        return _missing_safety_snapshot()
    if missing:
        return _strict_missing_safety_snapshot(config)

    return SafetyVetoSnapshot(
        tail_veto=(tail if tail is not None else False) if config.default_block_tail_veto else False,
        polarization_veto=polarization if polarization is not None else False,
        reversal_veto=(reversal if reversal is not None else False) if config.default_block_reversal_veto else False,
        quote_quality_pass=(
            True
            if not config.default_block_quote_quality_veto
            else (True if quote_quality_pass is None else quote_quality_pass)
        ),
        tail_fields={},
        polarization_fields={},
        reversal_fields={},
        quote_quality_fields={},
    )


def build_decision_input_from_row(row: Mapping[str, Any], *, config: DecisionReplayConfig) -> DecisionInput:
    return DecisionInput(
        timestamp=_to_timestamp_text(_first_present(row, ["timestamp", "ts"])),
        market_id=str(_first_present(row, ["market_id", "market", "slug"]) or "unknown"),
        probability=_build_probability(row),
        quote=_build_quote(row),
        tau_policy=_build_tau_policy(row, config=config),
        hmm_policy_state=_build_hmm_policy_state(row, config=config),
        safety_veto=_build_safety_veto(row, config=config),
        expected_growth=_build_expected_growth(row, config=config),
        side_context=_first_present(row, ["side_context", "vanilla_side"]),
        diagnostics={
            "row_has_outcome": _first_present(row, ["realized_outcome", "outcome"]) is not None,
            "min_posterior_confidence": config.min_posterior_confidence,
            "min_next_same_state_confidence": config.min_next_same_state_confidence,
            "min_persistence_count": config.min_persistence,
        },
    )


def _serialize_blockers(blockers: list[str]) -> str:
    return _stable_json(blockers)


def _simple_pnl_proxy(row: pd.Series) -> float | None:
    outcome = _to_float(_first_present(row, ["realized_outcome", "outcome"]))
    action = row.get("decision_action")
    q_yes = _to_float(_first_present(row, ["q_yes", "q_yes_ask"]))
    q_no = _to_float(_first_present(row, ["q_no", "q_no_ask"]))
    if outcome is None or action not in {"buy_yes", "buy_no"}:
        return None
    if action == "buy_yes" and q_yes is not None:
        return (1.0 - q_yes) if int(outcome) == 1 else -q_yes
    if action == "buy_no" and q_no is not None:
        return (1.0 - q_no) if int(outcome) == 0 else -q_no
    return None


def evaluate_decision_replay_frame(df: pd.DataFrame, *, config: DecisionReplayConfig) -> pd.DataFrame:
    if not {"p_yes"}.issubset(df.columns):
        raise ValueError("input frame missing required probability field: p_yes")
    if not ({"q_yes", "q_yes_ask"} & set(df.columns)):
        raise ValueError("input frame missing required quote field: q_yes or q_yes_ask")
    if not ({"q_no", "q_no_ask"} & set(df.columns)):
        raise ValueError("input frame missing required quote field: q_no or q_no_ask")

    rows: list[dict[str, Any]] = []
    for _, series in df.iterrows():
        mapping = series.to_dict()
        decision_input = build_decision_input_from_row(mapping, config=config)
        decision_output = evaluate_replay_decision(decision_input)
        blockers = list(decision_output.blocking_reasons)
        diag = decision_output.diagnostics if config.output_diagnostics else {}
        rows.append(
            {
                **mapping,
                "decision_action": decision_output.action,
                "decision_allowed": decision_output.allowed,
                "decision_chosen_side": decision_output.chosen_side,
                "decision_reason": decision_output.reason,
                "decision_blocking_reasons": _serialize_blockers(blockers),
                "decision_edge_yes": diag.get("edge_yes"),
                "decision_edge_no": diag.get("edge_no"),
                "decision_hmm_policy_state": diag.get("hmm_policy_state"),
                "decision_expected_growth_passes": None if decision_input.expected_growth is None else decision_input.expected_growth.passes,
                "decision_tail_veto_blocked": "tail_veto" in blockers,
                "decision_reversal_veto_blocked": "reversal_veto" in blockers,
                "decision_quote_quality_blocked": "quote_quality_veto" in blockers,
                "decision_diagnostics": _stable_json(diag) if config.output_diagnostics else "",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["decision_simple_replay_pnl_proxy"] = out.apply(_simple_pnl_proxy, axis=1)
    return out


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.dropna().tolist()
    if values and isinstance(values[0], str) and values[0].startswith("["):
        counter: Counter[str] = Counter()
        for raw in values:
            try:
                counter.update(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return dict(sorted(counter.items()))
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def _allowed_rate_by(df: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in df.columns:
        return {}
    grouped = df.groupby(column, dropna=False)["decision_allowed"].mean()
    return {str(index): float(value) for index, value in grouped.items()}


def _hit_rate_by_action(df: pd.DataFrame) -> dict[str, float]:
    outcome_col = "realized_outcome" if "realized_outcome" in df.columns else "outcome" if "outcome" in df.columns else None
    if outcome_col is None:
        return {}
    out: dict[str, float] = {}
    for action, group in df.groupby("decision_action", dropna=False):
        if action not in {"buy_yes", "buy_no"}:
            continue
        outcome = pd.to_numeric(group[outcome_col], errors="coerce")
        if action == "buy_yes":
            valid = outcome.notna()
            if valid.any():
                out[str(action)] = float(outcome[valid].mean())
        else:
            valid = outcome.notna()
            if valid.any():
                out[str(action)] = float((1.0 - outcome[valid]).mean())
    return out


def _brier_and_log_loss(df: pd.DataFrame) -> dict[str, float]:
    outcome_col = "realized_outcome" if "realized_outcome" in df.columns else "outcome" if "outcome" in df.columns else None
    if outcome_col is None or "p_yes" not in df.columns:
        return {}
    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    p_yes = pd.to_numeric(df["p_yes"], errors="coerce")
    valid = outcome.notna() & p_yes.notna()
    if not valid.any():
        return {}
    clipped = p_yes[valid].clip(1e-6, 1 - 1e-6)
    obs = outcome[valid]
    return {
        "brier_p_yes": float(((clipped - obs) ** 2).mean()),
        "log_loss_p_yes": float((-(obs * np.log(clipped) + (1 - obs) * np.log(1 - clipped))).mean()),
    }


def summarize_decision_replay(results_df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(results_df))
    allowed = int(results_df["decision_allowed"].fillna(False).sum()) if "decision_allowed" in results_df.columns else 0
    abstained = total - allowed
    summary: dict[str, Any] = {
        "n_rows": total,
        "n_allowed": allowed,
        "n_abstained": abstained,
        "allowed_rate": float(allowed / total) if total else 0.0,
        "action_counts": _counts(results_df["decision_action"]) if "decision_action" in results_df.columns else {},
        "reason_counts": _counts(results_df["decision_reason"]) if "decision_reason" in results_df.columns else {},
        "blocking_reason_counts": _counts(results_df["decision_blocking_reasons"]) if "decision_blocking_reasons" in results_df.columns else {},
        "allowed_rate_by_hmm_state": _allowed_rate_by(results_df, "decision_hmm_policy_state"),
        "allowed_rate_by_tau_bucket": _allowed_rate_by(results_df, "tau_bucket"),
        "allowed_rate_by_chosen_side": _allowed_rate_by(results_df, "decision_chosen_side"),
        "expected_growth_pass_counts": _counts(results_df["decision_expected_growth_passes"]) if "decision_expected_growth_passes" in results_df.columns else {},
        "safety_veto_block_counts": {
            "tail_veto": int(results_df.get("decision_tail_veto_blocked", pd.Series(dtype=bool)).fillna(False).sum()),
            "reversal_veto": int(results_df.get("decision_reversal_veto_blocked", pd.Series(dtype=bool)).fillna(False).sum()),
            "quote_quality_veto": int(results_df.get("decision_quote_quality_blocked", pd.Series(dtype=bool)).fillna(False).sum()),
        },
    }

    if "decision_simple_replay_pnl_proxy" in results_df.columns:
        pnl = pd.to_numeric(results_df["decision_simple_replay_pnl_proxy"], errors="coerce")
        valid = pnl.notna()
        if valid.any():
            summary["simple_replay_pnl_proxy"] = {
                "sum": float(pnl[valid].sum()),
                "mean_per_decision": float(pnl[valid].mean()),
                "n_scored_rows": int(valid.sum()),
            }
    hit_rate = _hit_rate_by_action(results_df)
    if hit_rate:
        summary["hit_rate_by_action"] = hit_rate
    summary.update(_brier_and_log_loss(results_df))
    return summary
