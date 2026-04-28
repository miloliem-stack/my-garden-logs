from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .hmm_dataset import read_table, write_table


ENTRY_ACTIONS = {"buy_yes", "buy_no", "YES", "NO", "yes", "no"}


def _series(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def _numeric(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return pd.to_numeric(_series(df, name, default), errors="coerce")


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _favoredness(df: pd.DataFrame) -> pd.Series:
    side = _series(df, "vanilla_side", _series(df, "vanilla_action", "unknown")).fillna("unknown").astype(str).str.upper()
    edge_yes = _numeric(df, "edge_yes_ask")
    edge_no = _numeric(df, "edge_no_ask")
    favored = pd.Series("unknown", index=df.index)
    favored[(side.str.contains("YES")) & (edge_yes >= 0)] = "favored"
    favored[(side.str.contains("YES")) & (edge_yes < 0)] = "contrarian"
    favored[(side.str.contains("NO")) & (edge_no >= 0)] = "favored"
    favored[(side.str.contains("NO")) & (edge_no < 0)] = "contrarian"
    return favored


def _add_grouping_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["confidence_bucket"] = _bucket_confidence(_series(out, "hmm_map_confidence"))
    out["q_tail_bucket"] = _bucket_tail(_series(out, "q_tail"))
    out["side"] = _series(out, "vanilla_side", _series(out, "vanilla_action", "unknown")).fillna("unknown")
    out["favoredness"] = _favoredness(out)
    out["reversal_status"] = np.where(_as_bool(_series(out, "reversal_veto_flag", False)), "reversal_fail", "reversal_pass")
    return out


def apply_policy_flags(
    df: pd.DataFrame,
    *,
    confidence_threshold: float = 0.70,
    next_same_threshold: float = 0.65,
    persistence_threshold: int = 2,
) -> pd.DataFrame:
    out = df.copy()
    action = _series(out, "vanilla_action", "").fillna("")
    is_entry = action.astype(str).isin(ENTRY_ACTIONS) | action.astype(str).str.startswith("buy_")
    growth = _numeric(out, "conservative_expected_log_growth")
    default_veto = (
        (growth <= 0)
        | _as_bool(_series(out, "tail_veto_flag", False))
        | _as_bool(_series(out, "polarization_veto_flag", False))
        | _as_bool(_series(out, "reversal_veto_flag", False))
    )
    hmm_block = (
        _series(out, "regime_policy_state", "").fillna("").eq("transition_uncertain")
        | (_numeric(out, "hmm_map_confidence", 1.0) < confidence_threshold)
        | (_numeric(out, "hmm_next_same_state_confidence", 1.0) < next_same_threshold)
        | (_numeric(out, "hmm_map_state_persistence_count", persistence_threshold) < persistence_threshold)
    )
    out["policy0_trade"] = is_entry
    out["policy1_trade"] = is_entry & ~(growth <= 0)
    out["policy2_trade"] = is_entry & ~default_veto
    out["policy3_trade"] = is_entry & ~default_veto & ~hmm_block
    out["default_ce_veto_block"] = is_entry & default_veto
    out["regime_gated_action"] = np.where(out["policy3_trade"], action, "blocked")
    return out


def _metrics(df: pd.DataFrame, trade_col: str) -> dict:
    trades = df[df[trade_col].fillna(False)]
    pnl = _numeric(trades, "realized_pnl_binary", 0.0).fillna(0.0)
    growth = _numeric(trades, "conservative_expected_log_growth")
    outcome = _numeric(trades, "realized_outcome")
    p_yes = _numeric(trades, "p_yes")
    q_yes = _numeric(trades, "q_yes_ask")
    result = {
        "n_decisions": int(len(df)),
        "n_trades": int(len(trades)),
        "abstention_count": int(len(df) - len(trades)),
        "realized_pnl": float(pnl.sum()),
        "realized_pnl_per_trade": float(pnl.mean()) if len(pnl) else 0.0,
        "mean_conservative_expected_log_growth": float(growth.mean()) if growth.notna().any() else np.nan,
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "bootstrap_ci_status": "computed" if len(pnl) >= 3 else "todo_min_3_trades",
    }
    valid = outcome.notna() & p_yes.notna()
    if valid.any():
        clipped = p_yes[valid].clip(1e-6, 1 - 1e-6)
        result["brier_p_yes"] = float(((clipped - outcome[valid]) ** 2).mean())
        result["log_loss_p_yes"] = float((-(outcome[valid] * np.log(clipped) + (1 - outcome[valid]) * np.log(1 - clipped))).mean())
    valid_q = outcome.notna() & q_yes.notna()
    if valid_q.any():
        clipped = q_yes[valid_q].clip(1e-6, 1 - 1e-6)
        result["market_brier_q_yes_ask"] = float(((clipped - outcome[valid_q]) ** 2).mean())
        result["market_log_loss_q_yes_ask"] = float((-(outcome[valid_q] * np.log(clipped) + (1 - outcome[valid_q]) * np.log(1 - clipped))).mean())
    if len(pnl) >= 3:
        boot = []
        rng = np.random.default_rng(42)
        values = pnl.to_numpy(dtype=float)
        for _ in range(200):
            boot.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        result["pnl_per_trade_ci_low"] = float(np.quantile(boot, 0.025))
        result["pnl_per_trade_ci_high"] = float(np.quantile(boot, 0.975))
    return result


def _bucket_confidence(series: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=[-np.inf, 0.7, 0.85, 0.95, np.inf], labels=["lt_70", "70_85", "85_95", "gt_95"])


def _bucket_tail(series: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=[-np.inf, 0.25, 0.5, 0.75, np.inf], labels=["low", "mid", "high", "extreme"])


def report_policy_replay(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    min_samples: int = 100,
    confidence_threshold: float = 0.70,
    next_same_threshold: float = 0.65,
    persistence_threshold: int = 2,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = apply_policy_flags(
        read_table(input_path),
        confidence_threshold=confidence_threshold,
        next_same_threshold=next_same_threshold,
        persistence_threshold=persistence_threshold,
    )
    for col, default in [
        ("regime_policy_state", "unknown"),
        ("tau_bucket", "unknown"),
        ("hmm_map_state", -1),
        ("fold_id", -1),
        ("timestamp", pd.NaT),
        ("realized_pnl_binary", 0.0),
        ("conservative_expected_log_growth", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    summary = pd.DataFrame(
        [{"policy": name, **_metrics(df, col)} for name, col in [
            ("vanilla", "policy0_trade"),
            ("expected_growth_veto", "policy1_trade"),
            ("expected_growth_default_vetoes", "policy2_trade"),
            ("hmm_abstention", "policy3_trade"),
        ]]
    )
    by_regime_tau = df.groupby(["regime_policy_state", "tau_bucket"], dropna=False).apply(lambda g: pd.Series(_metrics(g, "policy3_trade"))).reset_index()
    grouped_df = _add_grouping_dimensions(df)
    by_state_conf = grouped_df.groupby(
        ["hmm_map_state", "confidence_bucket"], dropna=False, observed=False
    ).apply(lambda g: pd.Series(_metrics(g, "policy3_trade"))).reset_index()
    group_cols = ["regime_policy_state", "hmm_map_state", "confidence_bucket", "tau_bucket", "q_tail_bucket", "side", "favoredness", "reversal_status"]
    by_policy_dimensions = grouped_df.groupby(group_cols, dropna=False, observed=False).apply(lambda g: pd.Series(_metrics(g, "policy3_trade"))).reset_index()
    candidates = grouped_df[grouped_df["default_ce_veto_block"]].copy()
    candidate_cells = candidates.groupby(group_cols, dropna=False, observed=False).agg(
        n_decisions=("timestamp", "count"),
        realized_pnl=("realized_pnl_binary", "sum"),
        mean_realized_pnl=("realized_pnl_binary", "mean"),
        mean_conservative_expected_log_growth=("conservative_expected_log_growth", "mean"),
        fold_count=("fold_id", "nunique"),
    ).reset_index()
    candidate_cells = candidate_cells[candidate_cells["n_decisions"] >= min_samples]
    candidate_cells["positive_realized_pnl"] = candidate_cells["realized_pnl"] > 0
    candidate_cells["positive_conservative_expected_growth"] = candidate_cells["mean_conservative_expected_log_growth"] > 0
    candidate_cells["stable_across_folds"] = candidate_cells["fold_count"] >= 2
    state_duration = df.groupby(["fold_id", "hmm_map_state"], dropna=False).size().reset_index(name="rows") if "fold_id" in df.columns else pd.DataFrame()
    fold_level_stability = df.groupby(["fold_id"], dropna=False).apply(lambda g: pd.Series(_metrics(g, "policy3_trade"))).reset_index()
    transition_summary = (
        df.assign(prev_state=df.get("hmm_map_state", pd.Series(index=df.index)).shift(1))
        .groupby(["prev_state", "hmm_map_state"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    outputs = {
        "policy_summary": summary,
        "policy_by_regime_tau": by_regime_tau,
        "policy_by_state_confidence": by_state_conf,
        "policy_by_override_dimensions": by_policy_dimensions,
        "candidate_override_cells": candidate_cells,
        "state_duration_summary": state_duration,
        "fold_level_stability": fold_level_stability,
        "transition_matrix_summary": transition_summary,
    }
    for name, table in outputs.items():
        write_table(table, output_dir / f"{name}.csv", fmt="csv")
    (output_dir / "readme_summary.txt").write_text(
        "HMM policy replay is offline analysis only. Candidate override cells are reports, not live permissions.\n",
        encoding="utf-8",
    )
    return outputs
