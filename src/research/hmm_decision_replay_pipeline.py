from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .decision_replay_adapter import (
    DecisionReplayConfig,
    evaluate_decision_replay_frame,
    summarize_decision_replay,
)
from .hmm_dataset import read_table


@dataclass(frozen=True)
class HMMDecisionReplayConfig:
    decision_replay: DecisionReplayConfig = field(default_factory=DecisionReplayConfig)
    hmm_map_state_aliases: tuple[str, ...] = ("hmm_map_state", "map_state", "state", "filtered_map_state")
    posterior_confidence_aliases: tuple[str, ...] = (
        "posterior_confidence",
        "hmm_map_confidence",
        "map_confidence",
        "filtered_map_confidence",
    )
    next_same_state_confidence_aliases: tuple[str, ...] = (
        "next_same_state_confidence",
        "hmm_next_same_state_confidence",
    )
    persistence_count_aliases: tuple[str, ...] = (
        "persistence_count",
        "hmm_map_state_persistence_count",
        "map_state_persistence",
    )
    policy_state_aliases: tuple[str, ...] = ("policy_state", "regime_policy_state")
    strict_schema: bool = True
    allow_missing_expected_growth: bool = False
    allow_missing_hmm: bool = False
    allow_missing_safety_fields: bool = True
    output_diagnostics: bool = True


def load_frame(path: str | Path) -> pd.DataFrame:
    return read_table(path)


def _first_existing(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _alias_groups(config: HMMDecisionReplayConfig) -> dict[str, tuple[str, ...]]:
    return {
        "hmm_map_state": config.hmm_map_state_aliases,
        "posterior_confidence": config.posterior_confidence_aliases,
        "next_same_state_confidence": config.next_same_state_confidence_aliases,
        "persistence_count": config.persistence_count_aliases,
        "policy_state": config.policy_state_aliases,
    }


def _outcome_columns(df: pd.DataFrame) -> list[str]:
    return [name for name in ["realized_outcome", "outcome"] if name in df.columns]


def build_schema_report(
    df: pd.DataFrame,
    *,
    config: HMMDecisionReplayConfig,
    require_outcome_metrics: bool = False,
) -> dict[str, Any]:
    columns = set(df.columns)
    alias_groups = _alias_groups(config)
    used_mappings: dict[str, str] = {}
    for canonical, aliases in alias_groups.items():
        match = _first_existing(columns, aliases)
        if match is not None:
            used_mappings[canonical] = match

    probability_missing = [name for name in ["p_yes"] if name not in columns]
    quote_missing: list[str] = []
    if not ({"q_yes", "q_yes_ask"} & columns):
        quote_missing.append("q_yes|q_yes_ask")
    if not ({"q_no", "q_no_ask"} & columns):
        quote_missing.append("q_no|q_no_ask")

    hmm_missing: list[str] = []
    if not config.allow_missing_hmm:
        for canonical, aliases in alias_groups.items():
            if canonical == "policy_state":
                continue
            if _first_existing(columns, aliases) is None:
                hmm_missing.append(canonical)

    expected_growth_missing: list[str] = []
    if not config.allow_missing_expected_growth:
        expected_candidates = {
            "expected_log_growth",
            "conservative_expected_log_growth",
            "expected_growth_passes",
            "expected_growth_pass",
        }
        if not (expected_candidates & columns):
            expected_growth_missing.extend(
                ["expected_log_growth", "conservative_expected_log_growth", "expected_growth_passes"]
            )

    safety_candidates = {
        "tail_veto_flag",
        "tail_veto",
        "polarization_veto_flag",
        "polarization_veto",
        "reversal_veto_flag",
        "reversal_veto",
        "quote_quality_pass",
        "quote_quality_veto_flag",
        "quote_quality_veto",
    }
    safety_missing = [] if config.allow_missing_safety_fields or (safety_candidates & columns) else ["safety_veto_fields"]
    outcome_missing = [] if _outcome_columns(df) else ["realized_outcome|outcome"]

    return {
        "row_count": int(len(df)),
        "columns": sorted(df.columns.tolist()),
        "used_mappings": used_mappings,
        "missing_probability_fields": probability_missing,
        "missing_quote_fields": quote_missing,
        "missing_hmm_fields": hmm_missing,
        "missing_expected_growth_fields": expected_growth_missing,
        "missing_safety_fields": safety_missing,
        "missing_outcome_fields": outcome_missing if require_outcome_metrics else [],
        "outcome_fields_present": _outcome_columns(df),
    }


def validate_hmm_decision_replay_schema(
    df: pd.DataFrame,
    *,
    config: HMMDecisionReplayConfig,
    require_outcome_metrics: bool = False,
) -> list[str]:
    report = build_schema_report(df, config=config, require_outcome_metrics=require_outcome_metrics)
    issues: list[str] = []
    for key, label in [
        ("missing_probability_fields", "missing probability fields"),
        ("missing_quote_fields", "missing quote fields"),
        ("missing_hmm_fields", "missing HMM fields"),
        ("missing_expected_growth_fields", "missing expected-growth fields"),
        ("missing_safety_fields", "missing safety fields"),
        ("missing_outcome_fields", "missing outcome fields"),
    ]:
        if report[key]:
            issues.append(f"{label}: {', '.join(report[key])}")
    return issues


def align_hmm_decision_schema(df: pd.DataFrame, *, config: HMMDecisionReplayConfig) -> pd.DataFrame:
    out = df.copy()
    used_mappings: dict[str, str] = {}
    for canonical, aliases in _alias_groups(config).items():
        if canonical in out.columns:
            used_mappings[canonical] = canonical
            continue
        match = _first_existing(set(out.columns), aliases)
        if match is not None:
            out[canonical] = out[match]
            used_mappings[canonical] = match
    out.attrs["schema_mappings"] = used_mappings
    out.attrs["boundary_note"] = "decision-layer inputs only; no HMM feature/model mutation"
    return out


def _effective_adapter_config(config: HMMDecisionReplayConfig) -> DecisionReplayConfig:
    base = config.decision_replay
    return DecisionReplayConfig(
        yes_edge_threshold=base.yes_edge_threshold,
        no_edge_threshold=base.no_edge_threshold,
        min_posterior_confidence=base.min_posterior_confidence,
        min_next_same_state_confidence=base.min_next_same_state_confidence,
        min_persistence=base.min_persistence,
        require_expected_growth_pass=base.require_expected_growth_pass,
        default_block_tail_veto=base.default_block_tail_veto,
        default_block_reversal_veto=base.default_block_reversal_veto,
        default_block_quote_quality_veto=base.default_block_quote_quality_veto,
        allow_missing_hmm_state=config.allow_missing_hmm,
        allow_missing_expected_growth=config.allow_missing_expected_growth,
        allow_missing_safety_fields=config.allow_missing_safety_fields,
        output_diagnostics=config.output_diagnostics,
    )


def run_hmm_decision_replay(
    df: pd.DataFrame,
    *,
    config: HMMDecisionReplayConfig,
    require_outcome_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aligned = align_hmm_decision_schema(df, config=config)
    issues = validate_hmm_decision_replay_schema(
        aligned,
        config=config,
        require_outcome_metrics=require_outcome_metrics,
    )
    if config.strict_schema and issues:
        raise ValueError("; ".join(issues))

    p_yes_before = aligned["p_yes"].copy(deep=True) if "p_yes" in aligned.columns else None
    results = evaluate_decision_replay_frame(aligned, config=_effective_adapter_config(config))
    if p_yes_before is not None and not results["p_yes"].equals(p_yes_before):
        raise AssertionError("pipeline must not mutate p_yes")
    summary = summarize_decision_replay(results)
    report = build_schema_report(aligned, config=config, require_outcome_metrics=require_outcome_metrics)
    report["issues"] = issues
    report["boundary_validation"] = {
        "mutates_hmm_features": False,
        "mutates_hmm_model": False,
        "mutates_probability_outputs": False,
        "queries_live_state": False,
    }
    summary["schema_report"] = report
    return results, summary
