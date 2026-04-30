import inspect
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research import hmm_decision_replay_pipeline as pipeline
from src.research.decision_replay_adapter import DecisionReplayConfig
from src.research.hmm_decision_replay_pipeline import (
    HMMDecisionReplayConfig,
    align_hmm_decision_schema,
    run_hmm_decision_replay,
    validate_hmm_decision_replay_schema,
)


def _row(**overrides):
    base = {
        "timestamp": "2026-05-01T12:00:00Z",
        "market_id": "M1",
        "p_yes": 0.60,
        "p_no": 0.40,
        "q_yes_ask": 0.50,
        "q_no_ask": 0.50,
        "tau_minutes": 20,
        "tau_bucket": "mid",
        "state": 1,
        "filtered_map_confidence": 0.90,
        "hmm_next_same_state_confidence": 0.85,
        "map_state_persistence": 3,
        "regime_policy_state": "state_1_confident",
        "expected_log_growth": 0.02,
        "conservative_expected_log_growth": 0.01,
        "tail_veto_flag": False,
        "polarization_veto_flag": False,
        "reversal_veto_flag": False,
        "quote_quality_pass": True,
    }
    base.update(overrides)
    return base


def _config(**overrides):
    base = HMMDecisionReplayConfig(
        decision_replay=DecisionReplayConfig(),
        allow_missing_safety_fields=True,
    )
    merged = {**base.__dict__, **overrides}
    return HMMDecisionReplayConfig(**merged)


def test_alias_mapping_into_canonical_adapter_columns():
    aligned = align_hmm_decision_schema(pd.DataFrame([_row()]), config=_config())

    assert aligned.loc[0, "hmm_map_state"] == 1
    assert aligned.loc[0, "posterior_confidence"] == pytest.approx(0.90)
    assert aligned.loc[0, "next_same_state_confidence"] == pytest.approx(0.85)
    assert aligned.loc[0, "persistence_count"] == 3
    assert aligned.loc[0, "policy_state"] == "state_1_confident"


def test_strict_schema_catches_missing_probability_fields():
    issues = validate_hmm_decision_replay_schema(pd.DataFrame([_row(p_yes=None)]).drop(columns=["p_yes"]), config=_config())
    assert issues == ["missing probability fields: p_yes"]


def test_strict_schema_catches_missing_quote_fields():
    issues = validate_hmm_decision_replay_schema(
        pd.DataFrame([_row()]).drop(columns=["q_yes_ask", "q_no_ask"]),
        config=_config(),
    )
    assert issues == ["missing quote fields: q_yes|q_yes_ask, q_no|q_no_ask"]


def test_strict_schema_catches_missing_hmm_fields():
    issues = validate_hmm_decision_replay_schema(
        pd.DataFrame([_row()]).drop(columns=["state", "filtered_map_confidence", "hmm_next_same_state_confidence", "map_state_persistence"]),
        config=_config(),
    )
    assert "missing HMM fields: hmm_map_state, posterior_confidence, next_same_state_confidence, persistence_count" in issues


def test_strict_schema_catches_missing_expected_growth_fields_by_default():
    issues = validate_hmm_decision_replay_schema(
        pd.DataFrame([_row()]).drop(columns=["expected_log_growth", "conservative_expected_log_growth"]),
        config=_config(),
    )
    assert issues == ["missing expected-growth fields: expected_log_growth, conservative_expected_log_growth, expected_growth_passes"]


def test_permissive_config_can_allow_missing_hmm():
    issues = validate_hmm_decision_replay_schema(
        pd.DataFrame([_row()]).drop(columns=["state", "filtered_map_confidence", "hmm_next_same_state_confidence", "map_state_persistence"]),
        config=_config(allow_missing_hmm=True),
    )
    assert issues == []


def test_permissive_config_can_allow_missing_expected_growth():
    issues = validate_hmm_decision_replay_schema(
        pd.DataFrame([_row()]).drop(columns=["expected_log_growth", "conservative_expected_log_growth"]),
        config=_config(allow_missing_expected_growth=True),
    )
    assert issues == []


def test_end_to_end_synthetic_frame_produces_decision_outputs():
    results, summary = run_hmm_decision_replay(pd.DataFrame([_row(), _row(p_yes=0.35, p_no=0.65)]), config=_config())

    assert "decision_action" in results.columns
    assert results["decision_action"].tolist() == ["buy_yes", "buy_no"]
    assert summary["n_rows"] == 2


def test_outcome_aware_summary_works_when_outcome_exists():
    results, summary = run_hmm_decision_replay(
        pd.DataFrame([_row(realized_outcome=1), _row(p_yes=0.35, p_no=0.65, realized_outcome=0)]),
        config=_config(),
        require_outcome_metrics=True,
    )

    assert "simple_replay_pnl_proxy" in summary
    assert "brier_p_yes" in summary
    assert results["decision_simple_replay_pnl_proxy"].notna().all()


def test_outcome_metrics_skipped_when_absent_and_not_required():
    _, summary = run_hmm_decision_replay(pd.DataFrame([_row()]), config=_config(), require_outcome_metrics=False)
    assert "simple_replay_pnl_proxy" not in summary
    assert "brier_p_yes" not in summary


def test_cli_help_works(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scripts/run_hmm_decision_replay_pipeline.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT / "scripts" / "run_hmm_decision_replay_pipeline.py"), run_name="__main__")

    assert excinfo.value.code == 0
    assert "--require-outcome-metrics" in capsys.readouterr().out


def test_cli_writes_results_summary_schema_and_readme(tmp_path, monkeypatch):
    input_path = tmp_path / "walk_forward.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame([_row(realized_outcome=1), _row(p_yes=0.35, p_no=0.65, realized_outcome=0)]).to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/run_hmm_decision_replay_pipeline.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--results-format",
            "csv",
            "--require-outcome-metrics",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT / "scripts" / "run_hmm_decision_replay_pipeline.py"), run_name="__main__")

    assert excinfo.value.code == 0
    assert (output_dir / "hmm_decision_replay_results.csv").exists()
    assert (output_dir / "hmm_decision_replay_summary.json").exists()
    assert (output_dir / "schema_report.json").exists()
    assert (output_dir / "README.txt").exists()
    summary = json.loads((output_dir / "hmm_decision_replay_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((output_dir / "schema_report.json").read_text(encoding="utf-8"))
    assert summary["n_rows"] == 2
    assert schema["used_mappings"]["hmm_map_state"] == "state"


def test_pipeline_does_not_import_operational_live_or_archived_modules():
    source = inspect.getsource(pipeline)
    forbidden = [
        "storage",
        "execution",
        "wallet_state",
        "redeemer",
        "polymarket_client",
        "run_bot",
        "strategy_manager",
        "archive.",
    ]
    for token in forbidden:
        assert token not in source


def test_pipeline_does_not_mutate_p_yes_and_outputs_decision_columns():
    df = pd.DataFrame([_row(), _row(p_yes=0.35, p_no=0.65)])
    before = df["p_yes"].copy(deep=True)
    results, _ = run_hmm_decision_replay(df, config=_config())

    assert results["p_yes"].equals(before)
    assert {"decision_action", "decision_reason", "decision_blocking_reasons"}.issubset(results.columns)
