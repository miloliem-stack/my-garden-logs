import inspect
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research import decision_replay_adapter as adapter
from src.research.decision_replay_adapter import (
    DecisionReplayConfig,
    build_decision_input_from_row,
    evaluate_decision_replay_frame,
    summarize_decision_replay,
)


def _row(**overrides):
    base = {
        "timestamp": "2026-04-30T12:00:00Z",
        "market_id": "M1",
        "p_yes": 0.60,
        "p_no": 0.40,
        "engine_name": "gaussian_vol",
        "q_yes": 0.50,
        "q_no": 0.50,
        "tau_minutes": 20,
        "tau_bucket": "mid",
        "hmm_map_state": 1,
        "posterior_confidence": 0.90,
        "next_same_state_confidence": 0.85,
        "persistence_count": 3,
        "policy_state": "state_1_confident",
        "expected_log_growth": 0.02,
        "conservative_expected_log_growth": 0.01,
        "expected_growth_passes": True,
        "tail_veto_flag": False,
        "polarization_veto_flag": False,
        "reversal_veto_flag": False,
        "quote_quality_pass": True,
    }
    base.update(overrides)
    return base


def _config(**overrides):
    base = DecisionReplayConfig()
    return DecisionReplayConfig(**{**base.__dict__, **overrides})


def test_row_mapping_creates_valid_decision_input():
    decision_input = build_decision_input_from_row(_row(), config=_config())

    assert decision_input.market_id == "M1"
    assert decision_input.probability.p_yes == pytest.approx(0.60)
    assert decision_input.quote.q_yes == pytest.approx(0.50)
    assert decision_input.tau_policy.tau_bucket == "mid"
    assert decision_input.hmm_policy_state.policy_state == "state_1_confident"
    assert decision_input.expected_growth.passes is True


def test_positive_yes_edge_passes_when_all_gates_pass():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row()]), config=_config())

    assert bool(out.loc[0, "decision_allowed"]) is True
    assert out.loc[0, "decision_action"] == "buy_yes"


def test_positive_no_edge_passes_when_all_gates_pass():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row(p_yes=0.35, p_no=0.65)]), config=_config())

    assert bool(out.loc[0, "decision_allowed"]) is True
    assert out.loc[0, "decision_action"] == "buy_no"


def test_no_edge_blocks():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row(p_yes=0.51, p_no=0.49)]), config=_config())

    assert bool(out.loc[0, "decision_allowed"]) is False
    assert out.loc[0, "decision_reason"] == "no_edge_above_threshold"


def test_missing_hmm_state_blocks_by_default():
    out = evaluate_decision_replay_frame(
        pd.DataFrame([_row(hmm_map_state=None, posterior_confidence=None, next_same_state_confidence=None, persistence_count=None)]),
        config=_config(),
    )

    blockers = json.loads(out.loc[0, "decision_blocking_reasons"])
    assert "hmm_transition_uncertain" in blockers


def test_missing_expected_growth_blocks_by_default():
    out = evaluate_decision_replay_frame(
        pd.DataFrame([_row(expected_log_growth=None, conservative_expected_log_growth=None, expected_growth_passes=None)]),
        config=_config(),
    )

    assert out.loc[0, "decision_reason"] == "expected_growth_veto"


def test_expected_growth_nonpositive_blocks():
    out = evaluate_decision_replay_frame(
        pd.DataFrame([_row(conservative_expected_log_growth=0.0, expected_growth_passes=False)]),
        config=_config(),
    )

    assert out.loc[0, "decision_reason"] == "expected_growth_veto"


def test_tail_and_polarization_veto_blocks():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row(tail_veto_flag=True, polarization_veto_flag=True)]), config=_config())

    blockers = json.loads(out.loc[0, "decision_blocking_reasons"])
    assert "tail_veto" in blockers
    assert "polarization_veto" in blockers


def test_reversal_veto_blocks():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row(reversal_veto_flag=True)]), config=_config())

    assert out.loc[0, "decision_reason"] == "reversal_veto"


def test_quote_quality_veto_blocks():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row(quote_quality_pass=False)]), config=_config())

    assert out.loc[0, "decision_reason"] == "quote_quality_veto"


def test_low_hmm_thresholds_block():
    out = evaluate_decision_replay_frame(
        pd.DataFrame(
            [
                _row(posterior_confidence=0.60),
                _row(next_same_state_confidence=0.50),
                _row(persistence_count=1),
            ]
        ),
        config=_config(),
    )

    assert out.loc[0, "decision_reason"] == "hmm_low_posterior_confidence"
    assert out.loc[1, "decision_reason"] == "hmm_low_next_same_state_confidence"
    assert out.loc[2, "decision_reason"] == "hmm_insufficient_persistence"


def test_multiple_blockers_are_preserved():
    out = evaluate_decision_replay_frame(
        pd.DataFrame(
            [
                _row(
                    p_yes=0.51,
                    p_no=0.49,
                    expected_log_growth=None,
                    conservative_expected_log_growth=None,
                    expected_growth_passes=None,
                    tail_veto_flag=True,
                    reversal_veto_flag=True,
                    quote_quality_pass=False,
                    posterior_confidence=0.50,
                    next_same_state_confidence=0.50,
                    persistence_count=0,
                    policy_state="transition_uncertain",
                )
            ]
        ),
        config=_config(),
    )

    blockers = json.loads(out.loc[0, "decision_blocking_reasons"])
    assert blockers == [
        "no_edge_above_threshold",
        "expected_growth_veto",
        "tail_veto",
        "reversal_veto",
        "quote_quality_veto",
        "hmm_transition_uncertain",
        "hmm_low_posterior_confidence",
        "hmm_low_next_same_state_confidence",
        "hmm_insufficient_persistence",
    ]


def test_output_frame_contains_expected_columns():
    out = evaluate_decision_replay_frame(pd.DataFrame([_row()]), config=_config())

    expected = {
        "decision_action",
        "decision_allowed",
        "decision_chosen_side",
        "decision_reason",
        "decision_blocking_reasons",
        "decision_edge_yes",
        "decision_edge_no",
        "decision_hmm_policy_state",
        "decision_expected_growth_passes",
        "decision_tail_veto_blocked",
        "decision_reversal_veto_blocked",
        "decision_quote_quality_blocked",
    }
    assert expected.issubset(out.columns)


def test_summary_contains_reason_and_blocker_counts():
    out = evaluate_decision_replay_frame(
        pd.DataFrame(
            [
                _row(),
                _row(p_yes=0.51, p_no=0.49),
                _row(reversal_veto_flag=True),
            ]
        ),
        config=_config(),
    )
    summary = summarize_decision_replay(out)

    assert summary["n_rows"] == 3
    assert summary["n_allowed"] == 1
    assert summary["reason_counts"]["ok"] == 1
    assert summary["blocking_reason_counts"]["reversal_veto"] == 1


def test_outcome_aware_summary_computes_metrics_only_when_outcome_exists():
    with_outcome = evaluate_decision_replay_frame(
        pd.DataFrame(
            [
                _row(realized_outcome=1),
                _row(p_yes=0.35, p_no=0.65, realized_outcome=0),
            ]
        ),
        config=_config(),
    )
    without_outcome = evaluate_decision_replay_frame(pd.DataFrame([_row()]), config=_config())

    summary_with = summarize_decision_replay(with_outcome)
    summary_without = summarize_decision_replay(without_outcome.drop(columns=["realized_outcome"], errors="ignore"))

    assert "simple_replay_pnl_proxy" in summary_with
    assert "brier_p_yes" in summary_with
    assert "simple_replay_pnl_proxy" not in summary_without
    assert "brier_p_yes" not in summary_without


def test_adapter_does_not_import_operational_or_archived_modules():
    source = inspect.getsource(adapter)

    forbidden = [
        "storage",
        "execution",
        "wallet_state",
        "redeemer",
        "polymarket_client",
        "archive.",
        "policy_replay",
    ]
    for token in forbidden:
        assert token not in source


def test_cli_help_works(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scripts/run_decision_replay_adapter.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT / "scripts" / "run_decision_replay_adapter.py"), run_name="__main__")

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--input" in out
    assert "--output-dir" in out


def test_cli_writes_results_and_summary_artifacts(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            _row(realized_outcome=1),
            _row(p_yes=0.35, p_no=0.65, realized_outcome=0),
        ]
    )
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "out"
    df.to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/run_decision_replay_adapter.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "csv",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT / "scripts" / "run_decision_replay_adapter.py"), run_name="__main__")

    assert excinfo.value.code == 0
    results_path = output_dir / "decision_replay_results.csv"
    summary_path = output_dir / "decision_replay_summary.json"
    assert results_path.exists()
    assert summary_path.exists()
    results = pd.read_csv(results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "decision_action" in results.columns
    assert summary["n_rows"] == 2
