import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.hmm_dataset import build_hmm_replay_dataset
from src.research.hmm_features import HMM_FEATURE_COLUMNS, build_hmm_feature_frame, hmm_feature_matrix, transition_entropy_15state
from src.research.hmm_policy_replay import apply_policy_flags, report_policy_replay
from src.research.hmm_visuals import compress_regime_segments, state_color
from src.research.hmm_walk_forward import run_walk_forward


def _klines(n=220, start="2026-01-01T00:00:00Z"):
    ts = pd.date_range(start, periods=n, freq="min", tz="UTC")
    base = 50000 + np.cumsum(np.sin(np.arange(n) / 9.0) * 3 + np.linspace(-1, 1, n))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 8,
            "low": base - 8,
            "close": base + np.sin(np.arange(n) / 3.0),
            "volume": 10 + (np.arange(n) % 17),
        }
    )


def test_feature_builder_is_past_only_and_entropy_bounded():
    df = _klines()
    features = build_hmm_feature_frame(df)
    changed = df.copy()
    changed.loc[120:, "close"] += 10000
    changed_features = build_hmm_feature_frame(changed)

    pd.testing.assert_series_equal(features.loc[:80, "r_15m"], changed_features.loc[:80, "r_15m"], check_names=False)
    pd.testing.assert_series_equal(features.loc[:80, "realized_vol_30m"], changed_features.loc[:80, "realized_vol_30m"], check_names=False)
    entropy = features["transition_entropy_15state"].dropna()
    assert ((entropy >= 0.0) & (entropy <= 1.0)).all()
    assert features["transition_entropy_percentile"].dropna().between(0.0, 1.0).all()


def test_transition_entropy_handles_missing_states():
    entropy = transition_entropy_15state([0, 0, 0, 0, 0], window=5)
    assert entropy.dropna().between(0.0, 1.0).all()


def test_replay_exporter_csv_parquet_and_hmm_matrix_excludes_policy_columns(tmp_path):
    klines = _klines()
    csv_path = tmp_path / "klines.csv"
    pq_path = tmp_path / "klines.parquet"
    klines.to_csv(csv_path, index=False)
    klines.to_parquet(pq_path, index=False)
    decision = pd.DataFrame(
        {
            "timestamp": klines["timestamp"].iloc[50:55],
            "vanilla_action": ["buy_yes"] * 5,
            "p_yes": [0.55] * 5,
            "realized_outcome": [1] * 5,
        }
    )
    decision_path = tmp_path / "decision.jsonl"
    decision.to_json(decision_path, orient="records", lines=True, date_format="iso")

    out_csv = build_hmm_replay_dataset(csv_path, decision_log_path=decision_path)
    out_pq = build_hmm_replay_dataset(pq_path)
    assert len(out_csv) == len(klines)
    assert len(out_pq) == len(klines)
    matrix = hmm_feature_matrix(out_csv)
    assert list(matrix.columns) == HMM_FEATURE_COLUMNS
    assert "p_yes" not in matrix.columns
    assert "realized_outcome" not in matrix.columns


def test_walk_forward_outputs_filtered_posteriors_and_uncertain_policy_state(tmp_path):
    dataset = build_hmm_feature_frame(_klines(360))
    dataset["vanilla_action"] = "buy_yes"
    dataset["conservative_expected_log_growth"] = 0.01
    input_path = tmp_path / "replay.csv"
    dataset.to_csv(input_path, index=False)

    result = run_walk_forward(
        input_path,
        tmp_path / "run",
        n_states=3,
        train_start="2026-01-01T00:00:00Z",
        train_end="2026-01-01T03:00:00Z",
        test_start="2026-01-01T03:01:00Z",
        test_end="2026-01-01T05:00:00Z",
        confidence_threshold=1.01,
    )
    assert {"hmm_state_prob_0", "hmm_next_state_prob_0", "hmm_map_state", "hmm_entropy"}.issubset(result.columns)
    assert (result["regime_policy_state"] == "transition_uncertain").all()
    state = int(result["hmm_map_state"].iloc[0])
    assert result["hmm_next_same_state_confidence"].iloc[0] == pytest.approx(result[f"hmm_next_state_prob_{state}"].iloc[0])
    metadata = json.loads((tmp_path / "run" / "fold_metadata.json").read_text(encoding="utf-8"))
    assert metadata["folds"][0]["train_period"][0].startswith("2026-01-01T00:00:00")
    assert "hmm_transition_matrix" in metadata["folds"][0]


def test_policy_replay_blocks_growth_and_default_vetoes_and_reports(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "vanilla_action": ["buy_yes", "buy_no", "hold", "buy_yes"],
            "vanilla_side": ["YES", "NO", None, "YES"],
            "conservative_expected_log_growth": [-0.01, 0.02, 0.0, 0.03],
            "tail_veto_flag": [False, True, False, False],
            "polarization_veto_flag": [False, False, False, False],
            "reversal_veto_flag": [False, False, False, False],
            "regime_policy_state": ["state_0_confident", "state_0_confident", "transition_uncertain", "transition_uncertain"],
            "hmm_map_confidence": [0.9, 0.9, 0.5, 0.9],
            "hmm_next_same_state_confidence": [0.9, 0.9, 0.4, 0.9],
            "hmm_map_state": [0, 0, 1, 1],
            "tau_bucket": ["mid"] * 4,
            "q_tail": [0.2] * 4,
            "realized_pnl_binary": [1.0, -1.0, 0.0, 1.0],
            "realized_outcome": [1, 0, 1, 1],
            "p_yes": [0.6, 0.4, 0.5, 0.7],
            "q_yes_ask": [0.55, 0.45, 0.5, 0.65],
            "fold_id": [0, 0, 0, 1],
        }
    )
    flagged = apply_policy_flags(df)
    assert not flagged.loc[0, "policy1_trade"]
    assert not flagged.loc[1, "policy2_trade"]
    assert not flagged.loc[3, "policy3_trade"]

    input_path = tmp_path / "hmm.csv"
    df.to_csv(input_path, index=False)
    report_policy_replay(input_path, tmp_path / "report", min_samples=1)
    for name in [
        "policy_summary.csv",
        "policy_by_regime_tau.csv",
        "policy_by_state_confidence.csv",
        "candidate_override_cells.csv",
        "state_duration_summary.csv",
        "transition_matrix_summary.csv",
        "readme_summary.txt",
    ]:
        assert (tmp_path / "report" / name).exists()


def test_visual_segment_compression_and_html_creation(tmp_path):
    df = _klines(10)
    df["regime_policy_state"] = ["quiet_confident"] * 3 + ["transition_uncertain"] * 4 + ["high_entropy"] * 3
    df["hmm_map_confidence"] = np.linspace(0.5, 0.9, 10)
    input_path = tmp_path / "plot.csv"
    output_path = tmp_path / "plot.html"
    df.to_csv(input_path, index=False)
    segments = compress_regime_segments(df)
    assert len(segments) == 3
    assert state_color("transition_uncertain") == state_color("transition")

    pytest.importorskip("plotly")
    argv = [
        "scripts/plot_hmm_regime_overlay.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(ROOT / "scripts" / "plot_hmm_regime_overlay.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
    assert output_path.exists()
    assert "<html" in output_path.read_text(encoding="utf-8").lower()

