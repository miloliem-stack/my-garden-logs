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
from src.research.hmm_features import A_FEATURE_COLUMNS, D_FEATURE_COLUMNS, HMM_FEATURE_COLUMNS, add_training_entropy_percentiles, build_hmm_feature_frame, hmm_feature_matrix, transition_entropy_15state
from src.research.hmm_policy_replay import apply_policy_flags, report_policy_replay
from src.research.hmm_visuals import compress_regime_segments, state_color
from src.research.hmm_walk_forward import _clean_feature_frame, run_walk_forward


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


def test_all_hmm_features_are_causal_when_future_rows_change():
    df = _klines(180)
    cutoff = 90
    baseline = build_hmm_feature_frame(df)
    changed = df.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close"]] += 25000.0
    changed.loc[cutoff + 1 :, "volume"] *= 100.0
    mutated = build_hmm_feature_frame(changed)

    focus = [
        "forecast_abs_move_5m_baseline",
        "forecast_abs_move_15m_baseline",
        "forecast_large_move_probability_baseline",
        "transition_entropy_percentile",
        "spectral_entropy_32",
        "low_freq_power_ratio_32",
        "high_freq_power_ratio_32",
        "smoothness_score_32",
        "entropy_slope",
        "entropy_shock_score",
    ]
    for col in HMM_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(baseline.loc[:cutoff, col], mutated.loc[:cutoff, col], check_names=False)
    for col in focus:
        pd.testing.assert_series_equal(baseline.loc[:cutoff, col], mutated.loc[:cutoff, col], check_names=False)


def test_transition_entropy_handles_missing_states():
    entropy = transition_entropy_15state([0, 0, 0, 0, 0], window=5)
    assert entropy.dropna().between(0.0, 1.0).all()


def test_hmm_feature_matrix_is_strict_a_d_whitelist():
    features = build_hmm_feature_frame(_klines())
    forbidden = {
        "q_yes": 0.5,
        "q_no": 0.5,
        "p_yes": 0.55,
        "p_no": 0.45,
        "edge_yes_ask": 0.02,
        "tau_minutes": 15,
        "realized_outcome": 1,
        "realized_pnl_binary": 1.0,
        "inventory_qty": 2,
        "wallet_balance": 100,
        "open_order_count": 1,
        "fill_qty": 1,
    }
    for col, value in forbidden.items():
        features[col] = value

    matrix = hmm_feature_matrix(features)
    assert list(matrix.columns) == A_FEATURE_COLUMNS + D_FEATURE_COLUMNS
    assert set(matrix.columns).isdisjoint(forbidden)
    assert list(hmm_feature_matrix(features, feature_columns=A_FEATURE_COLUMNS).columns) == A_FEATURE_COLUMNS
    with pytest.raises(ValueError, match="forbidden|non A/D"):
        hmm_feature_matrix(features, feature_columns=["r_1m", "p_yes"])
    with pytest.raises(ValueError, match="forbidden|non A/D"):
        hmm_feature_matrix(features, feature_columns=["r_1m", "wallet_balance"])


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
    assert metadata["folds"][0]["posterior_method"] == "filtered_forward_probabilities_no_viterbi_no_smoothing"


def test_walk_forward_scaler_and_hmm_fit_ignore_test_rows(tmp_path):
    dataset = build_hmm_feature_frame(_klines(360))
    input_a = tmp_path / "replay_a.csv"
    input_b = tmp_path / "replay_b.csv"
    changed_test = dataset.copy()
    test_mask = changed_test["timestamp"] >= pd.Timestamp("2026-01-01T03:01:00Z")
    for col in HMM_FEATURE_COLUMNS:
        if changed_test[col].dtype == bool:
            changed_test.loc[test_mask, col] = ~changed_test.loc[test_mask, col]
        else:
            changed_test.loc[test_mask, col] = changed_test.loc[test_mask, col].fillna(0.0) + 999.0
    dataset.to_csv(input_a, index=False)
    changed_test.to_csv(input_b, index=False)

    kwargs = dict(
        n_states=3,
        train_start="2026-01-01T00:00:00Z",
        train_end="2026-01-01T03:00:00Z",
        test_start="2026-01-01T03:01:00Z",
        test_end="2026-01-01T05:00:00Z",
    )
    run_walk_forward(input_a, tmp_path / "run_a", **kwargs)
    run_walk_forward(input_b, tmp_path / "run_b", **kwargs)
    meta_a = json.loads((tmp_path / "run_a" / "fold_metadata.json").read_text(encoding="utf-8"))["folds"][0]
    meta_b = json.loads((tmp_path / "run_b" / "fold_metadata.json").read_text(encoding="utf-8"))["folds"][0]
    train_mask = (dataset["timestamp"] >= pd.Timestamp(kwargs["train_start"])) & (dataset["timestamp"] <= pd.Timestamp(kwargs["train_end"]))
    train = add_training_entropy_percentiles(dataset, train_mask=train_mask).loc[train_mask]
    expected_mean = _clean_feature_frame(train, HMM_FEATURE_COLUMNS).mean().to_numpy()

    assert np.allclose(meta_a["scaler_mean"], expected_mean)
    assert meta_a["scaler_mean"] == meta_b["scaler_mean"]
    assert meta_a["hmm_transition_matrix"] == meta_b["hmm_transition_matrix"]
    assert meta_a["hmm_means"] == meta_b["hmm_means"]
    assert meta_a["hmm_covariances"] == meta_b["hmm_covariances"]
    with pytest.raises(ValueError, match="forbidden|non A/D"):
        run_walk_forward(input_a, tmp_path / "run_forbidden", feature_columns=["r_1m", "p_yes"], **kwargs)


def test_policy_replay_blocks_growth_default_vetoes_hmm_abstention_and_reports(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="min", tz="UTC"),
            "vanilla_action": ["buy_yes", "buy_no", "buy_yes", "buy_yes", "buy_yes", "buy_yes", "buy_yes"],
            "vanilla_side": ["YES", "NO", "YES", "YES", "YES", "YES", "YES"],
            "conservative_expected_log_growth": [-0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03],
            "tail_veto_flag": [False, True, False, False, False, False, False],
            "polarization_veto_flag": [False, False, True, False, False, False, False],
            "reversal_veto_flag": [False, False, False, True, False, False, False],
            "regime_policy_state": ["state_0_confident"] * 7,
            "hmm_map_confidence": [0.9, 0.9, 0.9, 0.9, 0.4, 0.9, 0.9],
            "hmm_next_same_state_confidence": [0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.9],
            "hmm_map_state_persistence_count": [3, 3, 3, 3, 3, 3, 1],
            "hmm_map_state": [0, 0, 0, 0, 1, 1, 2],
            "tau_bucket": ["mid", "mid", "late", "late", "mid", "mid", "early"],
            "q_tail": [0.2, 0.2, 0.7, 0.7, 0.2, 0.2, 0.9],
            "realized_pnl_binary": [1.0, -1.0, 0.5, 0.25, 1.0, 1.0, -0.5],
            "realized_outcome": [1, 0, 1, 1, 1, 1, 0],
            "p_yes": [0.6, 0.4, 0.6, 0.6, 0.7, 0.7, 0.45],
            "q_yes_ask": [0.55, 0.45, 0.55, 0.55, 0.65, 0.65, 0.5],
            "edge_yes_ask": [0.05, 0.0, 0.05, 0.05, 0.05, 0.05, -0.05],
            "edge_no_ask": [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0],
            "fold_id": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    flagged = apply_policy_flags(df)
    assert not flagged.loc[0, "policy1_trade"]
    assert not flagged.loc[1, "policy2_trade"]
    assert not flagged.loc[2, "policy2_trade"]
    assert not flagged.loc[3, "policy2_trade"]
    assert not flagged.loc[4, "policy3_trade"]
    assert not flagged.loc[5, "policy3_trade"]
    assert not flagged.loc[6, "policy3_trade"]

    input_path = tmp_path / "hmm.csv"
    df.to_csv(input_path, index=False)
    outputs = report_policy_replay(input_path, tmp_path / "report", min_samples=1)
    for name in [
        "policy_summary.csv",
        "policy_by_regime_tau.csv",
        "policy_by_state_confidence.csv",
        "policy_by_override_dimensions.csv",
        "candidate_override_cells.csv",
        "state_duration_summary.csv",
        "fold_level_stability.csv",
        "transition_matrix_summary.csv",
        "readme_summary.txt",
    ]:
        assert (tmp_path / "report" / name).exists()
    summary = pd.read_csv(tmp_path / "report" / "policy_summary.csv")
    required_metric_cols = {
        "n_trades",
        "realized_pnl",
        "realized_pnl_per_trade",
        "hit_rate",
        "brier_p_yes",
        "log_loss_p_yes",
        "market_brier_q_yes_ask",
        "market_log_loss_q_yes_ask",
        "bootstrap_ci_status",
    }
    assert required_metric_cols.issubset(summary.columns)
    dimensions = pd.read_csv(tmp_path / "report" / "policy_by_override_dimensions.csv")
    assert {"regime_policy_state", "hmm_map_state", "confidence_bucket", "tau_bucket", "q_tail_bucket", "side", "favoredness", "reversal_status"}.issubset(dimensions.columns)
    assert (tmp_path / "report" / "fold_level_stability.csv").exists()
    assert {"policy_summary", "candidate_override_cells"}.issubset(outputs.keys())


def test_override_cells_are_analysis_only_and_do_not_mutate_replay_decisions(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
            "vanilla_action": ["buy_yes", "buy_yes", "buy_yes"],
            "vanilla_side": ["YES", "YES", "YES"],
            "conservative_expected_log_growth": [0.02, 0.02, 0.02],
            "tail_veto_flag": [True, True, True],
            "polarization_veto_flag": [False, False, False],
            "reversal_veto_flag": [False, False, False],
            "regime_policy_state": ["state_1_confident"] * 3,
            "hmm_map_confidence": [0.95, 0.95, 0.95],
            "hmm_next_same_state_confidence": [0.95, 0.95, 0.95],
            "hmm_map_state_persistence_count": [3, 4, 5],
            "hmm_map_state": [1, 1, 1],
            "tau_bucket": ["mid"] * 3,
            "q_tail": [0.8, 0.8, 0.8],
            "realized_pnl_binary": [1.0, 1.0, 1.0],
            "fold_id": [0, 1, 2],
        }
    )
    input_path = tmp_path / "override.csv"
    df.to_csv(input_path, index=False)
    before = apply_policy_flags(df)
    outputs = report_policy_replay(input_path, tmp_path / "report", min_samples=1)
    after = apply_policy_flags(df)

    assert before["policy2_trade"].sum() == 0
    assert before["policy3_trade"].sum() == 0
    assert after["policy2_trade"].equals(before["policy2_trade"])
    assert after["policy3_trade"].equals(before["policy3_trade"])
    assert not outputs["candidate_override_cells"].empty


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


def test_plot_cli_help_works_without_importing_plotly(capsys):
    argv = ["scripts/plot_hmm_regime_overlay.py", "--help"]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(ROOT / "scripts" / "plot_hmm_regime_overlay.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
    assert exc.value.code == 0
    assert "Plot BTC OHLCV" in capsys.readouterr().out
