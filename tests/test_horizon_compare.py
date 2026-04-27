from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import probability_backtest
from src.horizon_compare import build_horizon_splits, run_horizon_comparison, select_horizon_splits
from src.horizon_event_dataset import build_fixed_horizon_event_dataset
from src.horizon_feature_builder import build_horizon_event_features


def _synthetic_minute_df(days: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=days * 24 * 60, freq="min", tz="UTC")
    base = 50000 + np.linspace(0, 250, len(idx))
    wave = 150 * np.sin(np.arange(len(idx)) / 120.0)
    close = base + wave
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 5
    low = np.minimum(open_, close) - 5
    volume = 100 + 10 * np.cos(np.arange(len(idx)) / 30.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_build_fixed_horizon_event_dataset_for_4h():
    minute_df = _synthetic_minute_df(days=2)
    events = build_fixed_horizon_event_dataset(minute_df, horizon="4h", decision_step_minutes=60)
    assert not events.empty
    assert set(events["horizon"].unique()) == {"4h"}
    assert int(events.iloc[0]["tau_minutes"]) == 240
    assert float(events.iloc[0]["strike_price"]) == float(minute_df.iloc[0]["open"])


def test_run_horizon_comparison_smoke(tmp_path):
    minute_df = _synthetic_minute_df(days=8)
    out_dir = tmp_path / "horizon_compare"
    report = run_horizon_comparison(
        minute_df=minute_df,
        horizons=["1h", "4h"],
        output_dir=out_dir,
        model_names=["naive_directional", "ar_only"],
        decision_step_overrides={"1h": 15, "4h": 60},
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
        edge_thresholds=[0.05],
    )
    assert Path(report["summary_path"]).exists()
    assert Path(report["by_model_by_horizon_path"]).exists()
    summary = pd.read_csv(report["summary_path"])
    assert {"horizon", "model_name", "brier_score", "log_loss", "accuracy"}.issubset(summary.columns)
    assert set(summary["horizon"]) == {"1h", "4h"}


def test_base_result_row_handles_duplicate_label_event_series():
    event = pd.Series(
        [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z"), 0, 5, 50000.0, 50010.0, 50020.0, 1, 7],
        index=[
            "event_hour_start",
            "event_hour_end",
            "minute_in_hour",
            "tau_minutes",
            "strike_price",
            "spot_now",
            "settlement_price",
            "realized_yes",
            "minute_in_hour",
        ],
    )
    row = probability_backtest._base_result_row(event, pd.Timestamp("2026-01-01T00:05:00Z"), 100, True)
    assert row["minute_in_hour"] == 0
    assert row["tau_minutes"] == 5


def test_horizon_feature_builder_regression_for_minute_overlap():
    minute_df = _synthetic_minute_df(days=2)
    events = build_fixed_horizon_event_dataset(minute_df, horizon="4h", decision_step_minutes=60)
    feature_df = build_horizon_event_features(minute_df, events)
    assert "minute_in_event" in feature_df.columns
    assert "decision_minute_of_hour" in feature_df.columns
    assert list(feature_df.columns).count("minute_in_hour") == 0


def test_probability_backtest_rejects_duplicate_columns():
    minute_df = _synthetic_minute_df(days=2)
    events = build_fixed_horizon_event_dataset(minute_df, horizon="1h", decision_step_minutes=30)
    bad_events = events.rename(columns={"event_start": "event_hour_start", "event_end": "event_hour_end", "minute_in_event": "minute_in_hour"})
    bad_events["minute_in_hour_copy"] = bad_events["minute_in_hour"]
    bad_events = bad_events.rename(columns={"minute_in_hour_copy": "minute_in_hour"})
    try:
        probability_backtest.run_probability_backtest(
            close_series=minute_df["close"],
            events=bad_events,
            fit_window=100,
            min_history=100,
        )
        assert False, "expected duplicate-column guard to raise"
    except ValueError as exc:
        assert "Duplicate columns passed to probability backtest" in str(exc)


def test_run_horizon_comparison_emits_progress_callback(tmp_path):
    minute_df = _synthetic_minute_df(days=6)
    progress_events = []
    run_horizon_comparison(
        minute_df=minute_df,
        horizons=["1h"],
        output_dir=tmp_path / "horizon_compare_progress",
        model_names=["naive_directional", "gaussian_vol"],
        decision_step_overrides={"1h": 30},
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
        edge_thresholds=[0.05],
        progress_callback=progress_events.append,
        heartbeat_seconds=0.0,
    )
    assert progress_events
    assert any(event.get("stage") == "file_ingest" or event.get("stage") == "horizon_setup" for event in progress_events)
    assert any(event.get("stage") == "calibration" for event in progress_events)
    assert any(event.get("stage") == "validation" for event in progress_events)
    assert any(event.get("current_window_index") is not None for event in progress_events)
    assert any(event.get("total_windows") is not None for event in progress_events)
    assert all("last_updated" in event for event in progress_events)


def test_select_horizon_splits_uses_most_recent_windows():
    minute_df = _synthetic_minute_df(days=8)
    events = build_fixed_horizon_event_dataset(minute_df, horizon="1h", decision_step_minutes=30)
    splits = build_horizon_splits(
        events,
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
    )
    selected, meta = select_horizon_splits(splits, max_windows=3)
    assert meta["total_candidate_windows"] == len(splits)
    assert meta["evaluated_windows"] == 3
    assert meta["mode"] == "limited_windows"
    assert [split["fold_id"] for split in selected] == [split["fold_id"] for split in splits[-3:]]


def test_single_split_mode_evaluates_exactly_one_window(tmp_path):
    minute_df = _synthetic_minute_df(days=8)
    report = run_horizon_comparison(
        minute_df=minute_df,
        horizons=["1h"],
        output_dir=tmp_path / "single_split",
        model_names=["naive_directional"],
        decision_step_overrides={"1h": 30},
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
        edge_thresholds=[0.05],
        max_windows=1,
    )
    metrics = json.loads(Path(report["metrics_path"]).read_text())
    window_selection = metrics["1h"]["window_selection"]
    assert window_selection["evaluated_windows"] == 1
    assert window_selection["mode"] == "single_split"


def test_limited_windows_evaluates_at_most_requested_count(tmp_path):
    minute_df = _synthetic_minute_df(days=8)
    report = run_horizon_comparison(
        minute_df=minute_df,
        horizons=["1h"],
        output_dir=tmp_path / "limited_windows",
        model_names=["naive_directional"],
        decision_step_overrides={"1h": 30},
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
        edge_thresholds=[0.05],
        max_windows=3,
    )
    metrics = json.loads(Path(report["metrics_path"]).read_text())
    window_selection = metrics["1h"]["window_selection"]
    assert window_selection["evaluated_windows"] == 3
    assert window_selection["mode"] == "limited_windows"


def test_default_behavior_remains_full_window_selection(tmp_path):
    minute_df = _synthetic_minute_df(days=8)
    report = run_horizon_comparison(
        minute_df=minute_df,
        horizons=["1h"],
        output_dir=tmp_path / "full_windows",
        model_names=["naive_directional"],
        decision_step_overrides={"1h": 30},
        train_events=3,
        calibration_events=1,
        validation_events=1,
        step_events=1,
        edge_thresholds=[0.05],
    )
    metrics = json.loads(Path(report["metrics_path"]).read_text())
    window_selection = metrics["1h"]["window_selection"]
    assert window_selection["mode"] == "full"
    assert window_selection["evaluated_windows"] == window_selection["total_candidate_windows"]
