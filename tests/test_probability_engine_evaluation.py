import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hourly_event_dataset import build_hourly_event_dataset
from src.probability_engine_evaluation import (
    build_event_hour_splits,
    evaluate_gaussian_vol_configs,
    evaluate_probability_engines,
)
from src.probability_engine_factory import get_default_probability_engine_name


class FakeLGBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        score = np.clip(0.5 + 0.05 * np.tanh(X.iloc[:, 0].to_numpy()), 0.01, 0.99)
        return np.column_stack([1.0 - score, score])


def _make_minute_df(hours: int = 12) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=hours * 60, freq="min", tz="UTC")
    rows = []
    for i, _ in enumerate(index):
        base = 100.0 + 0.03 * i + 0.2 * np.sin(i / 8)
        rows.append(
            {
                "open": base,
                "high": base + 0.2,
                "low": base - 0.2,
                "close": base + 0.02,
                "volume": 10.0 + (i % 20),
            }
        )
    return pd.DataFrame(rows, index=index)


def test_fold_assignment_keeps_event_hours_together() -> None:
    minute_df = _make_minute_df(hours=8)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    splits = build_event_hour_splits(events, train_hours=3, calibration_hours=2, validation_hours=2, step_hours=2)
    assert splits
    for split in splits:
        train_hours = set(split["train_hours"])
        calib_hours = set(split["calibration_hours"])
        val_hours = set(split["validation_hours"])
        assert train_hours.isdisjoint(calib_hours)
        assert train_hours.isdisjoint(val_hours)
        assert calib_hours.isdisjoint(val_hours)


def test_engine_comparison_smoke_runs_on_tiny_dataset(tmp_path: Path) -> None:
    minute_df = _make_minute_df(hours=10)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    report = evaluate_probability_engines(
        minute_df=minute_df,
        events=events,
        output_dir=tmp_path,
        train_hours=4,
        calibration_hours=2,
        validation_hours=2,
        step_hours=2,
        engine_configs=[
            {"engine_name": "gaussian_vol", "engine_kwargs": {"fit_window": 120, "vol_window": 120, "refit_every_n_events": 2, "min_history": 60}},
            {"engine_name": "ar_egarch", "engine_kwargs": {"fit_window": 120, "residual_buffer_size": 120, "refit_every_n_events": 2, "n_sims": 10, "min_history": 60}},
            {"engine_name": "lgbm", "engine_kwargs": {"model_cls": FakeLGBMClassifier}},
        ],
    )
    assert Path(report["summary_path"]).exists()
    summary = pd.read_csv(report["summary_path"])
    assert {"gaussian_vol", "ar_egarch", "lgbm"} <= set(summary["engine_name"])


def test_gaussian_sweep_can_skip_predictions_but_keeps_summaries(tmp_path: Path) -> None:
    minute_df = _make_minute_df(hours=10)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    report = evaluate_gaussian_vol_configs(
        minute_df=minute_df,
        events=events,
        output_dir=tmp_path,
        gaussian_configs=[
            {
                "config_id": "gaussian_default",
                "vol_window": 120,
                "min_periods": 60,
                "calibration_mode": "none",
                "fit_window": 120,
            }
        ],
        train_hours=4,
        calibration_hours=2,
        validation_hours=2,
        step_hours=2,
        write_predictions=False,
    )
    assert Path(report["summary_path"]).exists()
    assert Path(report["tau_summary_path"]).exists()
    assert Path(report["calibration_summary_path"]).exists()
    assert Path(report["ranked_summary_path"]).exists()
    assert report["predictions_path"] is None
    assert not (tmp_path / "engine_validation_predictions.csv").exists()


def test_factory_default_engine_name_is_gaussian_vol(monkeypatch) -> None:
    for env_name in ("AR_EGARCH", "GAUSSIAN_VOL", "KALMAN_BLENDED_SIGMA_V1_CFG1", "GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "LGBM"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("GAUSSIAN_VOL", "true")
    assert get_default_probability_engine_name() == "gaussian_vol"
