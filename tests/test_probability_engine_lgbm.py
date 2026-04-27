import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probability_engine_factory import build_probability_engine


class FakeLGBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        self.columns_ = list(X.columns)
        return self

    def predict_proba(self, X):
        base = np.clip(0.5 + 0.1 * np.tanh(X.iloc[:, 0].to_numpy()), 0.01, 0.99)
        return np.column_stack([1.0 - base, base])


def _make_feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_ts": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min", tz="UTC"),
            "event_hour_start": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="15min", tz="UTC").floor("h"),
            "tau_minutes": [60, 45, 30, 15],
            "strike_price": [100.0] * 4,
            "spot_now": [100.0, 100.5, 101.0, 99.5],
            "realized_yes": [0, 1, 1, 0],
            "ret_lag_1": [0.1, 0.2, -0.1, 0.0],
            "rv_15": [0.01, 0.02, 0.02, 0.03],
            "signed_distance_to_strike": [0.0, 0.005, 0.01, -0.005],
        }
    )


def test_lgbm_engine_fit_and_predict_frame_via_factory() -> None:
    engine = build_probability_engine("lgbm", model_cls=FakeLGBMClassifier)
    rows = _make_feature_rows()
    engine.fit_frame(rows, label_col="realized_yes")
    preds = engine.predict_frame(rows)
    assert preds["p_yes"].between(0.0, 1.0).all()
    assert preds["p_no"].between(0.0, 1.0).all()


def test_lgbm_engine_predict_uses_online_feature_row() -> None:
    engine = build_probability_engine("lgbm", model_cls=FakeLGBMClassifier)
    rows = _make_feature_rows()
    engine.fit_frame(rows, label_col="realized_yes")
    engine.set_online_feature_row(rows.iloc[0].to_dict())
    out = engine.predict(strike_price=100.0, tau_minutes=30)
    assert out["failed"] is False
    assert 0.0 <= out["p_yes"] <= 1.0


def test_lgbm_engine_predict_fails_cleanly_without_feature_row() -> None:
    engine = build_probability_engine("lgbm", model_cls=FakeLGBMClassifier)
    rows = _make_feature_rows()
    engine.fit_frame(rows, label_col="realized_yes")
    out = engine.predict(strike_price=100.0, tau_minutes=30)
    assert out["failed"] is True
    assert out["reason"] == "tabular_engine_requires_feature_row"
