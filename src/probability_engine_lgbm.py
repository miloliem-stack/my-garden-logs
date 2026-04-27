"""LightGBM probability-engine adapter for offline event-row classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_builder import get_default_feature_columns


class LGBMProbabilityEngine:
    engine_name = "lgbm"

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        model_params: dict[str, Any] | None = None,
        model_cls=None,
        **_unused,
    ):
        self.feature_columns = feature_columns
        self.model_params = model_params or {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "objective": "binary",
        }
        self.model_cls = model_cls
        self.model = None
        self._current_spot = None
        self._online_feature_row = None

    def _resolve_model_cls(self):
        if self.model_cls is not None:
            return self.model_cls
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise ImportError("lightgbm is required for LGBMProbabilityEngine") from exc
        return LGBMClassifier

    def fit_history(self, prices: pd.Series) -> None:
        prices = prices.dropna().astype(float)
        if prices.empty:
            raise ValueError("Need non-empty prices to initialize lgbm engine")
        self._current_spot = float(prices.iloc[-1])

    def observe_bar(self, price: float, ts: pd.Timestamp | None = None, finalized: bool = True) -> None:
        self._current_spot = float(price)

    def fit_frame(self, feature_df: pd.DataFrame, label_col: str = "realized_yes") -> None:
        if label_col not in feature_df.columns:
            raise ValueError(f"Missing label column: {label_col}")
        feature_columns = self.feature_columns or get_default_feature_columns(feature_df)
        X = feature_df[feature_columns].astype(float)
        y = feature_df[label_col].astype(int)
        model_cls = self._resolve_model_cls()
        self.model = model_cls(**self.model_params)
        self.model.fit(X, y)
        self.feature_columns = feature_columns

    def predict_frame(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("LGBMProbabilityEngine is not fitted")
        feature_columns = self.feature_columns or get_default_feature_columns(feature_df)
        X = feature_df[feature_columns].astype(float)
        proba = np.asarray(self.model.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba must return an (n, 2) probability array")
        p_yes = np.clip(proba[:, 1], 0.0, 1.0)
        return pd.DataFrame({"p_yes": p_yes, "p_no": 1.0 - p_yes}, index=feature_df.index)

    def set_online_feature_row(self, feature_row: dict[str, Any] | pd.Series) -> None:
        self._online_feature_row = pd.DataFrame([dict(feature_row)])
        if "spot_now" in self._online_feature_row.columns:
            self._current_spot = float(self._online_feature_row.iloc[0]["spot_now"])

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        if self.model is None or self._online_feature_row is None:
            return {
                "engine_name": self.engine_name,
                "p_yes": None,
                "p_no": None,
                "failed": True,
                "reason": "tabular_engine_requires_feature_row",
                "raw_output": {},
            }
        pred = self.predict_frame(self._online_feature_row).iloc[0]
        return {
            "engine_name": self.engine_name,
            "p_yes": float(pred["p_yes"]),
            "p_no": float(pred["p_no"]),
            "failed": False,
            "reason": None,
            "raw_output": {},
        }

    def current_spot(self) -> float | None:
        return self._current_spot

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "is_fitted": self.model is not None,
            "feature_count": 0 if self.feature_columns is None else len(self.feature_columns),
            "current_spot": self._current_spot,
        }
