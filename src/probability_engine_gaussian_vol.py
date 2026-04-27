"""Production-grade driftless Gaussian-volatility probability engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


ENGINE_VERSION = "1.0"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else str(raw)


class GaussianVolProbabilityEngine:
    engine_name = "gaussian_vol"

    def __init__(
        self,
        fit_window: int = 2000,
        vol_window: int | None = None,
        min_periods: int | None = None,
        sigma_floor: float | None = None,
        sigma_cap: float | None = None,
        fallback_sigma: float | None = None,
        calibration_mode: str | None = None,
        calibration_params: dict[str, float] | None = None,
        calibration_params_path: str | None = None,
        **_unused,
    ):
        self.fit_window = int(fit_window)
        self.vol_window = int(vol_window or _env_int("GAUSSIAN_VOL_WINDOW", 120))
        self.min_periods = int(min_periods or _env_int("GAUSSIAN_MIN_PERIODS", 60))
        self.sigma_floor = float(sigma_floor if sigma_floor is not None else _env_float("GAUSSIAN_SIGMA_FLOOR", 1e-8))
        self.sigma_cap = float(sigma_cap if sigma_cap is not None else _env_float("GAUSSIAN_SIGMA_CAP", 0.25))
        self.fallback_sigma = float(fallback_sigma if fallback_sigma is not None else _env_float("GAUSSIAN_FALLBACK_SIGMA", 5e-4))
        self.calibration_mode = str(calibration_mode or _env_str("GAUSSIAN_CALIBRATION_MODE", "none")).lower()

        self.prices = pd.Series(dtype=float)
        self._current_spot: float | None = None
        self._sigma_per_sqrt_min: float | None = None
        self._fallback_sigma_used = False
        self._last_prediction: dict[str, Any] | None = None
        self._calibration_params: dict[str, float] | None = None

        if calibration_params_path:
            self.load_calibration_params(calibration_params_path)
        elif calibration_params is not None:
            self.set_calibration_params(calibration_params)
        else:
            self._calibration_params = None

    def set_calibration_params(self, params: dict[str, float] | None) -> None:
        if params is None:
            self._calibration_params = None
            return
        intercept = float(params.get("intercept", 0.0))
        slope = float(params.get("slope", 1.0))
        self._calibration_params = {"intercept": intercept, "slope": slope}

    def load_calibration_params(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "mode" in payload:
            self.calibration_mode = str(payload["mode"]).lower()
        self.set_calibration_params(payload)

    def fit_logistic_calibration(self, probabilities: pd.Series, outcomes: pd.Series) -> dict[str, float]:
        p = probabilities.astype(float).clip(1e-6, 1 - 1e-6)
        y = outcomes.astype(float).to_numpy()
        x = np.log(p / (1.0 - p)).to_numpy()

        def objective(params: np.ndarray) -> float:
            a, b = params
            logits = np.clip(a + b * x, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs, 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

        result = optimize.minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
        params = {"intercept": 0.0, "slope": 1.0} if not result.success else {"intercept": float(result.x[0]), "slope": float(result.x[1])}
        self.set_calibration_params(params)
        return params

    def fit_history(self, prices: pd.Series) -> None:
        prices = prices.dropna().astype(float)
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to fit gaussian_vol engine")
        if len(prices) > self.fit_window:
            prices = prices.iloc[-self.fit_window :]
        self.prices = prices.copy()
        self._current_spot = float(prices.iloc[-1])
        self._recompute_sigma()

    def _trim_history(self) -> None:
        self.prices = self.prices.sort_index()
        if len(self.prices) > self.fit_window:
            self.prices = self.prices.iloc[-self.fit_window :]

    def _recompute_sigma(self) -> None:
        self._fallback_sigma_used = False
        if len(self.prices) < 2:
            self._sigma_per_sqrt_min = self._bounded_sigma(self.fallback_sigma)
            self._fallback_sigma_used = True
            return
        log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        if len(log_returns) > self.vol_window:
            log_returns = log_returns.iloc[-self.vol_window :]
        if len(log_returns) < self.min_periods:
            self._sigma_per_sqrt_min = self._bounded_sigma(self.fallback_sigma)
            self._fallback_sigma_used = True
            return
        sigma = float(log_returns.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 0:
            self._sigma_per_sqrt_min = self._bounded_sigma(self.fallback_sigma)
            self._fallback_sigma_used = True
            return
        self._sigma_per_sqrt_min = self._bounded_sigma(sigma)

    def _bounded_sigma(self, sigma: float | None) -> float:
        value = self.fallback_sigma if sigma is None or not np.isfinite(sigma) else float(sigma)
        value = max(value, self.sigma_floor)
        value = min(value, self.sigma_cap)
        return float(value)

    def observe_bar(self, price: float, ts: pd.Timestamp | None = None, finalized: bool = True) -> None:
        self._current_spot = float(price)
        if finalized:
            ts = ts or pd.Timestamp.now(tz="UTC")
            self.prices.loc[ts] = float(price)
            self._trim_history()
            self._recompute_sigma()

    def _deterministic_probability(self, spot: float, strike: float) -> float:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5

    def _apply_calibration(self, raw_p_yes: float) -> float:
        if self.calibration_mode == "none":
            return raw_p_yes
        if self.calibration_mode == "logistic":
            params = self._calibration_params or {"intercept": 0.0, "slope": 1.0}
            p = float(np.clip(raw_p_yes, 1e-6, 1 - 1e-6))
            logit = np.log(p / (1.0 - p))
            calibrated_logit = params["intercept"] + params["slope"] * logit
            return float(1.0 / (1.0 + np.exp(-np.clip(calibrated_logit, -30.0, 30.0))))
        raise ValueError(f"Unsupported calibration mode: {self.calibration_mode}")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        spot = self._current_spot
        if spot is None or spot <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "missing_spot", "raw_output": {}}
        if strike_price <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "invalid_strike", "raw_output": {}}

        tau_minutes = int(tau_minutes)
        sigma = self._bounded_sigma(self._sigma_per_sqrt_min)
        raw_p_yes: float
        z_score: float | None
        horizon_sigma: float

        if tau_minutes <= 0:
            raw_p_yes = self._deterministic_probability(float(spot), float(strike_price))
            z_score = None
            horizon_sigma = 0.0
        else:
            horizon_sigma = float(sigma * np.sqrt(float(tau_minutes)))
            if horizon_sigma <= self.sigma_floor:
                raw_p_yes = self._deterministic_probability(float(spot), float(strike_price))
                z_score = None
            else:
                log_distance = float(np.log(float(strike_price) / float(spot)))
                z_score = float(log_distance / horizon_sigma)
                raw_p_yes = float(1.0 - stats.norm.cdf(z_score))

        raw_p_yes = float(np.clip(raw_p_yes, 0.0, 1.0))
        calibrated_p_yes = float(np.clip(self._apply_calibration(raw_p_yes), 0.0, 1.0))
        p_yes = calibrated_p_yes
        self._last_prediction = {
            "spot_now": float(spot),
            "strike_price": float(strike_price),
            "tau_minutes": tau_minutes,
            "raw_p_yes": raw_p_yes,
            "calibrated_p_yes": calibrated_p_yes,
            "sigma_per_sqrt_min": sigma,
            "horizon_sigma": horizon_sigma,
            "z_score": z_score,
        }
        return {
            "engine_name": self.engine_name,
            "p_yes": p_yes,
            "p_no": float(1.0 - p_yes),
            "failed": False,
            "reason": None,
            "raw_output": {
                "raw_p_yes": raw_p_yes,
                "calibrated_p_yes": calibrated_p_yes,
                "sigma_per_sqrt_min": sigma,
                "horizon_sigma": horizon_sigma,
                "z_score": z_score,
                "calibration_mode": self.calibration_mode,
            },
        }

    def current_spot(self) -> float | None:
        return self._current_spot

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "engine_version": ENGINE_VERSION,
            "current_spot": self._current_spot,
            "sigma_per_sqrt_min": self._sigma_per_sqrt_min,
            "vol_window": self.vol_window,
            "min_periods": self.min_periods,
            "history_len": int(len(self.prices)),
            "fallback_sigma": self.fallback_sigma,
            "fallback_sigma_used": self._fallback_sigma_used,
            "sigma_floor": self.sigma_floor,
            "sigma_cap": self.sigma_cap,
            "calibration_mode": self.calibration_mode,
            "calibration_params_loaded": self._calibration_params is not None,
            "last_prediction": self._last_prediction,
        }
