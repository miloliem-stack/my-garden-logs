"""Offline-only short-memory probability engines for BTC hourly research sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .probability_engine_gaussian_vol import GaussianVolProbabilityEngine


SHORT_MEMORY_RESEARCH_VERSION = "research_v2"


class _ShortMemoryResearchBase(GaussianVolProbabilityEngine):
    def __init__(self, **kwargs):
        self.vol_window_short = int(max(2, kwargs.pop("vol_window_short", kwargs.get("vol_window", 45))))
        self.vol_window_medium = int(max(self.vol_window_short, kwargs.pop("vol_window_medium", kwargs.get("fit_window", 120))))
        self.ewma_alpha = float(np.clip(kwargs.pop("ewma_alpha", 0.2), 1e-6, 1.0))
        super().__init__(**kwargs)
        self._state_summary: dict[str, Any] = {}

    def _compute_component_sigma(self, returns: pd.Series, window: int) -> float | None:
        if len(returns) < max(2, int(window)):
            return None
        recent = returns.iloc[-int(window) :].astype(float)
        sigma = float(recent.std(ddof=0))
        return None if not np.isfinite(sigma) or sigma <= 0 else sigma

    def _compute_ewma_sigma(self, returns: pd.Series) -> float | None:
        if len(returns) < 2:
            return None
        squared = returns.astype(float).pow(2)
        ewma_var = float(squared.ewm(alpha=self.ewma_alpha, adjust=False).mean().iloc[-1])
        if not np.isfinite(ewma_var) or ewma_var <= 0:
            return None
        sigma = float(np.sqrt(ewma_var))
        return None if not np.isfinite(sigma) or sigma <= 0 else sigma

    def _apply_calibration(
        self,
        raw_p_yes: float,
        *,
        tau_minutes: int | None = None,
        z_score: float | None = None,
    ) -> tuple[float, dict[str, Any]]:
        calibrated = super()._apply_calibration(raw_p_yes)
        return calibrated, {
            "bucket_key": None if tau_minutes is None else f"tau_{int(tau_minutes)}",
            "params_used": {
                "mode": self.calibration_mode,
                "z_score": None if z_score is None else float(z_score),
            },
        }

    def _reset_sigma_state(self) -> None:
        self._fallback_sigma_used = False
        self._sigma_short_raw = None
        self._sigma_medium_raw = None
        self._sigma_ewma_raw = None
        self._sigma_blended_raw = None
        self._sigma_blend_weights_used = {}

    def _insufficient_history(self) -> bool:
        return len(self.prices) < 2 or (len(self.prices) - 1) < self.min_periods

    def _log_returns(self) -> pd.Series:
        return np.log(self.prices / self.prices.shift(1)).dropna().astype(float)

    def _apply_sigma_result(self, sigma: float | None, *, fallback_reason: str | None = None) -> None:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            self._sigma_per_sqrt_min = self._bounded_sigma(self.fallback_sigma)
            self._fallback_sigma_used = True
            if fallback_reason:
                self._state_summary["fallback_reason"] = str(fallback_reason)
            return
        self._sigma_per_sqrt_min = self._bounded_sigma(sigma)

    def _augment_prediction(self, out: dict[str, Any], extra_fields: dict[str, Any]) -> dict[str, Any]:
        if out.get("failed"):
            return out
        raw = out.setdefault("raw_output", {})
        raw.update(extra_fields)
        if self._last_prediction is not None:
            self._last_prediction.update(extra_fields)
        return out

    def _kalman_step(
        self,
        state: float,
        variance: float,
        observation: float,
        *,
        persistence: float,
        process_var: float,
        measurement_var: float,
        floor: float = 0.0,
    ) -> tuple[float, float, float]:
        state_prior = float(max(floor, persistence * state))
        variance_prior = float(max((persistence**2 * variance) + process_var, 1e-12))
        gain = float(variance_prior / (variance_prior + measurement_var))
        updated_state = float(max(floor, state_prior + gain * (observation - state_prior)))
        updated_variance = float(max((1.0 - gain) * variance_prior, 1e-12))
        return updated_state, updated_variance, gain

    def _compute_winsorized_sigma(self, returns: pd.Series, window: int, winsor_z: float = 2.0) -> float | None:
        if len(returns) < max(2, int(window)):
            return None
        recent = returns.iloc[-int(window) :].astype(float)
        center = float(recent.mean())
        scale = float(recent.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            return None
        clipped = recent.clip(lower=center - winsor_z * scale, upper=center + winsor_z * scale)
        value = float(clipped.std(ddof=0))
        return None if not np.isfinite(value) or value <= 0 else value

    def _sigma_observations(self, log_returns: pd.Series) -> dict[str, float | None]:
        self._sigma_short_raw = self._compute_component_sigma(log_returns, self.vol_window_short)
        self._sigma_medium_raw = self._compute_component_sigma(log_returns, self.vol_window_medium)
        self._sigma_ewma_raw = self._compute_ewma_sigma(log_returns)
        return {
            "short": self._sigma_short_raw,
            "medium": self._sigma_medium_raw,
            "ewma": self._sigma_ewma_raw,
            "winsorized": self._compute_winsorized_sigma(log_returns, self.vol_window_short),
        }

    def _predict_gaussian_with_sigma(self, strike_price: float, tau_minutes: int, sigma: float, extra_fields: dict[str, Any]) -> dict[str, Any]:
        spot = self._current_spot
        if spot is None or spot <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "missing_spot", "raw_output": {}}
        if strike_price <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "invalid_strike", "raw_output": {}}

        tau_minutes = int(tau_minutes)
        bounded_sigma = self._bounded_sigma(sigma)
        if tau_minutes <= 0:
            raw_p_yes = self._deterministic_probability(float(spot), float(strike_price))
            horizon_sigma = 0.0
            z_score = None
        else:
            horizon_sigma = float(bounded_sigma * np.sqrt(float(tau_minutes)))
            if horizon_sigma <= self.sigma_floor:
                raw_p_yes = self._deterministic_probability(float(spot), float(strike_price))
                z_score = None
            else:
                log_distance = float(np.log(float(strike_price) / float(spot)))
                z_score = float(log_distance / horizon_sigma)
                raw_p_yes = float(1.0 - stats.norm.cdf(z_score))

        raw_p_yes = float(np.clip(raw_p_yes, 0.0, 1.0))
        calibrated_p_yes, calibration_details = self._apply_calibration(raw_p_yes, tau_minutes=tau_minutes, z_score=z_score)
        p_yes = float(np.clip(calibrated_p_yes, 0.0, 1.0))
        self._last_prediction = {
            "spot_now": float(spot),
            "strike_price": float(strike_price),
            "tau_minutes": tau_minutes,
            "raw_p_yes": raw_p_yes,
            "calibrated_p_yes": p_yes,
            "sigma_per_sqrt_min": bounded_sigma,
            "sigma_short": self._sigma_short_raw,
            "sigma_medium": self._sigma_medium_raw,
            "sigma_ewma": self._sigma_ewma_raw,
            "sigma_blended": self._sigma_blended_raw,
            "sigma_blend_weights_used": dict(self._sigma_blend_weights_used),
            "horizon_sigma": horizon_sigma,
            "z_score": z_score,
            "abs_z_score": None if z_score is None else abs(float(z_score)),
            "calibration_bucket_key": calibration_details["bucket_key"],
            "calibration_params_used": calibration_details["params_used"],
            **extra_fields,
        }
        return {
            "engine_name": self.engine_name,
            "p_yes": p_yes,
            "p_no": float(1.0 - p_yes),
            "failed": False,
            "reason": None,
            "raw_output": dict(self._last_prediction),
        }

    def get_diagnostics(self) -> dict[str, Any]:
        out = super().get_diagnostics()
        out["engine_version"] = SHORT_MEMORY_RESEARCH_VERSION
        out.update(self._state_summary)
        return out


class MemoryDecaySigmaProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "memory_decay_sigma_v1"

    def __init__(self, decay: float = 0.94, **kwargs):
        super().__init__(**kwargs)
        self.decay = float(np.clip(decay, 0.0, 0.9999))
        self._variance_state: float | None = None

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._variance_state = None
        self._state_summary = {
            "state_model": "memory_decay_sigma",
            "decay": self.decay,
            "variance_state": None,
            "effective_memory_minutes": None if self.decay >= 1.0 else float(1.0 / max(1.0 - self.decay, 1e-8)),
        }
        if self._insufficient_history():
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        variance_state: float | None = None
        blend = 1.0 - self.decay
        for ret in log_returns.to_numpy(dtype=float):
            squared = float(ret * ret)
            variance_state = squared if variance_state is None else (self.decay * variance_state) + (blend * squared)
        self._variance_state = variance_state
        self._state_summary["variance_state"] = None if variance_state is None else float(variance_state)
        sigma = None if variance_state is None else float(np.sqrt(max(variance_state, 0.0)))
        self._apply_sigma_result(sigma, fallback_reason="invalid_variance_state")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        out = super().predict(strike_price=strike_price, tau_minutes=tau_minutes, n_sims=n_sims, seed=seed)
        return self._augment_prediction(
            out,
            {
                "state_model": "memory_decay_sigma",
                "decay": self.decay,
                "variance_state": self._variance_state,
            },
        )


class KalmanVol1StateProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "kalman_vol_1state"

    def __init__(
        self,
        state_persistence: float = 0.985,
        process_var: float = 1e-8,
        measurement_var: float = 2.5e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_persistence = float(np.clip(state_persistence, 0.0, 0.9999))
        self.process_var = float(max(process_var, 1e-12))
        self.measurement_var = float(max(measurement_var, 1e-12))
        self._latent_sigma_state: float | None = None
        self._latent_sigma_var: float | None = None
        self._last_abs_return: float | None = None

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._latent_sigma_state = None
        self._latent_sigma_var = None
        self._last_abs_return = None
        self._state_summary = {
            "state_model": "kalman_vol_1state",
            "state_persistence": self.state_persistence,
            "process_var": self.process_var,
            "measurement_var": self.measurement_var,
            "latent_sigma_state": None,
            "latent_sigma_var": None,
            "last_abs_return": None,
        }
        if self._insufficient_history():
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        abs_returns = log_returns.abs()
        init_window = abs_returns.iloc[: min(len(abs_returns), max(self.min_periods, 5))]
        sigma_state = float(max(init_window.mean(), self.fallback_sigma, self.sigma_floor))
        state_var = float(max(init_window.var(ddof=0) if len(init_window) > 1 else self.measurement_var, self.process_var))
        for obs in abs_returns.to_numpy(dtype=float):
            sigma_state, state_var, _ = self._kalman_step(
                sigma_state,
                state_var,
                float(obs),
                persistence=self.state_persistence,
                process_var=self.process_var,
                measurement_var=self.measurement_var,
                floor=self.sigma_floor,
            )
            self._last_abs_return = float(obs)
        self._latent_sigma_state = sigma_state
        self._latent_sigma_var = state_var
        self._state_summary["latent_sigma_state"] = sigma_state
        self._state_summary["latent_sigma_var"] = state_var
        self._state_summary["last_abs_return"] = self._last_abs_return
        self._apply_sigma_result(sigma_state, fallback_reason="invalid_sigma_state")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        out = super().predict(strike_price=strike_price, tau_minutes=tau_minutes, n_sims=n_sims, seed=seed)
        return self._augment_prediction(
            out,
            {
                "state_model": "kalman_vol_1state",
                "state_persistence": self.state_persistence,
                "process_var": self.process_var,
                "measurement_var": self.measurement_var,
                "latent_sigma_state": self._latent_sigma_state,
                "latent_sigma_var": self._latent_sigma_var,
                "last_abs_return": self._last_abs_return,
            },
        )


class KalmanDriftVol2StateProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "kalman_drift_vol_2state"

    def __init__(
        self,
        drift_persistence: float = 0.95,
        vol_persistence: float = 0.985,
        drift_process_var: float = 5e-9,
        vol_process_var: float = 1e-8,
        return_measurement_var: float = 2.5e-7,
        abs_return_measurement_var: float = 2.5e-7,
        drift_cap_per_minute: float = 3e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.drift_persistence = float(np.clip(drift_persistence, 0.0, 0.9999))
        self.vol_persistence = float(np.clip(vol_persistence, 0.0, 0.9999))
        self.drift_process_var = float(max(drift_process_var, 1e-12))
        self.vol_process_var = float(max(vol_process_var, 1e-12))
        self.return_measurement_var = float(max(return_measurement_var, 1e-12))
        self.abs_return_measurement_var = float(max(abs_return_measurement_var, 1e-12))
        self.drift_cap_per_minute = float(max(drift_cap_per_minute, 0.0))
        self._latent_drift_state: float | None = None
        self._latent_drift_var: float | None = None
        self._latent_sigma_state: float | None = None
        self._latent_sigma_var: float | None = None
        self._drift_clipped: bool = False

    def _clip_drift(self, value: float) -> float:
        if self.drift_cap_per_minute <= 0:
            return float(value)
        clipped = float(np.clip(value, -self.drift_cap_per_minute, self.drift_cap_per_minute))
        self._drift_clipped = self._drift_clipped or not np.isclose(clipped, value)
        return clipped

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._latent_drift_state = None
        self._latent_drift_var = None
        self._latent_sigma_state = None
        self._latent_sigma_var = None
        self._drift_clipped = False
        self._state_summary = {
            "state_model": "kalman_drift_vol_2state",
            "drift_persistence": self.drift_persistence,
            "vol_persistence": self.vol_persistence,
            "drift_process_var": self.drift_process_var,
            "vol_process_var": self.vol_process_var,
            "return_measurement_var": self.return_measurement_var,
            "abs_return_measurement_var": self.abs_return_measurement_var,
            "drift_cap_per_minute": self.drift_cap_per_minute,
            "latent_drift_state": None,
            "latent_drift_var": None,
            "latent_sigma_state": None,
            "latent_sigma_var": None,
            "drift_clipped": False,
        }
        if self._insufficient_history():
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        init_window = log_returns.iloc[: min(len(log_returns), max(self.min_periods, 5))]
        drift_state = float(init_window.mean())
        drift_var = float(max(init_window.var(ddof=0) if len(init_window) > 1 else self.return_measurement_var, self.drift_process_var))
        sigma_state = float(max(init_window.abs().mean(), self.fallback_sigma, self.sigma_floor))
        sigma_var = float(max(init_window.abs().var(ddof=0) if len(init_window) > 1 else self.abs_return_measurement_var, self.vol_process_var))

        for ret in log_returns.to_numpy(dtype=float):
            ret = float(ret)
            drift_state, drift_var, _ = self._kalman_step(
                drift_state,
                drift_var,
                ret,
                persistence=self.drift_persistence,
                process_var=self.drift_process_var,
                measurement_var=self.return_measurement_var,
            )
            drift_state = self._clip_drift(drift_state)

            vol_obs = abs(ret - drift_state)
            sigma_state, sigma_var, _ = self._kalman_step(
                sigma_state,
                sigma_var,
                vol_obs,
                persistence=self.vol_persistence,
                process_var=self.vol_process_var,
                measurement_var=self.abs_return_measurement_var,
                floor=self.sigma_floor,
            )

        self._latent_drift_state = drift_state
        self._latent_drift_var = drift_var
        self._latent_sigma_state = sigma_state
        self._latent_sigma_var = sigma_var
        self._state_summary["latent_drift_state"] = drift_state
        self._state_summary["latent_drift_var"] = drift_var
        self._state_summary["latent_sigma_state"] = sigma_state
        self._state_summary["latent_sigma_var"] = sigma_var
        self._state_summary["drift_clipped"] = self._drift_clipped
        self._apply_sigma_result(sigma_state, fallback_reason="invalid_sigma_state")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        spot = self._current_spot
        if spot is None or spot <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "missing_spot", "raw_output": {}}
        if strike_price <= 0:
            return {"engine_name": self.engine_name, "p_yes": None, "p_no": None, "failed": True, "reason": "invalid_strike", "raw_output": {}}

        tau_minutes = int(tau_minutes)
        sigma = self._bounded_sigma(self._sigma_per_sqrt_min)
        drift = self._clip_drift(float(self._latent_drift_state or 0.0))
        drift_horizon = 0.0
        z_score: float | None
        if tau_minutes <= 0:
            raw_p_yes = self._deterministic_probability(float(spot), float(strike_price))
            horizon_sigma = 0.0
            z_score = None
        else:
            horizon_sigma = float(sigma * np.sqrt(float(tau_minutes)))
            drift_horizon = float(drift * tau_minutes)
            if horizon_sigma <= self.sigma_floor:
                shifted_spot = float(spot * np.exp(drift_horizon))
                raw_p_yes = self._deterministic_probability(shifted_spot, float(strike_price))
                z_score = None
            else:
                log_distance = float(np.log(float(strike_price) / float(spot)))
                z_score = float((log_distance - drift_horizon) / horizon_sigma)
                raw_p_yes = float(1.0 - stats.norm.cdf(z_score))

        raw_p_yes = float(np.clip(raw_p_yes, 0.0, 1.0))
        calibrated_p_yes, calibration_details = self._apply_calibration(raw_p_yes, tau_minutes=tau_minutes, z_score=z_score)
        p_yes = float(np.clip(calibrated_p_yes, 0.0, 1.0))
        extra = {
            "raw_p_yes": raw_p_yes,
            "calibrated_p_yes": p_yes,
            "sigma_per_sqrt_min": sigma,
            "horizon_sigma": horizon_sigma,
            "drift_per_minute": drift,
            "drift_horizon": drift_horizon,
            "z_score": z_score,
            "abs_z_score": None if z_score is None else abs(float(z_score)),
            "state_model": "kalman_drift_vol_2state",
            "latent_drift_state": self._latent_drift_state,
            "latent_drift_var": self._latent_drift_var,
            "latent_sigma_state": self._latent_sigma_state,
            "latent_sigma_var": self._latent_sigma_var,
            "drift_clipped": self._drift_clipped,
            "calibration_bucket_key": calibration_details["bucket_key"],
            "calibration_params_used": calibration_details["params_used"],
        }
        self._last_prediction = {
            "spot_now": float(spot),
            "strike_price": float(strike_price),
            "tau_minutes": tau_minutes,
            **extra,
        }
        return {
            "engine_name": self.engine_name,
            "p_yes": p_yes,
            "p_no": float(1.0 - p_yes),
            "failed": False,
            "reason": None,
            "raw_output": dict(extra),
        }


class GaussianKalmanSigmaProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "gaussian_kalman_sigma_v1"

    def __init__(
        self,
        observation_mode: str = "short_ewma",
        state_persistence: float = 0.992,
        process_var: float = 5e-9,
        measurement_var: float = 2e-7,
        raw_sigma_blend_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observation_mode = str(observation_mode).strip().lower()
        self.state_persistence = float(np.clip(state_persistence, 0.0, 0.9999))
        self.process_var = float(max(process_var, 1e-12))
        self.measurement_var = float(max(measurement_var, 1e-12))
        self.raw_sigma_blend_weight = float(np.clip(raw_sigma_blend_weight, 0.0, 1.0))
        self._sigma_raw_observation: float | None = None
        self._sigma_kalman_state: float | None = None
        self._sigma_kalman_variance: float | None = None
        self._kalman_ready = False
        self._kalman_reason: str | None = None

    def _combine_observations(self, observations: dict[str, float | None]) -> float | None:
        if self.observation_mode == "short_ewma":
            candidates = [observations.get("short"), observations.get("ewma")]
        elif self.observation_mode == "winsorized_medium":
            candidates = [observations.get("winsorized"), observations.get("medium")]
        else:
            candidates = [observations.get("short"), observations.get("medium"), observations.get("ewma")]
        valid = [float(value) for value in candidates if value is not None and np.isfinite(value) and value > 0]
        if not valid:
            return None
        return float(np.mean(valid))

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._sigma_raw_observation = None
        self._sigma_kalman_state = None
        self._sigma_kalman_variance = None
        self._kalman_ready = False
        self._kalman_reason = None
        self._state_summary = {
            "state_model": "gaussian_kalman_sigma",
            "observation_mode": self.observation_mode,
            "state_persistence": self.state_persistence,
            "process_var": self.process_var,
            "measurement_var": self.measurement_var,
            "raw_sigma_blend_weight": self.raw_sigma_blend_weight,
            "sigma_raw_observation": None,
            "sigma_kalman_state": None,
            "sigma_kalman_variance": None,
            "kalman_ready": False,
            "kalman_reason": None,
        }
        if self._insufficient_history():
            self._kalman_reason = "insufficient_history"
            self._state_summary["kalman_reason"] = self._kalman_reason
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        observations = self._sigma_observations(log_returns)
        sigma_obs = self._combine_observations(observations)
        self._sigma_raw_observation = sigma_obs
        if sigma_obs is None:
            self._kalman_reason = "missing_sigma_observation"
            self._state_summary["sigma_raw_observation"] = None
            self._state_summary["kalman_reason"] = self._kalman_reason
            self._apply_sigma_result(None, fallback_reason=self._kalman_reason)
            return

        state = float(max(sigma_obs, self.sigma_floor))
        variance = float(max(self.measurement_var, self.process_var))
        state, variance, _ = self._kalman_step(
            state,
            variance,
            sigma_obs,
            persistence=self.state_persistence,
            process_var=self.process_var,
            measurement_var=self.measurement_var,
            floor=self.sigma_floor,
        )
        final_sigma = ((1.0 - self.raw_sigma_blend_weight) * state) + (self.raw_sigma_blend_weight * sigma_obs)
        self._sigma_blended_raw = float(final_sigma)
        self._sigma_blend_weights_used = {"kalman": float(1.0 - self.raw_sigma_blend_weight), "raw_observation": float(self.raw_sigma_blend_weight)}
        self._sigma_kalman_state = state
        self._sigma_kalman_variance = variance
        self._kalman_ready = True
        self._state_summary.update(
            {
                "sigma_raw_observation": sigma_obs,
                "sigma_kalman_state": state,
                "sigma_kalman_variance": variance,
                "kalman_ready": True,
                "kalman_reason": "ok",
            }
        )
        self._apply_sigma_result(final_sigma, fallback_reason="invalid_kalman_sigma")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        return self._predict_gaussian_with_sigma(
            strike_price,
            tau_minutes,
            self._sigma_per_sqrt_min or self.fallback_sigma,
            {
                "state_model": "gaussian_kalman_sigma",
                "sigma_raw_observation": self._sigma_raw_observation,
                "sigma_kalman_state": self._sigma_kalman_state,
                "sigma_kalman_variance": self._sigma_kalman_variance,
                "kalman_ready": self._kalman_ready,
                "kalman_reason": "ok" if self._kalman_ready else self._kalman_reason,
            },
        )


class KalmanBlendedSigmaProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "kalman_blended_sigma_v1"

    def __init__(
        self,
        state_persistence: float = 0.992,
        process_var: float = 5e-9,
        measurement_var: float = 2e-7,
        short_measurement_scale: float = 1.0,
        medium_measurement_scale: float = 1.1,
        ewma_measurement_scale: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_persistence = float(np.clip(state_persistence, 0.0, 0.9999))
        self.process_var = float(max(process_var, 1e-12))
        self.measurement_var = float(max(measurement_var, 1e-12))
        self.short_measurement_scale = float(max(short_measurement_scale, 0.1))
        self.medium_measurement_scale = float(max(medium_measurement_scale, 0.1))
        self.ewma_measurement_scale = float(max(ewma_measurement_scale, 0.1))
        self._latent_effective_sigma: float | None = None
        self._latent_sigma_variance: float | None = None
        self._kalman_ready = False
        self._kalman_reason: str | None = None

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._latent_effective_sigma = None
        self._latent_sigma_variance = None
        self._kalman_ready = False
        self._kalman_reason = None
        self._state_summary = {
            "state_model": "kalman_blended_sigma",
            "state_persistence": self.state_persistence,
            "process_var": self.process_var,
            "measurement_var": self.measurement_var,
            "observed_sigma_short": None,
            "observed_sigma_medium": None,
            "observed_sigma_ewma": None,
            "latent_effective_sigma": None,
            "latent_sigma_variance": None,
            "kalman_ready": False,
            "kalman_reason": None,
        }
        if self._insufficient_history():
            self._kalman_reason = "insufficient_history"
            self._state_summary["kalman_reason"] = self._kalman_reason
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        observations = self._sigma_observations(log_returns)
        available = [
            ("short", observations.get("short"), self.measurement_var * self.short_measurement_scale),
            ("medium", observations.get("medium"), self.measurement_var * self.medium_measurement_scale),
            ("ewma", observations.get("ewma"), self.measurement_var * self.ewma_measurement_scale),
        ]
        valid = [(name, float(value), float(measurement)) for name, value, measurement in available if value is not None and np.isfinite(value) and value > 0]
        self._state_summary["observed_sigma_short"] = observations.get("short")
        self._state_summary["observed_sigma_medium"] = observations.get("medium")
        self._state_summary["observed_sigma_ewma"] = observations.get("ewma")
        if not valid:
            self._kalman_reason = "missing_sigma_observations"
            self._state_summary["kalman_reason"] = self._kalman_reason
            self._apply_sigma_result(None, fallback_reason=self._kalman_reason)
            return

        state = float(max(np.mean([value for _, value, _ in valid]), self.sigma_floor))
        variance = float(max(self.measurement_var, self.process_var))
        gains: dict[str, float] = {}
        state_prior = float(max(self.sigma_floor, self.state_persistence * state))
        variance_prior = float(max((self.state_persistence**2 * variance) + self.process_var, 1e-12))
        state = state_prior
        variance = variance_prior
        for name, observation, measurement in valid:
            gain = float(variance / (variance + measurement))
            state = float(max(self.sigma_floor, state + gain * (observation - state)))
            variance = float(max((1.0 - gain) * variance, 1e-12))
            gains[name] = gain

        self._sigma_blended_raw = state
        self._sigma_blend_weights_used = gains
        self._latent_effective_sigma = state
        self._latent_sigma_variance = variance
        self._kalman_ready = True
        self._state_summary.update(
            {
                "latent_effective_sigma": state,
                "latent_sigma_variance": variance,
                "kalman_ready": True,
                "kalman_reason": "ok",
            }
        )
        self._apply_sigma_result(state, fallback_reason="invalid_latent_effective_sigma")

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        return self._predict_gaussian_with_sigma(
            strike_price,
            tau_minutes,
            self._sigma_per_sqrt_min or self.fallback_sigma,
            {
                "state_model": "kalman_blended_sigma",
                "observed_sigma_short": self._sigma_short_raw,
                "observed_sigma_medium": self._sigma_medium_raw,
                "observed_sigma_ewma": self._sigma_ewma_raw,
                "latent_effective_sigma": self._latent_effective_sigma,
                "latent_sigma_variance": self._latent_sigma_variance,
                "kalman_ready": self._kalman_ready,
                "kalman_reason": "ok" if self._kalman_ready else self._kalman_reason,
            },
        )


class GaussianPDEDiffusionKalmanProbabilityEngine(_ShortMemoryResearchBase):
    engine_name = "gaussian_pde_diffusion_kalman_v1"

    def __init__(
        self,
        sigma_state_persistence: float = 0.992,
        sigma_process_var: float = 5e-9,
        sigma_measurement_var: float = 2e-7,
        relaxation_rate: float = 0.08,
        stress_process_var: float = 5e-6,
        stress_measurement_var: float = 2.5e-4,
        distance_weight: float = 1.0,
        velocity_weight: float = 12.0,
        persistence_weight: float = 8.0,
        stress_sigma_scale: float = 0.35,
        drift_lookback: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_state_persistence = float(np.clip(sigma_state_persistence, 0.0, 0.9999))
        self.sigma_process_var = float(max(sigma_process_var, 1e-12))
        self.sigma_measurement_var = float(max(sigma_measurement_var, 1e-12))
        self.relaxation_rate = float(np.clip(relaxation_rate, 0.0, 0.99))
        self.stress_process_var = float(max(stress_process_var, 1e-12))
        self.stress_measurement_var = float(max(stress_measurement_var, 1e-12))
        self.distance_weight = float(max(distance_weight, 0.0))
        self.velocity_weight = float(max(velocity_weight, 0.0))
        self.persistence_weight = float(max(persistence_weight, 0.0))
        self.stress_sigma_scale = float(max(stress_sigma_scale, 0.0))
        self.drift_lookback = int(max(drift_lookback, 2))
        self._base_sigma_state: float | None = None
        self._base_sigma_variance: float | None = None
        self._stress_state: float | None = None
        self._stress_variance: float | None = None
        self._stress_ready = False
        self._stress_reason: str | None = None
        self._signed_distance_from_strike: float | None = None
        self._standardized_distance_from_strike: float | None = None
        self._distance_velocity: float | None = None
        self._effective_sigma_after_stress: float | None = None

    def _recompute_sigma(self) -> None:
        self._reset_sigma_state()
        self._base_sigma_state = None
        self._base_sigma_variance = None
        self._stress_state = None
        self._stress_variance = None
        self._stress_ready = False
        self._stress_reason = None
        self._signed_distance_from_strike = None
        self._standardized_distance_from_strike = None
        self._distance_velocity = None
        self._effective_sigma_after_stress = None
        self._state_summary = {
            "state_model": "gaussian_pde_diffusion_kalman",
            "sigma_state_persistence": self.sigma_state_persistence,
            "sigma_process_var": self.sigma_process_var,
            "sigma_measurement_var": self.sigma_measurement_var,
            "relaxation_rate": self.relaxation_rate,
            "stress_process_var": self.stress_process_var,
            "stress_measurement_var": self.stress_measurement_var,
            "distance_weight": self.distance_weight,
            "velocity_weight": self.velocity_weight,
            "persistence_weight": self.persistence_weight,
            "stress_sigma_scale": self.stress_sigma_scale,
            "drift_lookback": self.drift_lookback,
            "base_sigma_state": None,
            "base_sigma_variance": None,
            "stress_state": None,
            "stress_variance": None,
            "stress_ready": False,
            "stress_reason": None,
        }
        if self._insufficient_history():
            self._stress_reason = "insufficient_history"
            self._state_summary["stress_reason"] = self._stress_reason
            self._apply_sigma_result(None, fallback_reason="insufficient_history")
            return

        log_returns = self._log_returns()
        observations = self._sigma_observations(log_returns)
        valid = [float(value) for key, value in observations.items() if key != "winsorized" and value is not None and np.isfinite(value) and value > 0]
        if not valid:
            self._stress_reason = "missing_sigma_observations"
            self._state_summary["stress_reason"] = self._stress_reason
            self._apply_sigma_result(None, fallback_reason=self._stress_reason)
            return
        sigma_obs = float(np.mean(valid))
        state = float(max(sigma_obs, self.sigma_floor))
        variance = float(max(self.sigma_measurement_var, self.sigma_process_var))
        state, variance, _ = self._kalman_step(
            state,
            variance,
            sigma_obs,
            persistence=self.sigma_state_persistence,
            process_var=self.sigma_process_var,
            measurement_var=self.sigma_measurement_var,
            floor=self.sigma_floor,
        )
        self._base_sigma_state = state
        self._base_sigma_variance = variance
        self._sigma_blended_raw = state
        self._apply_sigma_result(state, fallback_reason="invalid_base_sigma")
        self._state_summary["base_sigma_state"] = state
        self._state_summary["base_sigma_variance"] = variance

    def _stress_update(self, strike_price: float, tau_minutes: int) -> tuple[float, float]:
        if self._current_spot is None or strike_price <= 0 or len(self.prices) < max(self.min_periods, 3):
            self._stress_state = 0.0
            self._stress_variance = self.stress_measurement_var
            self._stress_ready = False
            self._stress_reason = "insufficient_strike_history"
            return 0.0, self._sigma_per_sqrt_min or self.fallback_sigma

        price_window = self.prices.iloc[-max(self.min_periods, self.drift_lookback + 2) :].astype(float)
        distances = np.log(price_window / float(strike_price))
        away_motion = distances.diff().fillna(0.0) * np.sign(distances.shift(1).fillna(distances))
        away_motion = away_motion.clip(lower=0.0)
        persistent_away = away_motion.rolling(window=self.drift_lookback, min_periods=2).mean().fillna(0.0)

        stress_state = 0.0
        stress_var = float(max(self.stress_measurement_var, self.stress_process_var))
        for distance, velocity, persistence_drive in zip(
            distances.to_numpy(dtype=float),
            away_motion.to_numpy(dtype=float),
            persistent_away.to_numpy(dtype=float),
            strict=False,
        ):
            observation = float(
                (self.distance_weight * abs(distance))
                + (self.velocity_weight * max(velocity, 0.0))
                + (self.persistence_weight * max(persistence_drive, 0.0))
            )
            stress_state, stress_var, _ = self._kalman_step(
                stress_state,
                stress_var,
                observation,
                persistence=max(0.0, 1.0 - self.relaxation_rate),
                process_var=self.stress_process_var,
                measurement_var=self.stress_measurement_var,
                floor=0.0,
            )

        self._signed_distance_from_strike = float(distances.iloc[-1])
        self._distance_velocity = float(away_motion.iloc[-1])
        horizon_scale = max(float(self._sigma_per_sqrt_min or self.fallback_sigma) * np.sqrt(max(tau_minutes, 1)), self.sigma_floor)
        self._standardized_distance_from_strike = float(self._signed_distance_from_strike / horizon_scale)
        self._stress_state = float(stress_state)
        self._stress_variance = float(stress_var)
        self._stress_ready = True
        self._stress_reason = "ok"
        effective_sigma = float((self._sigma_per_sqrt_min or self.fallback_sigma) * (1.0 + self.stress_sigma_scale * stress_state))
        self._effective_sigma_after_stress = self._bounded_sigma(effective_sigma)
        return stress_state, self._effective_sigma_after_stress

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        stress_state, effective_sigma = self._stress_update(float(strike_price), int(tau_minutes))
        self._state_summary.update(
            {
                "stress_state": self._stress_state,
                "stress_variance": self._stress_variance,
                "stress_ready": self._stress_ready,
                "stress_reason": self._stress_reason,
            }
        )
        return self._predict_gaussian_with_sigma(
            strike_price,
            tau_minutes,
            effective_sigma,
            {
                "state_model": "gaussian_pde_diffusion_kalman",
                "signed_distance_from_strike": self._signed_distance_from_strike,
                "standardized_distance_from_strike": self._standardized_distance_from_strike,
                "distance_velocity": self._distance_velocity,
                "stress_state": stress_state,
                "stress_variance": self._stress_variance,
                "relaxation_rate_used": self.relaxation_rate,
                "stress_ready": self._stress_ready,
                "stress_reason": self._stress_reason,
                "effective_sigma_after_stress": self._effective_sigma_after_stress,
                "base_sigma_state": self._base_sigma_state,
                "base_sigma_variance": self._base_sigma_variance,
            },
        )


class KalmanMicrostructureSmoother:
    engine_name = "kalman_microstructure_smoother_v1"

    def __init__(
        self,
        fit_window: int = 240,
        min_periods: int = 60,
        smooth_window: int = 48,
        spectral_window: int = 64,
        state_persistence: float = 0.985,
        process_var: float = 2.5e-4,
        measurement_var: float = 2.5e-3,
    ):
        self.fit_window = int(max(fit_window, 10))
        self.min_periods = int(max(min_periods, 5))
        self.smooth_window = int(max(smooth_window, 8))
        self.spectral_window = int(max(spectral_window, 16))
        self.state_persistence = float(np.clip(state_persistence, 0.0, 0.9999))
        self.process_var = float(max(process_var, 1e-12))
        self.measurement_var = float(max(measurement_var, 1e-12))
        self.prices = pd.Series(dtype=float)
        self._current_spot: float | None = None
        self._diagnostics: dict[str, Any] = {}

    def _trim_history(self) -> None:
        self.prices = self.prices.sort_index()
        if len(self.prices) > self.fit_window:
            self.prices = self.prices.iloc[-self.fit_window :]

    def _log_returns(self) -> pd.Series:
        return np.log(self.prices / self.prices.shift(1)).dropna().astype(float)

    def fit_history(self, prices: pd.Series) -> None:
        prices = prices.dropna().astype(float)
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to fit kalman_microstructure_smoother_v1")
        self.prices = prices.iloc[-self.fit_window :].copy()
        self._current_spot = float(self.prices.iloc[-1])
        self._recompute()

    def observe_bar(self, price: float, ts: pd.Timestamp | None = None, finalized: bool = True) -> None:
        self._current_spot = float(price)
        if finalized:
            ts = ts or pd.Timestamp.now(tz="UTC")
            self.prices.loc[ts] = float(price)
            self._trim_history()
            self._recompute()

    def _spectral_features(self, returns: pd.Series) -> tuple[float, float, float]:
        window = returns.iloc[-self.spectral_window :].to_numpy(dtype=float)
        if len(window) < 8:
            return np.nan, np.nan, np.nan
        centered = window - float(window.mean())
        power = np.abs(np.fft.rfft(centered)) ** 2
        if len(power) <= 1 or float(power.sum()) <= 0:
            return 0.0, np.nan, np.nan
        usable = power[1:]
        probs = usable / float(usable.sum())
        spectral_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))) / np.log(len(probs)))
        low_cut = max(1, len(usable) // 4)
        low_ratio = float(usable[:low_cut].sum() / usable.sum())
        high_ratio = float(usable[-low_cut:].sum() / usable.sum())
        return spectral_entropy, low_ratio, high_ratio

    def _recompute(self) -> None:
        log_returns = self._log_returns()
        diagnostics = {
            "engine_name": self.engine_name,
            "engine_version": SHORT_MEMORY_RESEARCH_VERSION,
            "history_len": int(len(self.prices)),
            "kalman_microstructure_ready": False,
            "kalman_microstructure_reason": None,
            "smoothness_score": None,
            "spectral_entropy": None,
            "low_frequency_power_ratio": None,
            "high_frequency_power_ratio": None,
            "microstructure_smoothness_smoothed": None,
            "microstructure_noise_score_smoothed": None,
            "microstructure_regime_smoothed": None,
        }
        if len(log_returns) < self.min_periods:
            diagnostics["kalman_microstructure_reason"] = "insufficient_history"
            self._diagnostics = diagnostics
            return

        recent = log_returns.iloc[-self.smooth_window :]
        gross = float(recent.abs().sum())
        smoothness = 0.0 if gross <= 0 else float(abs(recent.sum()) / gross)
        spectral_entropy, low_ratio, high_ratio = self._spectral_features(log_returns)
        noise_score = float(np.clip(0.5 * (1.0 - smoothness) + 0.5 * (spectral_entropy if np.isfinite(spectral_entropy) else 1.0), 0.0, 1.0))

        smooth_state = smoothness
        smooth_var = self.measurement_var
        noise_state = noise_score
        noise_var = self.measurement_var
        smooth_state = max(0.0, min(1.0, self.state_persistence * smooth_state))
        noise_state = max(0.0, min(1.0, self.state_persistence * noise_state))
        regime = "balanced"
        if smooth_state >= 0.6 and noise_state <= 0.4:
            regime = "trend_clean"
        elif noise_state >= 0.6:
            regime = "noise_dominant"

        diagnostics.update(
            {
                "kalman_microstructure_ready": True,
                "kalman_microstructure_reason": "ok",
                "smoothness_score": smoothness,
                "spectral_entropy": spectral_entropy,
                "low_frequency_power_ratio": low_ratio,
                "high_frequency_power_ratio": high_ratio,
                "microstructure_smoothness_smoothed": smooth_state,
                "microstructure_noise_score_smoothed": noise_state,
                "microstructure_regime_smoothed": regime,
                "microstructure_smoothness_variance": smooth_var,
                "microstructure_noise_variance": noise_var,
            }
        )
        self._diagnostics = diagnostics

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)
