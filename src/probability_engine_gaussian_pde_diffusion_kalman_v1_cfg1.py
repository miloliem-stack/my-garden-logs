"""Standalone production wrapper for `gaussian_pde_diffusion_kalman_v1_cfg1`."""

from __future__ import annotations

from .probability_engine_short_memory_research import GaussianPDEDiffusionKalmanProbabilityEngine


GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1_DEFAULTS = {
    "sigma_state_persistence": 0.992,
    "sigma_process_var": 5e-9,
    "sigma_measurement_var": 2e-7,
    "relaxation_rate": 0.08,
    "stress_process_var": 5e-6,
    "stress_measurement_var": 2.5e-4,
    "distance_weight": 1.0,
    "velocity_weight": 12.0,
    "persistence_weight": 8.0,
    "stress_sigma_scale": 0.35,
    "drift_lookback": 12,
    "min_periods": 45,
    "fallback_sigma": 5e-4,
    "sigma_floor": 1e-8,
    "sigma_cap": 0.25,
}


class GaussianPDEDiffusionKalmanV1Cfg1ProbabilityEngine(GaussianPDEDiffusionKalmanProbabilityEngine):
    engine_name = "gaussian_pde_diffusion_kalman_v1_cfg1"

    def __init__(self, **kwargs):
        super().__init__(**{**GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1_DEFAULTS, **kwargs})
