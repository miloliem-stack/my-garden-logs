"""Standalone production wrapper for `kalman_blended_sigma_v1_cfg1`."""

from __future__ import annotations

from .probability_engine_short_memory_research import KalmanBlendedSigmaProbabilityEngine


KALMAN_BLENDED_SIGMA_V1_CFG1_DEFAULTS = {
    "state_persistence": 0.992,
    "process_var": 5e-9,
    "measurement_var": 2e-7,
    "short_measurement_scale": 1.0,
    "medium_measurement_scale": 1.1,
    "ewma_measurement_scale": 0.9,
    "min_periods": 45,
    "fallback_sigma": 5e-4,
    "sigma_floor": 1e-8,
    "sigma_cap": 0.25,
}


class KalmanBlendedSigmaV1Cfg1ProbabilityEngine(KalmanBlendedSigmaProbabilityEngine):
    engine_name = "kalman_blended_sigma_v1_cfg1"

    def __init__(self, **kwargs):
        super().__init__(**{**KALMAN_BLENDED_SIGMA_V1_CFG1_DEFAULTS, **kwargs})
