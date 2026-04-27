"""Factory for probability-engine adapters."""

from __future__ import annotations

from .probability_engine_ar_egarch import AREGARCHProbabilityEngine
from .probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1 import (
    GaussianPDEDiffusionKalmanV1Cfg1ProbabilityEngine,
)
from .probability_engine_gaussian_vol import GaussianVolProbabilityEngine
from .probability_engine_kalman_blended_sigma_v1_cfg1 import (
    KalmanBlendedSigmaV1Cfg1ProbabilityEngine,
)
from .probability_engine_lgbm import LGBMProbabilityEngine

PROBABILITY_ENGINE_ENV_FLAGS = {
    "ar_egarch": "AR_EGARCH",
    "gaussian_vol": "GAUSSIAN_VOL",
    "kalman_blended_sigma_v1_cfg1": "KALMAN_BLENDED_SIGMA_V1_CFG1",
    "gaussian_pde_diffusion_kalman_v1_cfg1": "GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1",
    "lgbm": "LGBM",
}


def _env_flag(name: str) -> bool:
    from os import getenv

    raw = getenv(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def get_default_probability_engine_name() -> str:
    selected = [engine_name for engine_name, env_name in PROBABILITY_ENGINE_ENV_FLAGS.items() if _env_flag(env_name)]
    if len(selected) > 1:
        raise SystemExit("too many engines selected!")
    if len(selected) < 1:
        raise SystemExit("no probability engine selected!")
    return selected[0]


def build_probability_engine(engine_name: str | None = None, **kwargs):
    normalized = get_default_probability_engine_name() if engine_name is None else str(engine_name).strip().lower()
    if normalized == "ar_egarch":
        return AREGARCHProbabilityEngine(**kwargs)
    if normalized == "gaussian_vol":
        return GaussianVolProbabilityEngine(**kwargs)
    if normalized == "kalman_blended_sigma_v1_cfg1":
        return KalmanBlendedSigmaV1Cfg1ProbabilityEngine(**kwargs)
    if normalized == "gaussian_pde_diffusion_kalman_v1_cfg1":
        return GaussianPDEDiffusionKalmanV1Cfg1ProbabilityEngine(**kwargs)
    if normalized == "lgbm":
        return LGBMProbabilityEngine(**kwargs)
    raise ValueError(f"Unknown probability engine: {engine_name}")
