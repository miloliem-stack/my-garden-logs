"""AR/EGARCH/t/FHS probability-engine adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .model_ar_egarch_fhs import ModelAR1EGARCHStudentT


class AREGARCHProbabilityEngine:
    engine_name = "ar_egarch"

    def __init__(
        self,
        residual_buffer_size: int = 2000,
        fit_window: int = 2000,
        ar_lags: int = 1,
        egarch_o: int = 0,
        use_jump_augmentation: bool = False,
        jump_augment_weight: int = 10,
        **_unused,
    ):
        self.model = ModelAR1EGARCHStudentT(
            residual_buffer_size=residual_buffer_size,
            fit_window=fit_window,
            ar_lags=ar_lags,
            egarch_o=egarch_o,
        )
        self.use_jump_augmentation = bool(use_jump_augmentation)
        self.jump_augment_weight = int(jump_augment_weight)

    def fit_history(self, prices: pd.Series) -> None:
        self.model.update_with_price_series(prices)

    def observe_bar(self, price: float, ts: pd.Timestamp | None = None, finalized: bool = True) -> None:
        self.model.update_on_bar(float(price), ts=ts, include_in_buffer=bool(finalized))

    def predict(self, strike_price: float, tau_minutes: int, n_sims: int | None = None, seed: int | None = None) -> dict[str, Any]:
        if self.use_jump_augmentation:
            raw = self.model.probability_up(
                float(strike_price),
                minutes=int(tau_minutes),
                n_sims=n_sims,
                seed=seed,
                include_jumps=True,
                jump_augment_weight=self.jump_augment_weight,
            )
        else:
            raw = self.model.simulate_probability(
                float(strike_price),
                tau_minutes=int(tau_minutes),
                n_sims=n_sims or 2000,
                seed=seed,
            )
        p_yes = raw.get("p_hat")
        return {
            "engine_name": self.engine_name,
            "p_yes": None if p_yes is None else float(p_yes),
            "p_no": None if p_yes is None else float(1.0 - p_yes),
            "failed": bool(raw.get("simulation_failed", False)),
            "reason": raw.get("failure_reason"),
            "raw_output": raw,
        }

    def current_spot(self) -> float | None:
        return self.model.last_price

    def get_diagnostics(self) -> dict[str, Any]:
        diag = self.model.get_diagnostics()
        return {
            "engine_name": self.engine_name,
            "sigma_now": diag.get("sigma_now"),
            "nu": diag.get("nu"),
            "residual_buffer_len": diag.get("residual_buffer_len"),
            "tail_prob": getattr(self.model, "tail_prob", None),
            "jump_flag": getattr(self.model, "jump_flag", None),
        }
