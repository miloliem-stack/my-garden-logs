"""Generic probability-engine interface for BTC hourly event forecasting."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class ProbabilityEngine(Protocol):
    engine_name: str

    def fit_history(self, prices: pd.Series) -> None: ...

    def observe_bar(self, price: float, ts: pd.Timestamp | None = None, finalized: bool = True) -> None: ...

    def predict(
        self,
        strike_price: float,
        tau_minutes: int,
        n_sims: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]: ...

    def current_spot(self) -> float | None: ...

    def get_diagnostics(self) -> dict[str, Any]: ...
