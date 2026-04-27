import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hourly_event_dataset import build_hourly_event_dataset
from src.probability_backtest import run_probability_backtest


class DummyModel:
    fit_calls = 0

    def __init__(self, residual_buffer_size: int = 2000, fit_window: int = 2000):
        self.residual_buffer_size = residual_buffer_size
        self.fit_window = fit_window
        self.last_fit_end = None
        self.last_price = None
        self.sigma_now = 1.23
        self.nu = 7.0
        self.z_buffer = [0.1, -0.1]
        self.jump_flag = False
        self.tail_prob = 0.9

    def update_with_price_series(self, prices: pd.Series) -> None:
        DummyModel.fit_calls += 1
        self.last_fit_end = prices.index[-1]
        self.last_price = float(prices.iloc[-1])

    def update_on_bar(self, new_price: float, ts=None, include_in_buffer: bool = True) -> None:
        assert self.last_fit_end < ts
        self.last_price = float(new_price)

    def simulate_probability(self, target_price: float, tau_minutes: int, n_sims: int = 2000, seed: int | None = None) -> dict:
        p_hat = 0.75 if self.last_price >= target_price else 0.25
        return {"p_hat": p_hat, "n_sims": n_sims, "target_price": target_price}


def _make_minute_data(hours: int = 2) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=hours * 60, freq="min", tz="UTC")
    rows = []
    for i, _ts in enumerate(index):
        base = 100.0 + i * 0.1
        rows.append({"open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05, "volume": 1.0})
    return pd.DataFrame(rows, index=index)


def test_backtest_runs_without_lookahead_and_returns_probabilities() -> None:
    DummyModel.fit_calls = 0
    minute_df = _make_minute_data(hours=3)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    results = run_probability_backtest(
        close_series=minute_df["close"],
        events=events.head(4),
        fit_window=120,
        residual_buffer_size=50,
        n_sims=100,
        seed=42,
        min_history=2,
        refit_every_n_events=2,
        model_factory=DummyModel,
    )

    assert not results.empty
    valid = results[results["skip_reason"].isna()]
    assert not valid.empty
    assert valid["p_yes"].between(0.0, 1.0).all()
    assert valid["p_no"].between(0.0, 1.0).all()
    assert (valid["training_window_length"] > 0).all()
    assert {"fit_failed", "simulation_failed", "skip_reason", "used_refit"} <= set(results.columns)
    assert DummyModel.fit_calls < len(valid)
