import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probability_engine_factory import build_probability_engine
from src.run_bot import compute_market_probabilities


def _prices(n: int = 120, start: float = 100.0, step: float = 0.01) -> pd.Series:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="min", tz="UTC")
    return pd.Series([start + step * i for i in range(n)], index=index)


def test_probability_direction_and_at_the_money_behavior() -> None:
    prices = _prices()
    engine = build_probability_engine("gaussian_vol", fit_window=120, vol_window=60, min_periods=20)
    engine.fit_history(prices)
    engine.observe_bar(101.0, ts=prices.index[-1] + pd.Timedelta(minutes=1), finalized=False)
    above = engine.predict(strike_price=100.0, tau_minutes=30)
    below = engine.predict(strike_price=102.0, tau_minutes=30)
    atm = engine.predict(strike_price=101.0, tau_minutes=30)
    assert above["p_yes"] > 0.5
    assert below["p_yes"] < 0.5
    assert abs(atm["raw_output"]["raw_p_yes"] - 0.5) < 1e-9


def test_smaller_tau_produces_more_extreme_probability() -> None:
    prices = _prices()
    engine = build_probability_engine("gaussian_vol", fit_window=120, vol_window=60, min_periods=200, fallback_sigma=0.01)
    engine.fit_history(prices)
    engine.observe_bar(100.01, ts=prices.index[-1] + pd.Timedelta(minutes=1), finalized=False)
    short_tau = engine.predict(strike_price=100.0, tau_minutes=5)
    long_tau = engine.predict(strike_price=100.0, tau_minutes=60)
    assert short_tau["p_yes"] > long_tau["p_yes"]


def test_zero_vol_and_expired_paths_are_deterministic() -> None:
    flat = pd.Series(
        [100.0] * 40,
        index=pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="min", tz="UTC"),
    )
    engine = build_probability_engine("gaussian_vol", fit_window=40, vol_window=20, min_periods=5, fallback_sigma=0.0, sigma_floor=0.0, sigma_cap=1.0)
    engine.fit_history(flat)
    engine.observe_bar(100.0, ts=flat.index[-1] + pd.Timedelta(minutes=1), finalized=False)
    atm = engine.predict(strike_price=100.0, tau_minutes=10)
    expired = engine.predict(strike_price=99.0, tau_minutes=0)
    assert atm["p_yes"] == 0.5
    assert expired["p_yes"] == 1.0


def test_insufficient_history_uses_fallback_sigma_and_sets_flag() -> None:
    prices = _prices(n=10)
    engine = build_probability_engine("gaussian_vol", fit_window=20, vol_window=20, min_periods=15, fallback_sigma=0.002)
    engine.fit_history(prices)
    out = engine.predict(strike_price=100.0, tau_minutes=30)
    diag = engine.get_diagnostics()
    assert out["failed"] is False
    assert diag["fallback_sigma_used"] is True
    assert diag["sigma_per_sqrt_min"] == 0.002


def test_calibration_path_changes_reported_probability() -> None:
    prices = _prices()
    engine = build_probability_engine(
        "gaussian_vol",
        fit_window=120,
        vol_window=60,
        min_periods=20,
        calibration_mode="logistic",
        calibration_params={"intercept": 0.5, "slope": 0.8},
    )
    engine.fit_history(prices)
    engine.observe_bar(101.0, ts=prices.index[-1] + pd.Timedelta(minutes=1), finalized=False)
    out = engine.predict(strike_price=100.0, tau_minutes=30)
    assert out["failed"] is False
    assert out["raw_output"]["calibrated_p_yes"] == out["p_yes"]
    assert out["raw_output"]["raw_p_yes"] != out["p_yes"]


def test_factory_and_run_bot_probability_path_work_with_gaussian_config() -> None:
    prices = _prices()
    engine = build_probability_engine(
        "gaussian_vol",
        fit_window=120,
        vol_window=60,
        min_periods=20,
        sigma_floor=1e-7,
        sigma_cap=0.1,
        fallback_sigma=0.001,
        calibration_mode="none",
    )
    engine.fit_history(prices)
    engine.observe_bar(float(prices.iloc[-1]), ts=prices.index[-1], finalized=False)
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "strike_price": 100.0,
        "end_time": (prices.index[-1] + pd.Timedelta(minutes=20)).isoformat(),
    }
    state = compute_market_probabilities(bundle, engine, now=prices.index[-1], n_sims=10, seed=1)
    assert state["blocked"] is False
    assert 0.0 <= state["p_yes"] <= 1.0
    diag = engine.get_diagnostics()
    assert diag["engine_version"]
    assert diag["vol_window"] == 60
    assert diag["min_periods"] == 20
    assert isinstance(diag["last_prediction"], dict)


def test_factory_default_gaussian_config_is_120_60_none(monkeypatch) -> None:
    for env_name in ("AR_EGARCH", "GAUSSIAN_VOL", "KALMAN_BLENDED_SIGMA_V1_CFG1", "GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "LGBM"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("GAUSSIAN_VOL", "true")
    monkeypatch.delenv("GAUSSIAN_VOL_WINDOW", raising=False)
    monkeypatch.delenv("GAUSSIAN_MIN_PERIODS", raising=False)
    monkeypatch.delenv("GAUSSIAN_CALIBRATION_MODE", raising=False)
    engine = build_probability_engine()
    diag = engine.get_diagnostics()
    assert diag["engine_name"] == "gaussian_vol"
    assert diag["vol_window"] == 120
    assert diag["min_periods"] == 60
    assert diag["calibration_mode"] == "none"


def test_factory_env_overrides_still_work(monkeypatch) -> None:
    for env_name in ("AR_EGARCH", "GAUSSIAN_VOL", "KALMAN_BLENDED_SIGMA_V1_CFG1", "GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "LGBM"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("GAUSSIAN_VOL", "true")
    monkeypatch.setenv("GAUSSIAN_VOL_WINDOW", "240")
    monkeypatch.setenv("GAUSSIAN_MIN_PERIODS", "80")
    monkeypatch.setenv("GAUSSIAN_CALIBRATION_MODE", "logistic")
    engine = build_probability_engine()
    diag = engine.get_diagnostics()
    assert diag["vol_window"] == 240
    assert diag["min_periods"] == 80
    assert diag["calibration_mode"] == "logistic"


def test_factory_requires_exactly_one_probability_engine_flag(monkeypatch) -> None:
    for env_name in ("AR_EGARCH", "GAUSSIAN_VOL", "KALMAN_BLENDED_SIGMA_V1_CFG1", "GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "LGBM"):
        monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(SystemExit, match="no probability engine selected!"):
        build_probability_engine()

    monkeypatch.setenv("GAUSSIAN_VOL", "true")
    monkeypatch.setenv("LGBM", "true")
    with pytest.raises(SystemExit, match="too many engines selected!"):
        build_probability_engine()
