import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from src.model_ar_egarch_fhs import ModelAR1EGARCHStudentT


def make_price_series(n=800, seed=1):
    np.random.seed(seed)
    # small baseline returns with Student-t noise
    df = 6
    eps = 0.0002 + 0.001 * np.random.standard_t(df=df, size=n - 1)
    prices = 50000 * np.exp(np.cumsum(np.concatenate([[0.0], eps])))
    idx = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=n, freq='min')
    return pd.Series(prices, index=idx)


def test_update_and_tail_prob():
    s = make_price_series(600)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    # calibrate on history
    m.update_with_price_series(s)

    last_price = m.last_price
    # small move
    small_price = last_price * np.exp(0.0001)
    m.update_on_bar(small_price)
    small_tail = m.tail_prob

    # large move (simulate a jump)
    large_price = last_price * np.exp(0.02)
    m.update_on_bar(large_price)
    large_tail = m.tail_prob

    assert 0.0 <= small_tail <= 1.0
    assert 0.0 <= large_tail <= 1.0
    # large move should have smaller tail_prob (more extreme)
    assert large_tail <= small_tail


def test_probability_bounds():
    s = make_price_series(700)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    m.update_with_price_series(s)
    target = m.last_price * 1.001
    out = m.probability_up(target, minutes=30, n_sims=200)
    p = out.get('p_hat')
    assert 0.0 <= p <= 1.0


def test_fit_keeps_positive_finite_sigma_and_scale():
    s = make_price_series(800, seed=2)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    m.update_with_price_series(s)
    assert np.isfinite(m.sigma_now)
    assert m.sigma_now > 0.0
    assert np.isfinite(m._scale)
    assert m._scale > 0.0


def test_simulate_probability_returns_bounded_probability_or_clean_failure():
    s = make_price_series(800, seed=3)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    m.update_with_price_series(s)
    out = m.simulate_probability(m.last_price * 1.0005, tau_minutes=15, n_sims=100, seed=123)
    if out.get("simulation_failed"):
        assert out.get("failure_reason")
    else:
        assert np.isfinite(out["p_hat"])
        assert 0.0 <= out["p_hat"] <= 1.0


def test_log_space_terminal_comparison_matches_price_space_equivalent():
    s = make_price_series(800, seed=4)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    m.update_with_price_series(s)

    target = m.last_price * 1.002
    out = m.simulate_probability(target, tau_minutes=20, n_sims=150, seed=321)

    if out.get("simulation_failed"):
        assert out.get("failure_reason")
        return

    rng = np.random.default_rng(321)
    z_pool = m.z_buffer[np.isfinite(m.z_buffer)]
    mu = float(m.params.get("mu", m.params.get("const", 0.0)))
    phi = float(m.params.get("ar.1", 0.0))
    omega = float(m.params.get("omega", 0.0))
    alpha = float(m.params.get("alpha[1]", m.params.get("alpha", 0.0)))
    gamma = float(m.params.get("gamma[1]", m.params.get("gamma", 0.0)))
    beta = float(m.params.get("beta[1]", m.params.get("beta", 0.0)))
    r_prev = float(m.last_return)
    log_sigma2_prev = float(m.log_sigma2_now)
    threshold = float(np.log(target / m.last_price))

    price_space_hits = 0
    log_space_hits = 0
    for _ in range(150):
        r_tmp = r_prev
        log_sigma2 = log_sigma2_prev
        r_sum = 0.0
        shocks = rng.choice(z_pool, size=20, replace=True)
        for zt in shocks:
            log_sigma2 = m._clip_log_sigma2(m._egarch_step(log_sigma2, zt, omega, alpha, gamma, beta))
            sigma_t = m._safe_sigma(float(np.exp(0.5 * log_sigma2)))
            r_new = mu + phi * r_tmp + sigma_t * float(zt)
            r_sum += r_new
            r_tmp = r_new
        r_sum_unscaled = r_sum / m._scale
        if m.last_price * np.exp(r_sum_unscaled) >= target:
            price_space_hits += 1
        if r_sum_unscaled >= threshold:
            log_space_hits += 1

    assert price_space_hits == log_space_hits


def test_simulate_probability_rejects_non_positive_prices():
    s = make_price_series(800, seed=5)
    m = ModelAR1EGARCHStudentT(residual_buffer_size=500, fit_window=500)
    m.update_with_price_series(s)

    with pytest.raises(ValueError, match="Target price must be positive"):
        m.simulate_probability(0.0, tau_minutes=10, n_sims=10, seed=1)

    m.last_price = 0.0
    with pytest.raises(ValueError, match="Current price must be positive"):
        m.simulate_probability(1.0, tau_minutes=10, n_sims=10, seed=1)
