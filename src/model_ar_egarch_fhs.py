import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model
except Exception as e:
    arch_model = None

try:
    from arch.utility.exceptions import ConvergenceWarning, DataScaleWarning
except Exception:
    ConvergenceWarning = Warning
    DataScaleWarning = Warning


# Config via environment variables (can be overridden per-deployment)
AR1_EGARCH_ENABLED = os.getenv('AR1_EGARCH_ENABLED', '1') == '1'
EGARCH_ALPHA = float(os.getenv('EGARCH_ALPHA', '0.1'))
EGARCH_BETA = float(os.getenv('EGARCH_BETA', '0.85'))
EGARCH_GAMMA = float(os.getenv('EGARCH_GAMMA', '0.0'))
EGARCH_OMEGA = float(os.getenv('EGARCH_OMEGA', '-0.5'))
AR_PHI = float(os.getenv('AR_PHI', '0.0'))
AR_MU = float(os.getenv('AR_MU', '0.0'))
STUDENT_T_DF = float(os.getenv('STUDENT_T_DF', '8.0'))
JUMP_TAIL_THRESHOLD = float(os.getenv('JUMP_TAIL_THRESHOLD', '0.01'))
SIMULATION_COUNT = int(os.getenv('SIMULATION_COUNT', '2000'))
SIGMA_FLOOR = float(os.getenv('EGARCH_SIGMA_FLOOR', '1e-8'))
LOG_SIGMA2_MIN = float(os.getenv('EGARCH_LOG_SIGMA2_MIN', '-40.0'))
LOG_SIGMA2_MAX = float(os.getenv('EGARCH_LOG_SIGMA2_MAX', '20.0'))
TARGET_SCALED_STD = float(os.getenv('EGARCH_TARGET_SCALED_STD', '1.0'))
MIN_RETURN_SCALE = float(os.getenv('EGARCH_MIN_RETURN_SCALE', '100.0'))
MAX_RETURN_SCALE = float(os.getenv('EGARCH_MAX_RETURN_SCALE', '1000000.0'))


class ModelAR1EGARCHStudentT:
    def __init__(
        self,
        residual_buffer_size: int = 2000,
        fit_window: int = 2000,
        ar_lags: int = 1,
        egarch_o: int = 0,
    ):
        if arch_model is None:
            raise ImportError("arch package is required. Install with: pip install arch")
        self.residual_buffer_size = residual_buffer_size
        self.fit_window = fit_window
        self.ar_lags = int(ar_lags)
        self.egarch_o = int(egarch_o)

        # fitted parameters
        self.params = None
        self.nu = None

        # structural params (slowly recalibrated)
        self.mu = AR_MU
        self.phi = AR_PHI
        self.omega = EGARCH_OMEGA
        self.alpha = EGARCH_ALPHA
        self.gamma = EGARCH_GAMMA
        self.beta = EGARCH_BETA

        # last observed state
        self.last_price: Optional[float] = None
        self.last_return: Optional[float] = None
        self.sigma_now: Optional[float] = None
        self.log_sigma2_now: Optional[float] = None
        self.conditional_mean: Optional[float] = None
        self.z_now: Optional[float] = None
        self.tail_prob: Optional[float] = None
        self.jump_flag: Optional[bool] = None

        # buffer of standardized residuals (z = resid / sigma)
        self.z_buffer = np.zeros(0)

        # sample mean of |z| used in EGARCH recursion (empirical)
        self.mean_abs_z = None

        # last fitted timestamp (optional)
        self.last_ts = None
        self._scale = 1.0
        self.sigma_floor = SIGMA_FLOOR
        self.log_sigma2_min = LOG_SIGMA2_MIN
        self.log_sigma2_max = LOG_SIGMA2_MAX

    def _choose_scale(self, returns: pd.Series) -> float:
        std = float(returns.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            return 10000.0
        scale = TARGET_SCALED_STD / std
        return float(np.clip(scale, MIN_RETURN_SCALE, MAX_RETURN_SCALE))

    def _clip_log_sigma2(self, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("Non-finite log_sigma2 encountered")
        return float(np.clip(value, self.log_sigma2_min, self.log_sigma2_max))

    def _safe_sigma(self, sigma: float) -> float:
        if not np.isfinite(sigma):
            raise ValueError("Non-finite sigma encountered")
        return float(max(sigma, self.sigma_floor))

    def update_with_price_series(self, prices: pd.Series):
        """Feed minute-level prices (indexed by datetime). Fits model using last `fit_window` returns."""
        if len(prices) < 2:
            raise ValueError("Need at least 2 price points to compute returns")

        prices = prices.dropna()
        S = prices.values
        r = np.log(S[1:] / S[:-1])
        r_series = pd.Series(r)

        # store last price and return
        self.last_price = float(S[-1])
        self.last_return = float(r[-1])

        # Fit on last fit_window returns
        fit_r = r_series if len(r_series) <= self.fit_window else r_series.iloc[-self.fit_window :]

        # Rescale returns for better numerical stability in the arch optimizer.
        scale = self._choose_scale(fit_r)
        fit_r_scaled = fit_r * scale

        mean = "AR" if self.ar_lags > 0 else "Constant"
        lags = self.ar_lags if self.ar_lags > 0 else 0
        am = arch_model(fit_r_scaled, mean=mean, lags=lags, vol='EGARCH', p=1, o=self.egarch_o, q=1, dist='t')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataScaleWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = am.fit(disp='off', options={"maxiter": 300})

        self.params = res.params.to_dict()
        if 'nu' in res.params.index:
            self.nu = float(res.params['nu'])
        else:
            self.nu = float(STUDENT_T_DF)

        # populate structural params from fit unless env overrides provided
        # mu/const
        if 'mu' in self.params:
            self.mu = float(self.params.get('mu'))
        elif 'const' in self.params:
            self.mu = float(self.params.get('const'))
        # ar
        self.phi = 0.0 if self.ar_lags <= 0 else self.phi
        if 'ar.1' in self.params:
            self.phi = float(self.params.get('ar.1'))
        else:
            for key in self.params:
                if key.startswith('ar.') or key.startswith('ar[') or key in ('ar1', 'ar1.'):
                    self.phi = float(self.params[key])
                    break
        if 'omega' in self.params:
            self.omega = float(self.params.get('omega'))
        if 'alpha[1]' in self.params or 'alpha' in self.params:
            self.alpha = float(self.params.get('alpha[1]', self.params.get('alpha')))
        if 'gamma[1]' in self.params or 'gamma' in self.params:
            self.gamma = float(self.params.get('gamma[1]', self.params.get('gamma')))
        if 'beta[1]' in self.params or 'beta' in self.params:
            self.beta = float(self.params.get('beta[1]', self.params.get('beta')))

        # conditional vol and standardized residuals from in-sample fit (scaled units)
        cond_vol_scaled = res.conditional_volatility
        resid_scaled = res.resid
        terminal_sigma = float(cond_vol_scaled.iloc[-1])
        if not np.isfinite(terminal_sigma) or terminal_sigma <= 0:
            raise ValueError("Model fit produced non-finite or zero terminal conditional volatility")
        z = resid_scaled / cond_vol_scaled

        # store last raw scaled resid and cond vol series for diagnostics
        self._resid_scaled = resid_scaled
        self._cond_vol_scaled = cond_vol_scaled

        # update sigma_now in SCALED units and remember scale
        self._scale = scale
        self.sigma_now = self._safe_sigma(terminal_sigma)
        self.log_sigma2_now = self._clip_log_sigma2(float(np.log(self.sigma_now ** 2)))

        # r_prev in scaled units
        self.last_return = float(r[-1] * scale)

        # clean non-finite standardized residuals before updating buffer
        z_clean = z[np.isfinite(z)].values
        if z_clean.size == 0:
            z_vals = np.array([])
        else:
            z_vals = z_clean

        if len(self.z_buffer) == 0:
            self.z_buffer = z_vals[-self.residual_buffer_size :]
        else:
            merged = np.concatenate([self.z_buffer, z_vals])
            merged = merged[~np.isnan(merged)]
            self.z_buffer = merged[-self.residual_buffer_size :]

        # empirical mean absolute z (handle empty buffer)
        if len(self.z_buffer) == 0:
            self.mean_abs_z = None
        else:
            self.mean_abs_z = float(np.mean(np.abs(self.z_buffer)))

        # set conditional mean at last observation (m_t = mu + phi*(r_{t-1} - mu))
        try:
            self.conditional_mean = float(self.mu + self.phi * (self.last_return - self.mu))
        except Exception:
            self.conditional_mean = None

        # set last timestamp if index provided
        try:
            self.last_ts = prices.index[-1]
        except Exception:
            self.last_ts = None

    # --- online update per new bar (no refit) ---
    def update_on_bar(self, new_price: float, ts: Optional[pd.Timestamp] = None, include_in_buffer: bool = True):
        """Update internal state with a new price/bar without refitting structural params.

        This updates latest return, conditional mean, z_obs, tail probability, jump flag,
        and advances the EGARCH recursion for sigma.
        """
        if self.last_price is None:
            # initialize
            self.last_price = float(new_price)
            self.last_ts = ts
            return

        # compute return (log) unscaled and scaled
        r_raw = np.log(float(new_price) / float(self.last_price))
        scale = getattr(self, '_scale', 1.0)
        r_scaled = float(r_raw * scale)

        # conditional mean m_t = mu + phi*(r_{t-1} - mu)
        m_t = float(self.mu + self.phi * (self.last_return - self.mu)) if self.last_return is not None else float(self.mu)
        self.conditional_mean = m_t

        # current sigma (in scaled units)
        sigma_t = self._safe_sigma(float(self.sigma_now) if self.sigma_now is not None else 1.0)

        # standardized residual observed
        z_obs = (r_scaled - m_t) / sigma_t if sigma_t != 0 else 0.0
        self.z_now = float(z_obs)

        # student-t tail probability for |z_obs|; standardize to unit variance if nu>2
        nu = float(self.nu) if self.nu is not None else float(STUDENT_T_DF)
        if nu > 2:
            scale_std = np.sqrt((nu - 2.0) / nu)
            z_std = z_obs * scale_std
            tail_prob = 2.0 * (1.0 - float(stats.t.cdf(abs(z_std), df=nu)))
        else:
            # fallback to normal tails when df too small
            tail_prob = 2.0 * (1.0 - float(stats.norm.cdf(abs(z_obs))))

        self.tail_prob = float(tail_prob)
        self.jump_flag = bool(tail_prob <= float(JUMP_TAIL_THRESHOLD))

        # advance EGARCH recursion using observed z_obs
        if self.log_sigma2_now is None:
            # initialize with sigma_now if available
            self.log_sigma2_now = self._clip_log_sigma2(float(np.log(sigma_t ** 2)))

        # ensure mean_abs_z exists for recursion
        if self.mean_abs_z is None and len(self.z_buffer) > 0:
            self.mean_abs_z = float(np.mean(np.abs(self.z_buffer)))

        # use structural params for egarch recursion
        log_sigma2_new = self._egarch_step(self.log_sigma2_now, z_obs, float(self.omega), float(self.alpha), float(self.gamma), float(self.beta))
        log_sigma2_new = self._clip_log_sigma2(log_sigma2_new)
        sigma_new = self._safe_sigma(float(np.exp(0.5 * log_sigma2_new)))
        self.log_sigma2_now = log_sigma2_new
        self.sigma_now = sigma_new

        # update last return and price
        self.last_return = r_scaled
        self.last_price = float(new_price)
        self.last_ts = ts

        # append to z_buffer optionally
        if include_in_buffer:
            if len(self.z_buffer) == 0:
                self.z_buffer = np.array([z_obs])[-self.residual_buffer_size:]
            else:
                merged = np.concatenate([self.z_buffer, np.array([z_obs])])
                merged = merged[np.isfinite(merged)]
                self.z_buffer = merged[-self.residual_buffer_size:]
            self.mean_abs_z = float(np.mean(np.abs(self.z_buffer)))

    def _egarch_step(self, log_sigma2_prev: float, z_t: float, omega: float, alpha: float, gamma: float, beta: float) -> float:
        # EGARCH(1,1) recursion for log(sigma^2)
        return omega + alpha * (abs(z_t) - self.mean_abs_z) + gamma * z_t + beta * log_sigma2_prev

    def simulate_probability(self, target_price: float, tau_minutes: int, n_sims: int = 2000, seed: Optional[int] = None) -> dict:
        """Estimate P(final_price >= target_price) by FHS using standardized residual buffer.

        Returns a dict with keys: p_hat, n_sims, target_price
        """
        if self.params is None or self.last_price is None or self.sigma_now is None:
            raise RuntimeError("Model not fitted or state not initialized. Call update_with_price_series() first.")
        if self.last_price <= 0:
            raise ValueError("Current price must be positive for probability simulation")
        if target_price <= 0:
            raise ValueError("Target price must be positive for probability simulation")

        rng = np.random.default_rng(seed)

        # extract params (these are in SCALED units)
        mu = float(self.params.get('mu', 0.0)) if 'mu' in self.params else float(self.params.get('const', 0.0))
        # AR(1) param name from arch: 'ar.1' or 'ar[1]'
        phi = 0.0
        for k in self.params:
            if k.startswith('ar.') or k.startswith('ar[') or k == 'ar1' or k == 'ar1.':
                phi = float(self.params[k])
        # common names: 'ar.1'
        if 'ar.1' in self.params:
            phi = float(self.params['ar.1'])

        # EGARCH params names: 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]'
        omega = float(self.params.get('omega', 0.0))
        alpha = float(self.params.get('alpha[1]', self.params.get('alpha', 0.0)))
        gamma = float(self.params.get('gamma[1]', self.params.get('gamma', 0.0)))
        beta = float(self.params.get('beta[1]', self.params.get('beta', 0.0)))

        # Use residual buffer as empirical standardized shock distribution
        if len(self.z_buffer) == 0:
            raise RuntimeError("Standardized residual buffer is empty.")

        z_pool = self.z_buffer[np.isfinite(self.z_buffer)]
        if len(z_pool) == 0:
            return {
                "p_hat": np.nan,
                "n_sims": n_sims,
                "target_price": float(target_price),
                "simulation_failed": True,
                "failure_reason": "empty_residual_buffer",
            }

        S0 = float(self.last_price)
        log_return_threshold = float(np.log(float(target_price) / S0))
        # last_return and sigma_now are stored in scaled units
        r_prev = float(self.last_return)
        log_sigma2_prev = self._clip_log_sigma2(float(np.log(self._safe_sigma(self.sigma_now) ** 2)))

        exceed_count = 0

        for i in range(n_sims):
            r_tmp = r_prev
            log_sigma2 = log_sigma2_prev
            R_sum = 0.0
            # draw tau shocks by sampling from residual buffer
            shocks = rng.choice(z_pool, size=tau_minutes, replace=True)
            for zt in shocks:
                log_sigma2 = self._clip_log_sigma2(self._egarch_step(log_sigma2, zt, omega, alpha, gamma, beta))
                sigma_t = self._safe_sigma(float(np.exp(0.5 * log_sigma2)))
                r_new = mu + phi * r_tmp + sigma_t * float(zt)
                if not np.isfinite(r_new):
                    return {
                        "p_hat": np.nan,
                        "n_sims": n_sims,
                        "target_price": float(target_price),
                        "simulation_failed": True,
                        "failure_reason": "non_finite_return_path",
                    }
                R_sum += r_new
                r_tmp = r_new

            # R_sum is in SCALED units; convert back to original returns before mapping to price
            scale = getattr(self, '_scale', 1.0)
            R_sum_unscaled = R_sum / scale
            if not np.isfinite(R_sum_unscaled):
                return {
                    "p_hat": np.nan,
                    "n_sims": n_sims,
                    "target_price": float(target_price),
                    "simulation_failed": True,
                    "failure_reason": "non_finite_path_sum",
                }
            if R_sum_unscaled >= log_return_threshold:
                exceed_count += 1

        p_hat = exceed_count / n_sims
        return {
            "p_hat": p_hat,
            "n_sims": n_sims,
            "target_price": float(target_price),
            "simulation_failed": False,
            "failure_reason": None,
        }

    def simulate_event_probability(self, event: dict, now_ts: Optional[pd.Timestamp] = None, n_sims: int = 2000, seed: Optional[int] = None) -> dict:
        """Compute probability for an event dict returned by `discover_current_hour_event`.

        If `now` is before event start, simulate to open then hour and compute P(close >= open).
        If `now` is within the hour, simulate remaining minutes to close and compute P(close >= open_fixed).
        """
        if now_ts is None:
            now_ts = pd.Timestamp.now(tz='UTC')

        start = event.get('startDate')
        end = event.get('endDate')
        if start is None or end is None:
            raise ValueError('Event missing startDate/endDate')

        if now_ts >= start:
            # in-hour: open is fixed; price open is the hour open (may be fetched externally)
            # here we expect caller to provide the open via event metadata if available
            open_price = event.get('open_price')
            if open_price is None:
                # fallback: treat current price as threshold
                open_price = float(self.last_price)
            minutes_remaining = max(0, int((end - now_ts).total_seconds() // 60))
            out = self.simulate_probability(open_price, minutes_remaining or 1, n_sims=n_sims, seed=seed)
            out['mode'] = 'in-hour'
            out['minutes_remaining'] = minutes_remaining
            return out
        else:
            # pre-open: simulate to open (tau1) and then simulate the hour (tau2=60)
            tau1 = max(0, int((start - now_ts).total_seconds() // 60))
            tau2 = int((end - start).total_seconds() // 60)
            rng = np.random.default_rng(seed)

            # extract params same as simulate_probability
            mu = float(self.params.get('mu', 0.0)) if 'mu' in self.params else float(self.params.get('const', 0.0))
            phi = 0.0
            if 'ar.1' in self.params:
                phi = float(self.params['ar.1'])
            omega = float(self.params.get('omega', 0.0))
            alpha = float(self.params.get('alpha[1]', self.params.get('alpha', 0.0)))
            gamma = float(self.params.get('gamma[1]', self.params.get('gamma', 0.0)))
            beta = float(self.params.get('beta[1]', self.params.get('beta', 0.0)))

            if len(self.z_buffer) == 0:
                raise RuntimeError('Standardized residual buffer is empty')
            z_pool = self.z_buffer

            S0 = float(self.last_price)
            r_prev = float(self.last_return)
            log_sigma2_prev = np.log(self.sigma_now ** 2)

            exceed = 0
            for i in range(n_sims):
                r_tmp = r_prev
                log_sigma2 = log_sigma2_prev
                # simulate to open
                shocks1 = rng.choice(z_pool, size=max(1, tau1), replace=True)
                for zt in shocks1:
                    log_sigma2 = self._egarch_step(log_sigma2, zt, omega, alpha, gamma, beta)
                    sigma_t = float(np.exp(0.5 * log_sigma2))
                    r_new = mu + phi * r_tmp + sigma_t * float(zt)
                    r_tmp = r_new
                # price at open
                scale = getattr(self, '_scale', 1.0)
                # accumulate returns during tau1: need to recompute S_open
                # For simplicity, recompute by simulating full path again (approx)
                # Here we use incremental approach: generate new shocks for full path including hour
                # Simulate tau1+tau2 path
                r_tmp2 = r_prev
                log_sigma2_2 = log_sigma2_prev
                R_sum = 0.0
                shocks = rng.choice(z_pool, size=max(1, tau1 + tau2), replace=True)
                for idx, zt in enumerate(shocks):
                    log_sigma2_2 = self._egarch_step(log_sigma2_2, zt, omega, alpha, gamma, beta)
                    sigma_t = float(np.exp(0.5 * log_sigma2_2))
                    r_new2 = mu + phi * r_tmp2 + sigma_t * float(zt)
                    R_sum += r_new2
                    r_tmp2 = r_new2
                    if idx == tau1 - 1:
                        R_to_open = R_sum
                # compute S_open and S_close
                R_open_unscaled = (R_to_open if tau1>0 else 0.0) / scale
                R_close_unscaled = R_sum / scale
                S_open = S0 * np.exp(R_open_unscaled)
                S_close = S0 * np.exp(R_close_unscaled)
                if S_close >= S_open:
                    exceed += 1

            p_hat = exceed / n_sims
            return {"p_hat": p_hat, "n_sims": n_sims, "mode": "pre-open", "tau_to_open": tau1, "tau_hour": tau2}

    def probability_up(self, target_price: float, minutes: int, n_sims: int = None, seed: Optional[int] = None, include_jumps: bool = False, jump_augment_weight: int = 10) -> dict:
        """Convenience wrapper returning P(final >= target_price) from current state.

        If `include_jumps` is True and a recent jump was detected, the simulation augments
        the empirical shock pool with the observed large shock to increase tail sampling.
        """
        if n_sims is None:
            n_sims = SIMULATION_COUNT

        if self.params is None and len(self.z_buffer) == 0:
            raise RuntimeError("Model has no fitted parameters or residual buffer; calibrate first.")

        # prepare z_pool
        z_pool = self.z_buffer.copy()
        if include_jumps and getattr(self, 'jump_flag', False) and self.z_now is not None:
            # boost presence of observed extreme shock in pool
            aug = np.full(int(jump_augment_weight), float(self.z_now))
            z_pool = np.concatenate([z_pool, aug])

        # call lower-level simulate but passing augmented z_pool via temporary replacement
        saved_pool = self.z_buffer
        try:
            self.z_buffer = z_pool
            return self.simulate_probability(target_price, minutes, n_sims=n_sims, seed=seed)
        finally:
            self.z_buffer = saved_pool

    def get_diagnostics(self) -> dict:
        """Return diagnostic info about the fitted model and residual buffer."""
        return {
            "params": self.params,
            "nu": self.nu,
            "last_price": self.last_price,
            "sigma_now": self.sigma_now,
            "last_return_scaled": self.last_return,
            "residual_buffer_len": len(self.z_buffer),
            "mean_abs_z": self.mean_abs_z,
            "z_quantiles": None if len(self.z_buffer) == 0 else {
                "0.01": float(np.quantile(self.z_buffer, 0.01)),
                "0.05": float(np.quantile(self.z_buffer, 0.05)),
                "0.5": float(np.quantile(self.z_buffer, 0.5)),
                "0.95": float(np.quantile(self.z_buffer, 0.95)),
                "0.99": float(np.quantile(self.z_buffer, 0.99)),
            },
            "resid_tail": None if not hasattr(self, '_resid_scaled') else [None if np.isnan(x) else float(x) for x in self._resid_scaled.tail(20).tolist()],
            "cond_vol_tail": None if not hasattr(self, '_cond_vol_scaled') else [None if np.isnan(x) else float(x) for x in self._cond_vol_scaled.tail(20).tolist()],
        }


if __name__ == "__main__":
    # Small example for quick local sanity check (requires pandas)
    import pandas as pd

    # create synthetic price series
    t = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=1500, freq='T')
    # simulate a random walk
    np.random.seed(1)
    returns = 0.0002 + 0.001 * np.random.standard_t(df=5, size=len(t) - 1)
    prices = 50000 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    series = pd.Series(prices, index=t)

    m = ModelAR1EGARCHStudentT()
    m.update_with_price_series(series)
    targ = m.last_price * 1.001
    out = m.simulate_probability(targ, tau_minutes=30, n_sims=500, seed=42)
    print(out)
