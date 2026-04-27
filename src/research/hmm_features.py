from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


BASE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

A_FEATURE_COLUMNS = [
    "r_1m",
    "r_3m",
    "r_5m",
    "r_15m",
    "abs_r_1m",
    "abs_r_5m",
    "realized_vol_5m",
    "realized_vol_15m",
    "realized_vol_30m",
    "range_15m",
    "trend_consistency_15m",
    "sign_flip_count_15m",
    "return_acceleration",
    "volume_zscore",
]

D_FEATURE_COLUMNS = [
    "spectral_entropy_32",
    "low_freq_power_ratio_32",
    "high_freq_power_ratio_32",
    "smoothness_score_32",
    "transition_entropy_15state",
    "transition_entropy_percentile",
    "low_entropy_flag",
    "entropy_slope",
    "entropy_shock_score",
    "forecast_abs_move_5m_baseline",
    "forecast_abs_move_15m_baseline",
    "forecast_large_move_probability_baseline",
]

HMM_FEATURE_COLUMNS = A_FEATURE_COLUMNS + D_FEATURE_COLUMNS


@dataclass(frozen=True)
class EntropyThresholds:
    low_entropy_percentile: float = 0.20
    shock_z_threshold: float = 2.0


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLCV columns: {missing}")


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    validate_ohlcv(df)
    out = df.loc[:, BASE_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _rolling_spectral_features(returns: pd.Series, window: int = 32) -> pd.DataFrame:
    rows = []
    values = returns.fillna(0.0).to_numpy(dtype=float)
    for idx in range(len(values)):
        if idx + 1 < window:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        sample = values[idx + 1 - window : idx + 1]
        sample = sample - np.nanmean(sample)
        power = np.abs(np.fft.rfft(sample)) ** 2
        if len(power) <= 1 or not np.isfinite(power).all() or power.sum() <= 0:
            rows.append((0.0, 0.0, 0.0, 1.0))
            continue
        power = power[1:]
        probs = power / power.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
        split = max(1, len(power) // 3)
        low_ratio = float(power[:split].sum() / power.sum())
        high_ratio = float(power[-split:].sum() / power.sum())
        smoothness = float(1.0 / (1.0 + np.nanstd(np.diff(sample))))
        rows.append((entropy, low_ratio, high_ratio, smoothness))
    return pd.DataFrame(
        rows,
        columns=[
            "spectral_entropy_32",
            "low_freq_power_ratio_32",
            "high_freq_power_ratio_32",
            "smoothness_score_32",
        ],
    )


def causal_volume_quintile(volume: pd.Series, window: int) -> pd.Series:
    values = volume.to_numpy(dtype=float)
    out = np.ones(len(values), dtype=int)
    for idx, value in enumerate(values):
        start = max(0, idx + 1 - window)
        sample = values[start : idx + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < 2 or not np.isfinite(value):
            out[idx] = 3
            continue
        thresholds = np.quantile(sample, [0.2, 0.4, 0.6, 0.8])
        out[idx] = int(np.searchsorted(thresholds, value, side="right") + 1)
    return pd.Series(out, index=volume.index, name="volume_quintile")


def build_15state_series(returns: pd.Series, volume: pd.Series, *, volume_window: int = 240) -> pd.Series:
    signs = np.sign(returns.fillna(0.0).to_numpy(dtype=float)).astype(int)
    signs = np.where(signs < 0, 0, np.where(signs > 0, 2, 1))
    quintiles = causal_volume_quintile(volume, volume_window).to_numpy(dtype=int) - 1
    states = signs * 5 + quintiles
    return pd.Series(states.astype(int), index=returns.index, name="state_15")


def transition_entropy_15state(states: Sequence[int], *, window: int = 120, n_states: int = 15) -> pd.Series:
    states = np.asarray(states, dtype=int)
    out = np.full(len(states), np.nan, dtype=float)
    uniform_entropy = 1.0
    for idx in range(len(states)):
        start = max(0, idx + 1 - window)
        sample = states[start : idx + 1]
        if len(sample) < 2:
            continue
        counts = np.zeros((n_states, n_states), dtype=float)
        for prev, cur in zip(sample[:-1], sample[1:]):
            if 0 <= prev < n_states and 0 <= cur < n_states:
                counts[prev, cur] += 1.0
        row_entropies = []
        weights = []
        for row in counts:
            total = row.sum()
            if total <= 0:
                row_entropies.append(uniform_entropy)
                weights.append(1.0)
            else:
                probs = row / total
                row_entropies.append(float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(n_states)))
                weights.append(total)
        out[idx] = float(np.average(row_entropies, weights=weights))
    return pd.Series(np.clip(out, 0.0, 1.0))


def add_training_entropy_percentiles(
    df: pd.DataFrame,
    *,
    train_mask: Iterable[bool] | None = None,
    thresholds: EntropyThresholds = EntropyThresholds(),
) -> pd.DataFrame:
    out = df.copy()
    source = out["transition_entropy_15state"].astype(float)
    if train_mask is None:
        values = source.to_numpy(dtype=float)
        percentiles = np.full(len(values), np.nan, dtype=float)
        seen: list[float] = []
        for idx, value in enumerate(values):
            if not np.isfinite(value):
                continue
            seen.append(float(value))
            sorted_values = np.sort(np.asarray(seen, dtype=float))
            percentiles[idx] = np.searchsorted(sorted_values, value, side="right") / len(sorted_values)
        out["transition_entropy_percentile"] = np.clip(percentiles, 0.0, 1.0)
        out["low_entropy_flag"] = out["transition_entropy_percentile"] <= thresholds.low_entropy_percentile
        return out
    mask = pd.Series(list(train_mask), index=out.index)
    train_values = source[mask & source.notna()]
    if train_values.empty:
        out["transition_entropy_percentile"] = np.nan
        out["low_entropy_flag"] = False
        return out
    sorted_values = np.sort(train_values.to_numpy(dtype=float))
    ranks = np.searchsorted(sorted_values, source.to_numpy(dtype=float), side="right") / len(sorted_values)
    out["transition_entropy_percentile"] = np.clip(ranks, 0.0, 1.0)
    out["low_entropy_flag"] = out["transition_entropy_percentile"] <= thresholds.low_entropy_percentile
    return out


def build_hmm_feature_frame(
    klines: pd.DataFrame,
    *,
    entropy_window: int = 120,
    volume_quintile_window: int = 240,
    large_move_quantile: float = 0.80,
) -> pd.DataFrame:
    out = _prepare_ohlcv(klines)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    log_close = np.log(close.replace(0, np.nan))
    r_1m = log_close.diff()

    out["r_1m"] = r_1m
    for minutes in [3, 5, 15]:
        out[f"r_{minutes}m"] = log_close.diff(minutes)
    out["abs_r_1m"] = out["r_1m"].abs()
    out["abs_r_5m"] = out["r_5m"].abs()
    for minutes in [5, 15, 30]:
        out[f"realized_vol_{minutes}m"] = r_1m.rolling(minutes, min_periods=minutes).std()
    out["range_15m"] = (high.rolling(15, min_periods=15).max() - low.rolling(15, min_periods=15).min()) / close
    sign_sum = np.sign(r_1m.fillna(0.0)).rolling(15, min_periods=15).sum().abs()
    out["trend_consistency_15m"] = sign_sum / 15.0
    sign = np.sign(r_1m.fillna(0.0))
    flips = ((sign != sign.shift(1)) & (sign != 0) & (sign.shift(1) != 0)).astype(int)
    out["sign_flip_count_15m"] = flips.rolling(15, min_periods=15).sum()
    out["return_acceleration"] = out["r_1m"] - out["r_1m"].shift(1)
    vol_mean = volume.rolling(60, min_periods=20).mean()
    vol_std = volume.rolling(60, min_periods=20).std().replace(0, np.nan)
    out["volume_zscore"] = (volume - vol_mean) / vol_std

    spectral = _rolling_spectral_features(r_1m, window=32)
    out = pd.concat([out, spectral], axis=1)

    states = build_15state_series(r_1m, volume, volume_window=volume_quintile_window)
    out["transition_entropy_15state"] = transition_entropy_15state(states, window=entropy_window).to_numpy()
    out = add_training_entropy_percentiles(out)
    out["entropy_slope"] = out["transition_entropy_15state"].diff()
    entropy_mean = out["transition_entropy_15state"].rolling(60, min_periods=20).mean()
    entropy_std = out["transition_entropy_15state"].rolling(60, min_periods=20).std().replace(0, np.nan)
    out["entropy_shock_score"] = (out["transition_entropy_15state"] - entropy_mean) / entropy_std

    out["forecast_abs_move_5m_baseline"] = out["abs_r_1m"].rolling(30, min_periods=5).mean() * np.sqrt(5.0)
    out["forecast_abs_move_15m_baseline"] = out["abs_r_1m"].rolling(60, min_periods=15).mean() * np.sqrt(15.0)
    threshold = out["abs_r_5m"].rolling(240, min_periods=30).quantile(large_move_quantile)
    out["forecast_large_move_probability_baseline"] = (
        (out["abs_r_5m"] > threshold).astype(float).rolling(60, min_periods=10).mean()
    )
    return out[BASE_COLUMNS + HMM_FEATURE_COLUMNS]


def hmm_feature_matrix(df: pd.DataFrame, *, feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    cols = list(feature_columns or HMM_FEATURE_COLUMNS)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing HMM feature columns: {missing}")
    return df.loc[:, cols].copy()
