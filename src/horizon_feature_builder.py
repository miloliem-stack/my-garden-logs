"""Generic strictly causal feature builder for fixed-horizon BTC events."""

from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_LAGS = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]
VOL_WINDOWS = [5, 15, 30, 60, 120, 240]
TREND_WINDOWS = [15, 60, 240]


def _validate_minute_df(minute_df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(minute_df.columns))
    if missing:
        raise ValueError(f"minute_df missing required columns: {missing}")
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        raise TypeError("minute_df index must be a DatetimeIndex")
    if minute_df.index.tz is None or str(minute_df.index.tz) != "UTC":
        raise ValueError("minute_df index must be timezone-aware UTC")
    return minute_df.sort_index()


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_ts", "event_start", "tau_minutes", "strike_price", "spot_now", "realized_yes", "horizon"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")
    out = events.copy()
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True)
    out["event_start"] = pd.to_datetime(out["event_start"], utc=True)
    out["event_end"] = pd.to_datetime(out["event_end"], utc=True)
    return out.sort_values("decision_ts").reset_index(drop=True)


def build_horizon_event_features(minute_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    minute_df = _validate_minute_df(minute_df)
    events = _validate_events(events)

    close = minute_df["close"].astype(float)
    high = minute_df["high"].astype(float)
    low = minute_df["low"].astype(float)
    volume = minute_df["volume"].astype(float) if "volume" in minute_df.columns else pd.Series(0.0, index=minute_df.index)

    log_close = np.log(close.clip(lower=1e-12))
    returns = log_close.diff()
    range_proxy = np.log((high / low).clip(lower=1e-12))
    abs_returns = returns.abs()

    feature_frame = pd.DataFrame(index=minute_df.index)
    for lag in RETURN_LAGS:
        feature_frame[f"ret_lag_{lag}"] = returns.shift(lag)

    past_returns = returns.shift(1)
    past_abs_returns = abs_returns.shift(1)
    past_ranges = range_proxy.shift(1)
    past_volume = volume.shift(1)

    for window in VOL_WINDOWS:
        rolled = past_returns.rolling(window, min_periods=window)
        feature_frame[f"rv_{window}"] = rolled.std(ddof=0)
        feature_frame[f"skew_{window}"] = rolled.skew()
        feature_frame[f"kurt_{window}"] = rolled.kurt()
        feature_frame[f"abs_ret_mean_{window}"] = past_abs_returns.rolling(window, min_periods=window).mean()
        feature_frame[f"range_mean_{window}"] = past_ranges.rolling(window, min_periods=window).mean()
        feature_frame[f"volume_mean_{window}"] = past_volume.rolling(window, min_periods=window).mean()

    for window in TREND_WINDOWS:
        feature_frame[f"trend_{window}"] = log_close.shift(1) - log_close.shift(window)
        feature_frame[f"reversal_{window}"] = returns.shift(1) - returns.shift(window)

    feature_frame["volume_prev_1"] = past_volume
    feature_frame["volume_z_60"] = (
        (past_volume - feature_frame["volume_mean_60"]) / feature_frame["volume_mean_60"].replace(0.0, np.nan)
    )

    joined = events.set_index("decision_ts").join(feature_frame, how="left")
    joined["signed_distance_to_strike"] = np.log(
        joined["spot_now"].astype(float).clip(lower=1e-12) / joined["strike_price"].astype(float).clip(lower=1e-12)
    )
    joined["distance_to_strike_scaled_vol"] = joined["signed_distance_to_strike"] / joined["rv_60"].clip(lower=1e-8)
    joined["current_event_return_so_far"] = np.log(
        joined["spot_now"].astype(float).clip(lower=1e-12) / joined["strike_price"].astype(float).clip(lower=1e-12)
    )
    decision_index = pd.to_datetime(joined.index, utc=True)
    joined["decision_minute_of_hour"] = decision_index.minute
    joined["hour_of_day"] = decision_index.hour
    joined["day_of_week"] = decision_index.dayofweek
    joined["month"] = decision_index.month
    joined["horizon_1h"] = (joined["horizon"] == "1h").astype(int)
    joined["horizon_4h"] = (joined["horizon"] == "4h").astype(int)
    joined["horizon_1d"] = (joined["horizon"] == "1d").astype(int)
    out = joined.reset_index()
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate columns in horizon dataset: {duplicated}")
    return out


def get_horizon_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    exclude = {
        "decision_ts",
        "event_start",
        "event_end",
        "realized_yes",
        "settlement_price",
        "horizon",
    }
    return [column for column in feature_df.columns if column not in exclude]
