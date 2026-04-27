"""Strictly causal feature builder for BTC hourly event probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_LAGS = [1, 2, 3, 5, 10, 15, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]


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
    required = {"decision_ts", "event_hour_start", "tau_minutes", "strike_price", "spot_now", "realized_yes"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")
    out = events.copy()
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True)
    out["event_hour_start"] = pd.to_datetime(out["event_hour_start"], utc=True)
    return out.sort_values("decision_ts").reset_index(drop=True)


def build_event_features(minute_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    minute_df = _validate_minute_df(minute_df)
    events = _validate_events(events)

    close = minute_df["close"].astype(float)
    high = minute_df["high"].astype(float)
    low = minute_df["low"].astype(float)
    volume = minute_df["volume"].astype(float) if "volume" in minute_df.columns else pd.Series(0.0, index=minute_df.index)

    log_close = np.log(close.clip(lower=1e-12))
    returns = log_close.diff()
    range_proxy = np.log((high / low).clip(lower=1e-12))

    feature_frame = pd.DataFrame(index=minute_df.index)
    for lag in RETURN_LAGS:
        feature_frame[f"ret_lag_{lag}"] = returns.shift(lag)

    past_returns = returns.shift(1)
    past_ranges = range_proxy.shift(1)
    past_volume = volume.shift(1)

    for window in ROLLING_WINDOWS:
        feature_frame[f"rv_{window}"] = past_returns.rolling(window, min_periods=window).std(ddof=0)
        feature_frame[f"range_mean_{window}"] = past_ranges.rolling(window, min_periods=window).mean()
        feature_frame[f"range_max_{window}"] = past_ranges.rolling(window, min_periods=window).max()
        feature_frame[f"volume_mean_{window}"] = past_volume.rolling(window, min_periods=window).mean()
        feature_frame[f"volume_std_{window}"] = past_volume.rolling(window, min_periods=window).std(ddof=0)

    feature_frame["volume_prev_1"] = past_volume
    volume_mean_15 = feature_frame["volume_mean_15"]
    volume_std_15 = feature_frame["volume_std_15"].replace(0.0, np.nan)
    feature_frame["volume_z_15"] = (past_volume - volume_mean_15) / volume_std_15

    event_indexed = events.set_index("decision_ts")
    joined = event_indexed.join(feature_frame, how="left")

    joined["signed_distance_to_strike"] = np.log(
        joined["spot_now"].astype(float).clip(lower=1e-12) / joined["strike_price"].astype(float).clip(lower=1e-12)
    )
    joined["distance_to_strike_scaled_vol"] = joined["signed_distance_to_strike"] / joined["rv_15"].clip(lower=1e-8)
    joined["current_hour_return_so_far"] = np.log(
        joined["spot_now"].astype(float).clip(lower=1e-12) / joined["strike_price"].astype(float).clip(lower=1e-12)
    )
    decision_index = pd.to_datetime(joined.index, utc=True)
    joined["minute_in_hour"] = decision_index.minute
    joined["hour_of_day"] = decision_index.hour
    joined["day_of_week"] = decision_index.dayofweek
    joined.index.name = "decision_ts"

    joined = joined.reset_index()
    return joined


def get_default_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    exclude = {
        "decision_ts",
        "event_hour_start",
        "event_hour_end",
        "realized_yes",
        "settlement_price",
    }
    return [column for column in feature_df.columns if column not in exclude]
