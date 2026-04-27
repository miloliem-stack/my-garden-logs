"""Strictly causal fixed-horizon BTC event datasets for 1h/4h/1d research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HORIZON_TO_MINUTES = {
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _validate_minute_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Minute frame missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Minute frame index must be a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("Minute frame index must be timezone-aware UTC")
    return df.sort_index()


def _horizon_freq(horizon: str) -> str:
    if horizon == "1h":
        return "1h"
    if horizon == "4h":
        return "4h"
    if horizon == "1d":
        return "1d"
    raise ValueError(f"Unsupported horizon: {horizon}")


def build_fixed_horizon_event_dataset(
    minute_df: pd.DataFrame,
    *,
    horizon: str,
    decision_step_minutes: int | None = None,
) -> pd.DataFrame:
    minute_df = _validate_minute_frame(minute_df)
    if horizon not in HORIZON_TO_MINUTES:
        raise ValueError(f"Unsupported horizon: {horizon}")
    horizon_minutes = HORIZON_TO_MINUTES[horizon]
    default_step = 5 if horizon == "1h" else 15 if horizon == "4h" else 60
    step_minutes = default_step if decision_step_minutes is None else int(decision_step_minutes)
    if step_minutes <= 0 or horizon_minutes % step_minutes != 0:
        raise ValueError("decision_step_minutes must be a positive divisor of the horizon length")

    freq = _horizon_freq(horizon)
    event_start_index = minute_df.index.floor(freq)
    rows: list[dict] = []
    for event_start, group in minute_df.groupby(event_start_index):
        event_end = event_start + pd.Timedelta(minutes=horizon_minutes)
        group = group.sort_index()
        if len(group) != horizon_minutes:
            continue
        expected = pd.date_range(event_start, periods=horizon_minutes, freq="min", tz="UTC")
        if not group.index.equals(expected):
            continue
        strike_price = float(group.iloc[0]["open"])
        settlement_price = float(group.iloc[-1]["close"])
        realized_yes = int(settlement_price >= strike_price)
        for minute_in_event in range(0, horizon_minutes, step_minutes):
            candle = group.iloc[minute_in_event]
            decision_ts = group.index[minute_in_event]
            tau_minutes = horizon_minutes - minute_in_event
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "event_start": event_start,
                    "event_end": event_end,
                    "horizon": horizon,
                    "horizon_minutes": horizon_minutes,
                    "minute_in_event": minute_in_event,
                    "tau_minutes": tau_minutes,
                    "strike_price": strike_price,
                    "spot_now": float(candle["open"]),
                    "settlement_price": settlement_price,
                    "realized_yes": realized_yes,
                }
            )
    return pd.DataFrame(rows)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
