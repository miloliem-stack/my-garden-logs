"""Helpers to build strictly causal BTC hourly-event datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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


def build_hourly_event_dataset(minute_df: pd.DataFrame, decision_step_minutes: int = 5) -> pd.DataFrame:
    if decision_step_minutes <= 0 or 60 % decision_step_minutes != 0:
        raise ValueError("decision_step_minutes must be a positive divisor of 60")
    minute_df = _validate_minute_frame(minute_df)
    hour_start = minute_df.index.floor("h")
    rows: list[dict] = []
    for event_hour_start, group in minute_df.groupby(hour_start):
        group = group.sort_index()
        if len(group) != 60:
            continue
        expected = pd.date_range(event_hour_start, periods=60, freq="min", tz="UTC")
        if not group.index.equals(expected):
            continue
        strike_price = float(group.iloc[0]["open"])
        settlement_price = float(group.iloc[-1]["close"])
        realized_yes = int(settlement_price >= strike_price)
        event_hour_end = event_hour_start + pd.Timedelta(hours=1)
        for minute_in_hour in range(0, 60, decision_step_minutes):
            candle = group.iloc[minute_in_hour]
            decision_ts = group.index[minute_in_hour]
            tau_minutes = 60 - minute_in_hour
            rows.append(
                {
                    "decision_ts": decision_ts,
                    "event_hour_start": event_hour_start,
                    "event_hour_end": event_hour_end,
                    "minute_in_hour": minute_in_hour,
                    "tau_minutes": tau_minutes,
                    "strike_price": strike_price,
                    "spot_now": float(candle["open"]),
                    "settlement_price": settlement_price,
                    "realized_yes": realized_yes,
                }
            )
    return pd.DataFrame(rows)


def filter_event_dataset(
    events: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    max_events: int | None = None,
) -> pd.DataFrame:
    out = events.copy()
    if start is not None:
        out = out[out["decision_ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out[out["decision_ts"] <= pd.Timestamp(end, tz="UTC")]
    out = out.sort_values("decision_ts")
    if max_events is not None:
        out = out.head(max_events)
    return out.reset_index(drop=True)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError("Parquet output requires pyarrow or fastparquet") from exc
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {path}; use .parquet or .csv")
