import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hourly_event_dataset import build_hourly_event_dataset


def _make_hour(start: str) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=60, freq="min", tz="UTC")
    rows = []
    for i, ts in enumerate(index):
        rows.append(
            {
                "open": 100.0 + i,
                "high": 100.5 + i,
                "low": 99.5 + i,
                "close": 100.25 + i,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows, index=index)


def test_hourly_event_dataset_uses_hour_open_and_hour_close() -> None:
    minute_df = _make_hour("2026-01-01T00:00:00Z")
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=5)

    first = events.iloc[0]
    last = events.iloc[-1]

    assert len(events) == 12
    assert first["strike_price"] == minute_df.iloc[0]["open"]
    assert first["settlement_price"] == minute_df.iloc[-1]["close"]
    assert first["spot_now"] == minute_df.iloc[0]["open"]
    assert first["tau_minutes"] == 60
    assert last["minute_in_hour"] == 55
    assert last["tau_minutes"] == 5


def test_hourly_event_dataset_skips_incomplete_hours() -> None:
    minute_df = _make_hour("2026-01-01T00:00:00Z").iloc[:-1]
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=5)
    assert events.empty
