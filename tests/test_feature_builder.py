import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_builder import build_event_features
from src.hourly_event_dataset import build_hourly_event_dataset


def _make_minute_df(hours: int = 3) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=hours * 60, freq="min", tz="UTC")
    rows = []
    for i, _ts in enumerate(index):
        base = 100.0 + i * 0.1
        rows.append(
            {
                "open": base,
                "high": base + 0.3,
                "low": base - 0.2,
                "close": base + 0.05,
                "volume": 10.0 + i,
            }
        )
    return pd.DataFrame(rows, index=index)


def test_feature_builder_is_causal_relative_to_decision_ts() -> None:
    minute_df = _make_minute_df(hours=3)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    target_ts = pd.Timestamp("2026-01-01T01:30:00Z")

    row_before = build_event_features(minute_df, events)
    row_before = row_before[row_before["decision_ts"] == target_ts].iloc[0]

    mutated = minute_df.copy()
    mutated.loc[mutated.index > target_ts, ["close", "high", "low", "volume"]] *= 10.0
    row_after = build_event_features(mutated, events)
    row_after = row_after[row_after["decision_ts"] == target_ts].iloc[0]

    feature_cols = [c for c in row_before.index if c not in {"realized_yes", "settlement_price"}]
    assert row_before[feature_cols].to_dict() == row_after[feature_cols].to_dict()


def test_feature_builder_includes_requested_feature_families() -> None:
    minute_df = _make_minute_df(hours=3)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=15)
    features = build_event_features(minute_df, events)
    expected = {
        "ret_lag_1",
        "ret_lag_60",
        "rv_15",
        "range_mean_30",
        "signed_distance_to_strike",
        "distance_to_strike_scaled_vol",
        "current_hour_return_so_far",
        "minute_in_hour",
        "hour_of_day",
        "day_of_week",
        "volume_mean_15",
        "volume_z_15",
    }
    assert expected <= set(features.columns)
