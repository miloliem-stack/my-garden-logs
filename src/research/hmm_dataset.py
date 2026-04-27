from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .hmm_features import BASE_COLUMNS, HMM_FEATURE_COLUMNS, build_hmm_feature_frame


POLICY_COLUMNS = [
    "q_yes_bid",
    "q_yes_ask",
    "q_no_bid",
    "q_no_ask",
    "spread_yes",
    "spread_no",
    "quote_age_sec",
    "q_tail",
    "quote_polarization",
    "p_yes",
    "p_no",
    "sigma_per_sqrt_min",
    "edge_yes_ask",
    "edge_no_ask",
    "model_market_disagreement_abs",
    "market_id",
    "tau_minutes",
    "tau_bucket",
    "strike_price",
    "vanilla_action",
    "vanilla_side",
    "vanilla_entry_price",
    "expected_log_growth",
    "conservative_expected_log_growth",
    "tail_veto_flag",
    "polarization_veto_flag",
    "reversal_veto_flag",
    "realized_outcome",
    "realized_pnl_binary",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path, *, fmt: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or path.suffix.lstrip(".") or "csv").lower()
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output format: {fmt}")


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns and "ts" in out.columns:
        out = out.rename(columns={"ts": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("input table missing timestamp/ts column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _load_optional(path: str | Path | None, allowed_columns: Iterable[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    df = normalize_timestamp(read_table(path))
    keep = ["timestamp"] + [col for col in allowed_columns if col in df.columns]
    return df.loc[:, keep]


def build_hmm_replay_dataset(
    klines_path: str | Path,
    *,
    decision_log_path: str | Path | None = None,
    quotes_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    klines = normalize_timestamp(read_table(klines_path))
    if start:
        klines = klines[klines["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        klines = klines[klines["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    features = build_hmm_feature_frame(klines.loc[:, BASE_COLUMNS])
    out = features.copy()
    for optional in [
        _load_optional(decision_log_path, POLICY_COLUMNS),
        _load_optional(quotes_path, POLICY_COLUMNS),
    ]:
        if optional is None:
            continue
        out = out.merge(optional, on="timestamp", how="left", suffixes=("", "_dup"))
        dup_cols = [col for col in out.columns if col.endswith("_dup")]
        for col in dup_cols:
            base = col[:-4]
            if base in out.columns:
                out[base] = out[base].combine_first(out[col])
        out = out.drop(columns=dup_cols)
    for col in POLICY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[BASE_COLUMNS + HMM_FEATURE_COLUMNS + POLICY_COLUMNS]


def write_manifest(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

