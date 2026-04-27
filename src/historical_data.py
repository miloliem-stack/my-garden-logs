"""Historical Binance BTCUSDT 1m loaders for offline research.

The Binance spot kline files expected here are headerless with 12 columns:
0 open_time
1 open
2 high
3 low
4 close
5 volume
6 close_time
7 quote_asset_volume
8 number_of_trades
9 taker_buy_base_asset_volume
10 taker_buy_quote_asset_volume
11 ignore
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd


BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


@dataclass(frozen=True)
class GapSummary:
    gap_count: int
    gap_minutes: list[int]
    gap_after: list[str]


def _resolve_input_files(
    input_dir: str | Path | None = None,
    glob: str | None = None,
    files: Sequence[str | Path] | None = None,
    max_files: int | None = None,
    tail_files: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if files:
        paths.extend(Path(p) for p in files)
    if input_dir is not None:
        base = Path(input_dir)
        pattern = glob or "*.csv"
        paths.extend(sorted(base.glob(pattern)))
    if not paths:
        raise FileNotFoundError("No input files matched the provided directory/files arguments")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {missing}")
    resolved = sorted(dict.fromkeys(path.resolve() for path in paths))
    if tail_files is not None:
        resolved = resolved[-int(tail_files):]
    if max_files is not None:
        resolved = resolved[: int(max_files)]
    return resolved


def resolve_binance_input_files(
    input_dir: str | Path | None = None,
    glob: str | None = None,
    files: Sequence[str | Path] | None = None,
    max_files: int | None = None,
    tail_files: int | None = None,
) -> list[Path]:
    return _resolve_input_files(
        input_dir=input_dir,
        glob=glob,
        files=files,
        max_files=max_files,
        tail_files=tail_files,
    )


def detect_timestamp_unit(open_times: Iterable[int | float | str]) -> str:
    """Detect Binance timestamp unit from magnitude."""
    values = pd.Series(list(open_times)).dropna()
    if values.empty:
        raise ValueError("Cannot detect timestamp unit from empty timestamp series")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("Timestamp series contains no numeric values")
    magnitude = int(numeric.abs().max())
    return "us" if magnitude >= 10**15 else "ms"


def _read_binance_file(path: Path) -> pd.DataFrame:
    compression = "zip" if path.suffix.lower() == ".zip" else "infer"
    df = pd.read_csv(path, header=None, compression=compression)
    if df.shape[1] != len(BINANCE_KLINE_COLUMNS):
        raise ValueError(
            f"{path} has {df.shape[1]} columns; expected {len(BINANCE_KLINE_COLUMNS)} Binance kline columns"
        )
    df.columns = BINANCE_KLINE_COLUMNS
    unit = detect_timestamp_unit(df["open_time"])
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="raise"), unit=unit, utc=True)
    for column in _NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["source_path"] = str(path)
    return df


def summarize_minute_gaps(index: pd.DatetimeIndex) -> GapSummary:
    if len(index) < 2:
        return GapSummary(gap_count=0, gap_minutes=[], gap_after=[])
    diffs = index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=1)]
    gap_minutes = [int(delta.total_seconds() // 60) - 1 for delta in gaps]
    gap_after = [
        (ts - delta).isoformat()
        for ts, delta in zip(gaps.index, gaps.values, strict=False)
    ]
    return GapSummary(gap_count=len(gaps), gap_minutes=gap_minutes, gap_after=gap_after)


def load_binance_1m_dataframe(
    input_dir: str | Path | None = None,
    glob: str | None = None,
    files: Sequence[str | Path] | None = None,
    max_files: int | None = None,
    tail_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    """Load Binance 1m kline files into a UTC-indexed DataFrame."""
    paths = _resolve_input_files(
        input_dir=input_dir,
        glob=glob,
        files=files,
        max_files=max_files,
        tail_files=tail_files,
    )
    frames = []
    total_files = len(paths)
    rows_loaded = 0
    for idx, path in enumerate(paths, start=1):
        frame = _read_binance_file(path)
        frames.append(frame)
        rows_loaded += len(frame)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "file_ingest",
                    "files_loaded": idx,
                    "files_total": total_files,
                    "rows_loaded": rows_loaded,
                    "current_file": str(path),
                }
            )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp")
    duplicate_count = int(df.duplicated(subset=["timestamp"]).sum())
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.set_index("timestamp")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Loaded minute data index is not monotonic increasing after sort/dedup")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("Loaded minute data index is not timezone-aware UTC")
    df = df[["open", "high", "low", "close", "volume"]]
    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df.index >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df.index < end_ts]
    gaps = summarize_minute_gaps(df.index)
    df.attrs["source_files"] = [str(path) for path in paths]
    df.attrs["duplicate_rows_dropped"] = duplicate_count
    df.attrs["gaps"] = {
        "gap_count": gaps.gap_count,
        "gap_minutes": gaps.gap_minutes,
        "gap_after": gaps.gap_after,
    }
    return df


def load_btc_1m_close_series(
    input_dir: str | Path | None = None,
    glob: str | None = None,
    files: Sequence[str | Path] | None = None,
) -> pd.Series:
    df = load_binance_1m_dataframe(input_dir=input_dir, glob=glob, files=files)
    series = df["close"].copy()
    series.name = "close"
    return series
