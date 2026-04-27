import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.historical_data import detect_timestamp_unit, load_binance_1m_dataframe, resolve_binance_input_files


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_detect_timestamp_unit_ms_and_us() -> None:
    assert detect_timestamp_unit([1693526400000, 1693526460000]) == "ms"
    assert detect_timestamp_unit([1735689600000000, 1735689660000000]) == "us"


def test_loader_returns_sorted_utc_index_and_expected_columns(tmp_path: Path) -> None:
    ms_path = tmp_path / "BTCUSDT-1m-ms.csv"
    us_path = tmp_path / "BTCUSDT-1m-us.csv"
    _write_rows(
        ms_path,
        [
            [1693526460000, 26010, 26015, 26005, 26012, 1, 1693526519999, 1, 10, 0.5, 0.5, 0],
            [1693526400000, 26000, 26005, 25995, 26001, 1, 1693526459999, 1, 10, 0.5, 0.5, 0],
        ],
    )
    _write_rows(
        us_path,
        [
            [1735689720000000, 43020, 43025, 43015, 43023, 2, 1735689779999999, 2, 20, 1, 1, 0],
            [1735689660000000, 43010, 43015, 43005, 43012, 2, 1735689719999999, 2, 20, 1, 1, 0],
            [1735689660000000, 43011, 43016, 43006, 43013, 3, 1735689719999999, 3, 21, 1, 1, 0],
        ],
    )

    df = load_binance_1m_dataframe(files=[ms_path, us_path])

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == 4
    assert df.attrs["duplicate_rows_dropped"] == 1


def test_loader_reports_small_gaps(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT-1m-gap.csv"
    _write_rows(
        path,
        [
            [1693526400000, 26000, 26005, 25995, 26001, 1, 1693526459999, 1, 10, 0.5, 0.5, 0],
            [1693526520000, 26020, 26025, 26015, 26021, 1, 1693526579999, 1, 10, 0.5, 0.5, 0],
        ],
    )
    df = load_binance_1m_dataframe(files=[path])
    assert df.attrs["gaps"]["gap_count"] == 1
    assert df.attrs["gaps"]["gap_minutes"] == [1]


def test_loader_raises_on_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([[1, 2, 3]]).to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="expected 12 Binance kline columns"):
        load_binance_1m_dataframe(files=[path])


def test_resolve_input_files_applies_tail_then_max_on_sorted_files(tmp_path: Path) -> None:
    paths = []
    for month in ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"]:
        path = tmp_path / f"BTCUSDT-1m-{month}.csv"
        _write_rows(path, [[1693526400000, 1, 1, 1, 1, 1, 1693526459999, 1, 1, 1, 1, 0]])
        paths.append(path)

    selected = resolve_binance_input_files(
        input_dir=tmp_path,
        glob="BTCUSDT-1m-*.csv",
        tail_files=3,
        max_files=2,
    )

    assert [path.name for path in selected] == [
        "BTCUSDT-1m-2025-03.csv",
        "BTCUSDT-1m-2025-04.csv",
    ]


def test_loader_filters_by_date_range(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT-1m-2025-01.csv"
    _write_rows(
        path,
        [
            [1735689540000000, 100, 101, 99, 100, 1, 1735689599999999, 1, 1, 1, 1, 0],  # 2024-12-31 23:59
            [1735689600000000, 101, 102, 100, 101, 1, 1735689659999999, 1, 1, 1, 1, 0],  # 2025-01-01 00:00
            [1735776000000000, 102, 103, 101, 102, 1, 1735776059999999, 1, 1, 1, 1, 0],  # 2025-01-02 00:00
            [1735862400000000, 103, 104, 102, 103, 1, 1735862459999999, 1, 1, 1, 1, 0],  # 2025-01-03 00:00
        ],
    )

    df = load_binance_1m_dataframe(
        files=[path],
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert len(df) == 2
    assert df.index.min() == pd.Timestamp("2025-01-01T00:00:00Z")
    assert df.index.max() == pd.Timestamp("2025-01-02T00:00:00Z")
