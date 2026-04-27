from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe
from src.hourly_event_dataset import build_hourly_event_dataset, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BTC hourly-event dataset from Binance 1m files")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision-step-minutes", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    minute_df = load_binance_1m_dataframe(input_dir=args.input_dir, glob=args.glob)
    events = build_hourly_event_dataset(minute_df, decision_step_minutes=args.decision_step_minutes)
    write_dataframe(events, args.output)
    print(
        {
            "rows": int(len(events)),
            "output": str(args.output),
            "source_files": minute_df.attrs.get("source_files", []),
            "gaps": minute_df.attrs.get("gaps", {}),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
