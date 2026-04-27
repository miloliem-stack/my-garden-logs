from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe
from src.probability_backtest import build_or_load_events
from src.probability_engine_evaluation import evaluate_probability_engines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling blocked evaluation of probability engines")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--events-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision-step-minutes", type=int, default=15)
    parser.add_argument("--train-hours", type=int, default=24 * 30)
    parser.add_argument("--calibration-hours", type=int, default=24 * 7)
    parser.add_argument("--validation-hours", type=int, default=24 * 7)
    parser.add_argument("--step-hours", type=int)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-events", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    minute_df = load_binance_1m_dataframe(input_dir=args.input_dir, glob=args.glob)
    events = build_or_load_events(
        close_minute_df=minute_df,
        events_path=args.events_path,
        decision_step_minutes=args.decision_step_minutes,
        start=args.start,
        end=args.end,
        max_events=args.max_events,
    )
    report = evaluate_probability_engines(
        minute_df=minute_df,
        events=events,
        output_dir=args.output_dir,
        train_hours=args.train_hours,
        calibration_hours=args.calibration_hours,
        validation_hours=args.validation_hours,
        step_hours=args.step_hours,
    )
    print(
        {
            "events": len(events),
            "fold_count": report["fold_count"],
            "summary_path": report["summary_path"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
