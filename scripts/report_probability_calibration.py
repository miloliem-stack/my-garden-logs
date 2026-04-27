from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.probability_backtest import load_event_dataset
from src.probability_calibration import (
    build_forecast_summary_table,
    build_reliability_comparison,
    build_reliability_table,
    build_tau_bucket_brier,
    compute_calibration_summary,
    write_calibration_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration metrics for hourly-event probability backtests")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-method", choices=["none", "logistic"], default="none")
    parser.add_argument("--gaussian-vol-window", type=int, default=288)
    parser.add_argument("--bucket-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_event_dataset(args.input)
    report = write_calibration_report(
        results,
        args.output_dir,
        bucket_count=args.bucket_count,
        calibration_method=args.calibration_method,
        gaussian_vol_window=args.gaussian_vol_window,
    )
    print(report["summary"])
    print(
        {
            "reliability_rows": len(build_reliability_table(results, bucket_count=args.bucket_count)),
            "tau_rows": len(build_tau_bucket_brier(results)),
            "forecast_rows": len(build_forecast_summary_table(results)),
            "reliability_comparison_rows": len(build_reliability_comparison(results, bucket_count=args.bucket_count)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
