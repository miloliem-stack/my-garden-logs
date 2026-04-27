from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe
from src.probability_backtest import build_or_load_events
from src.probability_engine_evaluation import evaluate_gaussian_vol_configs


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep gaussian_vol engine parameters offline")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--events-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision-step-minutes", type=int, default=15)
    parser.add_argument("--vol-windows", type=_parse_int_list, default=[720, 1440, 2880])
    parser.add_argument("--min-periods", type=_parse_int_list, default=[180, 360, 720])
    parser.add_argument("--calibration-modes", type=_parse_str_list, default=["none", "logistic"])
    parser.add_argument("--write-predictions", type=_parse_bool, default=False)
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
    configs = []
    for vol_window in args.vol_windows:
        for min_periods in args.min_periods:
            for calibration_mode in args.calibration_modes:
                configs.append(
                    {
                        "config_id": f"gaussian_vw{vol_window}_mp{min_periods}_{calibration_mode}",
                        "vol_window": vol_window,
                        "min_periods": min_periods,
                        "calibration_mode": calibration_mode,
                        "fit_window": max(vol_window, min_periods),
                    }
                )
    report = evaluate_gaussian_vol_configs(
        minute_df=minute_df,
        events=events,
        output_dir=args.output_dir,
        gaussian_configs=configs,
        train_hours=args.train_hours,
        calibration_hours=args.calibration_hours,
        validation_hours=args.validation_hours,
        step_hours=args.step_hours,
        write_predictions=args.write_predictions,
    )
    print(
        {
            "configs": len(configs),
            "summary_path": report["summary_path"],
            "ranked_summary_path": report.get("ranked_summary_path"),
            "predictions_path": report.get("predictions_path"),
            "fold_count": report["fold_count"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
