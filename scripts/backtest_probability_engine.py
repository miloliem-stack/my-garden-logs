from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe
from src.probability_backtest import build_or_load_events, run_probability_backtest, save_backtest_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward BTC hourly-event probability backtest")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--events-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-window", type=int, default=2000)
    parser.add_argument("--residual-buffer-size", type=int, default=2000)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--decision-step-minutes", type=int, default=5)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--refit-every-n-events", type=int, default=12)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--seed", type=int)
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
    results = run_probability_backtest(
        close_series=minute_df["close"],
        events=events,
        fit_window=args.fit_window,
        residual_buffer_size=args.residual_buffer_size,
        n_sims=args.n_sims,
        seed=args.seed,
        min_history=args.min_history,
        refit_every_n_events=args.refit_every_n_events,
    )
    save_backtest_results(results, args.output)
    print(
        {
            "events_requested": int(len(events)),
            "results_written": int(len(results)),
            "skipped_or_failed": int((results["skip_reason"].notna()).sum()) if not results.empty else 0,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
