from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe
from src.probability_backtest import build_or_load_events
from src.probability_challengers import build_challenger_config_grid, run_challenger_sweep


def _parse_bool_list(value: str | None) -> list[bool] | None:
    if value is None:
        return None
    mapping = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    out = []
    for item in value.split(","):
        key = item.strip().lower()
        if key not in mapping:
            raise argparse.ArgumentTypeError(f"Invalid boolean selector '{item}'. Use on/off.")
        out.append(mapping[key])
    return out


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep challenger configs for the offline hourly-event probability engine")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--events-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision-step-minutes", type=int, default=15)
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--gaussian-vol-window", type=int, default=288)
    parser.add_argument("--near-strike-log-threshold", type=float, default=0.001)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--ar", type=_parse_bool_list, help="Comma-separated AR filter: on,off")
    parser.add_argument("--asymmetry", type=_parse_bool_list, help="Comma-separated asymmetry filter: on,off")
    parser.add_argument("--jump-augmentation", type=_parse_bool_list, help="Comma-separated jump filter: on,off")
    parser.add_argument("--fit-windows", type=_parse_int_list, help="Comma-separated fit windows")
    parser.add_argument("--residual-buffer-sizes", type=_parse_int_list, help="Comma-separated residual buffer sizes")
    parser.add_argument("--refit-every-n-events-list", type=_parse_int_list, help="Comma-separated refit cadences")
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
    configs = build_challenger_config_grid(
        ar_values=args.ar,
        asymmetry_values=args.asymmetry,
        jump_values=args.jump_augmentation,
        fit_windows=args.fit_windows,
        residual_buffer_sizes=args.residual_buffer_sizes,
        refit_every_values=args.refit_every_n_events_list,
    )
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    report = run_challenger_sweep(
        close_series=minute_df["close"],
        events=events,
        output_dir=args.output_dir,
        configs=configs,
        n_sims=args.n_sims,
        min_history=args.min_history,
        gaussian_vol_window=args.gaussian_vol_window,
        near_strike_log_threshold=args.near_strike_log_threshold,
    )
    print(
        {
            "configs_evaluated": len(configs),
            "events": len(events),
            "winner_count": report["winner_count"],
            "summary_path": report["summary_path"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
