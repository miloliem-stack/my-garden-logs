#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data import load_binance_1m_dataframe, resolve_binance_input_files
from src.horizon_compare import build_horizon_splits, run_horizon_comparison, select_horizon_splits
from src.horizon_event_dataset import build_fixed_horizon_event_dataset


DEFAULT_MODELS = [
    "naive_directional",
    "gaussian_vol",
    "ar_only",
    "ar_vol",
    "ar_egarch",
    "lgbm",
    "blend_ar_lgbm",
]


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BTC directional prediction quality across 1h/4h/1d horizons.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--glob", default="BTCUSDT-1m-*.csv")
    parser.add_argument("--start-date", help="Inclusive UTC start date filter, e.g. 2025-01-01")
    parser.add_argument("--end-date", help="Inclusive UTC end date filter, e.g. 2025-03-31")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--tail-files", type=int)
    parser.add_argument("--horizons", default="1h,4h,1d", help="Comma-separated horizon list from {1h,4h,1d}")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-events", type=int, default=90)
    parser.add_argument("--calibration-events", type=int, default=30)
    parser.add_argument("--validation-events", type=int, default=30)
    parser.add_argument("--step-events", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--edge-thresholds", default="0.05,0.10")
    parser.add_argument("--decision-step-1h", type=int, default=5)
    parser.add_argument("--decision-step-4h", type=int, default=15)
    parser.add_argument("--decision-step-1d", type=int, default=60)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args()


class ProgressReporter:
    def __init__(self, output_dir: Path, heartbeat_seconds: float = 30.0) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.heartbeat_seconds = max(float(heartbeat_seconds), 0.0)
        self.started = time.time()
        self.last_print = 0.0
        self.last_stage = None
        self.state: dict = {}

    def update(self, payload: dict) -> None:
        now = time.time()
        merged = dict(self.state)
        merged.update(payload)
        merged["elapsed_seconds"] = merged.get("elapsed_seconds", now - self.started)
        merged["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state = merged
        self.progress_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        stage = self.state.get("stage")
        should_print = (
            self.last_stage != stage
            or self.heartbeat_seconds == 0.0
            or (now - self.last_print) >= self.heartbeat_seconds
        )
        if should_print:
            print(self._format_line(), flush=True)
            self.last_print = now
            self.last_stage = stage

    def _format_line(self) -> str:
        stage = self.state.get("stage", "unknown")
        parts = [f"[progress] stage={stage}"]
        if self.state.get("mode"):
            parts.append(f"mode={self.state.get('mode')}")
        if self.state.get("files_total") is not None:
            parts.append(f"files={self.state.get('files_loaded', 0)}/{self.state.get('files_total')}")
        if self.state.get("rows_loaded") is not None:
            parts.append(f"rows={self.state.get('rows_loaded')}")
        if self.state.get("total_windows") is not None:
            parts.append(f"windows={self.state.get('windows_completed', 0)}/{self.state.get('total_windows')}")
        if self.state.get("current_window_index") is not None:
            parts.append(f"window_idx={self.state.get('current_window_index')}")
        if self.state.get("current_window_label"):
            parts.append(f"window={self.state.get('current_window_label')}")
        if self.state.get("models_total") is not None:
            parts.append(f"models={self.state.get('models_completed', 0)}/{self.state.get('models_total')}")
        if self.state.get("total_model_tasks") is not None:
            parts.append(f"tasks={self.state.get('completed_model_tasks', 0)}/{self.state.get('total_model_tasks')}")
        if self.state.get("current_horizon"):
            parts.append(f"horizon={self.state.get('current_horizon')}")
        if self.state.get("current_model"):
            parts.append(f"model={self.state.get('current_model')}")
        if self.state.get("events_total") is not None:
            parts.append(f"events={self.state.get('events_completed', 0)}/{self.state.get('events_total')}")
        elapsed = self.state.get("elapsed_seconds")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed:.1f}s")
        eta = self.state.get("eta_seconds")
        if eta is not None:
            parts.append(f"eta={eta:.1f}s")
        return " ".join(parts)


def main() -> int:
    args = parse_args()
    max_windows = 1 if args.single_split else args.max_windows
    mode = "single_split" if args.single_split else ("limited_windows" if max_windows is not None else "full")
    output_dir = Path(args.output_dir)
    reporter = ProgressReporter(output_dir=output_dir, heartbeat_seconds=float(args.heartbeat_seconds))
    matched_files = resolve_binance_input_files(input_dir=args.input_dir, glob=args.glob)
    selected_files = resolve_binance_input_files(
        input_dir=args.input_dir,
        glob=args.glob,
        max_files=args.max_files,
        tail_files=args.tail_files,
    )
    minute_df = load_binance_1m_dataframe(
        files=selected_files,
        start_date=args.start_date,
        end_date=args.end_date,
        progress_callback=reporter.update,
    )
    retained_start = None if minute_df.empty else minute_df.index.min().isoformat()
    retained_end = None if minute_df.empty else minute_df.index.max().isoformat()
    print(
        (
            "[ingest] "
            f"matched_files={len(matched_files)} "
            f"loaded_files={len(selected_files)} "
            f"rows={len(minute_df)} "
            f"retained_start={retained_start} "
            f"retained_end={retained_end}"
        ),
        flush=True,
    )
    reporter.update(
        {
            "stage": "ingest_complete",
            "files_loaded": len(selected_files),
            "files_total": len(selected_files),
            "rows_loaded": len(minute_df),
            "selected_files": [str(path) for path in selected_files],
            "matched_files_total": len(matched_files),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "retained_start": retained_start,
            "retained_end": retained_end,
        }
    )
    horizons = _parse_csv_list(args.horizons)
    decision_steps = {
        "1h": int(args.decision_step_1h),
        "4h": int(args.decision_step_4h),
        "1d": int(args.decision_step_1d),
    }
    window_plan = []
    for horizon in horizons:
        events = build_fixed_horizon_event_dataset(minute_df, horizon=horizon, decision_step_minutes=decision_steps.get(horizon))
        candidate_splits = build_horizon_splits(
            events,
            train_events=int(args.train_events),
            calibration_events=int(args.calibration_events),
            validation_events=int(args.validation_events),
            step_events=args.step_events,
        )
        _, split_meta = select_horizon_splits(candidate_splits, max_windows=max_windows)
        window_plan.append({"horizon": horizon, **split_meta})
    print(
        "[windows] "
        + " ".join(
            f"{item['horizon']}:candidate={item['total_candidate_windows']},eval={item['evaluated_windows']},mode={item['mode']}"
            for item in window_plan
        ),
        flush=True,
    )
    reporter.update({"stage": "window_plan", "mode": mode, "window_plan": window_plan})
    report = run_horizon_comparison(
        minute_df=minute_df,
        horizons=horizons,
        output_dir=output_dir,
        model_names=_parse_csv_list(args.models),
        decision_step_overrides=decision_steps,
        train_events=int(args.train_events),
        calibration_events=int(args.calibration_events),
        validation_events=int(args.validation_events),
        step_events=args.step_events,
        edge_thresholds=[float(item) for item in _parse_csv_list(args.edge_thresholds)],
        progress_callback=reporter.update,
        heartbeat_seconds=float(args.heartbeat_seconds),
        max_windows=max_windows,
    )
    reporter.update({"stage": "complete"})
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
