from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.products.btc_1h.profile import DEFAULT_SERIES_ID
from src.runtime.market_recorder import MarketRecorder


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone Polymarket market-data recorder")
    parser.add_argument("--series", default=DEFAULT_SERIES_ID)
    parser.add_argument("--output-dir", default="artifacts/market_recorder")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--snapshot-interval-seconds", type=float)
    parser.add_argument("--use-websocket", type=_parse_bool, default=True)
    parser.add_argument("--flush-every-n", type=int, default=10)
    parser.add_argument("--log-mode", default="compact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recorder = MarketRecorder(
        series_id=args.series,
        output_dir=args.output_dir,
        poll_seconds=args.poll_seconds,
        snapshot_interval_seconds=args.snapshot_interval_seconds,
        flush_every_n=args.flush_every_n,
        use_websocket=args.use_websocket,
        log_mode=args.log_mode,
    )
    report = recorder.run(duration_seconds=args.duration_seconds)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
