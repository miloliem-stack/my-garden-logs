from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.market_event_enrichment import enrich_event_file_with_market_quotes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach market-implied quote fields to an hourly event dataset")
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--quotes-path", required=True)
    parser.add_argument("--quote-tolerance-seconds", type=int, default=300)
    parser.add_argument("--drop-unmatched", action="store_true")
    parser.add_argument("--no-summary-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = enrich_event_file_with_market_quotes(
        events_path=args.events_path,
        output_path=args.output_path,
        quotes_path=args.quotes_path,
        quote_tolerance_seconds=args.quote_tolerance_seconds,
        drop_unmatched=args.drop_unmatched,
        write_summary_json=not args.no_summary_json,
    )
    print(
        {
            "total_rows": report["total_rows"],
            "matched_rows": report["matched_rows"],
            "unmatched_rows": report["unmatched_rows"],
            "coverage_rate": report["coverage_rate"],
            "output_path": report["output_path"],
            "summary_path": report.get("summary_path"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
