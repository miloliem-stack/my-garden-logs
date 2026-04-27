import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.hmm_dataset import build_hmm_replay_dataset, write_table


def main():
    parser = argparse.ArgumentParser(description="Build offline BTC-1H HMM replay dataset.")
    parser.add_argument("--klines", required=True, type=Path)
    parser.add_argument("--decision-log", type=Path)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hmm_research/replay_dataset.csv"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None)
    args = parser.parse_args()

    df = build_hmm_replay_dataset(
        args.klines,
        decision_log_path=args.decision_log,
        quotes_path=args.quotes,
        start=args.start,
        end=args.end,
    )
    fmt = args.format or args.output.suffix.lstrip(".") or "csv"
    write_table(df, args.output, fmt=fmt)
    print(f"wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

