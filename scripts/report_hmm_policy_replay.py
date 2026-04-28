import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.hmm_policy_replay import report_policy_replay


def main():
    parser = argparse.ArgumentParser(description="Report offline HMM policy replay comparisons.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--next-same-threshold", type=float, default=0.65)
    parser.add_argument("--persistence-threshold", type=int, default=2)
    args = parser.parse_args()
    outputs = report_policy_replay(
        args.input,
        args.output_dir,
        min_samples=args.min_samples,
        confidence_threshold=args.confidence_threshold,
        next_same_threshold=args.next_same_threshold,
        persistence_threshold=args.persistence_threshold,
    )
    print(f"wrote {len(outputs)} reports to {args.output_dir}")


if __name__ == "__main__":
    main()
