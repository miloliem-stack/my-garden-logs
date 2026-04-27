import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.hmm_walk_forward import run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Run causal offline HMM walk-forward replay.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--test-start")
    parser.add_argument("--test-end")
    parser.add_argument("--train-window-days", type=int)
    parser.add_argument("--test-window-days", type=int)
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--covariance-type", default="diag")
    parser.add_argument("--feature-set", default="a_d_v0")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--next-same-threshold", type=float, default=0.65)
    parser.add_argument("--persistence-threshold", type=int, default=2)
    args = parser.parse_args()

    if args.feature_set != "a_d_v0":
        raise ValueError("only feature-set a_d_v0 is available in this scaffold")
    df = run_walk_forward(
        args.input,
        args.output_dir,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        random_seed=args.random_seed,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        confidence_threshold=args.confidence_threshold,
        next_same_threshold=args.next_same_threshold,
        persistence_threshold=args.persistence_threshold,
    )
    print(f"wrote {len(df)} rows to {args.output_dir / 'hmm_walk_forward_output.csv'}")


if __name__ == "__main__":
    main()

