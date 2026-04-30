from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.decision_replay_adapter import (
    DecisionReplayConfig,
    evaluate_decision_replay_frame,
    summarize_decision_replay,
)
from src.research.hmm_dataset import read_table, write_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline decision replay adapter against a CSV or Parquet dataset."
    )
    parser.add_argument("--input", required=True, help="Input replay dataset path (.csv or .parquet).")
    parser.add_argument("--output-dir", required=True, help="Directory for replay results and summary artifacts.")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="parquet", help="Results table format.")
    parser.add_argument("--yes-edge-threshold", type=float, default=0.03)
    parser.add_argument("--no-edge-threshold", type=float, default=0.03)
    parser.add_argument("--min-posterior-confidence", type=float, default=0.70)
    parser.add_argument("--min-next-same-state-confidence", type=float, default=0.65)
    parser.add_argument("--min-persistence", type=int, default=2)
    parser.add_argument("--allow-missing-hmm-state", action="store_true")
    parser.add_argument("--allow-missing-expected-growth", action="store_true")
    parser.add_argument(
        "--strict-missing-safety-fields",
        action="store_true",
        help="Block missing safety-veto annotations instead of treating them as unavailable optional fields.",
    )
    parser.add_argument("--disable-expected-growth-pass-requirement", action="store_true")
    parser.add_argument("--disable-tail-veto-block", action="store_true")
    parser.add_argument("--disable-reversal-veto-block", action="store_true")
    parser.add_argument("--disable-quote-quality-veto-block", action="store_true")
    parser.add_argument("--disable-output-diagnostics", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> DecisionReplayConfig:
    return DecisionReplayConfig(
        yes_edge_threshold=args.yes_edge_threshold,
        no_edge_threshold=args.no_edge_threshold,
        min_posterior_confidence=args.min_posterior_confidence,
        min_next_same_state_confidence=args.min_next_same_state_confidence,
        min_persistence=args.min_persistence,
        require_expected_growth_pass=not args.disable_expected_growth_pass_requirement,
        default_block_tail_veto=not args.disable_tail_veto_block,
        default_block_reversal_veto=not args.disable_reversal_veto_block,
        default_block_quote_quality_veto=not args.disable_quote_quality_veto_block,
        allow_missing_hmm_state=args.allow_missing_hmm_state,
        allow_missing_expected_growth=args.allow_missing_expected_growth,
        allow_missing_safety_fields=not args.strict_missing_safety_fields,
        output_diagnostics=not args.disable_output_diagnostics,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_from_args(args)
    df = read_table(input_path)
    results = evaluate_decision_replay_frame(df, config=config)
    summary = summarize_decision_replay(results)

    results_path = output_dir / f"decision_replay_results.{args.output_format}"
    summary_path = output_dir / "decision_replay_summary.json"
    write_table(results, results_path, fmt=args.output_format)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
