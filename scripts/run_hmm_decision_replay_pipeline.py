from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.decision_replay_adapter import DecisionReplayConfig
from src.research.hmm_dataset import write_table
from src.research.hmm_decision_replay_pipeline import (
    HMMDecisionReplayConfig,
    build_schema_report,
    load_frame,
    run_hmm_decision_replay,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline HMM-to-decision replay handoff pipeline on a CSV or Parquet frame."
    )
    parser.add_argument("--input", required=True, help="Input HMM walk-forward or replay dataset.")
    parser.add_argument("--output-dir", required=True, help="Output directory for replay artifacts.")
    parser.add_argument("--results-format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--yes-edge-threshold", type=float, default=0.03)
    parser.add_argument("--no-edge-threshold", type=float, default=0.03)
    parser.add_argument("--min-posterior-confidence", type=float, default=0.70)
    parser.add_argument("--min-next-same-state-confidence", type=float, default=0.65)
    parser.add_argument("--min-persistence", type=int, default=2)
    parser.add_argument("--allow-missing-hmm", action="store_true")
    parser.add_argument("--allow-missing-expected-growth", action="store_true")
    parser.add_argument("--strict-missing-safety-fields", action="store_true")
    parser.add_argument("--require-outcome-metrics", action="store_true")
    parser.add_argument("--disable-tail-veto-block", action="store_true")
    parser.add_argument("--disable-reversal-veto-block", action="store_true")
    parser.add_argument("--disable-quote-quality-veto-block", action="store_true")
    parser.add_argument("--disable-expected-growth-pass-requirement", action="store_true")
    parser.add_argument("--disable-output-diagnostics", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> HMMDecisionReplayConfig:
    decision_replay = DecisionReplayConfig(
        yes_edge_threshold=args.yes_edge_threshold,
        no_edge_threshold=args.no_edge_threshold,
        min_posterior_confidence=args.min_posterior_confidence,
        min_next_same_state_confidence=args.min_next_same_state_confidence,
        min_persistence=args.min_persistence,
        require_expected_growth_pass=not args.disable_expected_growth_pass_requirement,
        default_block_tail_veto=not args.disable_tail_veto_block,
        default_block_reversal_veto=not args.disable_reversal_veto_block,
        default_block_quote_quality_veto=not args.disable_quote_quality_veto_block,
        allow_missing_hmm_state=args.allow_missing_hmm,
        allow_missing_expected_growth=args.allow_missing_expected_growth,
        allow_missing_safety_fields=not args.strict_missing_safety_fields,
        output_diagnostics=not args.disable_output_diagnostics,
    )
    return HMMDecisionReplayConfig(
        decision_replay=decision_replay,
        strict_schema=True,
        allow_missing_expected_growth=args.allow_missing_expected_growth,
        allow_missing_hmm=args.allow_missing_hmm,
        allow_missing_safety_fields=not args.strict_missing_safety_fields,
        output_diagnostics=not args.disable_output_diagnostics,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_from_args(args)
    df = load_frame(args.input)
    schema_report = build_schema_report(df, config=config, require_outcome_metrics=args.require_outcome_metrics)
    schema_report_path = output_dir / "schema_report.json"
    schema_report_path.write_text(json.dumps(schema_report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    try:
        results, summary = run_hmm_decision_replay(
            df,
            config=config,
            require_outcome_metrics=args.require_outcome_metrics,
        )
    except ValueError as exc:
        raise SystemExit(f"schema validation failed: {exc}") from exc
    results_path = output_dir / f"hmm_decision_replay_results.{args.results_format}"
    summary_path = output_dir / "hmm_decision_replay_summary.json"
    readme_path = output_dir / "README.txt"
    write_table(results, results_path, fmt=args.results_format)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    readme_path.write_text(
        (
            "Artifacts in this directory are offline-only.\n"
            "- hmm_decision_replay_results.*: row-level replay decisions from the decision contract.\n"
            "- hmm_decision_replay_summary.json: aggregate replay summary and schema report.\n"
            "- schema_report.json: input schema mapping and missing-field report.\n"
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
