import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy_replay import load_named_schedules, load_scenarios, run_scenario_library


def build_variants(baselines):
    baseline = dict(baselines['baseline_conservative'])
    return {
        'baseline_conservative': baseline,
        'final_entries_allowed': {**baseline, 'POLICY_FINAL_ALLOW_NEW_ENTRIES': 'true'},
        'late_merge_allowed': {**baseline, 'POLICY_LATE_ALLOW_MERGE': 'true', 'POLICY_FINAL_ALLOW_MERGE': 'true'},
        'symmetric_thresholds': {**baseline, 'POLICY_LATE_EDGE_THRESHOLD_YES': 0.02, 'POLICY_LATE_EDGE_THRESHOLD_NO': 0.02},
        'aggressive_kelly': {**baseline, 'POLICY_MID_KELLY_MULTIPLIER': 1.0, 'POLICY_LATE_KELLY_MULTIPLIER': 0.75},
        'tight_quotes': {**baseline, 'POLICY_MID_QUOTE_MAX_AGE_SEC': 3, 'POLICY_LATE_QUOTE_MAX_AGE_SEC': 2, 'POLICY_MID_QUOTE_MAX_SPREAD': 0.06, 'POLICY_LATE_QUOTE_MAX_SPREAD': 0.04},
    }


def format_report(results):
    lines = ['Policy Ablation Comparison']
    for name, result in results.items():
        lines.append(
            f"- {name}: train_score={result['train_score']:.4f} "
            f"validation_trades={result['validation']['trades_attempted']} "
            f"stress_blocked={sum(result['stress']['blocked_opportunities_by_reason'].values())} "
            f"pnl_proxy={result['validation']['pnl_proxy']:.4f}"
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    baselines = load_named_schedules(args.baseline_file)
    variants = build_variants(baselines)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, config in variants.items():
        results[name] = {
            'train_score': run_scenario_library(scenarios, output_dir=args.output_dir / name / 'train', policy_overrides=config, seed=args.seed, split='train')['score'],
            'validation': run_scenario_library(scenarios, output_dir=args.output_dir / name / 'validation', policy_overrides=config, seed=args.seed, split='validation')['summary'],
            'stress': run_scenario_library(scenarios, output_dir=args.output_dir / name / 'stress', policy_overrides=config, seed=args.seed, split='stress')['summary'],
        }

    (args.output_dir / 'ablation_report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (args.output_dir / 'ablation_report.txt').write_text(format_report(results), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
