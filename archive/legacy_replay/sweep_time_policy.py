import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy_replay import load_named_schedules, load_scenarios, run_scenario_library


DEFAULT_SWEEP = {
    'POLICY_MID_EDGE_THRESHOLD_YES': [0.025, 0.03],
    'POLICY_MID_EDGE_THRESHOLD_NO': [0.025, 0.03],
    'POLICY_LATE_EDGE_THRESHOLD_YES': [0.015, 0.02],
    'POLICY_LATE_EDGE_THRESHOLD_NO': [0.015, 0.02],
    'POLICY_MID_KELLY_MULTIPLIER': [0.7, 0.8],
    'POLICY_LATE_KELLY_MULTIPLIER': [0.4, 0.5],
    'POLICY_FINAL_ALLOW_NEW_ENTRIES': ['false', 'true'],
}


def iter_configs(spec):
    keys = list(spec.keys())
    values = [spec[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_report(runs):
    lines = ['Top parameter sets by offline score:']
    for idx, run in enumerate(runs[:3], start=1):
        lines.append(
            f"{idx}. train_score={run['score']:.4f} train_trades={run['train']['trades_attempted']} "
            f"validation_trades={run['validation']['trades_attempted']} stress_trades={run['stress']['trades_attempted']} overrides={run['config']}"
        )
    lines.append('')
    lines.append('Recommended initial default policy schedule:')
    if runs:
        best = runs[0]['config']
        for key in sorted(best):
            lines.append(f'- {key}={best[key]}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sweep-file', type=Path, default=None)
    parser.add_argument('--baseline-file', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    sweep_spec = DEFAULT_SWEEP if args.sweep_file is None else json.loads(args.sweep_file.read_text(encoding='utf-8'))
    baselines = load_named_schedules(args.baseline_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for idx, config in enumerate(iter_configs(sweep_spec)):
        run_dir = args.output_dir / f'run_{idx:03d}'
        train = run_scenario_library(scenarios, output_dir=run_dir / 'train', policy_overrides=config, seed=args.seed, split='train')
        validation = run_scenario_library(scenarios, output_dir=run_dir / 'validation', policy_overrides=config, seed=args.seed, split='validation')
        stress = run_scenario_library(scenarios, output_dir=run_dir / 'stress', policy_overrides=config, seed=args.seed, split='stress')
        runs.append({
            'config': config,
            'score': train['score'],
            'train': train['summary'],
            'validation': validation['summary'],
            'stress': stress['summary'],
        })

    runs.sort(key=lambda item: item['score'], reverse=True)
    baseline_results = {}
    for name, config in baselines.items():
        baseline_results[name] = {
            'train': run_scenario_library(scenarios, output_dir=args.output_dir / f'baseline_{name}' / 'train', policy_overrides=config, seed=args.seed, split='train')['summary'],
            'validation': run_scenario_library(scenarios, output_dir=args.output_dir / f'baseline_{name}' / 'validation', policy_overrides=config, seed=args.seed, split='validation')['summary'],
            'stress': run_scenario_library(scenarios, output_dir=args.output_dir / f'baseline_{name}' / 'stress', policy_overrides=config, seed=args.seed, split='stress')['summary'],
        }
    summary = {
        'runs_evaluated': len(runs),
        'best_train_score': runs[0]['score'] if runs else None,
        'baselines': baseline_results,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output_dir / 'runs.jsonl').open('w', encoding='utf-8') as handle:
        for run in runs:
            handle.write(json.dumps(run) + '\n')
    (args.output_dir / 'best_configs.json').write_text(json.dumps(runs[:3], indent=2), encoding='utf-8')
    (args.output_dir / 'report.txt').write_text(build_report(runs), encoding='utf-8')
    print(json.dumps({'summary': summary, 'best_config': runs[0] if runs else None}, indent=2))


if __name__ == '__main__':
    main()
