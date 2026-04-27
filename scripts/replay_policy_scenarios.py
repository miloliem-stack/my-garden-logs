import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy_replay import load_scenarios, run_scenario_library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    result = run_scenario_library(scenarios, output_dir=args.output_dir, seed=args.seed, split=args.split)
    (args.output_dir / 'summary.json').write_text(json.dumps(result['summary'], indent=2), encoding='utf-8')
    with (args.output_dir / 'runs.jsonl').open('w', encoding='utf-8') as handle:
        for run in result['runs']:
            handle.write(json.dumps(run) + '\n')
    print(json.dumps({'score': result['score'], 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
