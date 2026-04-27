import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import run_bot
from src.policy_replay import filter_scenarios, load_named_schedules, load_scenarios, run_replay_scenario, run_scenario_library


SCENARIO_FILE = ROOT / 'scenarios' / 'policy' / 'scenario_library.json'


def test_replay_harness_uses_routed_decision_path_not_old_wrapper(tmp_path, monkeypatch):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'early_positive_yes_edge')
    monkeypatch.setattr(run_bot, 'decide_and_execute', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('legacy wrapper should not be used')))

    result = run_replay_scenario(scenario, db_path=tmp_path / 'replay.db', output_dir=tmp_path / 'out')

    assert result['summary']['trades_attempted'] >= 1


def test_policy_sweep_changes_outputs_when_tau_bucket_thresholds_change(tmp_path):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'late_edge_only_loose_thresholds')
    strict = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'strict.db',
        policy_overrides={'POLICY_LATE_EDGE_THRESHOLD_YES': 0.03, 'POLICY_LATE_EDGE_THRESHOLD_NO': 0.03},
    )
    loose = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'loose.db',
        policy_overrides={'POLICY_LATE_EDGE_THRESHOLD_YES': 0.01, 'POLICY_LATE_EDGE_THRESHOLD_NO': 0.01},
    )

    assert strict['summary']['trades_attempted'] != loose['summary']['trades_attempted'] or strict['summary']['fills'] != loose['summary']['fills']


def test_scenario_split_filtering_works_correctly():
    scenarios = load_scenarios(SCENARIO_FILE)
    assert len(filter_scenarios(scenarios, 'train')) > 0
    assert all(item['split'] == 'train' for item in filter_scenarios(scenarios, 'train'))
    assert all(item['split'] == 'validation' for item in filter_scenarios(scenarios, 'validation'))
    assert all(item['split'] == 'stress' for item in filter_scenarios(scenarios, 'stress'))


def test_validation_results_are_kept_separate_from_train_results(tmp_path):
    scenarios = load_scenarios(SCENARIO_FILE)
    train = run_scenario_library(scenarios, output_dir=tmp_path / 'train', split='train')
    validation = run_scenario_library(scenarios, output_dir=tmp_path / 'validation', split='validation')

    assert train['split'] == 'train'
    assert validation['split'] == 'validation'
    assert train['scenario_count'] != validation['scenario_count'] or train['summary'] != validation['summary']


def test_late_entry_cutoff_reduces_trade_count_in_final_bucket(tmp_path):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'stale_order_near_expiry')
    blocked = run_replay_scenario(scenario, db_path=tmp_path / 'blocked.db')
    allowed = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'allowed.db',
        policy_overrides={'POLICY_FINAL_ALLOW_NEW_ENTRIES': 'true'},
    )

    assert blocked['summary']['trades_attempted'] <= allowed['summary']['trades_attempted']


def test_tighter_stale_order_thresholds_increase_cancel_actions_near_expiry(tmp_path):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'stale_order_near_expiry')
    loose = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'loose.db',
        policy_overrides={'POLICY_FINAL_ALLOW_NEW_ENTRIES': 'true', 'POLICY_FINAL_CANCEL_OPEN_AFTER_SEC': 300},
    )
    tight = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'tight.db',
        policy_overrides={'POLICY_FINAL_ALLOW_NEW_ENTRIES': 'true', 'POLICY_FINAL_CANCEL_OPEN_AFTER_SEC': 45},
    )

    loose_cancels = sum(trace['stale_actions_count'] for trace in loose['decision_traces'])
    tight_cancels = sum(trace['stale_actions_count'] for trace in tight['decision_traces'])
    assert tight_cancels >= loose_cancels


def test_execution_realism_layer_changes_replay_outcomes_when_enabled(tmp_path):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'early_positive_yes_edge')
    plain = run_replay_scenario(scenario, db_path=tmp_path / 'plain.db')
    realistic = run_replay_scenario(
        scenario,
        db_path=tmp_path / 'realistic.db',
        execution_realism={'enabled': True, 'friction': 1.2},
    )

    assert plain['summary']['pnl_proxy'] != realistic['summary']['pnl_proxy']


def test_score_decomposition_sums_consistently(tmp_path):
    scenario = next(item for item in load_scenarios(SCENARIO_FILE) if item['name'] == 'early_positive_no_edge')
    result = run_replay_scenario(scenario, db_path=tmp_path / 'score.db')
    components = result['summary']['score_components']
    assert abs(sum(components.values()) - result['summary']['score']) < 1e-9
    assert abs(result['summary']['score'] - result['score']) < 1e-9


def test_scenario_library_produces_stable_deterministic_outputs(tmp_path):
    scenarios = load_scenarios(SCENARIO_FILE)
    first = run_scenario_library(scenarios, output_dir=tmp_path / 'run1', seed=7)
    second = run_scenario_library(scenarios, output_dir=tmp_path / 'run2', seed=7)

    assert first['score'] == second['score']
    assert first['summary'] == second['summary']


def test_report_artifacts_are_written_correctly(tmp_path):
    output_dir = tmp_path / 'sweep'
    sweep_file = tmp_path / 'sweep.json'
    sweep_file.write_text(json.dumps({'POLICY_FINAL_ALLOW_NEW_ENTRIES': ['false', 'true']}), encoding='utf-8')
    argv = [
        'scripts/sweep_time_policy.py',
        '--scenario-file',
        str(SCENARIO_FILE),
        '--output-dir',
        str(output_dir),
        '--sweep-file',
        str(sweep_file),
        '--seed',
        '3',
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(ROOT / 'scripts' / 'sweep_time_policy.py'), run_name='__main__')
    finally:
        sys.argv = old_argv

    assert (output_dir / 'summary.json').exists()
    assert (output_dir / 'runs.jsonl').exists()
    assert (output_dir / 'best_configs.json').exists()
    assert (output_dir / 'report.txt').exists()


def test_ablation_report_compares_named_variants_correctly(tmp_path):
    output_dir = tmp_path / 'ablation'
    argv = [
        'scripts/report_policy_ablation.py',
        '--scenario-file',
        str(SCENARIO_FILE),
        '--output-dir',
        str(output_dir),
        '--seed',
        '3',
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(ROOT / 'scripts' / 'report_policy_ablation.py'), run_name='__main__')
    finally:
        sys.argv = old_argv

    report = json.loads((output_dir / 'ablation_report.json').read_text(encoding='utf-8'))
    assert 'baseline_conservative' in report
    assert 'final_entries_allowed' in report
    assert 'validation' in report['baseline_conservative']


def test_baseline_schedules_can_be_loaded_and_reported_deterministically():
    schedules = load_named_schedules()
    assert set(schedules.keys()) >= {'baseline_conservative', 'candidate_tuned', 'stress_safe'}
    assert schedules['baseline_conservative']['POLICY_FINAL_ALLOW_NEW_ENTRIES'] == 'false'
