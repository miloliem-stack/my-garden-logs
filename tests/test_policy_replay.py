import runpy
import sys
from pathlib import Path

import pytest

from src import policy_replay


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "archive" / "legacy_replay"


def test_legacy_policy_replay_module_is_archived():
    assert policy_replay.LEGACY_REPLAY_ARCHIVED is True
    assert ARCHIVE_DIR.exists()
    assert (ARCHIVE_DIR / "policy_replay.py").exists()


def test_legacy_policy_replay_public_functions_raise_archive_error():
    archived_calls = [
        lambda: policy_replay.load_scenarios(ARCHIVE_DIR / "scenario_library.json"),
        lambda: policy_replay.filter_scenarios([], "train"),
        lambda: policy_replay.load_named_schedules(),
        lambda: policy_replay.run_replay_scenario({}),
        lambda: policy_replay.run_scenario_library([]),
    ]
    for call in archived_calls:
        with pytest.raises(RuntimeError, match="archived"):
            call()


def test_legacy_replay_archive_contains_original_configs_and_scenarios():
    assert (ARCHIVE_DIR / "policy_schedules.json").exists()
    assert (ARCHIVE_DIR / "scenario_library.json").exists()
    assert (ARCHIVE_DIR / "replay_policy_scenarios.py").exists()
    assert (ARCHIVE_DIR / "sweep_time_policy.py").exists()
    assert (ARCHIVE_DIR / "report_policy_ablation.py").exists()


def test_legacy_replay_scripts_exit_with_archive_message():
    scripts = [
        ROOT / "scripts" / "replay_policy_scenarios.py",
        ROOT / "scripts" / "sweep_time_policy.py",
        ROOT / "scripts" / "report_policy_ablation.py",
    ]
    old_argv = sys.argv[:]
    try:
        for script in scripts:
            sys.argv = [str(script)]
            with pytest.raises(SystemExit, match="archived"):
                runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
