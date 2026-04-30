"""Archived legacy replay compatibility shim.

The old live-style policy replay harness was moved to `archive/legacy_replay/`.
New replay work belongs in `src.research.decision_contract` and
`src.research.hmm_policy_replay`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


LEGACY_REPLAY_ARCHIVED = True
ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_replay"
ARCHIVE_MESSAGE = (
    "Legacy policy replay was archived to archive/legacy_replay and is not part of the "
    "active BTC-1H frame. Use src.research.decision_contract and "
    "src.research.hmm_policy_replay for new replay work."
)


def legacy_policy_replay_message() -> str:
    return ARCHIVE_MESSAGE


def _raise_archived(*_args: Any, **_kwargs: Any):
    raise RuntimeError(ARCHIVE_MESSAGE)


def load_scenarios(*args: Any, **kwargs: Any):
    _raise_archived(*args, **kwargs)


def filter_scenarios(*args: Any, **kwargs: Any):
    _raise_archived(*args, **kwargs)


def load_named_schedules(*args: Any, **kwargs: Any):
    _raise_archived(*args, **kwargs)


def run_replay_scenario(*args: Any, **kwargs: Any):
    _raise_archived(*args, **kwargs)


def run_scenario_library(*args: Any, **kwargs: Any):
    _raise_archived(*args, **kwargs)
