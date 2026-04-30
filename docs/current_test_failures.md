# Current Pytest Status

Command run:

```bash
.venv/bin/python -m pytest
```

Result on this workspace after Phase 4B HMM-to-decision replay pipeline:

- 484 passed
- 1 skipped

Previous result before Phase 4B:

- 470 passed
- 1 skipped

The full suite now collects 485 tests. Phase 4B added the offline HMM-to-decision replay handoff pipeline around the existing decision replay adapter, plus its CLI and replay-only tests, without changing live trading behavior. No HMM live wiring, probability output change, normal buy-execution change, wallet behavior change, order-status normalization change, DB table drop, normal redemption change, loser-finalization change, or storage audit-history removal was performed.

Targeted validation also ran:

```bash
.venv/bin/python -m pytest tests/test_decision_contract.py
.venv/bin/python -m pytest tests/test_decision_replay_adapter.py
.venv/bin/python -m pytest tests/test_hmm_decision_replay_pipeline.py
.venv/bin/python -m pytest tests/test_hmm_research_scaffold.py
```

Targeted results:

- `tests/test_decision_contract.py`: 13 passed
- `tests/test_decision_replay_adapter.py`: 18 passed
- `tests/test_hmm_decision_replay_pipeline.py`: 14 passed
- `tests/test_hmm_research_scaffold.py`: 10 passed, 1 skipped

## Current Result

There are no current pytest failures.

Phase 4A added:

- `src/research/decision_replay_adapter.py`
- `scripts/run_decision_replay_adapter.py`
- `docs/decision_replay_adapter.md`

The adapter is offline-only. It converts replay rows into `DecisionInput` objects, evaluates them through `evaluate_replay_decision()`, appends explicit decision columns, and emits replay summaries without importing storage, execution, wallet, redeemer, Polymarket client, or archived modules.

Phase 4B added:

- `src/research/hmm_decision_replay_pipeline.py`
- `scripts/run_hmm_decision_replay_pipeline.py`
- `docs/hmm_decision_replay_pipeline.md`

The pipeline is offline-only. It aligns HMM walk-forward output aliases into the canonical decision replay schema, validates missing fields strictly by default, and feeds the aligned dataframe into the decision replay adapter without importing storage, execution, wallet, redeemer, Polymarket client, live runtime modules, or archived modules.

Remaining non-failing caveat:

- `tests/test_hmm_research_scaffold.py` emits a `joblib/loky` warning about physical core detection in this environment. It does not fail the suite.
