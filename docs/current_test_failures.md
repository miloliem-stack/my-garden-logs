# Current Pytest Failure Audit

Command run:

```bash
.venv/bin/python -m pytest
```

Result on this workspace after Phase 2C live position-reevaluation cleanup:

- 449 passed
- 11 failed
- 1 skipped

Previous result before Phase 2C:

- 471 passed
- 11 failed
- 1 skipped

Earlier milestones:

- Before Phase 2B decision-contract scaffold: 458 passed, 11 failed, 1 skipped
- Before strategy-level merge/recycler cleanup: 459 passed, 14 failed, 1 skipped
- Before Phase 2A cleanup: 451 passed, 22 failed, 1 skipped
- Before Phase 1 cleanup: 448 passed, 25 failed, 1 skipped

The full suite now collects 461 tests. Phase 2C replaced active position-reevaluation tests with disabled/no-op compatibility assertions, so the collected and passing test counts dropped while the same 11 known legacy failures remain. No HMM live wiring, probability output change, normal buy-execution change, wallet behavior change, order-status normalization change, DB table drop, normal redemption change, loser-finalization change, or storage audit-history removal was performed.

Targeted validation also ran:

```bash
.venv/bin/python -m pytest tests/test_position_reevaluation.py tests/test_growth_phase1.py tests/test_decision_state.py tests/test_strategy_manager_dedupe.py
.venv/bin/python -m pytest tests/test_decision_contract.py
.venv/bin/python -m pytest tests/test_redeemer.py
```

Targeted results:

- `tests/test_position_reevaluation.py tests/test_growth_phase1.py tests/test_decision_state.py tests/test_strategy_manager_dedupe.py`: 63 passed, 4 failed
- `tests/test_decision_contract.py`: 13 passed
- `tests/test_redeemer.py`: 24 passed

The targeted failures are the known growth/shadow metric drift and same-side guard-order semantic drift.

## Phase 2C Position-Reevaluation Cleanup

Live add/reduce/flip position reevaluation was removed/deprecated:

- `src/run_bot.py` no longer calls `evaluate_position_reevaluation()` in the live decision path.
- `src/run_bot.py` no longer allows reevaluation to override the effective live trade state.
- `src/strategy_manager.py` no longer executes `add_same_side`, `reduce_position`, or `flip_position` reevaluation actions.
- `src/position_reevaluation.py` remains as a no-op compatibility hook returning `position_reeval_disabled`.
- Legacy `POSITION_REEVAL_*` flags are ignored by the no-op compatibility hook.
- `tests/test_position_reevaluation.py` now asserts disabled/no-op behavior.

Preserved deliberately:

- Normal first-entry buy execution.
- Storage position-management/audit compatibility tables and helpers.
- Trade stats and journal fields that may read historical reevaluation diagnostics.
- Settlement redemption, loser finalization, receipt reconciliation, and storage audit history.

Future pre-resolution inventory management, if needed, must return as an explicit sell-before-resolution or regime-switch policy with replay tests and operator controls.

## Merge / Recycler Cleanup Findings

Strategy-level merge, pair-lock, pair-recycling, and open-market recycler behavior was previously removed/deprecated from the live decision path:

- `src/run_bot.py` no longer calls `build_inventory_exit_action()` before normal entry logic.
- `src/strategy_manager.py` no longer contains recycler constants, pair-lock logic, open-market recycler sell execution, shortfall retry handling, or pair-recycling helpers.
- `build_inventory_exit_action()` remains only as a no-op compatibility hook returning `strategy_inventory_exit_disabled`.
- `src/time_policy.py`, `config/policy_schedules.json`, `scripts/report_policy_ablation.py`, and `src/policy_replay.py` no longer carry merge policy/scoring traces.

Preserved deliberately:

- `src/redeemer.py` settlement/redemption/loser-finalization behavior.
- `src/storage.py` historical `merged_lots` audit schema, merge receipt classification, reconciliation support, and existing storage merge helpers.
- Inventory accounting tests for storage-level historical merge/audit behavior.
- Redeemer tests; all pass.

## Phase 2B Decision Contract Scaffold

Added:

- `src/research/decision_contract.py`
- `tests/test_decision_contract.py`
- `docs/decision_contract.md`

The contract is offline/research-only. It defines typed snapshots for probability, quote, tau policy, HMM policy state, safety vetoes, expected growth, decision input, and decision output. `evaluate_replay_decision()` computes edge from `p_yes/p_no` versus `q_yes/q_no`, applies expected-growth, safety, tau, and HMM abstention gates, and returns deterministic blocking reasons.

The contract does not import or call storage, execution, wallet, redeemer, or Polymarket client modules. It does not modify `p_yes`.

## Remaining Failures

| Test | Error summary | Classification | Reason |
| --- | --- | --- | --- |
| `tests/test_decision_overlay.py::test_decision_state_blocks_extreme_contrarian_trade` | Expected `tail_contrarian_hard_block`, got `no_edge_above_threshold`. | Decision-layer semantic drift | Current decision reason differs from the old tail-overlay expectation. |
| `tests/test_growth_phase1.py::test_conservative_growth_metric_is_not_above_naive_in_tail_setup` | Compares `None <= None` for expected growth metrics. | Growth/shadow metric drift | Existing growth shadow metrics are absent for this setup; future design should move expected growth into replay veto semantics. |
| `tests/test_growth_phase1.py::test_polarized_minority_entry_gets_worse_growth_than_balanced_entry` | Compares `None < float` for conservative growth metric. | Growth/shadow metric drift | Existing growth shadow metrics are not consistently populated across scenarios. |
| `tests/test_growth_phase1.py::test_shadow_metrics_do_not_change_live_action_selection` | `KeyError: 'side'`. | Decision-layer semantic drift | Current live action result shape differs from the old test expectation. |
| `tests/test_identifier_propagation.py::test_no_trade_when_market_closed` | Expected `None`, got structured `skipped_market_not_open` result. | Stale expectation / decision semantic drift | Current code returns a structured skip result for closed markets. |
| `tests/test_identifier_propagation.py::test_strategy_sizing_uses_runtime_effective_bankroll_instead_of_static_env` | Expected quantity `1.6`, got `2.4`. | Decision/sizing semantic drift | Remaining failure is a sizing expectation mismatch. |
| `tests/test_policy_replay.py::test_replay_harness_uses_routed_decision_path_not_old_wrapper` | Expected at least one attempted trade, got zero. | Replay harness drift | Existing `src/policy_replay.py` exercises old live-style strategy behavior; future direction is HMM replay-first analysis. |
| `tests/test_policy_replay.py::test_policy_sweep_changes_outputs_when_tau_bucket_thresholds_change` | Strict and loose policies both produce zero trades/fills. | Replay harness drift | Remaining failure is stale replay semantics. |
| `tests/test_policy_replay.py::test_execution_realism_layer_changes_replay_outcomes_when_enabled` | Expected different PnL proxy, both are `0.0`. | Replay harness drift | Existing replay harness does not produce the expected trade/PnL variation. |
| `tests/test_regime_shadow_attribution.py::test_microstructure_regime_returns_disabled_state_when_env_off` | Expected `disabled`, got `unknown`. | Stale expectation / shadow regime drift | Existing `src/regime_detector.py` behavior differs from old shadow-regime test expectation. |
| `tests/test_strategy_manager_dedupe.py::test_same_side_inventory_blocks_new_entry_when_disabled` | Expected `skipped_existing_same_side_exposure`, got `skipped_regime_entry_guard`. | Decision-layer semantic drift | Current guard ordering differs from the old decision-layer expectation. |

## Failures Fixed By Phase 2C

Phase 2C did not reduce the remaining failure count. It converted active position-reevaluation tests into passing disabled/no-op assertions and left the known legacy semantic failures untouched.

## Earlier Fixed Failures

Phase 2A fixed stale operational fixtures and low-risk interface drift:

- `tests/test_decision_overlay.py::test_decision_state_allows_normal_edge_trade`
- `tests/test_decision_overlay.py::test_tail_guard_disabled_keeps_decision_behavior_unchanged`
- `tests/test_inventory.py::test_resolve_and_redeem`
- `tests/test_inventory.py::test_merge_consumes_equal_yes_and_no_and_preserves_market_status`
- `tests/test_migration.py::test_dry_run_leaves_db_unchanged`
- `tests/test_migration.py::test_migrate_clearly_mappable_row`
- `tests/test_migration.py::test_quarantine_missing_fields`
- `tests/test_quote_gating.py::test_binance_consumer_retries_after_open_timeout_without_crashing`

Phase 1 fixed low-risk `decision_context` mock incompatibilities:

- `scripts/test_startup_checks.py::test_bootstrap_first_trade_dry_run_works`
- `tests/test_policy_replay.py::test_report_artifacts_are_written_correctly`
- `tests/test_policy_replay.py::test_ablation_report_compares_named_variants_correctly`

## Grouped Diagnosis

### Decision-Layer Semantic Drift

- `tests/test_decision_overlay.py::test_decision_state_blocks_extreme_contrarian_trade`
- `tests/test_growth_phase1.py::test_shadow_metrics_do_not_change_live_action_selection`
- `tests/test_identifier_propagation.py::test_no_trade_when_market_closed`
- `tests/test_identifier_propagation.py::test_strategy_sizing_uses_runtime_effective_bankroll_instead_of_static_env`
- `tests/test_strategy_manager_dedupe.py::test_same_side_inventory_blocks_new_entry_when_disabled`

Recommendation: postpone broad behavior changes until the replay-first regime-conditioned decision layer is designed. Do not silently delete tests that protect closed-market, sizing, or exposure-blocking behavior.

### Growth / Shadow Metric Drift

- `tests/test_growth_phase1.py::test_conservative_growth_metric_is_not_above_naive_in_tail_setup`
- `tests/test_growth_phase1.py::test_polarized_minority_entry_gets_worse_growth_than_balanced_entry`

Recommendation: preserve the intended coverage concept, but rewrite around the future offline expected-growth replay veto contract.

### Replay Harness Drift

- `tests/test_policy_replay.py::test_replay_harness_uses_routed_decision_path_not_old_wrapper`
- `tests/test_policy_replay.py::test_policy_sweep_changes_outputs_when_tau_bucket_thresholds_change`
- `tests/test_policy_replay.py::test_execution_realism_layer_changes_replay_outcomes_when_enabled`

Recommendation: keep the legacy harness until useful scenarios are ported. Do not expand `src/policy_replay.py` as the future architecture; use the HMM replay-first path for new policy work.

### Stale Expectation / Shadow Regime Drift

- `tests/test_regime_shadow_attribution.py::test_microstructure_regime_returns_disabled_state_when_env_off`

Recommendation: decide whether old `regime_detector.py` shadow semantics remain supported. If not, mark the module/tests as legacy while HMM remains offline.

## Current Classification

No remaining failures are classified as caused by the offline HMM scaffold or by the position-reevaluation cleanup. The remaining failures are evidence of pre-existing repo drift in decision semantics, growth/shadow metrics, legacy replay behavior, and stale shadow-regime expectations.
