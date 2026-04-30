# Decision Replay Adapter

`src/research/decision_replay_adapter.py` is the first dataframe-oriented replay path in the revamped BTC-1H architecture.

It is offline-only. It must not be imported by `src/run_bot.py` or any other live runtime module.

## Purpose

The adapter converts replay rows into `DecisionInput` objects for `src/research/decision_contract.py`, evaluates them through `evaluate_replay_decision()`, and writes explicit replay outputs for analysis.

This replaces the old idea of extending `src/policy_replay.py`. The legacy live-style replay stack is archived and should stay archived.

## Input Schema

Required replay fields:

- probability:
  - `p_yes`
- quote:
  - `q_yes` or `q_yes_ask`
  - `q_no` or `q_no_ask`

Common optional fields the adapter will map when available:

- probability metadata:
  - `p_no`
  - `engine_name`
- quote details:
  - `q_yes_bid`, `q_yes_ask`, `q_no_bid`, `q_no_ask`
  - `spread_yes`, `spread_no`
  - `quote_age_sec`
- tau/policy:
  - `tau_minutes`
  - `tau_bucket`
  - row-level threshold overrides if present
- HMM policy state:
  - `hmm_map_state` or `map_state`
  - `posterior_confidence` or `hmm_map_confidence`
  - `next_same_state_confidence` or `hmm_next_same_state_confidence`
  - `persistence_count` or `hmm_map_state_persistence_count`
  - `policy_state` or `regime_policy_state`
- expected growth:
  - `expected_log_growth`
  - `conservative_expected_log_growth`
  - `expected_growth_passes`
- safety vetoes:
  - `tail_veto_flag`
  - `polarization_veto_flag`
  - `reversal_veto_flag`
  - `quote_quality_pass` or `quote_quality_veto_flag`

These are decision-layer inputs only. They are not HMM inputs.

## Strictness Defaults

The adapter is strict by default on:

- missing HMM policy-state fields
- missing expected-growth fields

Those rows block unless configuration explicitly allows missing values.

Missing safety-veto fields are permissive by default because current replay datasets may not yet carry every optional quote-quality annotation. If stricter replay is needed, enable strict missing-safety handling in the adapter config or CLI.

## Output Schema

The evaluated replay frame appends:

- `decision_action`
- `decision_allowed`
- `decision_chosen_side`
- `decision_reason`
- `decision_blocking_reasons`
- `decision_edge_yes`
- `decision_edge_no`
- `decision_hmm_policy_state`
- `decision_expected_growth_passes`
- `decision_tail_veto_blocked`
- `decision_reversal_veto_blocked`
- `decision_quote_quality_blocked`

It may also emit `decision_diagnostics` and `decision_simple_replay_pnl_proxy`.

`decision_simple_replay_pnl_proxy` is only a simple binary-outcome proxy. It is not realized PnL and does not simulate execution, queue position, partial fills, or orderbook behavior.

## Relationship To HMM Replay Data

The expected upstream flow is:

1. build a replay dataset with `src/research/hmm_dataset.py`
2. run HMM walk-forward to attach state/confidence fields
3. feed the resulting dataframe into the decision replay adapter

The adapter does not fit HMMs, modify probabilities, or simulate orders. It only evaluates trade eligibility under the explicit decision contract.

## Decision Semantics

The adapter enforces the current replay-first contract:

- expected growth is a real veto
- tail/polarization/reversal vetoes block by default when supplied
- HMM acts as an abstention/state-conditioning layer
- HMM never modifies `p_yes`

Merge, pair-recycling, and live position reevaluation are not part of this replay path.

## Limitations

This adapter does not:

- place orders
- query wallet or storage
- simulate fills or execution realism
- produce realized PnL
- alter live trading behavior

Use it for offline replay analysis only.
