# HMM Decision Replay Pipeline

`src/research/hmm_decision_replay_pipeline.py` is the end-to-end offline handoff from HMM walk-forward outputs into the decision replay adapter.

It is research-only. It must not be imported by the live trading path.

## Purpose

The pipeline takes a dataframe or file produced by HMM replay or walk-forward work, aligns HMM output aliases into the canonical decision-adapter schema, validates the required fields, and runs the explicit decision contract through the dataframe-oriented replay adapter.

This is the first end-to-end offline path that connects:

1. HMM state/confidence outputs
2. decision-layer replay evaluation
3. replay summary artifacts

## Expected Upstream Flow

Typical offline sequence:

1. build replay dataset with `src/research/hmm_dataset.py`
2. run HMM walk-forward with `src/research/hmm_walk_forward.py`
3. run this pipeline on the HMM output
4. inspect decision replay results and summary artifacts

The pipeline consumes HMM state/confidence outputs. It does not fit or mutate HMMs.

## Input Schema

Required decision-layer inputs:

- probability:
  - `p_yes`
- quote:
  - `q_yes` or `q_yes_ask`
  - `q_no` or `q_no_ask`

Expected HMM alias groups:

- `hmm_map_state`, `map_state`, `state`, `filtered_map_state`
- `posterior_confidence`, `hmm_map_confidence`, `map_confidence`, `filtered_map_confidence`
- `next_same_state_confidence`, `hmm_next_same_state_confidence`
- `persistence_count`, `hmm_map_state_persistence_count`, `map_state_persistence`
- `policy_state`, `regime_policy_state`

Expected optional policy inputs:

- `tau_minutes`
- `tau_bucket`
- `expected_log_growth`
- `conservative_expected_log_growth`
- `expected_growth_passes`
- `tail_veto_flag`
- `polarization_veto_flag`
- `reversal_veto_flag`
- `quote_quality_pass`

Outcome fields are optional unless outcome-aware metrics are explicitly required.

## Output Artifacts

`scripts/run_hmm_decision_replay_pipeline.py` writes:

- `hmm_decision_replay_results.csv` or `.parquet`
- `hmm_decision_replay_summary.json`
- `schema_report.json`
- `README.txt`

The results file contains the appended `decision_*` columns from the decision replay adapter.

The summary file contains:

- row counts
- allowed/abstained counts
- action and reason counts
- blocking reason counts
- allowed rates by HMM state and tau bucket
- expected-growth and safety-veto block counts
- optional outcome-aware offline metrics

The schema report records canonical mappings and missing-field diagnostics.

## Strict Missing-Field Behavior

By default the pipeline is strict on:

- missing probability fields
- missing quote fields
- missing HMM state/confidence fields
- missing expected-growth fields

Missing safety fields are permissive by default because some replay datasets may not yet carry every quote-quality annotation. That can be tightened with CLI/config flags.

Outcome fields do not block unless outcome-aware metrics are explicitly required.

## Relationship To Other Modules

- `src/research/hmm_walk_forward.py`
  - upstream HMM state/confidence producer
- `src/research/decision_replay_adapter.py`
  - dataframe-oriented decision evaluation and summary
- `src/research/decision_contract.py`
  - pure decision contract

This pipeline is the glue between the HMM output surface and the decision replay surface.

## Boundary Guarantees

This pipeline may consume:

- HMM posterior/state outputs
- probability fields
- quote fields
- tau fields
- expected-growth fields
- safety-veto fields
- outcomes for offline evaluation

It must not:

- alter HMM features
- alter HMM fitted models
- alter `p_yes`
- query wallet, storage, execution, or settlement state
- import archived replay code
- change live trading behavior

## Limitations

This pipeline does not:

- simulate execution realism
- model orderbook fills
- produce realized PnL
- wire anything into live trading

It remains an offline research pipeline only.
