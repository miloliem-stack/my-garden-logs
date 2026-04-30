# Decision Contract

`src/research/decision_contract.py` is the offline, replay-first decision-layer contract for the BTC-1H revamp.

It is research-only for now. It does not replace live code yet and must not be imported by the live trading path until a later explicit shadow-wiring phase.

## Purpose

The contract defines one explicit decision input:

- probability snapshot
- quote snapshot
- tau/time-policy snapshot
- expected-growth snapshot
- safety veto snapshot
- HMM policy-state snapshot

It returns one explicit decision output:

- `buy_yes`
- `buy_no`
- `abstain`

with deterministic blocking reasons and diagnostics.

## Architecture Boundary

The probability engine estimates `P(YES)`. The HMM estimates market condition and uncertainty. The decision contract decides whether the probability edge is tradable.

The HMM does not modify `p_yes`. HMM state is used only for abstention or future state-conditioned policy analysis.

B/F/H fields stay outside the HMM and enter the decision layer instead:

- B: Polymarket quote and executable market features
- F: probability engine outputs and model-vs-market edge
- H: tau/time-policy features

## Replay Vetoes

Expected growth is a real replay veto in this contract:

- `conservative_expected_log_growth <= 0` blocks.
- `passes=False` blocks.

Tail, polarization, reversal, and quote-quality vetoes block by default when supplied.

HMM abstention blocks when:

- regime policy state is `transition_uncertain`
- posterior confidence is below the configured floor
- next-same-state confidence is below the configured floor
- MAP state persistence is insufficient

## Not Included

This contract never:

- places orders
- queries wallet state
- queries storage
- reads open orders or fills
- modifies inventory
- performs settlement or redemption
- modifies probability outputs

Strategy-level merge, pair-lock, and pair-recycling are not part of the decision contract. If early inventory exits are ever reintroduced, they must be explicit sell-before-resolution policies with separate replay tests and operator controls.

## Relationship To Legacy Code

The legacy live decision stack remains scattered across `run_bot.py`, `strategy_manager.py`, `decision_overlay.py`, `polarization_credibility.py`, `growth_optimizer.py`, `reversal_evidence.py`, `position_reevaluation.py`, `time_policy.py`, and `regime_detector.py`.

The old live-style replay harness is archived under `archive/legacy_replay/`. New replay-first work should target this decision contract and the HMM research replay stack, not `src/policy_replay.py`.

This contract supersedes that architecture conceptually, but does not change live behavior yet.

The first dataframe-oriented caller for this contract is `src/research/decision_replay_adapter.py`; see `docs/decision_replay_adapter.md`.
