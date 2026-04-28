# Architecture Boundaries

This repo is being revived around an HMM regime revamp. These boundaries are current project rules.

## Probability Engine

The probability engine estimates `P(YES)`.

The current production candidate remains `gaussian_vol` unless explicitly changed. The future direction is a small explicit interface with one or a few supported engines, not an open-ended zoo of historical prototypes.

## HMM Input Boundary

The HMM estimates market condition and uncertainty. The offline HMM training matrix is restricted to A/D features only:

- A: BTC path, volatility, and path-shape features.
- D: microstructure, entropy, and magnitude-state features.

The canonical HMM feature source is:

- `src/research/hmm_features.py`

Old feature builders are not HMM feature sources unless data is explicitly routed through `src/research/hmm_features.py`.

## Fields That Stay Outside HMM

B/F/H fields stay outside HMM and belong to policy/replay:

- B: Polymarket quote and executable market features.
- F: probability engine outputs and model-vs-market edge.
- H: tau/time-policy features.

These must never be HMM training inputs:

- Polymarket quotes
- model edge
- tau/time-policy fields
- realized outcome
- realized PnL
- inventory
- wallet state
- orders
- fills
- stale orders
- execution state
- settlement state

## Decision Layer

The decision layer decides whether probability edge is tradable under quote, edge, tau, expected-growth, veto, and regime conditions.

The current decision layer is legacy and scattered across `run_bot.py`, `strategy_manager.py`, `decision_overlay.py`, `polarization_credibility.py`, `growth_optimizer.py`, `reversal_evidence.py`, the deprecated `position_reevaluation.py` compatibility hook, `time_policy.py`, and `regime_detector.py`.

The intended replacement is replay-first and regime-conditioned. HMM should not directly modify `p_yes` at first.

Expected growth should become a real offline replay veto. Tail/polarization and reversal evidence should block by default in replay unless HMM-conditioned out-of-sample analysis proves otherwise.

The offline scaffold for the future contract is:

- `src/research/decision_contract.py`
- `docs/decision_contract.md`

It is not live-wired and must not query execution, storage, wallet, redeemer, or venue clients.

Strategy-level merge, pair-lock, and pair-recycling are removed/deprecated and should not be reintroduced as hidden live behavior. If the bot needs pre-resolution inventory management later, design it as an explicit sell-before-resolution policy with its own replay tests and operator controls.

Live position reevaluation is also removed/deprecated. The bot should not perform hidden add/reduce/flip management after entry. Any future early inventory management must return as an explicit sell-before-resolution or regime-switch policy, replay-tested first and separated from wallet, execution, settlement, and storage accounting.

Legacy `POSITION_REEVAL_*` flags are ignored by the no-op compatibility hook.

## Operational Organs

Execution, inventory, settlement, wallet, and storage are operational organs, not alpha logic.

Protected files include:

- `src/execution.py`
- `src/storage.py`
- `src/wallet_state.py`
- `src/redeemer.py`
- `src/market_router.py`
- `src/polymarket_feed.py`
- `src/polymarket_client.py`
- `src/binance_feed.py`
- `src/runtime/market_recorder.py`
- `src/tools/*`

Do not mix new alpha logic into these modules. Do not delete audit/ledger history behavior.

Normal settlement redemption, loser finalization, receipt reconciliation, and storage audit history remain protected. Historical merge audit storage may remain for old records, but it is not a live strategy path.

## Legacy Replay

`src/policy_replay.py` is legacy. It currently exercises live-style strategy-manager behavior and should not be expanded as the future replay architecture.

The future replay direction is the offline dataframe-style HMM policy replay scaffold under:

- `src/research/hmm_policy_replay.py`
- `scripts/report_hmm_policy_replay.py`
