# BTC-1H Polymarket Bot

This repository contains the BTC hourly-event research and trading stack for Polymarket. It combines Binance BTC market data, Polymarket market discovery and quotes, a probability engine for `P(YES)`, offline replay/research tools, and live operational organs for order execution, storage, wallet state, settlement, and reconciliation.

The current revival work is focused on an HMM regime revamp. The HMM work is offline-only for now.

## Current Architecture

The intended architecture is:

- Probability engine: estimates `P(YES)`.
- HMM regime model: estimates market condition and uncertainty.
- Decision layer: decides whether probability edge is tradable under quote, edge, tau, expected-growth, veto, and regime conditions.
- Inventory / execution / settlement: standalone operational organs that protect live venue behavior and audit history.

## Live vs Research

Live runtime starts from `src/run_bot.py` and flows through market routing, quote feeds, the probability engine, the legacy decision/guard stack, wallet state, strategy action construction, execution, storage, and settlement.

Offline research lives in the probability backtest/evaluation scripts, dataset builders, loss forensics, and the new HMM scaffold under `src/research/` plus `scripts/*hmm*`.

Do not treat research code as live trading code unless a later PR explicitly wires it in.

## Probability Engine Status

`gaussian_vol` remains the current production probability-engine candidate unless explicitly changed. Other engines are retained for tests or research comparison, but the future direction is a smaller explicit probability-engine interface with one or a few supported engines.

Relevant files:

- `src/probability_engine_factory.py`
- `src/probability_engine_gaussian_vol.py`
- `src/probability_engine_ar_egarch.py`
- `src/probability_engine_lgbm.py`
- `src/probability_engine_short_memory_research.py`

## HMM Scaffold Status

The HMM scaffold is offline-only and not live-wired.

HMM training inputs are restricted to A/D feature groups:

- A: BTC path, volatility, and path-shape features.
- D: microstructure, entropy, and magnitude-state features.

The following must not enter the HMM feature matrix: Polymarket quotes, model edge, tau/time policy, realized outcome, PnL, inventory, wallet, orders, fills, stale orders, settlement state, or execution state.

Relevant files:

- `src/research/hmm_features.py`
- `src/research/hmm_dataset.py`
- `src/research/hmm_walk_forward.py`
- `src/research/hmm_policy_replay.py`
- `src/research/hmm_visuals.py`
- `scripts/build_hmm_replay_dataset.py`
- `scripts/run_hmm_walk_forward_replay.py`
- `scripts/report_hmm_policy_replay.py`
- `scripts/plot_hmm_regime_overlay.py`

## Decision Layer Status

The current decision layer is legacy and scattered across `src/run_bot.py`, `src/strategy_manager.py`, `src/decision_overlay.py`, `src/polarization_credibility.py`, `src/growth_optimizer.py`, `src/reversal_evidence.py`, `src/time_policy.py`, the deprecated `src/position_reevaluation.py` compatibility hook, and `src/regime_detector.py`.

The planned replacement is a replay-first, regime-conditioned decision layer. Expected growth should become a real offline replay veto. Tail/polarization and reversal evidence should block by default in replay unless HMM-conditioned out-of-sample analysis proves otherwise. The HMM should not directly modify `p_yes`.

The offline scaffold for that future contract lives in `src/research/decision_contract.py`; see [docs/decision_contract.md](docs/decision_contract.md). It is not live-wired.

The first real revamped replay path around that contract now lives in:

- `src/research/decision_replay_adapter.py`
- `scripts/run_decision_replay_adapter.py`
- [docs/decision_replay_adapter.md](decision_replay_adapter.md)

It converts replay rows into explicit decision-contract inputs, applies expected-growth, safety-veto, tau, and HMM abstention logic, and emits offline replay outputs. It does not query wallet/storage/execution, simulate fills, or change live behavior.

The first end-to-end HMM-to-decision handoff now lives in:

- `src/research/hmm_decision_replay_pipeline.py`
- `scripts/run_hmm_decision_replay_pipeline.py`
- [docs/hmm_decision_replay_pipeline.md](hmm_decision_replay_pipeline.md)

This pipeline aligns real HMM walk-forward output aliases into the canonical decision replay schema, validates missing fields strictly by default, and runs the decision replay adapter without mutating HMMs, `p_yes`, or live state.

Strategy-level merge, pair-lock, pair-recycling, and live position reevaluation behavior have been removed/deprecated. The bot no longer runs hidden add/reduce/flip inventory management after entry. If early inventory management is needed later, it should be designed as an explicit sell-before-resolution or regime-switch policy with replay tests and operator controls.

Legacy `POSITION_REEVAL_*` environment flags are ignored by the compatibility hook.

The old live-style policy replay stack is archived under `archive/legacy_replay/`. The active `src/policy_replay.py` path is now only a compatibility shim that raises an archive/deprecation error.

## Protected Operational Organs

Do not casually modify or delete code that protects live venue behavior, inventory accounting, settlement, or audit history:

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

Source cleanup must not change live execution, wallet, settlement, inventory, storage, Polymarket client, order status, partial-fill, stale-order, redemption, loser-finalization, or ledger semantics.

Settlement redemption remains protected. Historical merge audit tables and receipt classification are retained for storage/reconciliation history, but they are not an active trading strategy.

## Current Test Status

Current full-suite status:

- 484 passed
- 1 skipped

The current status note is in [docs/current_test_failures.md](docs/current_test_failures.md).

For the broader cleanup/refactor audit, see [docs/current_system_cleanup_audit.md](docs/current_system_cleanup_audit.md).

## Useful Commands

```bash
.venv/bin/python -m pytest

.venv/bin/python scripts/build_hmm_replay_dataset.py --help
.venv/bin/python scripts/run_hmm_walk_forward_replay.py --help
.venv/bin/python scripts/report_hmm_policy_replay.py --help
.venv/bin/python scripts/plot_hmm_regime_overlay.py --help
.venv/bin/python scripts/run_decision_replay_adapter.py --help
.venv/bin/python scripts/run_hmm_decision_replay_pipeline.py --help
```
