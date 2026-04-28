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

Offline research lives in the probability backtest/evaluation scripts, dataset builders, loss forensics, legacy policy replay, and the new HMM scaffold under `src/research/` plus `scripts/*hmm*`.

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

Strategy-level merge, pair-lock, pair-recycling, and live position reevaluation behavior have been removed/deprecated. The bot no longer runs hidden add/reduce/flip inventory management after entry. If early inventory management is needed later, it should be designed as an explicit sell-before-resolution or regime-switch policy with replay tests and operator controls.

Legacy `POSITION_REEVAL_*` environment flags are ignored by the compatibility hook.

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

- 449 passed
- 11 failed
- 1 skipped

The failures are documented and grouped in [docs/current_test_failures.md](docs/current_test_failures.md). They are treated as evidence of repo drift, not ignored.

For the broader cleanup/refactor audit, see [docs/current_system_cleanup_audit.md](docs/current_system_cleanup_audit.md).

## Useful Commands

```bash
.venv/bin/python -m pytest

.venv/bin/python scripts/build_hmm_replay_dataset.py --help
.venv/bin/python scripts/run_hmm_walk_forward_replay.py --help
.venv/bin/python scripts/report_hmm_policy_replay.py --help
.venv/bin/python scripts/plot_hmm_regime_overlay.py --help
```
