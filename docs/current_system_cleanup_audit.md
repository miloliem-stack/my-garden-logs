# BTC-1H Current System Cleanup Audit

Date: 2026-04-28

Scope: audit/report only. No files were deleted. No live trading, execution, wallet, settlement, inventory, storage, or HMM live-wiring behavior was changed.

Validation command refreshed during this audit:

```bash
.venv/bin/python -m pytest
```

Result:

- 448 passed
- 25 failed
- 1 skipped

Phase 1 cleanup later reduced the current full-suite result to:

- 451 passed
- 22 failed
- 1 skipped

Phase 2A fixture/interface cleanup later reduced the current full-suite result to:

- 459 passed
- 14 failed
- 1 skipped

Strategy-level merge/pair-recycling cleanup later removed the hidden open-market inventory recycler path while preserving settlement, redemption, loser finalization, receipt reconciliation, and storage audit tables. See `docs/current_test_failures.md` for the latest test count after that cleanup.

- 458 passed
- 11 failed
- 1 skipped

Phase 2B later added the offline decision contract scaffold without live wiring:

- 471 passed
- 11 failed
- 1 skipped

Phase 2C later removed/deprecated live position reevaluation. `src/position_reevaluation.py` is now a no-op compatibility hook, legacy `POSITION_REEVAL_*` flags are ignored, `src/run_bot.py` no longer evaluates live add/reduce/flip decisions, and `src/strategy_manager.py` no longer executes reevaluation add/reduce/flip actions. Storage/audit history, settlement, redemption, loser finalization, and normal buy execution remain preserved.

- 449 passed
- 11 failed
- 1 skipped

See `docs/current_test_failures.md` for the refreshed failure list.

The failure list and first-pass classification are in `docs/current_test_failures.md`. The current failures are useful evidence of repo drift: live-path interface drift, storage schema/test fixture drift, stale decision-layer expectations, replay harness drift, and shadow/regime semantic confusion.

## Evidence Used

This audit used:

- `rg --files`
- `find src scripts data_collection tests docs config scenarios -maxdepth 3 -type f`
- Python AST import scan across `src/`, `scripts/`, `data_collection/`, and `tests/`
- `rg` searches for CLI entrypoints, probability engine references, decision/gate/replay references
- Current full pytest result
- `docs/current_test_failures.md`

Important caveat: dynamic imports, environment-selected code paths, and shell-invoked scripts can bypass static import evidence. Files classified as unused are deletion candidates only after a dedicated PR confirms runtime and operator workflows.

## Merge / Pair-Recycling Reference Audit

Status after cleanup:

- Strategy-level merge / pair-lock / pair-recycling behavior to remove:
  - Removed from `src/run_bot.py`: live loop no longer calls the open-market inventory exit hook before normal entry decisions.
  - Removed from `src/strategy_manager.py`: recycler constants, pair-lock logic, one-sided recycler sells, pair recycler sells, shortfall retry handling, and active recycler disposal records.
  - Removed from `src/time_policy.py`, `config/policy_schedules.json`, and `scripts/report_policy_ablation.py`: merge permission flags/variants.
  - Removed from `src/policy_replay.py`: merge scoring/status fields.
- Settlement/redeemer behavior to preserve:
  - `src/redeemer.py` remains protected for winner redemption, loser finalization, settlement checks, and inventory disposal audit records.
- Storage schema/audit table to preserve:
  - `src/storage.py` still creates `merged_lots`, supports historical merge receipt classification, keeps reconciliation support for historical merge-like receipts, and reports historical pair inventory in storage snapshots.
  - No DB tables were dropped.
- Tests/docs references:
  - Obsolete strategy-level recycler tests were replaced with assertions that strategy inventory exit is disabled.
  - Storage/reconciliation tests that prove historical merge audit behavior remain.
- Unclear/manual review:
  - Any future pre-resolution inventory management should be a new explicit sell-before-resolution policy with replay coverage and operator controls, not a hidden merge/recycler path.

## A. Current System Summary

### Entrypoints And Scripts

Live/runtime entrypoints:

- `src/run_bot.py`: main live loop and CLI. Pulls Binance klines, resolves active Polymarket market, builds probability state, builds decision state, fetches wallet state, calls strategy manager, handles stale orders, settlement cadence, heartbeat output, and shadow probability engines.
- `scripts/check_clean_start.py`: clean DB/startup readiness check.
- `scripts/init_fresh_db.py`: DB initialization script.
- `scripts/check_first_trade_readiness.py`: read-only first-trade readiness check using storage, market discovery, execution assumptions, and redeemer checks.
- `scripts/check_live_order_readiness.py`: live-order readiness audit.
- `scripts/first_live_order_smoke.py`: first-live-order boundary probe.
- `scripts/bootstrap_first_trade_dry_run.py`: dry-run bootstrap through current live decision/action path.
- `scripts/list_active_orders.py`, `scripts/list_reconciliation.py`, `scripts/print_snapshot.py`: operational inspection scripts.
- `scripts/probe_polymarket_venue.py`, `scripts/probe_order_status_shape.py`, `scripts/probe_cancel_shape.py`, `scripts/probe_tx_hash.py`: venue/API/probe scripts.
- `scripts/redeemer_dryrun_harness.py`: settlement/redeemer harness.

Offline research/reporting entrypoints:

- `scripts/build_hourly_event_dataset.py`, `data_collection/build_hourly_event_dataset.py`: build event datasets from Binance 1m data.
- `scripts/backtest_probability_engine.py`: probability backtest.
- `scripts/evaluate_probability_engines.py`: rolling/blocked probability engine evaluation.
- `scripts/report_probability_calibration.py`: calibration reports.
- `scripts/sweep_gaussian_vol.py`, `scripts/sweep_probability_challengers.py`: parameter/config sweeps.
- `scripts/compare_prediction_horizons.py`: horizon comparison research.
- `scripts/replay_policy_scenarios.py`, `scripts/sweep_time_policy.py`, `scripts/report_policy_ablation.py`: legacy scenario replay / policy sweep path.
- `scripts/inspect_recent_entry_losses.py`: forensic analysis of recent loss patterns.
- `scripts/report_trade_stats.py`, `scripts/export_trade_journal.py`: trade stats/journal reports.

HMM offline research entrypoints:

- `scripts/build_hmm_replay_dataset.py`
- `scripts/run_hmm_walk_forward_replay.py`
- `scripts/report_hmm_policy_replay.py`
- `scripts/plot_hmm_regime_overlay.py`

Legacy/migration entrypoints:

- `migrate_inventory_legacy.py`
- `export_legacy_unmapped.py`
- `scripts/migrate_order_9493_to_dust.py`

Test-as-script entrypoints:

- `scripts/test_startup_checks.py`
- `scripts/test_probe_tx_hash.py`
- `scripts/test_live_smoke_scripts.py`

These live under `scripts/` but are pytest files.

### `src` Module Tree

Operational/live modules:

- `run_bot.py`: live orchestration and decision-state construction.
- `strategy_manager.py`: entry/exit action construction, guard stack, quote checks, exposure checks, sizing, order placement calls.
- `execution.py`: order submission, status refresh/recovery, stale-order management, partial-fill handling.
- `storage.py`: SQLite schema, order/fill/lot/ledger/market state, reconciliation, snapshots.
- `polymarket_client.py`: Polymarket Gamma/CLOB/API and chain/RPC integration helpers.
- `polymarket_feed.py`: market discovery, quote snapshot reads/classification.
- `market_router.py`: active market bundle resolution.
- `wallet_state.py`: wallet/free USDC/effective bankroll logic.
- `redeemer.py`: settlement, redemption, loser finalization, and historical receipt/audit workflows.
- `binance_feed.py`: Binance historical/live data helpers.
- `live_heartbeat.py`: console/heartbeat formatting.
- `runtime/market_recorder.py`: standalone market data recorder.
- `products/btc_1h/profile.py`: BTC-1H product profile for recorder.

Decision/alpha/policy modules:

- `decision_overlay.py`: polarized-tail overlay.
- `polarization_credibility.py`: polarization zones, credibility discounts, same-side reentry veto helpers.
- `reversal_evidence.py`: reversal evidence features.
- `regime_detector.py`: existing microstructure/regime shadow detector.
- `growth_optimizer.py`: expected-growth shadow/optimizer helpers.
- `position_reevaluation.py`: deprecated no-op compatibility hook; live add/reduce/flip reevaluation is disabled.
- `time_policy.py`: tau/time-bucket policy overlays.
- `strategy_sizing.py`: fractional Kelly sizing.
- `strategy_signal.py`: currently appears unimported; likely older/simple signal code.

Probability/research modules:

- `probability_engine_base.py`: protocol/interface, currently not imported by the factory.
- `probability_engine_factory.py`: selects/builds engines.
- `probability_engine_gaussian_vol.py`: current production candidate.
- `probability_engine_ar_egarch.py`, `model_ar_egarch_fhs.py`: AR/EGARCH path.
- `probability_engine_lgbm.py`: LightGBM path.
- `probability_engine_short_memory_research.py`: large research base and variants.
- `probability_engine_kalman_blended_sigma_v1_cfg1.py`: wrapper over short-memory research variant.
- `probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py`: wrapper over short-memory research variant.
- Duplicate cleanup note: `src/probability_engine_short_memory_research copy.py` was removed in Phase 1 after a final `rg` check found no imports, scripts, tests, config, or operational docs depending on it; only this cleanup audit referenced it as a candidate.
- `probability_backtest.py`, `probability_engine_evaluation.py`, `probability_calibration.py`, `probability_challengers.py`: offline evaluation/reporting.
- `feature_builder.py`, `hourly_event_dataset.py`, `historical_data.py`: event features/datasets/data loading.
- `horizon_event_dataset.py`, `horizon_feature_builder.py`, `horizon_compare.py`: fixed-horizon research.

HMM research modules:

- `src/research/hmm_features.py`: causal A/D-only HMM feature builder and whitelist.
- `src/research/hmm_dataset.py`: replay dataset builder, optional policy/quote fields outside HMM matrix.
- `src/research/hmm_walk_forward.py`: offline causal filtered HMM walk-forward.
- `src/research/hmm_policy_replay.py`: offline policy comparison and candidate override reports.
- `src/research/hmm_visuals.py`: Plotly overlay visualization helper.

Tools:

- `src/tools/backfill_filled_buy_from_tx.py`
- `src/tools/repair_misclassified_clob_order_ids.py`

### Live Trading Path

Current live path is:

1. `src/run_bot.py` starts via CLI.
2. Binance data comes from `binance_feed.py` and websocket handling inside `run_bot.py`.
3. Market bundle comes from `market_router.py`, `polymarket_feed.py`, and `polymarket_client.py`.
4. Probability engine is built by `probability_engine_factory.py` and selected by CLI/env flags.
5. `run_bot.build_market_decision_state` assembles quotes, probability, tau/time policy, polarization, growth shadow metrics, reversal evidence, microstructure/regime shadow diagnostics, and policy flags. Live position reevaluation is no longer evaluated.
6. `strategy_manager.build_trade_action` applies shared buy/admission guards, inventory/exposure checks, regime guard, quote checks, and sizing.
7. `execution.py` places/refreshes/cancels orders and records state via `storage.py`.
8. `wallet_state.py` provides effective bankroll/free balance.
9. `redeemer.py` and storage settlement helpers handle redemption, loser finalization, receipt reconciliation, and historical audit records.
10. `live_heartbeat.py` formats console/heartbeat output.

Current issue: the decision path is not one clean layer. It is spread across `run_bot.py`, `strategy_manager.py`, `decision_overlay.py`, `polarization_credibility.py`, `growth_optimizer.py`, the deprecated `position_reevaluation.py` hook, `reversal_evidence.py`, `time_policy.py`, and `regime_detector.py`.

### Research Path

Current non-HMM research path includes:

- Historical Binance loading via `historical_data.py`.
- Hourly event construction via `hourly_event_dataset.py`.
- Feature construction via `feature_builder.py`.
- Probability backtests via `probability_backtest.py`.
- Engine evaluation via `probability_engine_evaluation.py`.
- Calibration reports via `probability_calibration.py`.
- Challenger sweeps via `probability_challengers.py`.
- Horizon research via `horizon_*`.
- Legacy policy scenario replay via `policy_replay.py` and `scenarios/policy/scenario_library.json`.

This path is valuable but currently mixes old AR/EGARCH, Gaussian, LGBM, challenger, horizon, and policy replay concepts without a single current architecture boundary.

### Probability Engine Path

The current engine interface is informal:

- `probability_engine_base.py` defines a protocol but the factory does not depend on it for enforcement.
- `probability_engine_factory.py` imports all registered engines eagerly.
- Supported names in the factory:
  - `ar_egarch`
  - `gaussian_vol`
  - `kalman_blended_sigma_v1_cfg1`
  - `gaussian_pde_diffusion_kalman_v1_cfg1`
  - `lgbm`

The Gaussian path is the cleanest current production candidate. AR/EGARCH and LGBM are tested/researched. The two long Kalman/PDE wrappers are shadow/research variants. `probability_engine_short_memory_research.py` is a large dependency for those wrappers.

### HMM Research Path

The current HMM scaffold is offline-only:

- No live module imports `src.research.*`.
- HMM feature matrix is whitelisted to A/D features.
- Quote/probability/edge/tau/outcome/PnL/inventory/wallet/order/fill/execution fields are blocked from HMM fitting.
- Walk-forward HMM uses training-only scaler/HMM fitting and filtered forward posteriors.
- Policy replay is analysis only. Candidate override cells are reports, not live permissions.

This path aligns with the intended revamp and should be preserved while live shadow wiring remains deferred.

### Decision Layer Path

The current decision layer is scattered:

- `run_bot.py`: builds market decision state, applies time policy, polarization overlay, growth shadow, reversal evidence, shadow engines, and microstructure regime diagnostics. Live position reevaluation wiring has been removed.
- `strategy_manager.py`: applies buy admission, quote guards, tail guard, microstructure/regime guard, exposure/active-order guards, sizing, inventory exit actions, and order placement.
- `time_policy.py`: tau bucket policy.
- `decision_overlay.py`: polarized-tail overlay.
- `polarization_credibility.py`: quote-polarization discounts and vetoes.
- `growth_optimizer.py`: expected-growth metrics/shadow vetoes.
- `reversal_evidence.py`: reversal evidence.
- `position_reevaluation.py`: deprecated no-op compatibility hook.
- `regime_detector.py`: old shadow regime detection.
- `policy_replay.py`: legacy replay path that exercises live strategy code.

This is the largest mismatch with the target architecture. Future work should consolidate decision/replay semantics into a replay-first regime-conditioned decision layer before live wiring.

### Inventory / Execution / Settlement Path

Operational organs:

- `storage.py`: SQLite schema and all order/fill/lot/ledger state.
- `execution.py`: venue order lifecycle, status normalization, partial fills, stale-order maintenance, recovery.
- `strategy_manager.py`: entry/exit action construction and calls into execution. This file currently mixes decision and execution-adjacent concerns.
- `wallet_state.py`: wallet/effective bankroll.
- `redeemer.py`: settlement, redemption, loser finalization, and historical audit support.
- `polymarket_client.py`: API/CLOB/RPC integration.
- `polymarket_feed.py`: market/quote reads.
- `market_router.py`: market routing.
- `src/tools/*`: repair/backfill tools for operational state.

These must not be casually deleted or semantically changed. They preserve live venue behavior and audit/ledger history.

### Storage / DB Path

`storage.py` is the central SQLite access layer. It is imported by live runtime, operational scripts, tests, replay, execution, wallet, strategy manager, redeemer, tools, and market routing.

Current test failures involving missing tables (`open_lots`, `fills`, `redeemed_lots`, `merged_lots`) indicate schema/test-fixture drift. That is cleanup evidence, but not a reason to simplify storage casually.

### Recorder / Dataset Path

Recorder:

- `src/runtime/market_recorder.py`
- `data_collection/run_polymarket_market_recorder.py`
- `src/products/btc_1h/profile.py`

Dataset/research:

- `historical_data.py`
- `hourly_event_dataset.py`
- `feature_builder.py`
- `horizon_event_dataset.py`
- `horizon_feature_builder.py`
- `horizon_compare.py`
- HMM dataset scripts/modules under `src/research` and `scripts/*hmm*`.

The recorder is operational data infrastructure and should be preserved. Feature builders need clearer boundaries: old probability/event feature builders are not HMM inputs; HMM A/D features now live separately.

### Tests

The test suite is broad and valuable. It covers:

- Execution recovery/order status/SDK replay/live integration harness
- Storage/reconciliation/inventory/dust/redeemer
- Probability engines/evaluation/calibration/backtests
- Decision state/growth/polarization/reversal and deprecated position-reevaluation compatibility behavior
- Market routing/Polymarket feed/probes
- HMM research scaffold
- Startup/live smoke scripts

Current failures are not random noise. They identify drift in live-path interfaces, storage fixture schema, decision-layer semantics, growth metrics, replay harness, and stale expectations.

### Docs / Config

- `README.md`: current repo orientation after Phase 1.
- `README_INTERNAL_BOT_NOTES.md`: useful current project notes.
- `docs/hmm_research_scaffold.md`: current HMM scaffold documentation.
- `docs/current_test_failures.md`: current failure audit.
- `config/policy_schedules.json`: legacy time-policy schedules used by replay/policy scripts.
- `scenarios/policy/scenario_library.json`: legacy policy replay scenario library.
- `requirements.txt`: project dependency list.

## B. Dependency Map

Classification labels:

- `live runtime`: imported by `run_bot.py`, live operational modules, or live-readiness scripts.
- `research scripts`: imported by offline research/reporting scripts.
- `tests`: imported only by tests.
- `legacy scripts`: imported by migration/legacy/one-off operational scripts.
- `not imported`: no static importer found.
- `dynamic risk`: likely invoked by CLI, env selection, shell command, or static scan may miss runtime usage.

### Source Modules

| File | Classification | Evidence |
| --- | --- | --- |
| `src/run_bot.py` | live runtime | CLI entrypoint; imported by readiness/smoke scripts and many tests. |
| `src/strategy_manager.py` | live runtime | Imported by `run_bot.py`, `policy_replay.py`, startup script, many tests. |
| `src/execution.py` | live runtime | Imported by `run_bot.py`, `strategy_manager.py`, `policy_replay.py`, execution tests, readiness scripts. |
| `src/storage.py` | live runtime | Imported by live runtime, execution, strategy, redeemer, wallet, tools, scripts, many tests. |
| `src/polymarket_client.py` | live runtime | Imported by execution, storage, wallet, feed/venue scripts, redeemer, recorder. |
| `src/polymarket_feed.py` | live runtime | Imported by `run_bot.py`, `market_router.py`, recorder, readiness scripts. |
| `src/market_router.py` | live runtime | Imported by `run_bot.py`, recorder, market router tests. |
| `src/binance_feed.py` | live runtime | Imported by `run_bot.py`, `market_router.py`, `storage.py`. |
| `src/wallet_state.py` | live runtime | Imported by `run_bot.py`, `strategy_manager.py`; wallet tests. |
| `src/redeemer.py` | live runtime | Imported by `run_bot.py`, readiness script, dry-run harness, tests. |
| `src/live_heartbeat.py` | live runtime | Imported by `run_bot.py`, `redeemer.py`, console/heartbeat tests. |
| `src/runtime/market_recorder.py` | live/data collection | Imported by `data_collection/run_polymarket_market_recorder.py`, tests. |
| `src/products/btc_1h/profile.py` | data collection | Imported by recorder runner. |
| `src/decision_overlay.py` | live decision | Imported by `run_bot.py`, tests. |
| `src/polarization_credibility.py` | live decision | Imported by `run_bot.py`; also calls storage. |
| `src/reversal_evidence.py` | live decision | Imported by `run_bot.py` and tests. |
| `src/growth_optimizer.py` | live decision | Imported by `run_bot.py`; reevaluation-specific helpers are legacy after Phase 2C. |
| `src/position_reevaluation.py` | legacy compatibility hook | Imported by tests; live add/reduce/flip reevaluation is disabled. |
| `src/time_policy.py` | live decision | Imported by `run_bot.py`, tests. |
| `src/regime_detector.py` | live/shadow decision | Imported by `run_bot.py`, `strategy_manager.py`, tests. |
| `src/strategy_sizing.py` | live decision/sizing | Imported by `run_bot.py`, `strategy_manager.py`, `growth_optimizer.py`. |
| `src/strategy_signal.py` | not imported | No static importer found. |
| `src/probability_engine_factory.py` | probability/research/live | Imported by `run_bot.py`, backtest/evaluation modules, tests. |
| `src/probability_engine_gaussian_vol.py` | probability | Imported by factory and short-memory research base. |
| `src/probability_engine_ar_egarch.py` | probability | Imported by factory; wraps `model_ar_egarch_fhs.py`. |
| `src/model_ar_egarch_fhs.py` | probability/research | Imported by AR/EGARCH engine and tests. |
| `src/probability_engine_lgbm.py` | probability/research | Imported by factory and tests. |
| `src/probability_engine_short_memory_research.py` | probability research | Imported by Kalman/PDE wrapper engines. |
| `src/probability_engine_kalman_blended_sigma_v1_cfg1.py` | probability research/shadow | Imported by factory. |
| `src/probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py` | probability research/shadow | Imported by factory. |
| `src/probability_engine_base.py` | not imported / conceptual | Protocol file; no static importer found. |
| `src/probability_backtest.py` | research scripts | Imported by probability evaluation/backtest scripts and tests. |
| `src/probability_engine_evaluation.py` | research scripts | Imported by evaluate/sweep scripts and tests. |
| `src/probability_calibration.py` | research scripts | Imported by calibration script and tests. |
| `src/probability_challengers.py` | research scripts | Imported by challenger sweep script and tests. |
| `src/historical_data.py` | research/data | Imported by dataset/backtest/evaluation/sweep/horizon scripts and tests. |
| `src/hourly_event_dataset.py` | research/data | Imported by dataset scripts, probability backtest, tests. |
| `src/feature_builder.py` | research/probability features | Imported by probability evaluation, LGBM, horizon compare, tests. |
| `src/horizon_event_dataset.py` | research | Imported by horizon compare and tests. |
| `src/horizon_feature_builder.py` | research | Imported by horizon compare and tests. |
| `src/horizon_compare.py` | research | Imported by horizon comparison script and tests. |
| `src/policy_replay.py` | legacy research/replay | Imported by legacy replay/sweep/ablation scripts and tests. |
| `src/trade_stats.py` | reporting | Imported by trade report/export scripts and tests. |
| `src/market_event_enrichment.py` | data/research | Imported by enrichment script and tests. |
| `src/research/hmm_features.py` | HMM research | Imported by HMM dataset/walk-forward and HMM tests. |
| `src/research/hmm_dataset.py` | HMM research | Imported by HMM dataset script, HMM walk-forward/policy/visual modules, tests. |
| `src/research/hmm_walk_forward.py` | HMM research | Imported by HMM walk-forward script and tests. |
| `src/research/hmm_policy_replay.py` | HMM research/reporting | Imported by HMM policy report script and tests. |
| `src/research/hmm_visuals.py` | visualization | Imported by HMM plot script and tests. |
| `src/tools/backfill_filled_buy_from_tx.py` | operational tool/tests | Imported by tests; CLI entrypoint. |
| `src/tools/repair_misclassified_clob_order_ids.py` | operational tool/tests | Imported by execution recovery tests; CLI entrypoint. |
| `src/__init__.py` | package metadata | Not meaningful runtime logic. |
| `src/runtime/__init__.py`, `src/products/__init__.py`, `src/products/btc_1h/__init__.py`, `src/research/__init__.py`, `src/tools/__init__.py` | package metadata | Package markers; low logic. |

### Scripts And Data Collection

| File | Classification | Evidence |
| --- | --- | --- |
| `scripts/build_hmm_replay_dataset.py` | HMM research CLI | Imports `src.research.hmm_dataset`; CLI entrypoint. |
| `scripts/run_hmm_walk_forward_replay.py` | HMM research CLI | Imports `src.research.hmm_walk_forward`; CLI entrypoint. |
| `scripts/report_hmm_policy_replay.py` | HMM research CLI | Imports `src.research.hmm_policy_replay`; CLI entrypoint. |
| `scripts/plot_hmm_regime_overlay.py` | HMM visualization CLI | Imports `src.research.hmm_visuals`; CLI entrypoint. |
| `scripts/backtest_probability_engine.py` | research CLI | Imports `historical_data` and `probability_backtest`. |
| `scripts/evaluate_probability_engines.py` | research CLI | Imports `probability_engine_evaluation`. |
| `scripts/report_probability_calibration.py` | research CLI | Imports probability calibration/reporting. |
| `scripts/sweep_gaussian_vol.py` | research CLI | Imports historical data/evaluation. |
| `scripts/sweep_probability_challengers.py` | research CLI | Imports challenger evaluation. |
| `scripts/compare_prediction_horizons.py` | research CLI | Imports horizon modules. |
| `scripts/build_hourly_event_dataset.py` | data CLI | Imports historical data/hourly event dataset. |
| `data_collection/build_hourly_event_dataset.py` | data CLI duplicate | Same role as script version. |
| `data_collection/enrich_hourly_events_with_market.py` | data/research CLI | Imports market event enrichment. |
| `data_collection/run_polymarket_market_recorder.py` | data collection CLI | Imports product profile and market recorder. |
| `scripts/replay_policy_scenarios.py`, `scripts/sweep_time_policy.py`, `scripts/report_policy_ablation.py` | legacy replay CLIs | Import `src.policy_replay`; current tests fail in this path. |
| `scripts/inspect_recent_entry_losses.py` | research/forensics CLI | Uses DB/logs/market data for loss analysis. |
| `scripts/report_trade_stats.py`, `scripts/export_trade_journal.py` | reporting CLIs | Import `trade_stats`/storage. |
| `scripts/check_*`, `scripts/init_fresh_db.py`, `scripts/first_live_order_smoke.py`, `scripts/bootstrap_first_trade_dry_run.py` | live operational/readiness CLIs | Import live runtime/storage/feed/execution modules. |
| `scripts/probe_*`, `scripts/report_venue_assumptions.py` | live/venue probe CLIs | Import Polymarket client/storage/feed; high operational value. |
| `scripts/list_*`, `scripts/print_snapshot.py` | operational inspection CLIs | Import storage. |
| `scripts/migrate_order_9493_to_dust.py`, `migrate_inventory_legacy.py`, `export_legacy_unmapped.py` | legacy/migration scripts | One-off or migration workflows; retain until migration status known. |
| `scripts/test_*.py` | tests | Pytest files under `scripts/`. |

### Docs, Config, Scenarios, Generated Files

| File | Classification | Evidence |
| --- | --- | --- |
| `README.md` | docs | Current BTC-1H orientation after Phase 1. |
| `README_INTERNAL_BOT_NOTES.md` | docs | Current bot/project notes. |
| `docs/hmm_research_scaffold.md` | docs/HMM | Current HMM scaffold docs. |
| `docs/current_test_failures.md` | docs/test status | Current failure audit. |
| `docs/current_system_cleanup_audit.md` | docs/audit | This report. |
| `config/policy_schedules.json` | config/legacy replay | Loaded by `src.policy_replay`. |
| `scenarios/policy/scenario_library.json` | config/legacy replay tests | Loaded by policy replay tests/scripts. |
| `requirements.txt` | dependency config | Required for environment. |
| `__pycache__/`, `.DS_Store` files | generated junk | Delete candidates; not source. |

## C. Architecture Alignment Matrix

| File/module | Current role | Future? | Future organ | Reason | Recommended action | Removal risk |
| --- | --- | --- | --- | --- | --- | --- |
| `src/run_bot.py` | Live orchestration, decision state, websocket loop | Yes | decision/execution orchestration | Central live entrypoint but too broad | Keep; later split decision assembly from runtime loop | Very high |
| `src/strategy_manager.py` | Guards, sizing, action building, execution calls | Yes, but rewrite boundary | decision/execution boundary | Protects live order path, but mixes alpha and operations | Keep; isolate operational calls before decision rewrite | Very high |
| `src/execution.py` | Order lifecycle, status/recovery, stale orders | Yes | execution | Protects live venue behavior | Keep; avoid semantic cleanup until tests green | Very high |
| `src/storage.py` | SQLite schema/ledger/orders/fills/lots | Yes | storage | Central audit/ledger state | Keep; schema cleanup only with migration plan | Very high |
| `src/polymarket_client.py` | Polymarket CLOB/API/RPC | Yes | execution/data collection | Venue integration | Keep | Very high |
| `src/polymarket_feed.py` | Market discovery/quotes | Yes | data collection/execution | Active market and quote reads | Keep | High |
| `src/market_router.py` | Active market routing | Yes | execution/data collection | Product routing | Keep | High |
| `src/binance_feed.py` | Binance OHLC helpers | Yes | data collection/settlement | Used by live, storage settlement, market router | Keep | High |
| `src/wallet_state.py` | Effective bankroll/wallet state | Yes | inventory/execution | Protects sizing and wallet behavior | Keep | High |
| `src/redeemer.py` | Merge/redeem/finalization | Yes | settlement | Protects settlement and redemption | Keep | Very high |
| `src/runtime/market_recorder.py` | Market recorder | Yes | data collection | Needed for replay/research data | Keep | Medium-high |
| `src/live_heartbeat.py` | Console heartbeat | Yes | execution/observability | Operational observability | Keep | Medium |
| `src/decision_overlay.py` | Tail overlay | Maybe | decision | Should be absorbed into new replay-first decision layer | Keep until replacement; then merge/deprecate | Medium |
| `src/polarization_credibility.py` | Polarization credibility/veto | Maybe | decision | Important behavior but scattered | Keep; later merge into decision layer | Medium-high |
| `src/reversal_evidence.py` | Reversal evidence | Maybe | decision/research | Useful as default veto evidence; not HMM input | Keep; later define replay contract | Medium |
| `src/growth_optimizer.py` | Expected-growth shadow metrics | Maybe | decision/research | Should become replay veto, not scattered live shadow | Keep; later rewrite into policy layer | Medium |
| `src/position_reevaluation.py` | Deprecated no-op compatibility hook | Maybe | obsolete/decision compatibility | Live add/reduce/flip reevaluation is disabled; retained to avoid broad import churn and document the removed behavior | Keep temporarily; delete candidate after callers/docs are migrated | Low |
| `src/time_policy.py` | Tau policy overlays | Maybe | decision | H features stay outside HMM; policy layer only | Keep; later simplify | Medium |
| `src/regime_detector.py` | Old microstructure/regime shadow detector | Maybe | obsolete/decision | Conflicts conceptually with new HMM regime path | Keep until HMM replacement; mark legacy shadow | Medium |
| `src/strategy_sizing.py` | Kelly sizing helper | Yes | decision | Small utility | Keep | Medium |
| `src/strategy_signal.py` | Old/simple signal | No/maybe | obsolete | No static importer found | Delete candidate after manual review | Low |
| `src/probability_engine_factory.py` | Engine registry | Yes, rewrite | probability | Eagerly imports too many engines | Keep; later simplify registry/interface | High |
| `src/probability_engine_gaussian_vol.py` | Gaussian probability engine | Yes | probability | Current production candidate | Keep | High |
| `src/probability_engine_ar_egarch.py` | AR/EGARCH engine wrapper | Maybe | probability/research | Tested and documented but likely not future primary | Deprecate/archive after replacement | Medium |
| `src/model_ar_egarch_fhs.py` | AR/EGARCH model | Maybe | probability/research | Supports AR/EGARCH engine/tests | Keep until AR/EGARCH decision made | Medium |
| `src/probability_engine_lgbm.py` | LGBM engine | Maybe | probability/research | Tested; optional dependency | Keep behind interface or archive | Medium |
| `src/probability_engine_short_memory_research.py` | Research engine zoo | Maybe/no | probability research | Large shadow/prototype base | Archive/deprecate after wrappers removed | Medium |
| `src/probability_engine_kalman_blended_sigma_v1_cfg1.py` | Wrapper over research engine | Maybe/no | probability research | Factory-supported but likely obsolete shadow candidate | Deprecate unless still actively compared | Medium |
| `src/probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py` | Wrapper over research engine | Maybe/no | probability research | Same as above | Deprecate unless active | Medium |
| `src/probability_engine_base.py` | Protocol | Yes, rewrite/activate | probability | Useful target interface, but not currently enforced | Keep; wire factory to it later | Low |
| `src/probability_backtest.py` | Probability backtest | Yes | research reporting | Useful offline evaluation | Keep | Medium |
| `src/probability_engine_evaluation.py` | Engine evaluation | Yes | research reporting | Useful for comparing supported engines | Keep; simplify engine list later | Medium |
| `src/probability_calibration.py` | Calibration metrics | Yes | research reporting | Useful | Keep | Low |
| `src/probability_challengers.py` | Challenger configs | Maybe | research reporting | May overlap future simplified engine set | Keep until engine cleanup | Low-medium |
| `src/historical_data.py` | Binance data loader | Yes | data collection/research | Core data utility | Keep | Medium |
| `src/hourly_event_dataset.py` | Hourly event dataset | Yes | data collection/research | Core probability dataset | Keep | Medium |
| `src/feature_builder.py` | Probability/event features | Maybe | probability/research | Not HMM input; keep boundary clear | Keep but document non-HMM role | Medium |
| `src/horizon_event_dataset.py` | Horizon datasets | Maybe | research | Research-only | Keep/archive later | Low-medium |
| `src/horizon_feature_builder.py` | Horizon features | Maybe | research | Research-only; not HMM | Keep/archive later | Low-medium |
| `src/horizon_compare.py` | Horizon comparison | Maybe | research reporting | Research-only | Keep/archive later | Low-medium |
| `src/policy_replay.py` | Legacy scenario replay | Maybe/no | decision research | Currently fails and exercises live strategy stack | Rewrite or supersede with HMM replay | Medium |
| `src/trade_stats.py` | Trade stats | Yes | research reporting/storage | Operational reporting | Keep | Medium |
| `src/market_event_enrichment.py` | Quote enrichment for datasets | Yes | data collection/research | Useful replay feature enrichment, outside HMM | Keep | Low |
| `src/research/hmm_*` | Hardened HMM scaffold | Yes | HMM/research reporting/visualization | Aligns with intended revamp | Keep | Medium |
| `src/tools/*` | Repair/backfill tools | Yes | storage/execution ops | Operational repair history | Keep | High |
| `scripts/*hmm*` | HMM CLIs | Yes | HMM/research reporting/visualization | Aligns with revamp | Keep | Low-medium |
| `scripts/replay_policy_scenarios.py`, `sweep_time_policy.py`, `report_policy_ablation.py` | Legacy replay CLIs | Maybe/no | obsolete/decision research | Current replay path fails and conflicts with new replay-first direction | Deprecate after HMM policy replay covers needs | Medium |
| `config/policy_schedules.json` | Legacy policy schedules | Maybe | decision config | Used by legacy replay | Keep until replay rewrite | Medium |
| `scenarios/policy/scenario_library.json` | Legacy scenario fixtures | Maybe | tests/research | Tests depend on it | Keep until replay rewrite | Medium |
| `README.md` | Wrong stale README | No | docs | Wrong project | Rewrite/delete stale content | Low |
| `README_INTERNAL_BOT_NOTES.md` | Internal docs | Yes | docs | Useful project notes | Keep/update | Low |
| `docs/current_test_failures.md` | Test failure audit | Yes | docs | Captures current drift | Keep/update | Low |
| `docs/hmm_research_scaffold.md` | HMM docs | Yes | docs | Current scaffold doc | Keep/update | Low |
| `tests/*` | Test suite | Yes | tests | Broad safety net, but stale failures need triage | Keep; quarantine/update stale tests carefully | Medium-high |

## D. Deletion / Refactor Candidates

Do not delete these yet. This is a candidate list for later PRs.

### 1. Definitely Dead / Unused

- `src/probability_engine_short_memory_research copy.py`
  - Removed in Phase 1 after proof that the canonical `src/probability_engine_short_memory_research.py` remains and no imports/scripts/tests/config referenced the duplicate.
- `.DS_Store` files under `src/`, `tests/`, `scenarios/`
  - Generated macOS metadata.
- `__pycache__/` files committed or present in source tree
  - Generated bytecode artifacts.

### 2. Obsolete Prototype

- `src/strategy_signal.py`
  - No static importer found.
  - Likely old/simple strategy signal prototype.
- `src/regime_detector.py`
  - Not deletion-ready because live imports it.
  - Conceptually obsolete once HMM regime layer replaces legacy shadow detector.
- `src/probability_challengers.py`
  - Useful for old research, but likely superseded by a smaller supported-engine list.

### 3. Duplicate Compatibility Shim

- `scripts/build_hourly_event_dataset.py` and `data_collection/build_hourly_event_dataset.py`
  - Similar role. Consider keeping one canonical CLI and making the other a thin compatibility wrapper or deleting after usage check.
- `src/probability_engine_kalman_blended_sigma_v1_cfg1.py`
  - Wrapper around a research engine class.
- `src/probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py`
  - Wrapper around a research engine class.

### 4. Superseded Probability Engine

Candidates only after active-engine policy is decided:

- `src/probability_engine_ar_egarch.py`
- `src/model_ar_egarch_fhs.py`
- `src/probability_engine_lgbm.py`
- `src/probability_engine_short_memory_research.py`
- `src/probability_engine_kalman_blended_sigma_v1_cfg1.py`
- `src/probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py`

Recommended direction: keep `gaussian_vol` as current production candidate; keep one explicit research comparison path; archive/remove the rest only after tests/reports are migrated.

### 5. Superseded Decision-Layer Code

Not safe to delete now because live code imports it, but target for rewrite/merge:

- `src/decision_overlay.py`
- `src/polarization_credibility.py`
- `src/growth_optimizer.py`
- `src/reversal_evidence.py`
- `src/time_policy.py`
- `src/regime_detector.py`
- legacy decision portions of `src/run_bot.py`
- legacy gate portions of `src/strategy_manager.py`

Target: one replay-first, regime-conditioned decision layer that consumes probability, quote/edge, tau, expected growth, C/E vetoes, and HMM regime state without modifying `p_yes`.

### 6. Superseded Feature Code

- `src/feature_builder.py`
  - Keep for probability/LGBM/research, but explicitly mark as non-HMM.
- `src/horizon_feature_builder.py`
  - Research-only; keep/archive depending on horizon research value.
- `src/horizon_event_dataset.py`
  - Research-only.
- `src/horizon_compare.py`
  - Research-only.

These do not conflict if documented as probability/research features outside HMM.

### 7. Stale Docs / Config

- `README.md`
  - Rewritten in Phase 1; keep current with cleanup phases.
- `README_INTERNAL_BOT_NOTES.md`
  - Keep but reconcile with HMM scaffold and current test status.
- `config/policy_schedules.json`
  - Legacy time-policy replay config; keep until replacement replay layer exists.
- `scenarios/policy/scenario_library.json`
  - Legacy scenario replay fixtures; keep until `policy_replay.py` is replaced or tests are migrated.

### 8. Stale Tests

Do not delete silently. Triage first:

- Tests expecting old function signatures:
  - `tests/test_quote_gating.py::test_binance_consumer_retries_after_open_timeout_without_crashing`
  - tests/mocks rejecting `decision_context`
- Tests expecting old storage schema or incomplete fixture setup:
  - inventory/migration/open-lot/fill/redeem failures.
- Tests expecting old decision semantics:
  - market-closed return semantics
  - same-side inventory vs regime guard ordering
  - tail/growth shadow metric expectations
- Legacy policy replay tests:
  - current `src/policy_replay.py` path conflicts with the future HMM replay-first direction.

### 9. Unclear, Needs Manual Review

- `migrate_inventory_legacy.py`
- `export_legacy_unmapped.py`
- `scripts/migrate_order_9493_to_dust.py`
- venue probe scripts
- repair/backfill tools under `src/tools/`

These may look old, but they protect operational audit/repair history. Review with DB history before deleting.

## E. Files That Must Not Be Touched Casually

| File | Protected behavior | Risk of accidental change |
| --- | --- | --- |
| `src/polymarket_client.py` | CLOB/API/RPC integration, auth, venue response shapes | Can break live order reads/writes, status normalization, tx probing, settlement probes. |
| `src/execution.py` | Order placement, status refresh, recovery, partial fills, stale-order maintenance | Can corrupt order lifecycle, duplicate/cancel wrong orders, mis-handle fills. |
| `src/storage.py` | SQLite schema, orders/fills/lots/ledger, reconciliation, snapshots | Can destroy audit trail, break inventory accounting, lose fill/order history. |
| `src/wallet_state.py` | Wallet/free balance/effective bankroll | Can mis-size orders or trade without available funds. |
| `src/redeemer.py` | Redemption, loser finalization, settlement, receipt reconciliation | Can leave redeemable positions stuck or incorrectly finalize inventory. |
| `src/market_router.py` | Active BTC-1H market routing | Can trade wrong market or route stale token IDs. |
| `src/polymarket_feed.py` | Market discovery and quote snapshots | Can use stale/malformed quotes or wrong market metadata. |
| `src/binance_feed.py` | BTC OHLC data and settlement price helpers | Can mis-price settlement/finalization and probability inputs. |
| `src/strategy_manager.py` | Entry/exit action building, live guard stack, execution calls | High risk because alpha, inventory, and execution are currently tangled. Refactor only with tests. |
| `src/run_bot.py` | Live loop orchestration and decision state | High risk because it wires all live organs together. |
| `src/runtime/market_recorder.py` | Quote recorder for replay data | Can damage future replay/research data quality. |
| `src/tools/backfill_filled_buy_from_tx.py` | Operational backfill repair | Can alter historical fills/lots incorrectly. |
| `src/tools/repair_misclassified_clob_order_ids.py` | Operational order ID repair | Can corrupt order identity/reconciliation. |
| `scripts/probe_*`, `scripts/check_*`, `scripts/first_live_order_smoke.py` | Operator safety checks and venue probes | Can remove guardrails used before live operations. |

## F. Current Failures As Cleanup Evidence

### 1. Interface Drift Failures

Tests:

- `scripts/test_startup_checks.py::test_bootstrap_first_trade_dry_run_works`
- `tests/test_identifier_propagation.py::test_strategy_sizing_uses_runtime_effective_bankroll_instead_of_static_env`
- `tests/test_policy_replay.py::test_policy_sweep_changes_outputs_when_tau_bucket_thresholds_change`
- `tests/test_policy_replay.py::test_report_artifacts_are_written_correctly`
- `tests/test_policy_replay.py::test_ablation_report_compares_named_variants_correctly`
- `tests/test_quote_gating.py::test_binance_consumer_retries_after_open_timeout_without_crashing`

Diagnosis:

- `place_marketable_buy` call sites now pass `decision_context`; some mocks still reject it.
- `consume_binance_klines` now takes `live_engine`; a test still passes `engine`.

Recommendation:

- Update tests/mocks if current code behavior is intended.
- If compatibility is important, add narrow adapter support deliberately.
- This is safe as a test/interface cleanup PR, not part of HMM modeling.

### 2. Schema / Test-Fixture Drift Failures

Tests:

- `tests/test_decision_overlay.py::*` failures with missing `open_lots`
- `tests/test_inventory.py::test_resolve_and_redeem`
- `tests/test_inventory.py::test_merge_consumes_equal_yes_and_no_and_preserves_market_status`
- `tests/test_migration.py::test_migrate_clearly_mappable_row`
- `tests/test_migration.py::test_quarantine_missing_fields`

Diagnosis:

- Tests are exercising storage-backed paths without schema fixtures matching current `storage.ensure_db()` expectations, or migration scripts target tables not present in fixture DBs.

Recommendation:

- Fix fixtures/setup for tests that protect live storage behavior.
- Do not delete storage tests.
- Migration tests may need a separate legacy-fixture harness or quarantine if migration is obsolete.

### 3. Decision-Layer Semantic Drift Failures

Tests:

- `tests/test_identifier_propagation.py::test_no_trade_when_market_closed`
- `tests/test_strategy_manager_dedupe.py::test_same_side_inventory_blocks_new_entry_when_disabled`
- `tests/test_decision_overlay.py::test_decision_state_blocks_extreme_contrarian_trade`

Diagnosis:

- Tests expect older return semantics or older guard ordering.
- Current code returns structured skip objects instead of `None` and can let regime entry guard block before same-side exposure guard.

Recommendation:

- Do not patch blindly. Decide intended behavior in the new regime-conditioned decision layer.
- For live safety, tests around market-closed and exposure blocking should not be deleted; update after semantics are explicitly chosen.

### 4. Growth / Shadow Metric Failures

Tests:

- `tests/test_growth_phase1.py::test_conservative_growth_metric_is_not_above_naive_in_tail_setup`
- `tests/test_growth_phase1.py::test_polarized_minority_entry_gets_worse_growth_than_balanced_entry`
- `tests/test_growth_phase1.py::test_shadow_metrics_do_not_change_live_action_selection`

Diagnosis:

- Expected-growth shadow metrics can be `None`, and action shape expectations have drifted.

Recommendation:

- Postpone behavior change until decision-layer replay rewrite.
- In the new design, expected growth should be a real offline replay veto first.
- Preserve tests conceptually, but rewrite them around the new replay policy contract.

### 5. Replay Harness Failures

Tests:

- `tests/test_policy_replay.py::test_replay_harness_uses_routed_decision_path_not_old_wrapper`
- `tests/test_policy_replay.py::test_execution_realism_layer_changes_replay_outcomes_when_enabled`
- Other legacy policy replay failures involving `decision_context`.

Diagnosis:

- `src/policy_replay.py` depends on live strategy manager behavior and test mocks that no longer match.
- This conflicts with the desired HMM replay-first design where policy variants are offline analysis, not live action path mutations.

Recommendation:

- Keep until HMM policy replay supersedes it.
- Then migrate any still-useful scenario tests to `src/research/hmm_policy_replay.py` style data-frame replay.
- Archive/delete obsolete replay harness after parity is documented.

### 6. Stale Expectation Failures

Tests:

- `tests/test_regime_shadow_attribution.py::test_microstructure_regime_returns_disabled_state_when_env_off`
- `tests/test_migration.py::test_dry_run_leaves_db_unchanged`

Diagnosis:

- Expected labels/legacy migration behavior no longer match current implementation.

Recommendation:

- For `regime_detector`, decide whether old shadow regime semantics are still supported. If not, mark legacy and adjust tests.
- For migration tests, determine whether legacy migration is still operator-relevant. If yes, fix; if no, archive.

## G. Current Repo Vs Intended HMM Revamp

Major differences still to address:

1. Decision layer is scattered.
   - Current logic spans `run_bot.py`, `strategy_manager.py`, `decision_overlay.py`, `polarization_credibility.py`, `growth_optimizer.py`, `reversal_evidence.py`, the deprecated `position_reevaluation.py` hook, `time_policy.py`, and `regime_detector.py`.
   - Future should have a cleaner replay-first decision layer with explicit inputs: probability, market quote/edge, tau policy, expected growth, default C/E vetoes, and HMM regime state.

2. Old regime detector conflicts conceptually with HMM.
   - `regime_detector.py` is currently live/shadow imported.
   - It is not the new HMM. It should be marked legacy until HMM shadow wiring is designed.

3. Old feature files can confuse HMM boundaries.
   - `feature_builder.py`, `horizon_feature_builder.py`, and `horizon_compare.py` are not HMM feature sources.
   - HMM A/D features must remain under `src/research/hmm_features.py`.
   - B/F/H, quote, edge, tau, outcome, PnL, inventory, wallet, order, fill, execution fields must remain outside HMM training.

4. Probability engine zoo is too large.
   - Factory currently registers five engines and imports all eagerly.
   - `probability_engine_short_memory_research.py` contains many variants.
   - Future should have one explicit interface and one/few supported engines. Keep Gaussian until explicitly replaced.

5. Shadow-only components create confusion.
   - Shadow probability engines, microstructure shadow regime, expected-growth shadow, and regime entry guard all coexist.
   - They are useful diagnostics but muddy what is live behavior.

6. Tests/docs still reflect old architecture.
   - `README.md` is wrong.
   - Test failures encode stale assumptions around interfaces, schemas, and decision semantics.
   - Legacy replay tests still target old live strategy path instead of offline policy-dataframe replay.

7. Legacy policy replay conflicts with HMM replay-first design.
   - `src/policy_replay.py` executes live-style strategy manager code.
   - HMM policy replay should remain data-frame/offline, with expected growth and safety vetoes explicit.

## H. Proposed Cleanup Plan

### Phase 1: Docs And Inventory Only

- No behavior changes.
- Keep this `docs/current_system_cleanup_audit.md` current.
- Rewrite `README.md` to describe BTC-1H, current live/research paths, and HMM offline status.
- Add a concise repo map if useful.
- Add a short architecture note distinguishing:
  - probability engine
  - HMM regime research
  - decision layer
  - operational organs

### Phase 2: Fix Or Quarantine Stale Tests

- Separate truly broken live-path tests from obsolete expectations.
- Update mocks for `decision_context` only if current call signature is intended.
- Update `consume_binance_klines` tests or add compatibility only if desired.
- Fix storage fixture setup for tests protecting inventory/settlement.
- Do not silently delete tests that protect venue behavior.
- Quarantine or mark legacy migration tests only after deciding migration support status.

### Phase 3: Delete / Archive Definitely Dead Files

- Delete generated junk: `.DS_Store`, `__pycache__/`.
- Keep the canonical `src/probability_engine_short_memory_research.py`; the duplicate copy file was removed in Phase 1 after final `rg` proof.
- Review `src/strategy_signal.py`; delete if no operator usage exists.
- Avoid deleting operational probes or repair tools in this phase.

### Phase 4: Isolate Live Operational Organs

- Document stable contracts for:
  - `execution.py`
  - `storage.py`
  - `wallet_state.py`
  - `redeemer.py`
  - `polymarket_client.py`
  - `polymarket_feed.py`
  - `market_router.py`
- Move alpha/decision code away from operational modules where possible.
- Add tests around order/fill/ledger semantics before refactors.

### Phase 5: Simplify Probability Engine Interface

- Preserve current Gaussian production candidate.
- Make `probability_engine_base.py` the actual interface or replace it with a small concrete adapter contract.
- Stop eager-importing all experimental engines if not needed.
- Keep AR/EGARCH/LGBM only if active tests/reports require them.
- Archive/delete obsolete engines only after dependency proof and test migration.

### Phase 6: Replace Decision Layer With Regime-Conditioned Replay-First Design

- Do not live-wire HMM yet.
- Build a clean offline policy interface:
  - probability/edge input
  - quote/tau input
  - expected growth veto
  - C/E vetoes
  - HMM abstention
  - override candidate reporting only
- Port useful legacy replay scenarios to dataframe replay.
- Keep HMM from modifying `p_yes`.

### Phase 7: Update Docs And Tests Around New Architecture

- Rewrite decision-layer tests against the new replay contract.
- Keep operational tests separate from alpha tests.
- Update docs for HMM A/D boundary and B/F/H policy boundary.
- Mark legacy modules clearly.

### Phase 8: Only Then Consider Live Shadow Wiring

- Add HMM live shadow output only after:
  - offline walk-forward reports are stable
  - policy replay reports are stable
  - live operational tests are green or explicitly quarantined
  - no HMM output can alter live actions
- Live gating should wait for a later explicit decision.

## I. Recommended First Cleanup PR

Recommended first PR: documentation and test-fixture triage only.

Contents:

- Rewrite `README.md`.
- Keep/update this audit.
- Add a short repo map.
- Add a pytest status note that links to `docs/current_test_failures.md`.
- Fix test mocks for `decision_context` only if current live signature is confirmed.
- Do not delete source files.
- Do not touch execution/storage/wallet/redeemer semantics.

Reason: the repo has live operational risk and 25 known failures. Cleanup should first make the current system legible and separate stale expectations from real broken behavior.
