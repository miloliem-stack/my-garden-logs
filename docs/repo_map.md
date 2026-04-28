# Repo Map

This is a concise orientation map for the BTC-1H repo. It is not a deletion plan.

## Live Runtime

- `src/run_bot.py` - protected, keep. Main live orchestration and decision-state assembly.
- `scripts/check_clean_start.py` - protected, keep. Startup safety check.
- `scripts/init_fresh_db.py` - protected, keep. DB initialization.
- `scripts/check_first_trade_readiness.py` - protected, keep. Readiness audit.
- `scripts/check_live_order_readiness.py` - protected, keep. Live-order readiness audit.
- `scripts/first_live_order_smoke.py` - protected, keep. First-order boundary smoke probe.
- `scripts/bootstrap_first_trade_dry_run.py` - protected, keep. Dry-run bootstrap path.

## Operational Organs

- `src/execution.py` - protected, keep. Order lifecycle, status refresh, recovery, partial fills, stale orders.
- `src/storage.py` - protected, keep. SQLite schema, orders, fills, lots, ledger, reconciliation.
- `src/wallet_state.py` - protected, keep. Wallet/effective bankroll.
- `src/redeemer.py` - protected, keep. Settlement, redemption, loser finalization.
- `src/market_router.py` - protected, keep. Active market routing.
- `src/polymarket_feed.py` - protected, keep. Market discovery and quotes.
- `src/polymarket_client.py` - protected, keep. Polymarket API/CLOB/RPC integration.
- `src/binance_feed.py` - protected, keep. Binance OHLC and settlement price helpers.
- `src/runtime/market_recorder.py` - protected, keep. Quote recorder for replay/research data.

## Probability Engines

- `src/probability_engine_factory.py` - keep, rewrite later. Current engine registry.
- `src/probability_engine_gaussian_vol.py` - keep. Current production candidate.
- `src/probability_engine_base.py` - keep, rewrite later. Protocol should become a real interface.
- `src/probability_engine_ar_egarch.py` - legacy but retained for now.
- `src/model_ar_egarch_fhs.py` - legacy but retained for now.
- `src/probability_engine_lgbm.py` - legacy/research but retained for tests.
- `src/probability_engine_short_memory_research.py` - legacy/research but retained for wrapper engines.
- `src/probability_engine_kalman_blended_sigma_v1_cfg1.py` - legacy but retained for now.
- `src/probability_engine_gaussian_pde_diffusion_kalman_v1_cfg1.py` - legacy but retained for now.

## HMM Research Scaffold

- `src/research/hmm_features.py` - keep. A/D-only causal HMM features and whitelist.
- `src/research/hmm_dataset.py` - keep. Offline replay dataset builder.
- `src/research/hmm_walk_forward.py` - keep. Causal walk-forward HMM replay.
- `src/research/hmm_policy_replay.py` - keep. Offline policy comparison.
- `src/research/hmm_visuals.py` - keep. Offline visualization.
- `scripts/build_hmm_replay_dataset.py` - keep.
- `scripts/run_hmm_walk_forward_replay.py` - keep.
- `scripts/report_hmm_policy_replay.py` - keep.
- `scripts/plot_hmm_regime_overlay.py` - keep.

## Decision Contract Scaffold

- `src/research/decision_contract.py` - keep. Offline replay-first future decision-layer contract.
- `docs/decision_contract.md` - keep. Contract documentation and boundaries.

## Legacy Decision Layer

- `src/strategy_manager.py` - protected, keep, rewrite later around clearer boundaries.
- `src/decision_overlay.py` - legacy but retained for now.
- `src/polarization_credibility.py` - legacy but retained for now.
- `src/growth_optimizer.py` - legacy but retained for now.
- `src/reversal_evidence.py` - legacy but retained for now.
- `src/time_policy.py` - legacy but retained for now.
- `src/position_reevaluation.py` - legacy/no-op compatibility hook, retained for now. Live add/reduce/flip reevaluation is disabled.
- `src/regime_detector.py` - legacy but retained for now. Not the new HMM.
- `src/strategy_sizing.py` - keep.
- `src/strategy_signal.py` - delete candidate later after manual confirmation.

Strategy-level merge, pair-lock, pair-recycling, and live position reevaluation paths are removed/deprecated. `build_inventory_exit_action()` and `evaluate_position_reevaluation()` are no-op compatibility hooks; future early inventory management should be a separately designed sell-before-resolution or regime-switch policy with replay tests.

## Legacy Replay / Policy Harness

- `src/policy_replay.py` - legacy but retained for now. Do not expand as future replay architecture.
- `scripts/replay_policy_scenarios.py` - legacy but retained for now.
- `scripts/sweep_time_policy.py` - legacy but retained for now.
- `scripts/report_policy_ablation.py` - legacy but retained for now.
- `config/policy_schedules.json` - legacy but retained for now.
- `scenarios/policy/scenario_library.json` - legacy but retained for tests/replay migration.

## Data Collection / Recorder

- `src/historical_data.py` - keep.
- `src/hourly_event_dataset.py` - keep.
- `src/feature_builder.py` - keep, but not an HMM feature source.
- `src/market_event_enrichment.py` - keep.
- `src/horizon_event_dataset.py` - legacy/research retained for now.
- `src/horizon_feature_builder.py` - legacy/research retained for now.
- `src/horizon_compare.py` - legacy/research retained for now.
- `data_collection/run_polymarket_market_recorder.py` - keep.
- `data_collection/build_hourly_event_dataset.py` - keep for now; duplicate role with script version.
- `data_collection/enrich_hourly_events_with_market.py` - keep.

## Tools / Repair Scripts

- `src/tools/backfill_filled_buy_from_tx.py` - protected, keep.
- `src/tools/repair_misclassified_clob_order_ids.py` - protected, keep.
- `scripts/probe_*` - protected, keep.
- `scripts/list_*` - protected, keep.
- `scripts/report_trade_stats.py` - keep.
- `scripts/export_trade_journal.py` - keep.
- `migrate_inventory_legacy.py` - legacy but retained for manual review.
- `export_legacy_unmapped.py` - legacy but retained for manual review.
- `scripts/migrate_order_9493_to_dust.py` - legacy/operational, retained for manual review.

## Docs / Config / Tests

- `README.md` - keep, current orientation.
- `README_INTERNAL_BOT_NOTES.md` - keep.
- `docs/hmm_research_scaffold.md` - keep.
- `docs/current_test_failures.md` - keep and update with test status.
- `docs/current_system_cleanup_audit.md` - keep.
- `docs/architecture_boundaries.md` - keep.
- `docs/repo_map.md` - keep.
- `requirements.txt` - keep.
- `tests/` - keep. Some tests are stale, but many protect live venue behavior.
