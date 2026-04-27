# Polymarket BTC Up/Down Bot

This repository contains a BTC hourly-event trading bot for Polymarket plus an offline probability-research pipeline. The live bot combines:

- Binance BTCUSDT 1m market data
- Polymarket market discovery and quote reads
- SQLite-backed local state for orders, fills, inventory, and reconciliation
- Pluggable probability engines:
  - `gaussian_vol`
  - `ar_egarch`
  - `lgbm`

For the 1-hour BTC hourly event, `gaussian_vol` is the current default production candidate.

Core implementation:
- `src/run_bot.py`
- `src/probability_engine_gaussian_vol.py`
- `src/model_ar_egarch_fhs.py`

## Setup

Python 3.10+ is a reasonable target for this codebase.

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS note for LightGBM:

```bash
brew install libomp
```

## Environment

The bot reads environment variables from the shell and also loads a local `.env` automatically through `python-dotenv`.

Common runtime variables:

- `BOT_DB_PATH`: SQLite state path. Default: `./bot_state.db`
- `DECISION_LOG_PATH`: JSONL decision log path. Default: `./decision_state.jsonl`
- `LIVE`: enables Polymarket live mode when `true`
- `EXECUTION_ENABLED`: enables actual execution when `true`
- `PROBABILITY_ENGINE`: `gaussian_vol`, `ar_egarch`, or `lgbm`
- `POLY_GAMMA_BASE`, `POLY_CLOB_BASE`, `POLY_API_BASE`
- `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_API_PASSPHRASE`
- `POLY_WALLET_PRIVATE_KEY`
- `POLY_SIGNATURE_TYPE`
- `POLY_FUNDER`
- `POLY_WS_URL`
- `POLYGON_RPC`

Risk, quote, and policy controls:

- `BOT_BANKROLL`
- `PER_TRADE_CAP_PCT`
- `TOTAL_EXPOSURE_CAP`
- `KELLY_K`
- `EDGE_THRESHOLD`
- `ALLOW_OPPOSITE_SIDE_ENTRY`
- `QUOTE_MAX_AGE_SEC`
- `QUOTE_MAX_SPREAD`
- `QUOTE_REQUIRE_BOTH_SIDES`
- `QUOTE_MIN_DEPTH`
- `ORDER_MAX_OPEN_AGE_SEC`
- `ORDER_MAX_PENDING_SUBMIT_AGE_SEC`
- `ORDER_CANCEL_RETRY_SEC`
- `STALE_ORDER_MAINTENANCE_SEC`
- `MERGE_TRIGGER_SUM`
- `MERGE_FEE`

Time-policy controls are implemented in `src/time_policy.py` and can be tuned with env vars such as:

- `POLICY_LATE_ALLOW_NEW_ENTRIES`
- `POLICY_LATE_ALLOW_MERGE`
- `POLICY_FINAL_ALLOW_NEW_ENTRIES`
- `POLICY_FINAL_ALLOW_MERGE`

Gaussian volatility engine controls:

- `GAUSSIAN_VOL_WINDOW`
- `GAUSSIAN_MIN_PERIODS`
- `GAUSSIAN_SIGMA_FLOOR`
- `GAUSSIAN_SIGMA_CAP`
- `GAUSSIAN_FALLBACK_SIGMA`
- `GAUSSIAN_CALIBRATION_MODE`

## Quick Start

Run the bot in offline/non-live mode:

```bash
python -m src.run_bot
```

Run the live repricing loop for a specific series:

```bash
python -m src.run_bot \
  --live \
  --series <SERIES_ID> \
  --probability-engine gaussian_vol \
  --duration 300
```

Guarded startup with an explicit production DB:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
BOT_REQUIRE_CLEAN_START=true \
LIVE=true \
EXECUTION_ENABLED=true \
python -m src.run_bot \
  --live \
  --series <SERIES_ID> \
  --probability-engine gaussian_vol
```

Useful CLI flags for `src/run_bot.py`:

- `--live`
- `--series <SERIES_ID>`
- `--probability-engine gaussian_vol|ar_egarch|lgbm`
- `--duration <seconds>`
- `--limit <backfill_bars>`
- `--tau <minutes>`
- `--sims <n>`
- `--allow-dirty-start`
- `--token-yes <TOKEN_ID>`
- `--token-no <TOKEN_ID>`
- `--market-id <MARKET_ID>`

## Production Checklist

Before enabling real order flow:

- credentials and wallet env vars are present
- production DB path is explicit and persistent
- clean-start status is confirmed
- market discovery returns valid market and token ids
- quote snapshots are tradable
- first-trade dry run passes
- live order readiness audit passes
- reconciliation and snapshot tooling are available

Recommended first sequence:

```bash
python scripts/init_fresh_db.py --db-path /path/to/prod-bot.db
BOT_DB_PATH=/path/to/prod-bot.db python scripts/check_clean_start.py
BOT_DB_PATH=/path/to/prod-bot.db python scripts/check_first_trade_readiness.py --series-id <SERIES_ID>
BOT_DB_PATH=/path/to/prod-bot.db python scripts/check_live_order_readiness.py --series-id <SERIES_ID>
BOT_DB_PATH=/path/to/prod-bot.db python scripts/bootstrap_first_trade_dry_run.py --series-id <SERIES_ID> --q-market 0.45
```

## Production Commands

Initialize or replace a fresh DB:

```bash
python scripts/init_fresh_db.py --db-path /path/to/prod-bot.db
python scripts/init_fresh_db.py --db-path /path/to/prod-bot.db --force
```

Verify clean-start state:

```bash
BOT_DB_PATH=/path/to/prod-bot.db python scripts/check_clean_start.py
```

Read-only first-trade readiness audit:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
python scripts/check_first_trade_readiness.py --series-id <SERIES_ID>
```

Read-only live-order readiness audit:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
python scripts/check_live_order_readiness.py --series-id <SERIES_ID>
```

Rebuild and export the derived trade journal:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
DECISION_LOG_PATH=/path/to/decision_state.jsonl \
python scripts/export_trade_journal.py --output artifacts/trade_journal.csv
```

Build realized trade-statistics artifacts:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
DECISION_LOG_PATH=/path/to/decision_state.jsonl \
python scripts/report_trade_stats.py --output-dir artifacts
```

Dry-run the first decision path without mutating inventory:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
python scripts/bootstrap_first_trade_dry_run.py \
  --series-id <SERIES_ID> \
  --q-market 0.45 \
  --p-model 0.55
```

Run the bot in live-routing mode without order submission:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
LIVE=false \
EXECUTION_ENABLED=false \
python -m src.run_bot \
  --live \
  --series <SERIES_ID> \
  --probability-engine gaussian_vol \
  --duration 300
```

Run the bot with real execution enabled:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
LIVE=true \
EXECUTION_ENABLED=true \
PROBABILITY_ENGINE=gaussian_vol \
python -m src.run_bot \
  --live \
  --series <SERIES_ID> \
  --duration 300
```

Print current inventory snapshot:

```bash
BOT_DB_PATH=/path/to/prod-bot.db python scripts/print_snapshot.py
```

List active orders from local state:

```bash
BOT_DB_PATH=/path/to/prod-bot.db python scripts/list_active_orders.py
```

List pending receipts and reconciliation issues:

```bash
BOT_DB_PATH=/path/to/prod-bot.db python scripts/list_reconciliation.py
```

Probe Polymarket venue connectivity and auth wiring without writing orders:

```bash
BOT_DB_PATH=/path/to/prod-bot.db python scripts/probe_polymarket_venue.py
```

First live order smoke test. Dry-run by default:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
python scripts/first_live_order_smoke.py \
  --series-id <SERIES_ID> \
  --side buy \
  --outcome-side YES \
  --qty 1 \
  --poll-seconds 10
```

Explicitly opt into a real smoke order:

```bash
BOT_DB_PATH=/path/to/prod-bot.db \
LIVE=true \
python scripts/first_live_order_smoke.py \
  --series-id <SERIES_ID> \
  --side buy \
  --outcome-side YES \
  --qty 1 \
  --live \
  --confirm-live
```

Probe transaction-hash reconciliation:

```bash
python scripts/probe_tx_hash.py --tx-hash <TX_HASH> --wallet <WALLET_ADDRESS>
```

## Testing

Run the full test suite:

```bash
pytest
```

Run focused tests:

```bash
pytest tests/test_probability_engine_gaussian_vol.py
pytest tests/test_execution_live.py
pytest tests/test_reconciliation.py
```

Smoke-test startup and live helper scripts:

```bash
python scripts/test_startup_checks.py
python scripts/test_live_smoke_scripts.py
```

## Offline Probability Research

The offline research pipeline does not touch live execution, inventory, or Polymarket order state.

Expected Binance input schema:

- headerless BTCUSDT spot 1m klines
- 12 columns in this order:
  `open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore`
- timestamp unit is auto-detected:
  - older spot files: milliseconds
  - newer spot files: microseconds

The loader keeps timestamps in UTC, sorts rows, drops duplicate timestamps, and reports minute gaps without forward-filling.

Build the hourly event dataset:

```bash
python scripts/build_hourly_event_dataset.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --output artifacts/hourly_events.parquet \
  --decision-step-minutes 5
```

Run the walk-forward backtest:

```bash
python scripts/backtest_probability_engine.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --events-path artifacts/hourly_events.parquet \
  --output artifacts/probability_backtest.parquet \
  --fit-window 2000 \
  --min-history 2000 \
  --residual-buffer-size 2000 \
  --n-sims 2000 \
  --refit-every-n-events 12 \
  --seed 42
```

Generate a calibration report:

```bash
python scripts/report_probability_calibration.py \
  --input artifacts/probability_backtest.parquet \
  --output-dir artifacts/calibration_report
```

Evaluate engines on rolling windows:

```bash
python scripts/evaluate_probability_engines.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --events-path artifacts/hourly_events.parquet \
  --output-dir artifacts/engine_eval \
  --train-hours 720 \
  --calibration-hours 168 \
  --validation-hours 168 \
  --step-hours 168
```

Sweep `gaussian_vol` parameters:

```bash
python scripts/sweep_gaussian_vol.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --events-path artifacts/hourly_events.parquet \
  --output-dir artifacts/gaussian_vol_sweep \
  --vol-windows 720,1440,2880 \
  --min-periods 180,360,720 \
  --calibration-modes none,logistic \
  --write-predictions false
```

Sweep challenger models:

```bash
python scripts/sweep_probability_challengers.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --events-path artifacts/hourly_events.parquet \
  --output-dir artifacts/challenger_sweep
```

Compare forecast horizons:

```bash
python scripts/compare_prediction_horizons.py \
  --input-dir data/binance/btcusdt_1m \
  --glob "BTCUSDT-1m-*.csv" \
  --output-dir artifacts/horizon_compare
```

## Gaussian Volatility Engine

Primary formula:

```text
p_yes = 1 - Phi(log(strike_price / spot_now) / (sigma_per_sqrt_min * sqrt(tau_minutes)))
```

Recommended baseline configuration for the 1-hour event:

- `PROBABILITY_ENGINE=gaussian_vol`
- `GAUSSIAN_VOL_WINDOW=120`
- `GAUSSIAN_MIN_PERIODS=60`
- `GAUSSIAN_CALIBRATION_MODE=none`

The engine exposes diagnostics including:

- raw and calibrated probabilities
- sigma state
- horizon sigma
- z-score
- prediction inputs

## Outputs and Artifacts

Typical files produced during operation:

- SQLite state DB at `BOT_DB_PATH`
- decision log JSONL at `DECISION_LOG_PATH`
- smoke-test manifests under `artifacts/first_live_order_smoke/`
- offline datasets, reports, and sweep outputs under `artifacts/`

## Notes

- The codebase defaults to dry-run behavior unless live execution is explicitly enabled.
- Historical tx hashes in tests are reconciliation fixtures, not seed inventory.
- Use an explicit persistent DB path for production. Do not rely on the default working-directory DB for a real deployment.
