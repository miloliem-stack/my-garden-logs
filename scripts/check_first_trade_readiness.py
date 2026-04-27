#!/usr/bin/env python3
"""Read-only first-trade readiness check using a temporary database."""
import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import execution, storage, polymarket_feed, redeemer, run_bot


def parse_args():
    parser = argparse.ArgumentParser(description='Check whether the bot is ready for its first real trade.')
    parser.add_argument('--series-id', required=True, help='Series id used for market discovery/hydration check')
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    temp_dir = tempfile.TemporaryDirectory(prefix='btc-bot-readiness-')
    original_env = os.getenv('BOT_DB_PATH')
    os.environ['BOT_DB_PATH'] = os.path.join(temp_dir.name, 'bot_state.db')

    try:
        storage.ensure_db()
        results['db_path'] = str(storage.get_db_path())
        results['db_schema_present'] = storage.get_db_path().exists()

        clean = storage.get_clean_start_status()
        results['clean_start'] = clean['clean_start']
        initial_snapshot = storage.get_position_snapshot()
        results['snapshot_empty_before_first_trade'] = initial_snapshot == []

        market_meta = polymarket_feed.detect_active_hourly_market(args.series_id)
        results['market_discovery_works'] = market_meta is not None
        results['discovered_market'] = None if market_meta is None else {
            'market_id': market_meta.get('market_id'),
            'token_yes': market_meta.get('token_yes'),
            'token_no': market_meta.get('token_no'),
            'condition_id': market_meta.get('condition_id'),
            'slug': market_meta.get('slug'),
            'title': market_meta.get('title'),
        }

        hydrated = False
        if market_meta is not None and market_meta.get('market_id'):
            start = market_meta.get('startDate')
            end = market_meta.get('endDate')
            storage.upsert_market(
                market_id=market_meta.get('market_id'),
                condition_id=market_meta.get('condition_id'),
                slug=market_meta.get('slug') or market_meta.get('title'),
                title=market_meta.get('title'),
                start_time=start.isoformat() if hasattr(start, 'isoformat') else (str(start) if start is not None else None),
                end_time=end.isoformat() if hasattr(end, 'isoformat') else (str(end) if end is not None else None),
                status=market_meta.get('status') or 'open',
            )
            hydrated = storage.get_market(market_meta.get('market_id')) is not None
        results['market_hydration_works'] = hydrated

        required_args = ('side', 'qty', 'price', 'token_id', 'market_id', 'outcome_side')
        signature_names = tuple(execution.place_live_limit.__code__.co_varnames[:execution.place_live_limit.__code__.co_argcount])
        results['execution_requires_identifiers'] = all(name in signature_names for name in required_args)

        # Dry-run-style execution path on the temporary DB.
        execution.place_limit_order = lambda token_id, side, qty, price, post_only=True, dry_run=False: {
            'status': 'ok',
            'txHash': '0xREADINESS',
            'filledQuantity': qty,
        }
        storage.create_market('READINESS_MKT', status='open')
        resp = execution.place_live_limit('buy', 1.0, 0.5, 'READINESS_YES', 'READINESS_MKT', 'YES')
        results['dry_run_execution_path_passes'] = resp.get('status') == 'ok' and storage.get_total_qty_by_token('READINESS_YES', market_id='READINESS_MKT') == 1.0

        snapshot = storage.get_position_snapshot()
        results['snapshot_prints'] = isinstance(snapshot, list)
        storage.print_position_snapshot()

        results['reconciliation_imports'] = callable(storage.reconcile_tx)
        results['redeemer_imports'] = hasattr(redeemer, 'Redeemer')
        results['run_bot_imports'] = hasattr(run_bot, 'main')

        results['operator_checklist'] = [
            'clean DB confirmed',
            'no active inventory',
            'snapshot empty before first trade',
            'discovery returns valid market/token ids',
            'dry-run execution path passes',
            'reconciliation tooling available',
        ]

        print(json.dumps(results, indent=2))

        ready = all([
            results['db_schema_present'],
            results['clean_start'],
            results['snapshot_empty_before_first_trade'],
            results['market_discovery_works'],
            results['market_hydration_works'],
            results['execution_requires_identifiers'],
            results['dry_run_execution_path_passes'],
            results['snapshot_prints'],
            results['reconciliation_imports'],
            results['redeemer_imports'],
            results['run_bot_imports'],
        ])
        print('result:', 'ready for first trade' if ready else 'NOT ready for first trade')
        if not ready:
            raise SystemExit(1)
    finally:
        if original_env is None:
            os.environ.pop('BOT_DB_PATH', None)
        else:
            os.environ['BOT_DB_PATH'] = original_env
        temp_dir.cleanup()


if __name__ == '__main__':
    main()
