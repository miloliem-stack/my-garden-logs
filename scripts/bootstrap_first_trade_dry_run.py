#!/usr/bin/env python3
"""Hydrate a discovered market and run one dry-run decision path without mutating inventory."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import storage, polymarket_feed, strategy_manager


def parse_args():
    parser = argparse.ArgumentParser(description='Bootstrap first-trade dry run on a clean DB.')
    parser.add_argument('--series-id', required=True, help='Series id used for market discovery')
    parser.add_argument('--p-model', type=float, default=0.8, help='Model probability for the dry-run decision')
    parser.add_argument('--q-market', type=float, required=True, help='Market probability for the dry-run decision')
    return parser.parse_args()


def main():
    args = parse_args()
    storage.ensure_db()
    clean = storage.get_clean_start_status()
    if not clean['clean_start']:
        raise SystemExit(f'bootstrap requires a clean DB at {storage.get_db_path()}, got {clean}')

    market_meta = polymarket_feed.detect_active_hourly_market(args.series_id)
    if market_meta is None:
        raise SystemExit('market discovery failed')
    if not market_meta.get('market_id') or not market_meta.get('token_yes') or not market_meta.get('token_no'):
        raise SystemExit(f'discovery returned incomplete market metadata: {market_meta}')

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

    print('snapshot_before=')
    print(json.dumps(storage.get_position_snapshot(), indent=2))

    original_place = strategy_manager.place_marketable_buy
    try:
        strategy_manager.place_marketable_buy = lambda token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', **kwargs: {
            'dry_run': True,
            'token_id': token_id,
            'qty': qty,
            'limit_price': limit_price,
            'market_id': market_id,
            'outcome_side': outcome_side,
        }
        action = strategy_manager.decide_and_execute(
            args.p_model,
            args.q_market,
            market_meta.get('token_yes'),
            market_meta.get('token_no'),
            market_id=market_meta.get('market_id'),
            dry_run=True,
        )
    finally:
        strategy_manager.place_marketable_buy = original_place

    print('dry_run_action=')
    print(json.dumps(action, indent=2))
    print('snapshot_after=')
    print(json.dumps(storage.get_position_snapshot(), indent=2))


if __name__ == '__main__':
    main()
