#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import polymarket_client, polymarket_feed, run_bot, storage


def parse_args():
    parser = argparse.ArgumentParser(description='Safely probe the first live order boundary and capture payloads.')
    parser.add_argument('--series-id', required=True)
    parser.add_argument('--side', choices=['buy', 'sell'], default='buy')
    parser.add_argument('--outcome-side', choices=['YES', 'NO'], default='YES')
    parser.add_argument('--qty', type=float, default=1.0)
    parser.add_argument('--price', type=float, default=None)
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--artifact-dir', default='artifacts/first_live_order_smoke')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--confirm-live', action='store_true')
    return parser.parse_args()


def _now_ts():
    return datetime.now(timezone.utc).isoformat()


def _require_live_opt_in(args):
    if args.live and not args.confirm_live:
        raise SystemExit('Refusing live smoke order without --confirm-live')


def _check_env():
    required = ['POLY_API_BASE', 'POLY_API_KEY', 'POLY_API_SECRET', 'POLY_API_PASSPHRASE']
    status = {name: bool(os.getenv(name)) for name in required}
    status['wallet_address_derivable'] = bool(polymarket_client.WALLET_ADDRESS)
    return status


def _build_artifact_dir(base_dir: str):
    path = Path(base_dir) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = parse_args()
    _require_live_opt_in(args)
    ensure = run_bot.enforce_startup_gate(allow_dirty_start=False)
    market_meta = polymarket_feed.detect_active_hourly_market(args.series_id)
    if market_meta is None:
        raise SystemExit('Validated market detection failed')
    yes_quote = polymarket_feed.get_quote_snapshot(market_meta['token_yes'])
    no_quote = polymarket_feed.get_quote_snapshot(market_meta['token_no'])
    ctx = run_bot.build_trade_context(market_meta, yes_quote, no_quote)
    can_trade, reason = run_bot.can_trade_context(ctx)
    if not can_trade:
        raise SystemExit(f'Pre-trade gate failed: {reason}')

    env_status = _check_env()
    if args.live and not all(env_status.values()):
        raise SystemExit(f'Missing live credentials/env: {env_status}')

    token_id = market_meta['token_yes'] if args.outcome_side == 'YES' else market_meta['token_no']
    quote = yes_quote if args.outcome_side == 'YES' else no_quote
    price = args.price if args.price is not None else quote.get('best_ask') or quote.get('mid')
    if price is None or price <= 0 or price >= 1:
        raise SystemExit(f'Invalid executable price for smoke order: {price}')

    artifact_dir = _build_artifact_dir(args.artifact_dir)
    manifest = {
        'timestamp': _now_ts(),
        'db_path': str(storage.get_db_path()),
        'storage_gate': ensure,
        'env_status': env_status,
        'market': {
            'market_id': market_meta.get('market_id'),
            'token_yes': market_meta.get('token_yes'),
            'token_no': market_meta.get('token_no'),
            'status': market_meta.get('status'),
            'startDate': str(market_meta.get('startDate')),
            'endDate': str(market_meta.get('endDate')),
        },
        'quote': quote,
        'steps': [],
    }

    place_resp = polymarket_client.place_limit_order(token_id, args.side, args.qty, price, post_only=True, dry_run=(not args.live))
    place_norm = polymarket_client.normalize_client_response(place_resp, default_status='unknown')
    manifest['steps'].append({'step': 'place', 'raw': place_resp, 'normalized': place_norm})

    if place_norm.get('status') in ('submitted', 'accepted', 'open', 'partially_filled') and (place_norm.get('order_id') or place_norm.get('client_order_id')):
        status_resp = polymarket_client.get_order_status(order_id=place_norm.get('order_id'), client_order_id=place_norm.get('client_order_id'), dry_run=(not args.live))
        status_norm = polymarket_client.normalize_client_response(status_resp, default_status='unknown')
        manifest['steps'].append({'step': 'status', 'raw': status_resp, 'normalized': status_norm})
        if status_norm.get('status') in ('submitted', 'open', 'partially_filled'):
            cancel_resp = polymarket_client.cancel_order(order_id=status_norm.get('order_id') or place_norm.get('order_id'), client_order_id=status_norm.get('client_order_id') or place_norm.get('client_order_id'), dry_run=(not args.live))
            cancel_norm = polymarket_client.normalize_client_response(cancel_resp, default_status='unknown')
            manifest['steps'].append({'step': 'cancel', 'raw': cancel_resp, 'normalized': cancel_norm})

    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'artifact_dir': str(artifact_dir), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
