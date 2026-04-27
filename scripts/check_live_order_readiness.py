#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import polymarket_client, polymarket_feed, run_bot, storage


def parse_args():
    parser = argparse.ArgumentParser(description='Read-only readiness audit for the first live order.')
    parser.add_argument('--series-id', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    env_report = {
        'POLY_API_BASE': bool(os.getenv('POLY_API_BASE')),
        'POLY_API_KEY': bool(os.getenv('POLY_API_KEY')),
        'POLY_API_SECRET': bool(os.getenv('POLY_API_SECRET')),
        'POLY_API_PASSPHRASE': bool(os.getenv('POLY_API_PASSPHRASE')),
        'LIVE': os.getenv('LIVE'),
        'wallet_address': polymarket_client.WALLET_ADDRESS,
        'api_base_url': polymarket_client.POLY_API_BASE,
    }
    market_meta = polymarket_feed.detect_active_hourly_market(args.series_id)
    yes_quote = polymarket_feed.get_quote_snapshot(market_meta['token_yes']) if market_meta else {}
    no_quote = polymarket_feed.get_quote_snapshot(market_meta['token_no']) if market_meta else {}
    ctx = run_bot.build_trade_context(market_meta or {}, yes_quote, no_quote) if market_meta else {'market': {}, 'quotes': {'yes': yes_quote, 'no': no_quote}}
    trade_ok, trade_reason = run_bot.can_trade_context(ctx) if market_meta else (False, 'market_detection_failed')
    startup = run_bot.enforce_startup_gate(allow_dirty_start=False)
    fixture_coverage = polymarket_client.get_fixture_coverage_summary()
    report = {
        'env': env_report,
        'storage_gate': startup,
        'market_detected': market_meta is not None,
        'market': None if market_meta is None else {
            'market_id': market_meta.get('market_id'),
            'token_yes': market_meta.get('token_yes'),
            'token_no': market_meta.get('token_no'),
            'status': market_meta.get('status'),
        },
        'yes_quote_valid': polymarket_feed.classify_quote_snapshot(yes_quote).get('tradable') if yes_quote else False,
        'no_quote_valid': polymarket_feed.classify_quote_snapshot(no_quote).get('tradable') if no_quote else False,
        'trade_gate_ok': trade_ok,
        'trade_gate_reason': trade_reason,
        'fixture_coverage': fixture_coverage,
    }
    print(json.dumps(report, indent=2))
    missing_creds = not all([env_report['POLY_API_KEY'], env_report['POLY_API_SECRET'], env_report['POLY_API_PASSPHRASE']])
    if missing_creds:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
