#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import polymarket_client, storage


def parse_args():
    parser = argparse.ArgumentParser(description='Probe order-status response shape without mutating storage.')
    parser.add_argument('--order-id', default=None)
    parser.add_argument('--client-order-id', default=None)
    parser.add_argument('--live', action='store_true', help='Allow real network call')
    parser.add_argument('--save-fixture', default=None, help='Optional fixture file path to save raw+normalized JSON')
    return parser.parse_args()


def main():
    args = parse_args()
    before_db = storage.get_db_path()
    before_exists = before_db.exists()
    before_size = before_db.stat().st_size if before_exists else None
    response = polymarket_client.get_order_status(order_id=args.order_id, client_order_id=args.client_order_id, dry_run=(not args.live))
    normalized = polymarket_client.normalize_client_response(response, default_status='unknown')
    payload = {
        'request': {'order_id': args.order_id, 'client_order_id': args.client_order_id, 'live': bool(args.live)},
        'raw': response,
        'normalized': normalized,
    }
    print(json.dumps(payload, indent=2))
    if args.save_fixture:
        fixture_path = Path(args.save_fixture)
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(payload, indent=2))
    after_exists = before_db.exists()
    after_size = before_db.stat().st_size if after_exists else None
    print(json.dumps({
        'storage_mutated': before_exists != after_exists or before_size != after_size,
        'db_path': str(before_db),
    }, indent=2))


if __name__ == '__main__':
    main()
