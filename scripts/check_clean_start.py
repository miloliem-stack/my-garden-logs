#!/usr/bin/env python3
"""Verify the bot starts from a clean zero-inventory state."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import storage


def main():
    storage.ensure_db()
    status = storage.get_clean_start_status()
    print(f'db_path={storage.get_db_path()}')
    print(json.dumps(status, indent=2))
    if status['clean_start']:
        print('result: clean DB confirmed')
        return
    print('result: NOT clean')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
