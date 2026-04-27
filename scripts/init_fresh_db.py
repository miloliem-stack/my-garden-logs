#!/usr/bin/env python3
"""Initialize a fresh bot DB with schema and clean-start verification."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import storage


def parse_args():
    parser = argparse.ArgumentParser(description='Initialize a fresh DB for the bot.')
    parser.add_argument('--db-path', default=None, help='Target DB path. Overrides BOT_DB_PATH for this command.')
    parser.add_argument('--force', action='store_true', help='If target DB exists and is dirty, replace it with a fresh DB.')
    return parser.parse_args()


def main():
    args = parse_args()
    original_env = os.getenv('BOT_DB_PATH')
    if args.db_path:
        os.environ['BOT_DB_PATH'] = args.db_path

    try:
        db_path = storage.get_db_path()
        if db_path.exists():
            storage.ensure_db()
            status = storage.get_clean_start_status()
            if not status['clean_start'] and not args.force:
                print(f'refusing to reuse dirty DB at {db_path}')
                print(status)
                raise SystemExit(1)
            if not status['clean_start'] and args.force:
                db_path.unlink()

        storage.ensure_db()
        status = storage.get_clean_start_status()
        print(f'db_path={storage.get_db_path()}')
        print(status)
        print('result:', 'clean DB ready' if status['clean_start'] else 'DB initialized but not clean')
        if not status['clean_start']:
            raise SystemExit(1)
    finally:
        if args.db_path:
            if original_env is None:
                os.environ.pop('BOT_DB_PATH', None)
            else:
                os.environ['BOT_DB_PATH'] = original_env


if __name__ == '__main__':
    main()
