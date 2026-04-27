#!/usr/bin/env python3
from src import storage

if __name__ == '__main__':
    storage.ensure_db()
    storage.print_position_snapshot()
