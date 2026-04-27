#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import polymarket_client


def main():
    print(json.dumps(polymarket_client.describe_venue_assumptions(), indent=2))


if __name__ == '__main__':
    main()
