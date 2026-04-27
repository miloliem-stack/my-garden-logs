"""CLI to print reconciliation issues and pending receipts

Usage:
  python scripts/list_reconciliation.py
"""
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from src import storage


def main():
    sweep = storage.run_reconciliation_sweep()
    print('Reconciliation sweep summary:')
    print(json.dumps(sweep['summary'], indent=2))
    print('Pending receipts:')
    for r in storage.get_pending_receipts():
        print(f"- tx_hash={r['tx_hash']} parsed={r['parsed']} ts={r['ts']}")
        print(json.dumps(r, indent=2))
    print('\nReconciliation issues:')
    for r in storage.get_reconciliation_issues():
        print(f"- tx_hash={r['tx_hash']} status={r['reason']} ts={r['ts']}")
        print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
