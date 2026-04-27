import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import storage


def main():
    report = storage.get_active_order_diagnostics()
    print('Active order summary:')
    print(json.dumps(report['summary'], indent=2))
    print('\nOrders:')
    for order in report['orders']:
        print(json.dumps(order, indent=2))


if __name__ == '__main__':
    main()
