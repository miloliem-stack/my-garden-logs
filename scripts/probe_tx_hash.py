"""Read-only tx hash probe across configured EVM chains.

Usage:
  python scripts/probe_tx_hash.py --tx-hash 0x... --wallet 0x...

RPC endpoints are read from environment:
  POLYGON_RPC
  ETHEREUM_RPC
  ARBITRUM_RPC
  BASE_RPC
  OPTIMISM_RPC
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import requests

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.polymarket_client import TRANSFER_BATCH_SIG, TRANSFER_SINGLE_SIG, _try_decode_erc1155_log


ERC20_TRANSFER_SIG = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
CHAIN_ENVS = [
    ('polygon', 'POLYGON_RPC'),
    ('ethereum', 'ETHEREUM_RPC'),
    ('arbitrum', 'ARBITRUM_RPC'),
    ('base', 'BASE_RPC'),
    ('optimism', 'OPTIMISM_RPC'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Probe a tx hash across configured EVM chains without mutating local state.')
    parser.add_argument('--tx-hash', required=True, help='Transaction hash to probe')
    parser.add_argument('--wallet', required=True, help='Wallet address to check for direct or log-level involvement')
    return parser.parse_args()


def normalize_addr(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    addr = addr.lower()
    return addr if addr.startswith('0x') else f'0x{addr}'


def wallet_topic(wallet: str) -> str:
    return '0x' + ('0' * 24) + wallet[2:]


def rpc_call(rpc_url: str, method: str, params: List):
    payload = {'jsonrpc': '2.0', 'method': method, 'params': params, 'id': 1}
    resp = requests.post(rpc_url, json=payload, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if body.get('error'):
        raise RuntimeError(str(body['error']))
    return body.get('result')


def decode_addr_topic(topic: Optional[str]) -> Optional[str]:
    if not topic:
        return None
    topic = str(topic).lower()
    if topic.startswith('0x'):
        topic = topic[2:]
    return '0x' + topic[-40:]


def decode_uint256(data: Optional[str]) -> Optional[int]:
    if not data:
        return None
    data = str(data)
    if data.startswith('0x'):
        data = data[2:]
    if not data:
        return None
    return int(data, 16)


def direction_for_wallet(from_addr: Optional[str], to_addr: Optional[str], wallet: str) -> str:
    if normalize_addr(to_addr) == wallet:
        return 'in'
    if normalize_addr(from_addr) == wallet:
        return 'out'
    return 'other'


def wallet_in_log(log: Dict, wallet: str, wallet_topic_hex: str) -> bool:
    if normalize_addr(log.get('address')) == wallet:
        return True
    topics = [str(t).lower() for t in (log.get('topics') or [])]
    wallet_suffix = wallet[2:]
    return wallet_topic_hex in topics or any(topic.endswith(wallet_suffix) for topic in topics)


def decode_erc1155_transfers(log: Dict, wallet: str) -> List[Dict]:
    decoded = _try_decode_erc1155_log(log)
    if not decoded or not decoded.get('event') or not decoded.get('args'):
        return []

    args = decoded['args']
    from_addr = normalize_addr(args.get('from'))
    to_addr = normalize_addr(args.get('to'))
    direction = direction_for_wallet(from_addr, to_addr, wallet)
    if decoded['event'] == 'TransferSingle':
        return [{
            'token_id': str(args.get('id')),
            'from': from_addr,
            'to': to_addr,
            'qty': float(args.get('value') or 0.0),
            'direction': direction,
        }]

    ids = args.get('ids') or []
    values = args.get('values') or []
    return [
        {
            'token_id': str(token_id),
            'from': from_addr,
            'to': to_addr,
            'qty': float(value or 0.0),
            'direction': direction,
        }
        for token_id, value in zip(ids, values)
    ]


def decode_erc20_transfer(log: Dict, wallet: str) -> Optional[Dict]:
    topics = [str(t).lower() for t in (log.get('topics') or [])]
    if len(topics) < 3 or topics[0] != ERC20_TRANSFER_SIG:
        return None
    from_addr = decode_addr_topic(topics[1])
    to_addr = decode_addr_topic(topics[2])
    return {
        'token_contract': normalize_addr(log.get('address')),
        'from': from_addr,
        'to': to_addr,
        'value': decode_uint256(log.get('data')),
        'direction': direction_for_wallet(from_addr, to_addr, wallet),
    }


def classify_receipt(receipt: Dict, wallet: str) -> Dict:
    topics_wallet = wallet_topic(wallet)
    logs = receipt.get('logs') or []
    tx_from = normalize_addr(receipt.get('from'))
    tx_to = normalize_addr(receipt.get('to'))
    wallet_in_from_to = wallet in {tx_from, tx_to}
    wallet_in_logs = any(wallet_in_log(log, wallet, topics_wallet) for log in logs)

    erc1155_logs = []
    erc20_logs = []
    erc1155_transfers = []
    erc20_transfers = []
    for log in logs:
        topics = [str(t).lower() for t in (log.get('topics') or [])]
        if topics and topics[0] in {TRANSFER_SINGLE_SIG, TRANSFER_BATCH_SIG}:
            erc1155_logs.append(log)
            erc1155_transfers.extend(decode_erc1155_transfers(log, wallet))
        if topics and topics[0] == ERC20_TRANSFER_SIG:
            erc20_logs.append(log)
            decoded_erc20 = decode_erc20_transfer(log, wallet)
            if decoded_erc20 is not None:
                erc20_transfers.append(decoded_erc20)

    wallet_touch = wallet_in_from_to or wallet_in_logs
    if erc1155_logs and wallet_touch:
        classification = 'Polygon ERC-1155 candidate'
    elif erc20_logs and wallet_touch:
        classification = 'ERC-20 funding/withdrawal candidate'
    elif wallet_touch:
        classification = 'wallet-touched other tx'
    else:
        classification = 'unrelated to wallet'

    return {
        'status': receipt.get('status'),
        'block_number': int(receipt['blockNumber'], 16) if receipt.get('blockNumber') else None,
        'from': tx_from,
        'to': tx_to,
        'wallet_in_from_to': wallet_in_from_to,
        'wallet_in_logs': wallet_in_logs,
        'wallet_touched': wallet_touch,
        'erc1155_present': bool(erc1155_logs),
        'erc1155_log_count': len(erc1155_logs),
        'erc1155_transfers': erc1155_transfers,
        'erc20_present': bool(erc20_logs),
        'erc20_log_count': len(erc20_logs),
        'erc20_transfers': erc20_transfers,
        'log_count': len(logs),
        'classification': classification,
    }


def main():
    args = parse_args()
    tx_hash = args.tx_hash
    wallet = normalize_addr(args.wallet)
    if wallet is None or len(wallet) != 42:
        raise SystemExit('wallet must be a 20-byte hex address')

    configured = [(chain, env, os.getenv(env)) for chain, env in CHAIN_ENVS if os.getenv(env)]
    if not configured:
        raise SystemExit('No chain RPCs configured. Set at least one of: POLYGON_RPC, ETHEREUM_RPC, ARBITRUM_RPC, BASE_RPC, OPTIMISM_RPC')

    found = []
    print(f'tx_hash={tx_hash}')
    print(f'wallet={wallet}')
    print('')
    for chain, env_name, rpc_url in configured:
        try:
            receipt = rpc_call(rpc_url, 'eth_getTransactionReceipt', [tx_hash])
        except Exception as exc:
            print(f'[{chain}] rpc_error env={env_name} error={exc}')
            continue

        if not receipt:
            print(f'[{chain}] not_found')
            continue

        summary = classify_receipt(receipt, wallet)
        found.append((chain, summary))
        print(f'[{chain}] found')
        print(f"  status={summary['status']} block={summary['block_number']}")
        print(f"  from={summary['from']} to={summary['to']}")
        print(f"  wallet_in_from_to={summary['wallet_in_from_to']} wallet_in_logs={summary['wallet_in_logs']} wallet_touched={summary['wallet_touched']}")
        print(f"  erc1155_present={summary['erc1155_present']} count={summary['erc1155_log_count']}")
        if summary['erc1155_transfers']:
            print('  erc1155_transfers:')
            for transfer in summary['erc1155_transfers']:
                print(
                    f"    token_id={transfer['token_id']} from={transfer['from']} "
                    f"to={transfer['to']} qty={transfer['qty']} direction={transfer['direction']}"
                )
        print(f"  erc20_present={summary['erc20_present']} count={summary['erc20_log_count']}")
        if summary['erc20_transfers']:
            print('  erc20_transfers:')
            for transfer in summary['erc20_transfers']:
                print(
                    f"    token_contract={transfer['token_contract']} from={transfer['from']} "
                    f"to={transfer['to']} value={transfer['value']} direction={transfer['direction']}"
                )
        print(f"  classification={summary['classification']}")
        print('')

    if not found:
        print('summary: not found on configured chains')
        return

    preferred = next((item for item in found if item[0] == 'polygon' and item[1]['erc1155_present']), None)
    if preferred is not None:
        print(f"summary: Polygon ERC-1155 candidate on chain={preferred[0]}")
        return

    preferred = next((item for item in found if item[1]['erc20_present'] and item[1]['wallet_touched']), None)
    if preferred is not None:
        print(f"summary: ERC-20 funding/withdrawal candidate on chain={preferred[0]}")
        return

    preferred = next((item for item in found if not item[1]['wallet_touched']), None)
    if len(found) == 1 and preferred is not None:
        print(f"summary: unrelated to wallet on chain={preferred[0]}")
        return

    chains = ', '.join(chain for chain, _ in found)
    print(f'summary: found on configured chains ({chains})')


if __name__ == '__main__':
    main()
