import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_tx_hash


def test_classify_receipt_polygon_erc1155_candidate():
    wallet = '0x' + 'a' * 40
    receipt = {
        'status': '0x1',
        'blockNumber': hex(123),
        'from': '0x' + 'b' * 40,
        'to': '0x' + 'c' * 40,
        'logs': [
            {
                'address': '0x' + 'd' * 40,
                'topics': [
                    probe_tx_hash.TRANSFER_SINGLE_SIG,
                    '0x' + '00' * 32,
                    '0x' + '00' * 32,
                    '0x' + '00' * 24 + ('a' * 40),
                ],
                'data': '0x' + format(11, '064x') + format(5, '064x'),
            }
        ],
    }
    out = probe_tx_hash.classify_receipt(receipt, wallet)
    assert out['wallet_touched'] is True
    assert out['erc1155_present'] is True
    assert out['classification'] == 'Polygon ERC-1155 candidate'
    assert out['erc1155_transfers'] == [{
        'token_id': '11',
        'from': '0x0000000000000000000000000000000000000000',
        'to': wallet,
        'qty': 5.0,
        'direction': 'in',
    }]


def test_classify_receipt_erc20_candidate():
    wallet = '0x' + 'a' * 40
    receipt = {
        'status': '0x1',
        'blockNumber': hex(123),
        'from': wallet,
        'to': '0x' + 'c' * 40,
        'logs': [
            {
                'address': '0x' + 'd' * 40,
                'topics': [
                    probe_tx_hash.ERC20_TRANSFER_SIG,
                    '0x' + '00' * 24 + ('a' * 40),
                    '0x' + '00' * 24 + ('c' * 40),
                ],
                'data': hex(12345),
            }
        ],
    }
    out = probe_tx_hash.classify_receipt(receipt, wallet)
    assert out['erc20_present'] is True
    assert out['classification'] == 'ERC-20 funding/withdrawal candidate'
    assert out['erc20_transfers'] == [{
        'token_contract': '0x' + 'd' * 40,
        'from': wallet,
        'to': '0x' + 'c' * 40,
        'value': 12345,
        'direction': 'out',
    }]


def test_classify_receipt_unrelated():
    wallet = '0x' + 'a' * 40
    receipt = {
        'status': '0x1',
        'blockNumber': hex(123),
        'from': '0x' + 'b' * 40,
        'to': '0x' + 'c' * 40,
        'logs': [],
    }
    out = probe_tx_hash.classify_receipt(receipt, wallet)
    assert out['wallet_touched'] is False
    assert out['classification'] == 'unrelated to wallet'


def test_main_reports_not_found(monkeypatch, capsys):
    monkeypatch.setenv('POLYGON_RPC', 'https://example.invalid')
    monkeypatch.setattr(sys, 'argv', ['probe_tx_hash.py', '--tx-hash', '0x123', '--wallet', '0x' + 'a' * 40])
    monkeypatch.setattr(probe_tx_hash, 'rpc_call', lambda rpc_url, method, params: None)
    probe_tx_hash.main()
    out = capsys.readouterr().out
    assert 'summary: not found on configured chains' in out


def test_main_prints_transfer_details(monkeypatch, capsys):
    wallet = '0x' + 'a' * 40
    monkeypatch.setenv('POLYGON_RPC', 'https://example.invalid')
    monkeypatch.setattr(sys, 'argv', ['probe_tx_hash.py', '--tx-hash', '0x123', '--wallet', wallet])
    monkeypatch.setattr(
        probe_tx_hash,
        'rpc_call',
        lambda rpc_url, method, params: {
            'status': '0x1',
            'blockNumber': hex(55),
            'from': '0x' + 'b' * 40,
            'to': '0x' + 'c' * 40,
            'logs': [
                {
                    'address': '0x' + 'd' * 40,
                    'topics': [
                        probe_tx_hash.TRANSFER_SINGLE_SIG,
                        '0x' + '00' * 32,
                        '0x' + '00' * 32,
                        '0x' + '00' * 24 + ('a' * 40),
                    ],
                    'data': '0x' + format(9, '064x') + format(2, '064x'),
                },
                {
                    'address': '0x' + 'e' * 40,
                    'topics': [
                        probe_tx_hash.ERC20_TRANSFER_SIG,
                        '0x' + '00' * 24 + ('a' * 40),
                        '0x' + '00' * 24 + ('f' * 40),
                    ],
                    'data': hex(77),
                },
            ],
        },
    )
    probe_tx_hash.main()
    out = capsys.readouterr().out
    assert 'erc1155_transfers:' in out
    assert 'token_id=9' in out
    assert 'direction=in' in out
    assert 'erc20_transfers:' in out
    assert 'token_contract=0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' in out
    assert 'value=77' in out
