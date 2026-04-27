"""Polymarket authenticated client (minimal).

This module reads credentials from environment (use python-dotenv or set env vars).

Notes:
- Header names and signature scheme implemented here follow a generic HMAC-SHA256 pattern
  (timestamp + method + path + body) -> base64(HMAC(secret, prehash)). Adjust header
  names and prehash as required by Polymarket Gamma/CLOB API.
- By default the client performs a dry-run (no send) unless LIVE env var is set to 'true'.
"""
from typing import Optional, Dict, Any
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import importlib
import os
import time
import hmac
import hashlib
import base64
import binascii
import json
import requests

from dotenv import load_dotenv

load_dotenv()
try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
except Exception:
    Account = None
    encode_defunct = None
try:
    from web3 import Web3
except Exception:  # pragma: no cover - optional in some environments
    Web3 = None
try:
    from eth_utils import keccak
except Exception:
    keccak = None
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType, MarketOrderArgs, TradeParams
    from py_clob_client.exceptions import PolyApiException
    from py_clob_client.utilities import order_to_json
except Exception:  # pragma: no cover
    ClobClient = None
    ApiCreds = None
    OrderArgs = None
    OrderType = None
    MarketOrderArgs = None
    TradeParams = None
    PolyApiException = Exception
    order_to_json = None

POLY_GAMMA_BASE = os.getenv('POLY_GAMMA_BASE', 'https://gamma-api.polymarket.com')
POLY_CLOB_BASE = os.getenv('POLY_CLOB_BASE', 'https://clob.polymarket.com')
POLY_API_BASE = os.getenv('POLY_API_BASE', POLY_CLOB_BASE)
POLY_API_KEY = os.getenv('POLY_API_KEY')
POLY_API_SECRET = os.getenv('POLY_API_SECRET')
POLY_API_PASSPHRASE = os.getenv('POLY_API_PASSPHRASE')
POLY_WALLET_PRIVATE_KEY = os.getenv('POLY_WALLET_PRIVATE_KEY')
POLY_SIGNATURE_TYPE = int(os.getenv('POLY_SIGNATURE_TYPE', '0'))
POLY_FUNDER = os.getenv('POLY_FUNDER')
LIVE = str(os.getenv('LIVE', 'false')).lower() in ('1', 'true', 'yes')
FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'client_responses'
SDK_FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'sdk_clob'

# Precompute canonical ERC-1155 event signature hashes when keccak is available.
TS_TRANSFER_SINGLE = 'TransferSingle(address,address,address,uint256,uint256)'
TS_TRANSFER_BATCH = 'TransferBatch(address,address,address,uint256[],uint256[])'
if keccak is not None:
    TRANSFER_SINGLE_SIG = '0x' + keccak(text=TS_TRANSFER_SINGLE).hex()
    TRANSFER_BATCH_SIG = '0x' + keccak(text=TS_TRANSFER_BATCH).hex()
else:
    TRANSFER_SINGLE_SIG = None
    TRANSFER_BATCH_SIG = None


def _wallet_address_from_privkey(priv: Optional[str]) -> Optional[str]:
    if not priv or Account is None:
        return None
    acct = Account.from_key(priv)
    return acct.address


def _checksum_address(address: str) -> str:
    if Web3 is None:
        return address
    if hasattr(Web3, 'to_checksum_address'):
        return Web3.to_checksum_address(address)
    return Web3.toChecksumAddress(address)


def _inject_poa_middleware(web3):
    if web3 is None or not hasattr(web3, 'middleware_onion'):
        return web3
    middleware = None
    try:
        middleware = getattr(importlib.import_module('web3.middleware'), 'ExtraDataToPOAMiddleware')
    except Exception:
        middleware = None
    if middleware is None:
        try:
            middleware = getattr(importlib.import_module('web3.middleware'), 'geth_poa_middleware')
        except Exception:
            middleware = None
    if middleware is None:
        return web3
    try:
        web3.middleware_onion.inject(middleware, layer=0)
    except Exception:
        pass
    return web3


def _condition_id_to_bytes(condition_id: Optional[str]) -> Optional[bytes]:
    if condition_id is None:
        return None
    value = str(condition_id).strip()
    if not value:
        return None
    if value.startswith('0x'):
        value = value[2:]
    if len(value) != 64:
        raise ValueError('condition_id must be 32-byte hex')
    return bytes.fromhex(value)


WALLET_ADDRESS = _wallet_address_from_privkey(POLY_WALLET_PRIVATE_KEY)
POLY_WS_URL = os.getenv('POLY_WS_URL')
POLYGON_RPC = os.getenv('POLYGON_RPC')
_CLOB_CLIENT = None
_CLOB_API_CREDS = None
MIN_PRICE_DECIMAL = Decimal('0.01')
MAX_PRICE_DECIMAL = Decimal('0.99')
MAKER_NOTIONAL_QUANTUM = Decimal('0.01')
SHARE_QUANTUM = Decimal('0.0001')
MIN_MARKET_BUY_SPEND = Decimal(os.getenv('POLY_MARKET_BUY_MIN_SPEND', '1.00'))
POLYGON_CHAIN_ID = int(os.getenv('POLYGON_CHAIN_ID', '137'))
DEFAULT_CTF_CONTRACT_ADDRESS = os.getenv('POLY_CTF_CONTRACT_ADDRESS', '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045')
DEFAULT_USDC_E_TOKEN_ADDRESS = os.getenv('USDC_E_TOKEN_ADDRESS', '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')
REDEEM_RECEIPT_TIMEOUT_SEC = int(os.getenv('REDEEM_RECEIPT_TIMEOUT_SEC', '120'))

_CTF_ABI = [
    {
        'inputs': [{'internalType': 'bytes32', 'name': 'conditionId', 'type': 'bytes32'}],
        'name': 'payoutDenominator',
        'outputs': [{'internalType': 'uint256', 'name': '', 'type': 'uint256'}],
        'stateMutability': 'view',
        'type': 'function',
    },
    {
        'inputs': [
            {'internalType': 'address', 'name': 'collateralToken', 'type': 'address'},
            {'internalType': 'bytes32', 'name': 'parentCollectionId', 'type': 'bytes32'},
            {'internalType': 'bytes32', 'name': 'conditionId', 'type': 'bytes32'},
            {'internalType': 'uint256[]', 'name': 'indexSets', 'type': 'uint256[]'},
        ],
        'name': 'redeemPositions',
        'outputs': [],
        'stateMutability': 'nonpayable',
        'type': 'function',
    },
]


def _timestamp() -> str:
    # ISO-like timestamp in seconds precision
    return str(int(time.time()))


def _sign(secret: str, prehash: str) -> str:
    if secret is None:
        # allow missing secret for dry-run; return empty signature
        return ''
    if isinstance(secret, str):
        try:
            secret_bytes = base64.b64decode(secret, validate=True)
        except (binascii.Error, ValueError):
            # Provisional fallback until live probe evidence confirms all credential encodings.
            secret_bytes = secret.encode()
    else:
        secret_bytes = secret
    sig = hmac.new(secret_bytes, prehash.encode(), hashlib.sha256).digest()
    return base64.b64encode(sig).decode()


def _serialize_body_for_hmac(body: Any) -> str:
    if body is None:
        return ''
    if isinstance(body, str):
        return body
    if isinstance(body, (dict, list)):
        return json.dumps(body, separators=(',', ':'), sort_keys=False)
    return str(body)


def _redacted_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    lowered = str(name).lower()
    if any(token in lowered for token in ('secret', 'passphrase', 'private', 'sign', 'signature')):
        return '[redacted]'
    return value


def redact_mapping(mapping: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if mapping is None:
        return None
    return {key: _redacted_value(key, value) for key, value in dict(mapping).items()}


def _safe_json_parse(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _decimal_from_value(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        candidate = value
    else:
        candidate = Decimal(str(value))
    if not candidate.is_finite():
        raise InvalidOperation(f'non-finite decimal: {value}')
    return candidate


def _floor_decimal(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_DOWN)


def _clamp_marketable_price(price: Decimal) -> Decimal:
    if price < MIN_PRICE_DECIMAL:
        return MIN_PRICE_DECIMAL
    if price > MAX_PRICE_DECIMAL:
        return MAX_PRICE_DECIMAL
    return price


def _skipped_quantized_order_payload(
    *,
    status: str,
    reason: str,
    token_id: str,
    side: str,
    order_type: str,
    post_only: bool,
    marketable: bool,
    raw_quantity: Any,
    raw_limit_price: Any,
    raw_notional: Any = None,
    quantized_quantity: Any = 0.0,
    quantized_limit_price: Any = None,
    quantized_notional: Any = 0.0,
) -> Dict[str, Any]:
    return {
        'status': status,
        'reason': reason,
        'token_id': token_id,
        'side': side,
        'order_type': order_type,
        'post_only': bool(post_only),
        'marketable': bool(marketable),
        'raw_requested_qty': raw_quantity,
        'raw_requested_notional': raw_notional,
        'quantized_qty': quantized_quantity,
        'quantized_notional': quantized_notional,
        'limit_price': quantized_limit_price,
        'raw_limit_price': raw_limit_price,
        'skipped_due_to_quantization': True,
    }


def quantize_order_submission(
    *,
    token_id: str,
    side: str,
    quantity: float,
    limit_price: Optional[float],
    order_type: str = 'GTC',
    post_only: bool = False,
    marketable: bool = False,
) -> Dict[str, Any]:
    if not token_id:
        raise ValueError('token_id is required')
    side_value = str(side or '').lower()
    if side_value not in ('buy', 'sell'):
        raise ValueError('side must be buy or sell')
    if limit_price is None:
        raise ValueError('limit_price is required')
    tif_value = str(order_type or 'GTC').upper()
    if tif_value not in ('GTC', 'GTD', 'FAK', 'FOK'):
        raise ValueError(f'unsupported order_type: {order_type}')
    if post_only and tif_value not in ('GTC', 'GTD'):
        raise ValueError('post_only is only supported with GTC/GTD')
    if post_only and marketable:
        raise ValueError('post_only and marketable cannot both be true')
    if marketable and tif_value not in ('FAK', 'FOK'):
        raise ValueError('marketable orders must use FAK or FOK')

    try:
        raw_quantity = _decimal_from_value(quantity)
        raw_price = _decimal_from_value(limit_price)
    except (InvalidOperation, TypeError, ValueError):
        return _skipped_quantized_order_payload(
            status='skipped_invalid_quantized_order',
            reason='malformed_numeric_input',
            token_id=str(token_id),
            side=side_value,
            order_type=tif_value,
            post_only=post_only,
            marketable=marketable,
            raw_quantity=quantity,
            raw_limit_price=limit_price,
        )

    if raw_quantity <= 0:
        raise ValueError('quantity must be positive')
    if raw_price <= 0:
        return _skipped_quantized_order_payload(
            status='skipped_invalid_quantized_order',
            reason='invalid_price',
            token_id=str(token_id),
            side=side_value,
            order_type=tif_value,
            post_only=post_only,
            marketable=marketable,
            raw_quantity=float(raw_quantity),
            raw_limit_price=float(raw_price),
        )

    price = _clamp_marketable_price(raw_price) if marketable else raw_price
    raw_notional = raw_quantity * price
    quantized_quantity = raw_quantity
    quantized_notional = raw_notional
    quantization_policy = 'none'
    skipped = False
    skip_reason = None

    if marketable and side_value == 'buy':
        quantization_policy = 'marketable_buy_quote_notional_2dp_shares_4dp_floor'
        requested_notional = _floor_decimal(raw_notional, MAKER_NOTIONAL_QUANTUM)
        if requested_notional < MIN_MARKET_BUY_SPEND:
            skipped = True
            skip_reason = 'skipped_below_min_market_buy_notional'
            quantized_quantity = Decimal('0')
            quantized_notional = Decimal('0')
        else:
            quantized_quantity = _floor_decimal(min(raw_quantity, requested_notional / price), SHARE_QUANTUM)
            quantized_notional = requested_notional
            if quantized_quantity <= 0:
                skipped = True
                skip_reason = 'zero_qty_after_quantization'
    elif marketable and side_value == 'sell':
        quantization_policy = 'marketable_sell_shares_4dp_floor'
        quantized_quantity = _floor_decimal(raw_quantity, SHARE_QUANTUM)
        quantized_notional = _floor_decimal(quantized_quantity * price, MAKER_NOTIONAL_QUANTUM)
        if quantized_quantity <= 0:
            skipped = True
            skip_reason = 'zero_qty_after_quantization'

    quantization = {
        'policy': quantization_policy,
        'raw_requested_qty': float(raw_quantity),
        'raw_requested_notional': float(raw_notional),
        'quantized_qty': float(quantized_quantity),
        'quantized_notional': float(quantized_notional),
        'raw_limit_price': float(raw_price),
        'limit_price': float(price),
        'skipped_due_to_quantization': skipped,
        'skip_reason': skip_reason,
    }
    if skipped:
        return {
            **_skipped_quantized_order_payload(
                status='skipped_invalid_quantized_order',
                reason=skip_reason or 'invalid_quantized_order',
                token_id=str(token_id),
                side=side_value,
                order_type=tif_value,
                post_only=post_only,
                marketable=marketable,
                raw_quantity=float(raw_quantity),
                raw_limit_price=float(raw_price),
                raw_notional=float(raw_notional),
                quantized_quantity=float(quantized_quantity),
                quantized_limit_price=float(price),
                quantized_notional=float(quantized_notional),
            ),
            'quantization': quantization,
        }
    return {
        'token_id': str(token_id),
        'side': side_value,
        'quantity': float(quantized_quantity),
        'amount': float(quantized_notional) if marketable and side_value == 'buy' else None,
        'limit_price': float(price),
        'order_type': tif_value,
        'post_only': bool(post_only),
        'marketable': bool(marketable),
        'requested_qty_raw': float(raw_quantity),
        'requested_notional_raw': float(raw_notional),
        'effective_notional': float(quantized_notional),
        'quantization': quantization,
        'skipped': False,
    }


def build_marketable_order_intent(
    *,
    token_id: str,
    side: str,
    quantity: float,
    limit_price: Optional[float],
    order_type: str = 'FAK',
    client_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    quantized = quantize_order_submission(
        token_id=token_id,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        order_type=order_type,
        post_only=False,
        marketable=True,
    )
    if quantized.get('status') == 'skipped_invalid_quantized_order':
        return {
            'token_id': str(token_id),
            'side': str(side or '').lower(),
            'order_type': str(order_type or 'FAK').upper(),
            'post_only': False,
            'marketable': True,
            'client_order_id': client_order_id,
            'skipped': True,
            'skip_reason': quantized.get('reason'),
            'request_body': {
                'tokenId': str(token_id),
                'side': str(side or '').lower(),
                'amount': quantized.get('quantized_notional', 0.0) if str(side or '').lower() == 'buy' else None,
                'quantity': quantized.get('quantized_qty', 0.0) if str(side or '').lower() == 'sell' else None,
                'price': quantized.get('limit_price'),
                'orderType': str(order_type or 'FAK').upper(),
                'postOnly': False,
                'marketable': True,
            },
            'quantization': quantized.get('quantization'),
        }

    side_value = quantized['side']
    tif_value = quantized['order_type']
    submitted_qty = float(quantized['quantity'])
    submitted_notional = float(quantized['effective_notional'])
    price = float(quantized['limit_price'])
    venue_intent_mode = 'amount' if side_value == 'buy' else 'quantity'
    request_body = {
        'tokenId': str(token_id),
        'side': side_value,
        'price': price,
        'orderType': tif_value,
        'postOnly': False,
        'marketable': True,
    }
    if side_value == 'buy':
        request_body['amount'] = submitted_notional
    else:
        request_body['quantity'] = submitted_qty
    return {
        'token_id': str(token_id),
        'side': side_value,
        'order_type': tif_value,
        'post_only': False,
        'marketable': True,
        'client_order_id': client_order_id,
        'venue_intent_mode': venue_intent_mode,
        'submitted_qty': submitted_qty,
        'submitted_notional': submitted_notional,
        'reservation_qty': submitted_notional if side_value == 'buy' else submitted_qty,
        'request_body': request_body,
        'quantization': quantized['quantization'],
        'skipped': False,
    }


def _serialize_post_order_body(order: Any, owner: Optional[str], order_type: Any, post_only: bool, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if order_to_json is None or owner is None:
        return fallback
    try:
        return order_to_json(order, owner, order_type, post_only)
    except Exception:
        return fallback


def build_auth_headers(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Build provisional Polymarket CLOB L2 auth headers for debug/probe use only.

    Authenticated production CLOB transport is delegated to py-clob-client.
    """
    ts = _timestamp()
    body_str = _serialize_body_for_hmac(body)
    prehash = ts + method.upper() + path + body_str
    sig = _sign(POLY_API_SECRET, prehash)
    headers = {
        'Content-Type': 'application/json',
        'POLY_ADDRESS': WALLET_ADDRESS or '',
        'POLY_SIGNATURE': sig,
        'POLY_TIMESTAMP': ts,
        'POLY_API_KEY': POLY_API_KEY or '',
        'POLY_PASSPHRASE': POLY_API_PASSPHRASE or '',
    }
    return headers


def get_clob_client() -> Any:
    global _CLOB_CLIENT
    if _CLOB_CLIENT is not None:
        return _CLOB_CLIENT
    if ClobClient is None:
        raise RuntimeError('py-clob-client is not available')
    funder = POLY_FUNDER or WALLET_ADDRESS
    _CLOB_CLIENT = ClobClient(
        host=POLY_CLOB_BASE,
        key=POLY_WALLET_PRIVATE_KEY,
        chain_id=137,
        signature_type=POLY_SIGNATURE_TYPE,
        funder=funder,
    )
    return _CLOB_CLIENT


def get_clob_api_creds() -> Any:
    global _CLOB_API_CREDS
    if _CLOB_API_CREDS is not None:
        return _CLOB_API_CREDS
    client = get_clob_client()
    if ApiCreds is None:
        raise RuntimeError('py-clob-client ApiCreds is not available')
    if POLY_API_KEY and POLY_API_SECRET and POLY_API_PASSPHRASE:
        creds = ApiCreds(
            api_key=POLY_API_KEY,
            api_secret=POLY_API_SECRET,
            api_passphrase=POLY_API_PASSPHRASE,
        )
    else:
        creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    _CLOB_API_CREDS = creds
    return creds


def _sdk_probe_template(method: str, path: str, request_body: Any = None, *, logical_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        'method': method,
        'path': path,
        'url': POLY_CLOB_BASE.rstrip('/') + path,
        'params': logical_params,
        'request_headers': None,
        'request_body': request_body,
        'http_status': None,
        'ok': False,
        'latency_ms': None,
        'response_headers': None,
        'response_text': None,
        'response_json': None,
        'error_text': None,
        'transport': 'clob_sdk',
    }


def _normalize_sdk_result(result: Any, *, default_status: str, raw_probe: Dict[str, Any], client_order_id: Optional[str] = None) -> Dict[str, Any]:
    payload = dict(result) if isinstance(result, dict) else {'result': result}
    payload.setdefault('raw_probe', raw_probe)
    payload.setdefault('http_status', raw_probe.get('http_status'))
    if client_order_id is not None and not payload.get('clientOrderId') and not payload.get('client_order_id'):
        payload['client_order_id'] = client_order_id
    return normalize_client_response(payload, default_status=default_status)


def _normalize_sdk_exception(exc: Exception, *, default_status: str, raw_probe: Dict[str, Any], client_order_id: Optional[str] = None) -> Dict[str, Any]:
    http_status = getattr(exc, 'status_code', None)
    error_payload = getattr(exc, 'error_msg', None)
    raw_probe = dict(raw_probe)
    raw_probe['http_status'] = http_status
    raw_probe['error_text'] = str(exc)
    if isinstance(error_payload, dict):
        raw_probe['response_json'] = error_payload
    elif error_payload is not None:
        raw_probe['response_text'] = str(error_payload)
    status = default_status
    if http_status in (401, 403):
        status = 'rejected'
    elif http_status in (400, 404, 422):
        status = 'error'
    elif http_status is None:
        status = 'unknown'
    payload: Dict[str, Any] = {
        'status': status,
        'http_status': http_status,
        'error_message': str(error_payload if error_payload is not None else exc),
        'raw_probe': raw_probe,
    }
    if isinstance(error_payload, dict):
        payload.update(error_payload)
    if client_order_id is not None:
        payload['client_order_id'] = client_order_id
    return normalize_client_response(payload, default_status=status, ok_statuses=set())


def build_sdk_order_args(
    *,
    token_id: str,
    side: str,
    quantity: float,
    limit_price: Optional[float],
    order_type: str = 'GTC',
    post_only: bool = False,
    marketable: bool = False,
    client_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    if marketable and str(side or '').lower() == 'buy':
        raise ValueError('marketable buy orders must use build_marketable_order_intent/create_market_order')
    quantized = quantize_order_submission(
        token_id=token_id,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        order_type=order_type,
        post_only=post_only,
        marketable=marketable,
    )
    if quantized.get('status') == 'skipped_invalid_quantized_order':
        return {
            'order_args': None,
            'order_type': quantized['order_type'],
            'post_only': bool(post_only),
            'marketable': bool(marketable),
            'request_body': {
                'tokenId': str(token_id),
                'side': str(side or '').lower(),
                'quantity': quantized.get('quantized_qty', 0.0),
                'price': quantized.get('limit_price'),
                'orderType': quantized['order_type'],
                'postOnly': bool(post_only),
                'marketable': bool(marketable),
            },
            'client_order_id': client_order_id,
            'skipped': True,
            'skip_reason': quantized.get('reason'),
            'quantization': quantized.get('quantization'),
        }
    if OrderArgs is None or OrderType is None:
        raise RuntimeError('py-clob-client order types are not available')
    size = float(quantized['quantity'])
    price = float(quantized['limit_price'])
    side_value = quantized['side']
    tif_value = quantized['order_type']
    request_body = {
        'tokenId': str(token_id),
        'side': side_value,
        'quantity': size,
        'price': price,
        'orderType': tif_value,
        'postOnly': bool(post_only),
        'marketable': bool(marketable),
    }
    if client_order_id is not None:
        request_body['clientOrderId'] = client_order_id
    return {
        'order_args': OrderArgs(
            token_id=str(token_id),
            price=price,
            size=size,
            side=side_value.upper(),
        ),
        'order_type': getattr(OrderType, tif_value, tif_value),
        'post_only': bool(post_only),
        'marketable': bool(marketable),
        'request_body': request_body,
        'client_order_id': client_order_id,
        'skipped': False,
        'quantization': quantized['quantization'],
    }


def load_sdk_fixture(name: str) -> Dict[str, Any]:
    return json.loads((SDK_FIXTURE_DIR / name).read_text())


def replay_sdk_fixture(name: str, *, default_status: str = 'unknown', client_order_id: Optional[str] = None) -> Dict[str, Any]:
    fixture = load_sdk_fixture(name)
    raw_probe = fixture.get('raw_probe') or _sdk_probe_template(
        fixture.get('method', 'GET'),
        fixture.get('path', '/fixture'),
        fixture.get('request_body'),
    )
    if fixture.get('exception'):
        exc_payload = fixture['exception']
        exc = SimpleNamespace(
            status_code=exc_payload.get('status_code'),
            error_msg=exc_payload.get('error_msg'),
        )
        return _normalize_sdk_exception(exc, default_status=default_status, raw_probe=raw_probe, client_order_id=client_order_id)
    result = fixture.get('raw') or {}
    return _normalize_sdk_result(result, default_status=default_status, raw_probe=raw_probe, client_order_id=client_order_id)


def raw_http_probe(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    auth: bool = True,
    timeout: int = 10,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    method = method.upper()
    effective_base_url = (base_url or POLY_CLOB_BASE).rstrip('/')
    if effective_base_url == 'https://api.polymarket.com' and (path in {'/order', '/heartbeats'} or path.startswith('/order/')):
        raise ValueError(f'Refusing to send CLOB order-management request to legacy host: {effective_base_url}{path}')
    url = effective_base_url + path
    headers = build_auth_headers(method, path, body) if auth else {'Content-Type': 'application/json'}
    started = time.perf_counter()
    response = None
    error_text = None
    try:
        response = requests.request(method, url, json=body, params=params, headers=headers, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = response.text
        return {
            'method': method,
            'path': path,
            'url': url,
            'params': params,
            'request_headers': redact_mapping(headers),
            'request_body': body,
            'http_status': response.status_code,
            'ok': bool(response.ok),
            'latency_ms': latency_ms,
            'response_headers': redact_mapping(dict(response.headers)),
            'response_text': raw_text,
            'response_json': _safe_json_parse(raw_text),
            'error_text': None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response is not None:
            error_text = str(exc)
            raw_text = response.text
            return {
                'method': method,
                'path': path,
                'url': url,
                'params': params,
                'request_headers': redact_mapping(headers),
                'request_body': body,
                'http_status': response.status_code,
                'ok': False,
                'latency_ms': latency_ms,
                'response_headers': redact_mapping(dict(response.headers)),
                'response_text': raw_text,
                'response_json': _safe_json_parse(raw_text),
                'error_text': error_text,
            }
        return {
            'method': method,
            'path': path,
            'url': url,
            'params': params,
            'request_headers': redact_mapping(headers),
            'request_body': body,
            'http_status': None,
            'ok': False,
            'latency_ms': latency_ms,
            'response_headers': None,
            'response_text': None,
            'response_json': None,
            'error_text': str(exc),
        }


def raw_rpc_probe(method: str, params: list, timeout: int = 10) -> Dict[str, Any]:
    payload = {'jsonrpc': '2.0', 'method': method, 'params': params, 'id': 1}
    if not POLYGON_RPC:
        return {
            'method': 'POST',
            'path': None,
            'url': None,
            'params': params,
            'request_headers': {'Content-Type': 'application/json'},
            'request_body': payload,
            'http_status': None,
            'ok': False,
            'latency_ms': 0.0,
            'response_headers': None,
            'response_text': None,
            'response_json': None,
            'error_text': 'POLYGON_RPC not configured',
            'rpc_method': method,
        }
    started = time.perf_counter()
    response = None
    try:
        response = requests.post(POLYGON_RPC, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = response.text
        parsed = _safe_json_parse(raw_text)
        ok = bool(response.ok) and isinstance(parsed, dict) and parsed.get('error') is None
        return {
            'method': 'POST',
            'path': None,
            'url': POLYGON_RPC,
            'params': params,
            'request_headers': {'Content-Type': 'application/json'},
            'request_body': payload,
            'http_status': response.status_code,
            'ok': ok,
            'latency_ms': latency_ms,
            'response_headers': redact_mapping(dict(response.headers)),
            'response_text': raw_text,
            'response_json': parsed,
            'error_text': None if ok else (None if not isinstance(parsed, dict) else json.dumps(parsed.get('error')) if parsed.get('error') is not None else None),
            'rpc_method': method,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            'method': 'POST',
            'path': None,
            'url': POLYGON_RPC,
            'params': params,
            'request_headers': {'Content-Type': 'application/json'},
            'request_body': payload,
            'http_status': None if response is None else response.status_code,
            'ok': False,
            'latency_ms': latency_ms,
            'response_headers': None if response is None else redact_mapping(dict(response.headers)),
            'response_text': None if response is None else response.text,
            'response_json': None if response is None else _safe_json_parse(response.text),
            'error_text': str(exc),
            'rpc_method': method,
        }


def normalize_client_response(resp: Any, *, default_status: str = 'unknown', ok_statuses: Optional[set] = None) -> Dict[str, Any]:
    ok_statuses = ok_statuses or {'ok', 'accepted', 'open', 'filled', 'partially_filled', 'success'}
    if resp is None:
        return {
            'ok': False,
            'status': 'no_response',
            'tx_hash': None,
            'order_id': None,
            'client_order_id': None,
            'http_status': None,
            'error_code': None,
            'error_message': 'no response',
            'raw': None,
        }
    if not isinstance(resp, dict):
        return {
            'ok': False,
            'status': 'invalid_response',
            'tx_hash': None,
            'order_id': None,
            'client_order_id': None,
            'http_status': None,
            'error_code': None,
            'error_message': str(resp),
            'raw': resp,
        }
    raw = dict(resp)
    tx_hash = raw.get('tx_hash') or raw.get('txHash') or raw.get('transactionHash')
    if tx_hash is None:
        tx_hashes = raw.get('transactionsHashes')
        if isinstance(tx_hashes, list) and tx_hashes:
            tx_hash = tx_hashes[0]
    order_id = raw.get('order_id') or raw.get('orderId') or raw.get('orderID') or raw.get('id')
    client_order_id = raw.get('client_order_id') or raw.get('clientOrderId')
    raw_probe = raw.get('raw_probe') or {}
    raw_probe_path = str(raw_probe.get('path') or '')
    explicit_null_result = 'result' in raw and raw.get('result') is None and not raw.get('status') and not raw.get('state')
    venue_lookup_null = explicit_null_result and (bool(order_id or client_order_id) or raw_probe_path.startswith('/order/'))
    status = str(
        'not_found_on_venue'
        if venue_lookup_null
        else raw.get('status')
        or raw.get('state')
        or raw.get('result')
        or ('dry_run' if raw.get('dry_run') else default_status)
    ).lower()
    http_status = raw.get('http_status') or raw.get('status_code')
    error_code = raw.get('error_code') or raw.get('code')
    error_message = raw.get('error_message') or raw.get('reason') or raw.get('text') or raw.get('message')
    if venue_lookup_null and not error_message:
        error_message = 'venue returned null for order lookup'
    filled_qty = raw.get('filled_qty')
    if filled_qty is None and 'filledQuantity' in raw:
        filled_qty = raw.get('filledQuantity')
    if filled_qty is None and 'size_matched' in raw:
        filled_qty = raw.get('size_matched')
    remaining_qty = raw.get('remaining_qty')
    if remaining_qty is None and 'remainingQuantity' in raw:
        remaining_qty = raw.get('remainingQuantity')
    avg_price = raw.get('avg_price') or raw.get('averagePrice') or raw.get('avgPrice')
    fills = raw.get('fills')
    qty = raw.get('qty') if raw.get('qty') is not None else raw.get('size') if raw.get('size') is not None else raw.get('quantity')
    if qty is None and 'original_size' in raw:
        qty = raw.get('original_size')
    fill_id = raw.get('fill_id') or raw.get('fillId')
    non_terminal_statuses = {'pending_submit', 'cancel_requested', 'unknown', 'rejected', 'not_found_on_venue'}
    ok = False
    if raw.get('dry_run'):
        ok = True
    elif status in ok_statuses and status not in non_terminal_statuses and not error_message:
        ok = True
    elif status in ok_statuses and status not in non_terminal_statuses and tx_hash is not None:
        ok = True
    normalized = {
        'ok': ok,
        'status': status,
        'tx_hash': tx_hash,
        'order_id': order_id,
        'client_order_id': client_order_id,
        'http_status': http_status,
        'error_code': error_code,
        'error_message': error_message,
        'filled_qty': filled_qty,
        'remaining_qty': remaining_qty,
        'avg_price': avg_price,
        'fills': fills,
        'qty': qty,
        'fill_id': fill_id,
        'raw': raw,
    }
    if raw.get('dry_run'):
        normalized['dry_run'] = True
    return {**raw, **normalized}


def _normalize_probe_result(probe_result: Dict[str, Any], *, default_status: str) -> Dict[str, Any]:
    probe_result = probe_result or {}
    payload = probe_result.get('response_json')
    if isinstance(payload, dict):
        raw_payload = dict(payload)
    else:
        raw_payload = {}
    http_status = probe_result.get('http_status')
    error_text = probe_result.get('error_text')
    if 'http_status' not in raw_payload:
        raw_payload['http_status'] = http_status
    if 'raw_probe' not in raw_payload:
        raw_payload['raw_probe'] = probe_result
    if error_text and 'error_message' not in raw_payload and 'reason' not in raw_payload and 'message' not in raw_payload:
        raw_payload['error_message'] = error_text
    if 'status' not in raw_payload and 'state' not in raw_payload and 'result' not in raw_payload:
        if http_status in (401, 403):
            raw_payload['status'] = 'rejected'
        elif http_status in (400, 404, 422):
            raw_payload['status'] = 'error'
        elif http_status is None or error_text:
            raw_payload['status'] = 'unknown'
        else:
            raw_payload['status'] = default_status
    if http_status is not None and http_status >= 500 and raw_payload.get('status') == 'error':
        raw_payload['status'] = 'unknown'
    return normalize_client_response(raw_payload, default_status=default_status)


def get_fixture_coverage_summary() -> Dict[str, Any]:
    fixture_names = []
    if FIXTURE_DIR.exists():
        fixture_names = sorted(path.name for path in FIXTURE_DIR.glob('*.json'))
    canonical_states = ['pending_submit', 'submitted', 'open', 'partially_filled', 'filled', 'cancel_requested', 'canceled', 'expired', 'rejected', 'failed', 'unknown']
    covered = {
        'submitted': any('accepted' in name for name in fixture_names),
        'open': any('open' in name for name in fixture_names),
        'partially_filled': any('partial' in name for name in fixture_names),
        'filled': any('filled' in name for name in fixture_names),
        'canceled': any('cancel' in name or 'canceled' in name for name in fixture_names),
        'expired': any('expired' in name for name in fixture_names),
        'failed': any('error' in name for name in fixture_names),
        'unknown': any('ambiguous' in name for name in fixture_names),
        'pending_submit': any('pending_submit' in name for name in fixture_names),
        'cancel_requested': any('cancel_requested' in name for name in fixture_names),
        'rejected': any('rejected' in name for name in fixture_names),
    }
    return {
        'fixture_dir': str(FIXTURE_DIR),
        'fixtures': fixture_names,
        'canonical_states': canonical_states,
        'covered_states': covered,
        'uncovered_states': [state for state, is_covered in covered.items() if not is_covered],
    }


def describe_venue_assumptions() -> Dict[str, Any]:
    fixture_coverage = get_fixture_coverage_summary()
    return {
        'placeholder_helpers': {
            'place_limit_order': True,
            'place_marketable_order': True,
            'get_order_status': True,
            'cancel_order': True,
            'post_heartbeat': True,
            'redeem_market_onchain': True,
        },
        'assumed_unverified_endpoints': [
            'GET gamma:/markets',
            'SDK clob:post_order',
            'SDK clob:get_order',
            'SDK clob:cancel',
            'SDK clob:post_heartbeat',
        ],
        'normalized_fields_explicit_or_inferred': {
            'status': 'explicit if status/state/result present; otherwise inferred from default_status',
            'order_id': 'explicit if order_id/orderId/id present',
            'client_order_id': 'explicit if client_order_id/clientOrderId present',
            'tx_hash': 'explicit if tx_hash/txHash/transactionHash present',
            'filled_qty': 'explicit if filled_qty/filledQuantity present',
            'remaining_qty': 'explicit if remaining_qty/remainingQuantity present',
            'avg_price': 'explicit if avg_price/averagePrice/avgPrice present',
            'ok': 'inferred from dry_run or status class, never from HTTP 200 alone',
        },
        'fixture_coverage': fixture_coverage,
    }


def place_limit_order(token_id: str, side: str, qty: float, price: float, post_only: bool = True, dry_run: bool = True, client_order_id: Optional[str] = None) -> Dict[str, Any]:
    """Place a limit order on the Polymarket CLOB.

    - `token_id`: CLOB token id (YES/NO token)
    - `side`: 'buy' or 'sell'
    - `qty`: quantity (units of token)
    - `price`: price per token (0..1 for Polymarket or actual quoting asset, adapt as needed)
    - `dry_run`: if True, do not send network request; return constructed payload and headers.

    Returns response dict or the dry-run payload.
    """
    path = '/order'
    url = POLY_CLOB_BASE.rstrip('/') + path
    method = 'POST'
    compiled = build_sdk_order_args(
        token_id=token_id,
        side=side,
        quantity=qty,
        limit_price=price,
        order_type='GTC',
        post_only=post_only,
        marketable=False,
        client_order_id=client_order_id,
    )
    if compiled.get('skipped'):
        return normalize_client_response(
            {**compiled.get('quantization', {}), 'status': 'skipped_invalid_quantized_order', 'reason': compiled.get('skip_reason')},
            default_status='skipped_invalid_quantized_order',
            ok_statuses=set(),
        )
    body = compiled['request_body']
    if dry_run or not LIVE:
        return normalize_client_response({'dry_run': True, 'url': url, 'method': method, 'body': body}, default_status='dry_run', ok_statuses={'dry_run'})

    # POST /order payload details remain provisional until full signed-order generation
    # and venue response capture are confirmed against real py-clob-client traffic.
    raw_probe = _sdk_probe_template(method, path, body)
    try:
        get_clob_api_creds()
        client = get_clob_client()
        order = client.create_order(compiled['order_args'])
        result = client.post_order(order, orderType=compiled['order_type'], post_only=compiled['post_only'])
        raw_probe['ok'] = True
        raw_probe['response_json'] = result if isinstance(result, dict) else {'result': result}
        return _normalize_sdk_result(result, default_status='submitted', raw_probe=raw_probe, client_order_id=client_order_id)
    except Exception as exc:
        return _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe, client_order_id=client_order_id)


def place_marketable_order(token_id: str, side: str, qty: float, limit_price: Optional[float] = None, order_type: str = 'FAK', dry_run: bool = True, retries: int = 1, client_order_id: Optional[str] = None, compiled_intent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Place a marketable limit order (FAK/IOC) - best-effort fill.

    Clamps limit_price into safe bounds and constructs a signed order payload.
    Returns dry-run payload when dry_run True or LIVE is False.
    """
    px = 0.99 if limit_price is None and side.lower() == 'buy' else 0.01 if limit_price is None else float(limit_price)
    path = '/order'
    url = POLY_CLOB_BASE.rstrip('/') + path
    method = 'POST'
    compiled = compiled_intent or build_marketable_order_intent(
        token_id=token_id,
        side=side,
        quantity=qty,
        limit_price=px,
        order_type=order_type,
        client_order_id=client_order_id,
    )
    if compiled.get('skipped'):
        return normalize_client_response(
            {
                'status': 'skipped_invalid_quantized_order',
                'reason': compiled.get('skip_reason'),
                'client_order_id': client_order_id,
                'quantization': compiled.get('quantization'),
                **(compiled.get('quantization') or {}),
            },
            default_status='skipped_invalid_quantized_order',
            ok_statuses=set(),
        )
    body = compiled['request_body']
    if dry_run or not LIVE:
        return normalize_client_response(
            {
                'dry_run': True,
                'url': url,
                'method': method,
                'body': body,
                'quantization': compiled.get('quantization'),
                'venue_intent_mode': compiled.get('venue_intent_mode'),
                'submitted_qty': compiled.get('submitted_qty'),
                'submitted_notional': compiled.get('submitted_notional'),
                'submitted_amount': compiled.get('submitted_notional') if compiled.get('venue_intent_mode') == 'amount' else None,
                **(compiled.get('quantization') or {}),
            },
            default_status='dry_run',
            ok_statuses={'dry_run'},
        )

    last_result = None
    for attempt in range(retries):
        raw_probe = _sdk_probe_template(
            method,
            path,
            body,
            logical_params={'attempt': attempt + 1, 'venue_intent_mode': compiled.get('venue_intent_mode')},
        )
        try:
            creds = get_clob_api_creds()
            client = get_clob_client()
            if compiled.get('venue_intent_mode') == 'amount':
                if MarketOrderArgs is None or OrderType is None:
                    raise RuntimeError('py-clob-client market order types are not available')
                order_args = MarketOrderArgs(
                    token_id=str(token_id),
                    amount=float(compiled['submitted_notional']),
                    side=str(side or '').upper(),
                    price=float(compiled['request_body']['price']),
                    order_type=getattr(OrderType, compiled['order_type'], compiled['order_type']),
                )
                order = client.create_market_order(order_args)
            else:
                sdk_compiled = build_sdk_order_args(
                    token_id=token_id,
                    side=side,
                    quantity=compiled['submitted_qty'],
                    limit_price=compiled['request_body']['price'],
                    order_type=order_type,
                    post_only=False,
                    marketable=True,
                    client_order_id=client_order_id,
                )
                order = client.create_order(sdk_compiled['order_args'])
            raw_probe['request_body'] = _serialize_post_order_body(order, getattr(creds, 'api_key', None), getattr(OrderType, compiled['order_type'], compiled['order_type']) if OrderType is not None else compiled['order_type'], compiled['post_only'], body)
            result = client.post_order(order, orderType=compiled['order_type'], post_only=compiled['post_only'])
            raw_probe['ok'] = True
            raw_probe['response_json'] = result if isinstance(result, dict) else {'result': result}
            normalized = _normalize_sdk_result(result, default_status='submitted', raw_probe=raw_probe, client_order_id=client_order_id)
        except Exception as exc:
            normalized = _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe, client_order_id=client_order_id)
        normalized['quantization'] = compiled.get('quantization')
        normalized.update(compiled.get('quantization') or {})
        normalized['venue_intent_mode'] = compiled.get('venue_intent_mode')
        normalized['submitted_qty'] = compiled.get('submitted_qty')
        normalized['submitted_notional'] = compiled.get('submitted_notional')
        normalized['submitted_amount'] = compiled.get('submitted_notional') if compiled.get('venue_intent_mode') == 'amount' else None
        last_result = normalized
        if normalized.get('status') not in ('unknown', 'error') or normalized.get('http_status') in (400, 401, 403, 404, 422):
            return normalized
    return last_result or normalize_client_response({'status': 'unknown', 'reason': 'no response', 'raw_probe': None}, default_status='unknown', ok_statuses=set())


def _get_redeem_web3():
    if Web3 is None:
        raise RuntimeError('web3 is not installed')
    rpc_url = os.getenv('POLYGON_RPC') or POLYGON_RPC
    if not rpc_url:
        raise RuntimeError('POLYGON_RPC not configured')
    timeout_sec = float(os.getenv('POLYGON_RPC_TIMEOUT_SEC', '10'))
    provider = Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': timeout_sec})
    return _inject_poa_middleware(Web3(provider))


def _normalize_redeem_payload(payload: Dict[str, Any], *, default_status: str = 'unknown') -> Dict[str, Any]:
    return normalize_client_response(payload, default_status=default_status, ok_statuses={'ok', 'success', 'dry_run'})


def redeem_market_onchain(
    market_id: str,
    dry_run: bool = True,
    *,
    condition_id: Optional[str] = None,
    redeemable_qty: Optional[float] = None,
    winning_outcome: Optional[str] = None,
    collateral_token_address: Optional[str] = None,
    conditional_tokens_address: Optional[str] = None,
    parent_collection_id: Optional[str] = None,
    index_sets: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Redeem resolved CTF positions directly on Polygon via ConditionalTokens.redeemPositions."""

    path_used = 'direct_onchain'
    redeemable_qty = None if redeemable_qty is None else float(redeemable_qty)
    payload: Dict[str, Any] = {
        'market_id': market_id,
        'condition_id': condition_id,
        'winning_outcome': winning_outcome,
        'redeemable_qty': redeemable_qty,
        'redeemed_qty': 0.0,
        'tx_hash': None,
        'path_used': path_used,
    }
    if redeemable_qty is not None and redeemable_qty <= 0:
        return _normalize_redeem_payload(
            {**payload, 'status': 'skipped', 'skip_reason': 'no_redeemable_qty', 'error_message': 'no redeemable quantity'},
            default_status='skipped',
        )

    try:
        condition_bytes = _condition_id_to_bytes(condition_id)
    except Exception as exc:
        return _normalize_redeem_payload(
            {**payload, 'status': 'skipped', 'skip_reason': 'missing_condition_id', 'error_message': str(exc)},
            default_status='skipped',
        )
    if condition_bytes is None:
        return _normalize_redeem_payload(
            {**payload, 'status': 'skipped', 'skip_reason': 'missing_condition_id', 'error_message': 'condition_id is required'},
            default_status='skipped',
        )

    private_key = POLY_WALLET_PRIVATE_KEY or os.getenv('POLY_WALLET_PRIVATE_KEY')
    wallet_address = WALLET_ADDRESS or _wallet_address_from_privkey(private_key) or os.getenv('POLY_WALLET_ADDRESS')
    ctf_address = conditional_tokens_address or DEFAULT_CTF_CONTRACT_ADDRESS
    collateral_address = collateral_token_address or DEFAULT_USDC_E_TOKEN_ADDRESS
    parent_collection = parent_collection_id or ('0x' + '00' * 32)
    index_sets = list(index_sets or [1, 2])
    request_body = {
        'contract': ctf_address,
        'collateral_token': collateral_address,
        'parent_collection_id': parent_collection,
        'condition_id': condition_id,
        'index_sets': index_sets,
        'chain_id': POLYGON_CHAIN_ID,
        'wallet_address': wallet_address,
    }
    if dry_run:
        return _normalize_redeem_payload(
            {
                **payload,
                'status': 'dry_run',
                'dry_run': True,
                'redeemed_qty': redeemable_qty,
                'body': request_body,
            },
            default_status='dry_run',
        )

    if not private_key or not wallet_address or not ctf_address or not collateral_address:
        return _normalize_redeem_payload(
            {
                **payload,
                'status': 'skipped',
                'skip_reason': 'missing_contract_config',
                'error_message': 'wallet private key, wallet address, CTF contract, and collateral token must be configured',
                'wallet_address': wallet_address,
                'conditional_tokens_address': ctf_address,
                'collateral_token_address': collateral_address,
            },
            default_status='skipped',
        )

    try:
        web3 = _get_redeem_web3()
        checksum_wallet = _checksum_address(wallet_address)
        contract = web3.eth.contract(address=_checksum_address(ctf_address), abi=_CTF_ABI)
        payout_denominator = int(contract.functions.payoutDenominator(condition_bytes).call())
        if payout_denominator <= 0:
            return _normalize_redeem_payload(
                {
                    **payload,
                    'status': 'skipped',
                    'skip_reason': 'condition_not_resolved_onchain',
                    'error_message': 'condition payout not finalized onchain',
                    'body': request_body,
                },
                default_status='skipped',
            )
        nonce = web3.eth.get_transaction_count(checksum_wallet)
        txn = contract.functions.redeemPositions(
            _checksum_address(collateral_address),
            _condition_id_to_bytes(parent_collection),
            condition_bytes,
            [int(item) for item in index_sets],
        ).build_transaction(
            {
                'from': checksum_wallet,
                'nonce': nonce,
                'chainId': POLYGON_CHAIN_ID,
            }
        )
        if 'gas' not in txn:
            txn['gas'] = int(contract.functions.redeemPositions(
                _checksum_address(collateral_address),
                _condition_id_to_bytes(parent_collection),
                condition_bytes,
                [int(item) for item in index_sets],
            ).estimate_gas({'from': checksum_wallet}) * 1.2)
        has_dynamic_fee = (
            txn.get('maxFeePerGas') is not None
            or txn.get('maxPriorityFeePerGas') is not None
        )
        if has_dynamic_fee:
            txn.pop('gasPrice', None)
        elif 'gasPrice' not in txn:
            txn['gasPrice'] = int(web3.eth.gas_price)
        signed = web3.eth.account.sign_transaction(txn, private_key=private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, 'hex') else str(tx_hash)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=REDEEM_RECEIPT_TIMEOUT_SEC)
        receipt_status = receipt.get('status') if isinstance(receipt, dict) else getattr(receipt, 'status', None)
        if int(receipt_status or 0) != 1:
            return _normalize_redeem_payload(
                {
                    **payload,
                    'status': 'reverted',
                    'tx_hash': tx_hash_hex,
                    'error_message': 'tx_reverted',
                    'error_reason': 'tx_reverted',
                    'body': request_body,
                },
                default_status='reverted',
            )
        return _normalize_redeem_payload(
            {
                **payload,
                'status': 'ok',
                'tx_hash': tx_hash_hex,
                'redeemed_qty': redeemable_qty,
                'body': request_body,
            },
            default_status='ok',
        )
    except Exception as exc:
        return _normalize_redeem_payload(
            {
                **payload,
                'status': 'error',
                'error_message': str(exc),
                'error_reason': 'rpc_or_submission_failure',
                'body': request_body,
            },
            default_status='error',
        )


def get_order_status(order_id: Optional[str] = None, client_order_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    if order_id is None and client_order_id is not None:
        return normalize_client_response(
            {
                'status': 'unknown',
                'error_message': 'client_order_id lookup is provisional; documented CLOB lookup uses venue order id',
                'client_order_id': client_order_id,
                'http_status': None,
                'raw_probe': None,
            },
            default_status='unknown',
            ok_statuses=set(),
        )
    path = f'/order/{order_id}' if order_id is not None else '/order/'
    url = POLY_CLOB_BASE.rstrip('/') + path
    if dry_run:
        return normalize_client_response({'dry_run': True, 'url': url, 'method': 'GET', 'body': None}, default_status='dry_run', ok_statuses={'dry_run'})
    raw_probe = _sdk_probe_template('GET', path, None)
    try:
        get_clob_api_creds()
        result = get_clob_client().get_order(order_id)
        raw_probe['ok'] = True
        raw_probe['response_json'] = result if isinstance(result, dict) else {'result': result}
        if result is None:
            return normalize_client_response(
                {
                    'status': 'not_found_on_venue',
                    'reason': 'venue returned null for order lookup',
                    'order_id': order_id,
                    'client_order_id': client_order_id,
                    'result': None,
                    'raw_probe': raw_probe,
                },
                default_status='unknown',
                ok_statuses=set(),
            )
        return _normalize_sdk_result(result, default_status='unknown', raw_probe=raw_probe)
    except Exception as exc:
        return _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe)


def cancel_order(order_id: Optional[str] = None, client_order_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    path = '/order'
    url = POLY_CLOB_BASE.rstrip('/') + path
    body = {'orderID': order_id} if order_id is not None else {'clientOrderId': client_order_id}
    if dry_run:
        return normalize_client_response({'dry_run': True, 'url': url, 'method': 'DELETE', 'body': body}, default_status='dry_run', ok_statuses={'dry_run'})
    raw_probe = _sdk_probe_template('DELETE', path, body)
    try:
        get_clob_api_creds()
        if order_id is None:
            return normalize_client_response(
                {
                    'status': 'unknown',
                    'error_message': 'client_order_id cancel path is provisional; sdk cancel uses venue order id',
                    'client_order_id': client_order_id,
                    'raw_probe': raw_probe,
                },
                default_status='unknown',
                ok_statuses=set(),
            )
        result = get_clob_client().cancel(order_id)
        raw_probe['ok'] = True
        raw_probe['response_json'] = result if isinstance(result, dict) else {'result': result}
        return _normalize_sdk_result(result, default_status='unknown', raw_probe=raw_probe, client_order_id=client_order_id)
    except Exception as exc:
        return _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe, client_order_id=client_order_id)


def get_user_trades(market_id: Optional[str] = None, token_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    path = '/trades'
    url = POLY_CLOB_BASE.rstrip('/') + path
    params = {'market': market_id, 'asset_id': token_id}
    if dry_run:
        return normalize_client_response({'dry_run': True, 'url': url, 'method': 'GET', 'params': params, 'trades': []}, default_status='dry_run', ok_statuses={'dry_run'})
    raw_probe = _sdk_probe_template('GET', path, None, logical_params=params)
    try:
        get_clob_api_creds()
        if TradeParams is None:
            raise RuntimeError('py-clob-client TradeParams is not available')
        result = get_clob_client().get_trades(TradeParams(market=market_id, asset_id=token_id))
        raw_probe['ok'] = True
        raw_probe['response_json'] = {'data': result} if isinstance(result, list) else result
        return normalize_client_response(
            {
                'status': 'ok',
                'trades': result if isinstance(result, list) else (result.get('data') if isinstance(result, dict) else []),
                'raw_probe': raw_probe,
            },
            default_status='ok',
            ok_statuses={'ok'},
        )
    except Exception as exc:
        return _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe)


def post_heartbeat(heartbeat_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    path = '/heartbeats'
    url = POLY_CLOB_BASE.rstrip('/') + path
    body = {'heartbeat_id': heartbeat_id or ''}
    if dry_run:
        return normalize_client_response({'dry_run': True, 'url': url, 'method': 'POST', 'body': body}, default_status='dry_run', ok_statuses={'dry_run'})
    raw_probe = _sdk_probe_template('POST', path, body)
    try:
        get_clob_api_creds()
        result = get_clob_client().post_heartbeat(heartbeat_id or '')
        raw_probe['ok'] = True
        raw_probe['response_json'] = result if isinstance(result, dict) else {'result': result}
        normalized = _normalize_sdk_result(result, default_status='ok', raw_probe=raw_probe)
        normalized['heartbeat_id'] = normalized.get('heartbeat_id') or normalized.get('raw', {}).get('heartbeat_id')
        return normalized
    except Exception as exc:
        return _normalize_sdk_exception(exc, default_status='unknown', raw_probe=raw_probe)


def get_tx_receipt(tx_hash: str) -> Optional[Dict[str, Any]]:
    """Fetch a transaction receipt from a configured Ethereum JSON-RPC endpoint.

    If `POLYGON_RPC` env var is set, perform `eth_getTransactionReceipt`. Otherwise
    return None (caller may mock this in tests).
    """
    POLYGON_RPC = os.getenv('POLYGON_RPC')
    if not POLYGON_RPC:
        return None
    payload = {
        'jsonrpc': '2.0',
        'method': 'eth_getTransactionReceipt',
        'params': [tx_hash],
        'id': 1
    }
    try:
        r = requests.post(POLYGON_RPC, json=payload, timeout=10)
        r.raise_for_status()
        resp = r.json()
        return resp.get('result')
    except Exception:
        return None


def _hex_to_int(x: str) -> int:
    if x is None:
        return 0
    if x.startswith('0x'):
        x = x[2:]
    if x == '':
        return 0
    return int(x, 16)


def _hex_to_addr(x: str) -> str:
    if not x:
        return None
    if x.startswith('0x'):
        x = x[2:]
    # address is last 20 bytes (40 hex chars)
    return '0x' + x[-40:]


def _decode_uint256(data_hex: str, offset: int) -> int:
    # data_hex without 0x
    if data_hex.startswith('0x'):
        data_hex = data_hex[2:]
    start = offset * 64
    word = data_hex[start:start + 64]
    return int(word, 16)


def _decode_dynamic_uint256_array(data_hex: str, start_offset_word: int) -> list:
    # start_offset_word is the word index where the dynamic array offset is stored (usually 0 or 1)
    if data_hex.startswith('0x'):
        data_hex = data_hex[2:]
    # read the offset pointer (in bytes) at the given start_offset_word
    ptr_word = data_hex[start_offset_word * 64:(start_offset_word + 1) * 64]
    ptr = int(ptr_word, 16)
    # convert ptr (bytes) to word index
    ptr_index = ptr // 32
    # read length at ptr_index
    length_word = data_hex[ptr_index * 64:(ptr_index + 1) * 64]
    length = int(length_word, 16)
    vals = []
    for i in range(length):
        w = data_hex[(ptr_index + 1 + i) * 64:(ptr_index + 2 + i) * 64]
        vals.append(int(w, 16))
    return vals


def _try_decode_erc1155_log(log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Accept either already-decoded logs (with 'event' and 'args') or raw logs with 'topics' and 'data'
    if log.get('event') and log.get('args'):
        return log
    topics = log.get('topics') or []
    data = log.get('data')
    if not topics or data is None:
        return None
    # require topic0 to be one of the ERC-1155 event signatures
    # verify topic0 matches canonical ERC-1155 transfer signatures
    # require precomputed signature constants; if unavailable skip raw decoding
    if TRANSFER_SINGLE_SIG is None or TRANSFER_BATCH_SIG is None:
        return None
    t0 = topics[0]
    if t0 != TRANSFER_SINGLE_SIG and t0 != TRANSFER_BATCH_SIG:
        # not an ERC-1155 transfer event
        return None
    # require at least 4 topics: [sig, operator, from, to]
    if len(topics) < 4:
        return None
    operator = _hex_to_addr(topics[1])
    from_addr = _hex_to_addr(topics[2])
    to_addr = _hex_to_addr(topics[3])

    # TransferSingle likely: data is two uint256 words (id, value)
    # TransferBatch: data contains two dynamic arrays (ids, values)
    # Heuristic: if data length == 2 words (64 bytes each) -> single
    data_no0x = data[2:] if data.startswith('0x') else data
    if len(data_no0x) == 64 * 2:
        # single
        tid = int(data_no0x[0:64], 16)
        val = int(data_no0x[64:128], 16)
        return {'event': 'TransferSingle', 'args': {'operator': operator, 'from': from_addr, 'to': to_addr, 'id': str(tid), 'value': float(val)}}
    else:
        # attempt to decode as batch: two dynamic arrays; offsets at words 0 and 1
        try:
            ids = _decode_dynamic_uint256_array(data, 0)
            vals = _decode_dynamic_uint256_array(data, 1)
            ids = [str(i) for i in ids]
            vals = [float(v) for v in vals]
            return {'event': 'TransferBatch', 'args': {'operator': operator, 'from': from_addr, 'to': to_addr, 'ids': ids, 'values': vals}}
        except Exception:
            # unable to decode
            return {'decode_failed': True, 'raw': log}


if __name__ == '__main__':
    # quick dry-run example (will not place live unless LIVE env is true)
    print(place_limit_order('TOKEN_YES', 'buy', 1.0, 0.55, dry_run=True))
