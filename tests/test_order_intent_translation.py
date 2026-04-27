from types import SimpleNamespace

from src import polymarket_client


def setup_function(_fn):
    polymarket_client._CLOB_CLIENT = None
    polymarket_client._CLOB_API_CREDS = None


def test_buy_limit_compilation():
    compiled = polymarket_client.build_sdk_order_args(
        token_id='TOK1',
        side='buy',
        quantity=3.0,
        limit_price=0.44,
        order_type='GTC',
        post_only=False,
        marketable=False,
        client_order_id='CID1',
    )
    assert compiled['order_args'].token_id == 'TOK1'
    assert compiled['order_args'].side == 'BUY'
    assert compiled['order_args'].size == 3.0
    assert compiled['order_args'].price == 0.44
    assert compiled['request_body']['clientOrderId'] == 'CID1'
    assert compiled['marketable'] is False


def test_sell_limit_compilation():
    compiled = polymarket_client.build_sdk_order_args(
        token_id='TOK2',
        side='sell',
        quantity=2.5,
        limit_price=0.61,
        order_type='GTC',
        post_only=False,
        marketable=False,
    )
    assert compiled['order_args'].side == 'SELL'
    assert compiled['order_args'].price == 0.61
    assert compiled['request_body']['side'] == 'sell'


def test_marketable_buy_translation_uses_amount_intent():
    compiled = polymarket_client.build_marketable_order_intent(
        token_id='TOK3',
        side='buy',
        quantity=1.234567,
        limit_price=1.2,
        order_type='FAK',
    )
    assert compiled['venue_intent_mode'] == 'amount'
    assert compiled['request_body']['orderType'] == 'FAK'
    assert compiled['request_body']['marketable'] is True
    assert compiled['request_body']['price'] == 0.99
    assert compiled['request_body']['amount'] == 1.22
    assert compiled['submitted_qty'] == 1.2323
    assert compiled['submitted_notional'] == 1.22


def test_quantize_marketable_buy_floors_notional_and_shares():
    quantized = polymarket_client.quantize_order_submission(
        token_id='TOKQ1',
        side='buy',
        quantity=3.141592,
        limit_price=0.53827,
        order_type='FAK',
        marketable=True,
    )
    assert quantized['quantity'] == 3.1396
    assert quantized['effective_notional'] == 1.69
    assert quantized['effective_notional'] <= quantized['requested_notional_raw']


def test_quantize_marketable_buy_skips_zero_or_too_small_orders():
    quantized = polymarket_client.quantize_order_submission(
        token_id='TOKQ2',
        side='buy',
        quantity=0.00011,
        limit_price=0.11,
        order_type='FAK',
        marketable=True,
    )
    assert quantized['status'] == 'skipped_invalid_quantized_order'
    assert quantized['reason'] in {'zero_qty_after_quantization', 'zero_notional_after_quantization', 'skipped_below_min_market_buy_notional'}


def test_quantize_marketable_sell_floors_share_qty():
    quantized = polymarket_client.quantize_order_submission(
        token_id='TOKQ3',
        side='sell',
        quantity=4.987654,
        limit_price=0.42,
        order_type='FAK',
        marketable=True,
    )
    assert quantized['quantity'] == 4.9876
    assert quantized['effective_notional'] == 2.09


def test_post_only_compilation():
    compiled = polymarket_client.build_sdk_order_args(
        token_id='TOK4',
        side='sell',
        quantity=1.0,
        limit_price=0.55,
        order_type='GTC',
        post_only=True,
        marketable=False,
    )
    assert compiled['post_only'] is True
    assert compiled['request_body']['postOnly'] is True


def test_build_sdk_order_args_rejects_marketable_buy_path():
    try:
        polymarket_client.build_sdk_order_args(
            token_id='TOKM',
            side='buy',
            quantity=1.0,
            limit_price=0.5,
            order_type='FAK',
            post_only=False,
            marketable=True,
        )
        assert False, 'expected marketable buy sdk limit-args path to be rejected'
    except ValueError as exc:
        assert 'marketable buy orders must use' in str(exc)


def test_invalid_input_rejection():
    for kwargs in (
        {'token_id': 'TOK', 'side': 'hold', 'quantity': 1.0, 'limit_price': 0.5},
        {'token_id': 'TOK', 'side': 'buy', 'quantity': 0.0, 'limit_price': 0.5},
        {'token_id': 'TOK', 'side': 'buy', 'quantity': 1.0, 'limit_price': None},
        {'token_id': 'TOK', 'side': 'buy', 'quantity': 1.0, 'limit_price': 0.5, 'order_type': 'IOC'},
        {'token_id': 'TOK', 'side': 'buy', 'quantity': 1.0, 'limit_price': 0.5, 'order_type': 'FAK', 'post_only': True},
        {'token_id': 'TOK', 'side': 'buy', 'quantity': 1.0, 'limit_price': 0.5, 'order_type': 'GTC', 'marketable': True},
    ):
        try:
            polymarket_client.build_sdk_order_args(**kwargs)
            assert False, f'expected failure for {kwargs}'
        except ValueError:
            pass


def test_local_client_order_id_preserved_in_normalized_result():
    normalized = polymarket_client._normalize_sdk_result(
        {'orderID': 'OID1', 'status': 'accepted'},
        default_status='submitted',
        raw_probe={'method': 'POST', 'path': '/order', 'url': 'https://clob.polymarket.com/order'},
        client_order_id='LOCAL-CID',
    )
    assert normalized['client_order_id'] == 'LOCAL-CID'


def test_place_limit_order_uses_compiled_intent(monkeypatch):
    calls = {}

    class FakeClient:
        def create_order(self, order_args):
            calls['order_args'] = order_args
            return {'signed': True}

        def post_order(self, order, orderType=None, post_only=False):
            calls['order_type'] = orderType
            calls['post_only'] = post_only
            return {'orderID': 'OID-LIMIT', 'status': 'accepted'}

    monkeypatch.setattr(polymarket_client, 'LIVE', True)
    monkeypatch.setattr(polymarket_client, 'get_clob_client', lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, 'get_clob_api_creds', lambda: SimpleNamespace(api_key='k'))
    out = polymarket_client.place_limit_order('TOK5', 'buy', 2.0, 0.33, post_only=True, dry_run=False, client_order_id='CID-LIMIT')
    assert calls['order_args'].token_id == 'TOK5'
    assert calls['order_args'].side == 'BUY'
    assert calls['post_only'] is True
    assert out['client_order_id'] == 'CID-LIMIT'


def test_place_marketable_order_uses_compiled_intent(monkeypatch):
    calls = {}

    class FakeSignedOrder:
        def dict(self):
            return {'makerAmount': '82468000', 'takerAmount': '100000000', 'side': 'BUY'}

    class FakeMarketOrderArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeOrderType:
        FAK = 'FAK'

    class FakeClient:
        def create_market_order(self, order_args):
            calls['order_args'] = order_args
            return FakeSignedOrder()

        def post_order(self, order, orderType=None, post_only=False):
            calls['order_type'] = orderType
            calls['post_only'] = post_only
            return {'orderID': 'OID-MKT', 'status': 'accepted', 'clientOrderId': 'ECHO-CID'}

    monkeypatch.setattr(polymarket_client, 'LIVE', True)
    monkeypatch.setattr(polymarket_client, 'get_clob_client', lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, 'get_clob_api_creds', lambda: SimpleNamespace(api_key='k'))
    monkeypatch.setattr(polymarket_client, 'MarketOrderArgs', FakeMarketOrderArgs)
    monkeypatch.setattr(polymarket_client, 'OrderType', FakeOrderType)
    out = polymarket_client.place_marketable_order('TOK6', 'buy', 2.5, limit_price=0.41, order_type='FAK', dry_run=False, client_order_id='CID-MKT')
    assert calls['order_args'].side == 'BUY'
    assert calls['order_args'].price == 0.41
    assert calls['order_args'].amount == 1.02
    assert calls['post_only'] is False
    assert out['order_id'] == 'OID-MKT'
    assert out['client_order_id'] == 'ECHO-CID'
    assert out['venue_intent_mode'] == 'amount'
    assert out['submitted_notional'] == 1.02
    assert out['submitted_qty'] == 2.4878
    assert out['raw_probe']['request_body']['owner'] == 'k'


def test_place_marketable_sell_uses_share_quantity_path(monkeypatch):
    calls = {}

    class FakeClient:
        def create_order(self, order_args):
            calls['order_args'] = order_args
            return {'signed': True}

        def post_order(self, order, orderType=None, post_only=False):
            calls['order_type'] = orderType
            calls['post_only'] = post_only
            return {'orderID': 'OID-MKT-SELL', 'status': 'accepted', 'clientOrderId': 'ECHO-SELL'}

    monkeypatch.setattr(polymarket_client, 'LIVE', True)
    monkeypatch.setattr(polymarket_client, 'get_clob_client', lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, 'get_clob_api_creds', lambda: SimpleNamespace(api_key='k'))
    out = polymarket_client.place_marketable_order('TOK7', 'sell', 4.1234567, limit_price=0.02, order_type='FAK', dry_run=False, client_order_id='CID-SELL')
    assert calls['order_args'].side == 'SELL'
    assert calls['order_args'].price == 0.02
    assert calls['order_args'].size == 4.1234
    assert out['venue_intent_mode'] == 'quantity'
    assert out['submitted_qty'] == 4.1234
    assert out['raw_probe']['request_body']['quantity'] == 4.1234
