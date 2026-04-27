import json
import os
import sys
from pathlib import Path
import base64
import hashlib
import hmac
from types import SimpleNamespace

import src.polymarket_client as polymarket_client
import src.storage as storage
from scripts import probe_polymarket_venue


def setup_function(_fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    for key in (
        "BOT_DB_PATH",
        "PROBE_ENABLE_WRITE",
        "PROBE_MAX_NOTIONAL_USDC",
        "PROBE_TOKEN_ID",
        "PROBE_MARKET_ID",
        "PROBE_OUTCOME_SIDE",
        "LIVE",
        "POLY_API_SECRET",
        "POLY_API_PASSPHRASE",
        "POLY_WALLET_PRIVATE_KEY",
        "POLYGON_RPC",
    ):
        os.environ.pop(key, None)
    polymarket_client._CLOB_CLIENT = None
    polymarket_client._CLOB_API_CREDS = None
    storage.ensure_db()


def test_redaction_and_raw_http_probe_preserve_error_body(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.status_code = 401
            self.ok = False
            self.text = '{"error":"bad auth"}'
            self.headers = {"Content-Type": "application/json"}

    def fake_request(method, url, json=None, params=None, headers=None, timeout=None):
        return FakeResponse()

    monkeypatch.setattr(polymarket_client.requests, "request", fake_request)
    monkeypatch.setattr(polymarket_client, "POLY_API_SECRET", "secret-value")
    monkeypatch.setattr(polymarket_client, "POLY_API_PASSPHRASE", "passphrase-value")
    result = polymarket_client.raw_http_probe("GET", "/order/x", body=None, auth=True, timeout=1, base_url=polymarket_client.POLY_CLOB_BASE)

    assert result["http_status"] == 401
    assert result["response_text"] == '{"error":"bad auth"}'
    assert result["response_json"]["error"] == "bad auth"
    assert result["request_headers"]["POLY_SIGNATURE"] == "[redacted]"
    assert result["request_headers"]["POLY_PASSPHRASE"] == "[redacted]"
    assert "PM-ACCESS-SIGN" not in result["request_headers"]
    assert "PM-WALLET-SIGN" not in result["request_headers"]


def test_build_auth_headers_use_poly_l2_headers(monkeypatch):
    monkeypatch.setattr(polymarket_client, "POLY_API_KEY", "key-1")
    monkeypatch.setattr(polymarket_client, "POLY_API_PASSPHRASE", "pass-1")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xabc")
    monkeypatch.setattr(polymarket_client, "POLY_API_SECRET", base64.b64encode(b"secret-bytes").decode())
    headers = polymarket_client.build_auth_headers("POST", "/order", {"a": 1})
    assert headers["Content-Type"] == "application/json"
    assert headers["POLY_ADDRESS"] == "0xabc"
    assert headers["POLY_API_KEY"] == "key-1"
    assert headers["POLY_PASSPHRASE"] == "pass-1"
    assert headers["POLY_TIMESTAMP"]
    assert headers["POLY_SIGNATURE"]
    assert "PM-ACCESS-KEY" not in headers
    assert "PM-ACCESS-SIGN" not in headers
    assert "PM-WALLET-ADDRESS" not in headers
    assert "PM-WALLET-SIGN" not in headers


def test_sign_base64_decodes_secret_before_hmac():
    secret_bytes = b"secret-bytes"
    secret_b64 = base64.b64encode(secret_bytes).decode()
    prehash = "123POST/order{}"
    expected = base64.b64encode(hmac.new(secret_bytes, prehash.encode(), hashlib.sha256).digest()).decode()
    assert polymarket_client._sign(secret_b64, prehash) == expected


def test_serialize_body_for_hmac_does_not_double_encode_strings():
    assert polymarket_client._serialize_body_for_hmac(None) == ""
    assert polymarket_client._serialize_body_for_hmac('{"a":1}') == '{"a":1}'
    assert polymarket_client._serialize_body_for_hmac({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_probe_runner_persists_events_and_skips_live_write_unless_enabled(monkeypatch, capsys):
    calls = []
    sdk_calls = []

    def fake_http_probe(method, path, body=None, params=None, auth=True, timeout=10, base_url=None):
        calls.append((method, path, auth, base_url))
        return {
            "method": method,
            "path": path,
            "url": f"{base_url or 'https://example.test'}{path}",
            "params": params,
            "request_headers": {"Content-Type": "application/json"},
            "request_body": body,
            "http_status": 400 if auth else 200,
            "ok": False if auth else True,
            "latency_ms": 12.0,
            "response_headers": {"Content-Type": "application/json"},
            "response_text": '{"detail":"bad payload"}' if auth else '{"items":[]}',
            "response_json": {"detail": "bad payload"} if auth else {"items": []},
            "error_text": None,
        }

    def fake_get_order_status(order_id=None, client_order_id=None, dry_run=True):
        sdk_calls.append(("get_order_status", order_id, client_order_id, dry_run))
        return {
            "ok": False,
            "status": "error",
            "order_id": order_id,
            "raw_probe": {
                "method": "GET",
                "path": f"/order/{order_id}",
                "url": f"https://clob.polymarket.com/order/{order_id}",
                "params": None,
                "request_headers": None,
                "request_body": None,
                "http_status": 404,
                "ok": False,
                "latency_ms": 12.0,
                "response_headers": None,
                "response_text": '{"detail":"bad payload"}',
                "response_json": {"detail": "bad payload"},
                "error_text": None,
            },
        }

    def fake_cancel_order(order_id=None, client_order_id=None, dry_run=True):
        sdk_calls.append(("cancel_order", order_id, client_order_id, dry_run))
        return {
            "ok": False,
            "status": "error",
            "order_id": order_id,
            "client_order_id": client_order_id,
            "raw_probe": {
                "method": "DELETE",
                "path": "/order",
                "url": "https://clob.polymarket.com/order",
                "params": None,
                "request_headers": None,
                "request_body": {"orderID": order_id} if order_id else {"clientOrderId": client_order_id},
                "http_status": 400,
                "ok": False,
                "latency_ms": 12.0,
                "response_headers": None,
                "response_text": '{"detail":"bad payload"}',
                "response_json": {"detail": "bad payload"},
                "error_text": None,
            },
        }

    def fake_post_heartbeat(heartbeat_id=None, dry_run=True):
        sdk_calls.append(("post_heartbeat", heartbeat_id, dry_run))
        return {
            "ok": False,
            "status": "error",
            "heartbeat_id": heartbeat_id,
            "raw_probe": {
                "method": "POST",
                "path": "/heartbeats",
                "url": "https://clob.polymarket.com/heartbeats",
                "params": None,
                "request_headers": None,
                "request_body": {"heartbeat_id": heartbeat_id},
                "http_status": 400,
                "ok": False,
                "latency_ms": 12.0,
                "response_headers": None,
                "response_text": '{"detail":"bad payload"}',
                "response_json": {"detail": "bad payload"},
                "error_text": None,
            },
        }

    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "raw_http_probe", fake_http_probe)
    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "get_order_status", fake_get_order_status)
    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "cancel_order", fake_cancel_order)
    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "post_heartbeat", fake_post_heartbeat)
    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "POLY_GAMMA_BASE", "https://gamma-api.polymarket.com")
    monkeypatch.setattr(probe_polymarket_venue.polymarket_client, "POLY_CLOB_BASE", "https://clob.polymarket.com")
    monkeypatch.setattr(sys, "argv", ["probe_polymarket_venue.py", "--skip-ws"])

    probe_polymarket_venue.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    run_id = payload["run_id"]

    run = storage.get_probe_run(run_id)
    events = storage.list_probe_events(run_id)
    summary_steps = {step["step_name"]: step["classification"] for step in payload["summary"]["steps"]}
    assert run is not None
    assert run["write_enabled"] is False
    assert any(event["step_name"] == "public_markets" for event in events)
    assert any(event["step_name"] == "auth_order_status_bogus" for event in events)
    assert any(event["step_name"] == "auth_heartbeat" for event in events)
    assert any(event["step_name"] == "live_write_probe" for event in events)
    assert not any(event["step_name"] == "live_write_submit" for event in events)
    assert not any(event["step_name"] == "auth_redeem_bogus" for event in events)
    for step_name in ("auth_order_status_bogus", "auth_cancel_bogus", "auth_heartbeat"):
        step_events = [event for event in events if event["step_name"] == step_name]
        assert len(step_events) == 2
        assert any(event["direction"] == "request" for event in step_events)
        assert any(event["direction"] in ("response", "error") for event in step_events)
        response_event = next(event for event in step_events if event["direction"] in ("response", "error"))
        assert response_event["transport"] == "clob_sdk"
        assert response_event["response_body"] is not None
        assert response_event["classification"] in ("bad_payload", "endpoint_missing", "auth_error", "success", "unexpected_schema")
    assert ("get_order_status", "probe-bogus-order", None, False) in sdk_calls
    assert ("cancel_order", "probe-bogus-order", None, False) in sdk_calls
    assert ("post_heartbeat", "", False) in sdk_calls
    assert summary_steps["live_write_probe"] == "skipped"
    assert ("GET", "/markets", False, "https://gamma-api.polymarket.com") in calls
    assert all(call[1] != "/order/probe-bogus-order" for call in calls)
    assert all(call[1] != "/order" or call[2] is False for call in calls)
    assert all(call[1] != "/heartbeats" for call in calls)


def test_heartbeat_helper_and_redeem_helper_dry_run_is_safe(monkeypatch):
    class FakeClient:
        def post_heartbeat(self, heartbeat_id):
            assert heartbeat_id == ""
            return {"status": "ok", "heartbeat_id": "hb-1"}

    monkeypatch.setattr(polymarket_client, "get_clob_client", lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, "get_clob_api_creds", lambda: SimpleNamespace(api_key="k"))
    out = polymarket_client.post_heartbeat("", dry_run=False)
    assert out["status"] == "ok"
    assert out["heartbeat_id"] == "hb-1"
    assert out["raw_probe"]["path"] == "/heartbeats"
    assert out["raw_probe"]["transport"] == "clob_sdk"

    redeem = polymarket_client.redeem_market_onchain("M1", dry_run=True, condition_id="0x" + "11" * 32, redeemable_qty=4.0, winning_outcome="YES")
    assert redeem["status"] == "dry_run"
    assert redeem["ok"] is True
    assert redeem["redeemed_qty"] == 4.0
    assert redeem["path_used"] == "direct_onchain"


def test_redeem_onchain_success_normalized(monkeypatch):
    class _TxHash:
        def hex(self):
            return "0xredeemok"

    class FakeRedeemCall:
        def call(self):
            return 1

        def build_transaction(self, tx):
            return {**tx, "to": "0xctf", "data": "0xabc"}

        def estimate_gas(self, tx):
            return 100000

    class FakeFunctions:
        def payoutDenominator(self, _condition_id):
            return FakeRedeemCall()

        def redeemPositions(self, _collateral, _parent, _condition, _index_sets):
            return FakeRedeemCall()

    class FakeAccount:
        def sign_transaction(self, tx, private_key=None):
            assert tx["chainId"] == 137
            assert private_key == "priv"
            return SimpleNamespace(raw_transaction=b"signed")

    class FakeEth:
        gas_price = 123

        def __init__(self):
            self.account = FakeAccount()

        def contract(self, address=None, abi=None):
            return SimpleNamespace(functions=FakeFunctions())

        def get_transaction_count(self, wallet):
            assert wallet == "0xwallet"
            return 7

        def send_raw_transaction(self, raw):
            assert raw == b"signed"
            return _TxHash()

        def wait_for_transaction_receipt(self, tx_hash, timeout=None):
            assert timeout == polymarket_client.REDEEM_RECEIPT_TIMEOUT_SEC
            return {"status": 1}

    class FakeWeb3:
        def __init__(self):
            self.eth = FakeEth()

    monkeypatch.setattr(polymarket_client, "POLY_WALLET_PRIVATE_KEY", "priv")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xwallet")
    monkeypatch.setattr(polymarket_client, "_get_redeem_web3", lambda: FakeWeb3())

    result = polymarket_client.redeem_market_onchain(
        "M1",
        dry_run=False,
        condition_id="0x" + "22" * 32,
        redeemable_qty=5.0,
        winning_outcome="YES",
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["tx_hash"] == "0xredeemok"
    assert result["redeemable_qty"] == 5.0
    assert result["redeemed_qty"] == 5.0
    assert result["path_used"] == "direct_onchain"


def test_redeem_onchain_skips_when_condition_not_final(monkeypatch):
    class FakeSkippedCall:
        def call(self):
            return 0

    class FakeFunctions:
        def payoutDenominator(self, _condition_id):
            return FakeSkippedCall()

    class FakeEth:
        def contract(self, address=None, abi=None):
            return SimpleNamespace(functions=FakeFunctions())

    class FakeWeb3:
        def __init__(self):
            self.eth = FakeEth()

    monkeypatch.setattr(polymarket_client, "POLY_WALLET_PRIVATE_KEY", "priv")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xwallet")
    monkeypatch.setattr(polymarket_client, "_get_redeem_web3", lambda: FakeWeb3())

    result = polymarket_client.redeem_market_onchain(
        "M2",
        dry_run=False,
        condition_id="0x" + "33" * 32,
        redeemable_qty=2.0,
        winning_outcome="NO",
    )

    assert result["ok"] is False
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "condition_not_resolved_onchain"
    assert result["redeemed_qty"] == 0.0


def test_redeem_onchain_reverted_is_normalized(monkeypatch):
    class _TxHash:
        def hex(self):
            return "0xredeemrevert"

    class FakeRedeemCall:
        def call(self):
            return 1

        def build_transaction(self, tx):
            return {**tx, "to": "0xctf", "data": "0xabc"}

        def estimate_gas(self, tx):
            return 100000

    class FakeFunctions:
        def payoutDenominator(self, _condition_id):
            return FakeRedeemCall()

        def redeemPositions(self, _collateral, _parent, _condition, _index_sets):
            return FakeRedeemCall()

    class FakeAccount:
        def sign_transaction(self, tx, private_key=None):
            return SimpleNamespace(raw_transaction=b"signed")

    class FakeEth:
        gas_price = 123

        def __init__(self):
            self.account = FakeAccount()

        def contract(self, address=None, abi=None):
            return SimpleNamespace(functions=FakeFunctions())

        def get_transaction_count(self, wallet):
            return 1

        def send_raw_transaction(self, raw):
            return _TxHash()

        def wait_for_transaction_receipt(self, tx_hash, timeout=None):
            return {"status": 0}

    class FakeWeb3:
        def __init__(self):
            self.eth = FakeEth()

    monkeypatch.setattr(polymarket_client, "POLY_WALLET_PRIVATE_KEY", "priv")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xwallet")
    monkeypatch.setattr(polymarket_client, "_get_redeem_web3", lambda: FakeWeb3())

    result = polymarket_client.redeem_market_onchain(
        "M3",
        dry_run=False,
        condition_id="0x" + "44" * 32,
        redeemable_qty=1.0,
        winning_outcome="YES",
    )

    assert result["ok"] is False
    assert result["status"] == "reverted"
    assert result["tx_hash"] == "0xredeemrevert"
    assert result["error_reason"] == "tx_reverted"


def test_redeem_onchain_legacy_tx_adds_gas_price(monkeypatch):
    captured = {}

    class _TxHash:
        def hex(self):
            return "0xlegacyredeem"

    class FakeRedeemCall:
        def call(self):
            return 1

        def build_transaction(self, tx):
            return {**tx, "to": "0xctf", "data": "0xabc"}

        def estimate_gas(self, tx):
            return 100000

    class FakeFunctions:
        def payoutDenominator(self, _condition_id):
            return FakeRedeemCall()

        def redeemPositions(self, _collateral, _parent, _condition, _index_sets):
            return FakeRedeemCall()

    class FakeAccount:
        def sign_transaction(self, tx, private_key=None):
            captured["tx"] = dict(tx)
            return SimpleNamespace(raw_transaction=b"signed")

    class FakeEth:
        gas_price = 123

        def __init__(self):
            self.account = FakeAccount()

        def contract(self, address=None, abi=None):
            return SimpleNamespace(functions=FakeFunctions())

        def get_transaction_count(self, wallet):
            return 1

        def send_raw_transaction(self, raw):
            return _TxHash()

        def wait_for_transaction_receipt(self, tx_hash, timeout=None):
            return {"status": 1}

    class FakeWeb3:
        def __init__(self):
            self.eth = FakeEth()

    monkeypatch.setattr(polymarket_client, "POLY_WALLET_PRIVATE_KEY", "priv")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xwallet")
    monkeypatch.setattr(polymarket_client, "_get_redeem_web3", lambda: FakeWeb3())

    result = polymarket_client.redeem_market_onchain(
        "M4",
        dry_run=False,
        condition_id="0x" + "55" * 32,
        redeemable_qty=1.0,
        winning_outcome="YES",
    )

    assert result["status"] == "ok"
    assert captured["tx"]["gasPrice"] == 123
    assert "maxFeePerGas" not in captured["tx"]
    assert "maxPriorityFeePerGas" not in captured["tx"]


def test_redeem_onchain_dynamic_fee_tx_preserves_fee_fields_and_omits_gas_price(monkeypatch):
    captured = {}

    class _TxHash:
        def hex(self):
            return "0xdynamicredeem"

    class FakeRedeemCall:
        def call(self):
            return 1

        def build_transaction(self, tx):
            return {
                **tx,
                "to": "0xctf",
                "data": "0xabc",
                "maxFeePerGas": 200,
                "maxPriorityFeePerGas": 30,
                "gasPrice": 999,
            }

        def estimate_gas(self, tx):
            return 100000

    class FakeFunctions:
        def payoutDenominator(self, _condition_id):
            return FakeRedeemCall()

        def redeemPositions(self, _collateral, _parent, _condition, _index_sets):
            return FakeRedeemCall()

    class FakeAccount:
        def sign_transaction(self, tx, private_key=None):
            captured["tx"] = dict(tx)
            return SimpleNamespace(raw_transaction=b"signed")

    class FakeEth:
        gas_price = 123

        def __init__(self):
            self.account = FakeAccount()

        def contract(self, address=None, abi=None):
            return SimpleNamespace(functions=FakeFunctions())

        def get_transaction_count(self, wallet):
            return 1

        def send_raw_transaction(self, raw):
            return _TxHash()

        def wait_for_transaction_receipt(self, tx_hash, timeout=None):
            return {"status": 1}

    class FakeWeb3:
        def __init__(self):
            self.eth = FakeEth()

    monkeypatch.setattr(polymarket_client, "POLY_WALLET_PRIVATE_KEY", "priv")
    monkeypatch.setattr(polymarket_client, "WALLET_ADDRESS", "0xwallet")
    monkeypatch.setattr(polymarket_client, "_get_redeem_web3", lambda: FakeWeb3())

    result = polymarket_client.redeem_market_onchain(
        "M5",
        dry_run=False,
        condition_id="0x" + "66" * 32,
        redeemable_qty=1.0,
        winning_outcome="YES",
    )

    assert result["status"] == "ok"
    assert captured["tx"]["maxFeePerGas"] == 200
    assert captured["tx"]["maxPriorityFeePerGas"] == 30
    assert "gasPrice" not in captured["tx"]


def test_sdk_backed_helpers_delegate_without_raw_authenticated_http(monkeypatch):
    calls = []

    class FakeClient:
        def get_order(self, order_id):
            calls.append(("get_order", order_id))
            return {"id": order_id, "status": "open", "filledQuantity": 0.0}

        def cancel(self, order_id):
            calls.append(("cancel", order_id))
            return {"id": order_id, "status": "canceled", "remainingQuantity": 1.0}

        def create_market_order(self, order_args):
            calls.append(("create_market_order", order_args.token_id, order_args.side, order_args.price, order_args.amount))
            return SimpleNamespace(dict=lambda: {"makerAmount": "88000000", "takerAmount": "200000000", "side": "BUY"})

        def create_order(self, order_args):
            calls.append(("create_order", order_args.token_id, order_args.side, order_args.price, order_args.size))
            return {"signed": True, "token_id": order_args.token_id}

        def post_order(self, order, orderType=None, post_only=False):
            calls.append(("post_order", orderType, post_only))
            return {"orderID": "OID1", "status": "accepted", "fills": [{"id": "F1", "price": 0.44, "size": 2.0}]}

        def post_heartbeat(self, heartbeat_id):
            calls.append(("post_heartbeat", heartbeat_id))
            return {"status": "ok", "heartbeat_id": "hb-2"}

    monkeypatch.setattr(polymarket_client, "get_clob_client", lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, "get_clob_api_creds", lambda: SimpleNamespace(api_key="k"))
    monkeypatch.setattr(polymarket_client, "LIVE", True)
    monkeypatch.setattr(polymarket_client, "raw_http_probe", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw_http_probe should not be used for authenticated helpers")))

    status = polymarket_client.get_order_status(order_id="OID1", dry_run=False)
    cancel = polymarket_client.cancel_order(order_id="OID1", dry_run=False)
    placed = polymarket_client.place_marketable_order("TOKEN1", "buy", 2.3, limit_price=0.44, dry_run=False, client_order_id="CID1")
    hb = polymarket_client.post_heartbeat("", dry_run=False)

    assert status["order_id"] == "OID1"
    assert cancel["status"] == "canceled"
    assert placed["order_id"] == "OID1"
    assert placed["client_order_id"] == "CID1"
    assert placed["fills"][0]["id"] == "F1"
    assert hb["heartbeat_id"] == "hb-2"
    assert ("get_order", "OID1") in calls
    assert ("cancel", "OID1") in calls
    assert ("create_market_order", "TOKEN1", "BUY", 0.44, 1.01) in calls
    assert ("post_heartbeat", "") in calls


def test_sdk_exception_returns_structured_error(monkeypatch):
    class FakeClient:
        def post_heartbeat(self, heartbeat_id):
            raise polymarket_client.PolyApiException(error_msg="boom")

    monkeypatch.setattr(polymarket_client, "get_clob_client", lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, "get_clob_api_creds", lambda: SimpleNamespace(api_key="k"))
    monkeypatch.setattr(polymarket_client, "LIVE", True)
    result = polymarket_client.post_heartbeat("", dry_run=False)
    assert result["ok"] is False
    assert result["status"] == "unknown"
    assert "boom" in result["error_message"]


def test_sdk_null_order_lookup_maps_to_not_found_on_venue(monkeypatch):
    class FakeClient:
        def get_order(self, order_id):
            return {"result": None}

    monkeypatch.setattr(polymarket_client, "get_clob_client", lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, "get_clob_api_creds", lambda: SimpleNamespace(api_key="k"))
    monkeypatch.setattr(polymarket_client, "LIVE", True)

    result = polymarket_client.get_order_status(order_id="OID-MISSING", dry_run=False)

    assert result["ok"] is False
    assert result["status"] == "not_found_on_venue"
    assert result["error_message"] == "venue returned null for order lookup"


def test_sdk_order_lookup_transport_failure_remains_unknown(monkeypatch):
    class FakeClient:
        def get_order(self, order_id):
            raise polymarket_client.PolyApiException(error_msg="auth failed")

    monkeypatch.setattr(polymarket_client, "get_clob_client", lambda: FakeClient())
    monkeypatch.setattr(polymarket_client, "get_clob_api_creds", lambda: SimpleNamespace(api_key="k"))
    monkeypatch.setattr(polymarket_client, "LIVE", True)

    result = polymarket_client.get_order_status(order_id="OID1", dry_run=False)

    assert result["ok"] is False
    assert result["status"] == "unknown"
    assert "auth failed" in result["error_message"]
