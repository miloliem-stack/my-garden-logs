import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import polymarket_client, wallet_state


class _FakeOnion:
    def __init__(self):
        self.calls = []

    def inject(self, middleware, layer=0):
        self.calls.append((middleware, layer))


class _FakeWeb3Instance:
    def __init__(self):
        self.middleware_onion = _FakeOnion()


def test_inject_poa_middleware_uses_v7_extradata_middleware(monkeypatch):
    fake_module = types.SimpleNamespace(ExtraDataToPOAMiddleware="v7_middleware")

    def fake_import(name):
        assert name == "web3.middleware"
        return fake_module

    monkeypatch.setattr(polymarket_client.importlib, "import_module", fake_import)
    web3 = _FakeWeb3Instance()

    result = polymarket_client._inject_poa_middleware(web3)

    assert result is web3
    assert web3.middleware_onion.calls == [("v7_middleware", 0)]


def test_inject_poa_middleware_falls_back_to_v6_geth_poa(monkeypatch):
    fake_module = types.SimpleNamespace(geth_poa_middleware="v6_middleware")

    def fake_import(name):
        assert name == "web3.middleware"
        return fake_module

    monkeypatch.setattr(polymarket_client.importlib, "import_module", fake_import)
    web3 = _FakeWeb3Instance()

    result = polymarket_client._inject_poa_middleware(web3)

    assert result is web3
    assert web3.middleware_onion.calls == [("v6_middleware", 0)]


def test_get_redeem_web3_calls_poa_injection(monkeypatch):
    class FakeWeb3:
        HTTPProvider = staticmethod(lambda url, request_kwargs=None: {"url": url, "request_kwargs": request_kwargs})

        def __init__(self, provider):
            self.provider = provider

    injected = []

    def fake_inject(web3):
        injected.append(web3)
        return web3

    monkeypatch.setenv("POLYGON_RPC", "https://polygon.example")
    monkeypatch.setattr(polymarket_client, "Web3", FakeWeb3)
    monkeypatch.setattr(polymarket_client, "_inject_poa_middleware", fake_inject)

    web3 = polymarket_client._get_redeem_web3()

    assert injected == [web3]
    assert web3.provider["url"] == "https://polygon.example"
    assert web3.provider["request_kwargs"] == {"timeout": 10.0}


def test_wallet_balance_reader_calls_poa_injection(monkeypatch):
    class FakeContractFunctions:
        @staticmethod
        def balanceOf(_wallet):
            return type("Call", (), {"call": staticmethod(lambda: 10_500_000)})()

        @staticmethod
        def decimals():
            return type("Call", (), {"call": staticmethod(lambda: 6)})()

    class FakeContract:
        functions = FakeContractFunctions()

    class FakeEth:
        @staticmethod
        def get_balance(_wallet):
            return 2 * 10**18

        @staticmethod
        def contract(address=None, abi=None):
            return FakeContract()

    class FakeWeb3:
        HTTPProvider = staticmethod(lambda url, request_kwargs=None: {"url": url, "request_kwargs": request_kwargs})

        def __init__(self, provider):
            self.provider = provider
            self.eth = FakeEth()

        @staticmethod
        def to_checksum_address(address):
            return address

        @staticmethod
        def from_wei(value, unit):
            assert unit == "ether"
            return value / 10**18

    injected = []

    def fake_inject(web3):
        injected.append(web3)
        return web3

    monkeypatch.setenv("POLYGON_RPC", "https://polygon.example")
    monkeypatch.setattr(wallet_state, "Web3", FakeWeb3)
    monkeypatch.setattr(wallet_state.polymarket_client, "_inject_poa_middleware", fake_inject)

    usdc, pol = wallet_state._read_onchain_balances("0xabc")

    assert injected and injected[0].provider["url"] == "https://polygon.example"
    assert usdc == 10.5
    assert pol == 2.0
