import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import wallet_state


class _FakeStorage:
    @staticmethod
    def get_inflight_exposure():
        return 3.0


class _RaisingStorage:
    @staticmethod
    def get_inflight_exposure():
        return 5.0


def test_fetch_wallet_state_success_path(monkeypatch):
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
            assert address
            assert abi
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

    monkeypatch.setenv("POLY_WALLET_ADDRESS", "0xabc")
    monkeypatch.setenv("BOT_BANKROLL", "1000")
    monkeypatch.setattr(wallet_state.polymarket_client, "WALLET_ADDRESS", None)
    monkeypatch.setattr(wallet_state, "Web3", FakeWeb3)
    monkeypatch.setattr(wallet_state, "_CACHE", {"state": None, "fetched_monotonic": None})

    state = wallet_state.fetch_wallet_state(storage=_FakeStorage(), force_refresh=True)

    assert state["wallet_address"] == "0xabc"
    assert state["usdc_e_balance"] == 10.5
    assert state["pol_balance"] == 2.0
    assert state["reserved_exposure_usdc"] == 3.0
    assert state["free_usdc"] == 7.5
    assert state["effective_bankroll"] == 7.5
    assert state["bankroll_source"] == "wallet_live"
    assert state["fetch_failed"] is False


def test_fetch_wallet_state_falls_back_to_env_bankroll_on_rpc_failure(monkeypatch):
    class BrokenWeb3:
        HTTPProvider = staticmethod(lambda url, request_kwargs=None: {"url": url, "request_kwargs": request_kwargs})

        def __init__(self, provider):
            self.provider = provider
            self.eth = type("BrokenEth", (), {"get_balance": staticmethod(lambda _wallet: (_ for _ in ()).throw(RuntimeError("rpc down")))})()

        @staticmethod
        def to_checksum_address(address):
            return address

    monkeypatch.setenv("POLY_WALLET_ADDRESS", "0xdef")
    monkeypatch.setenv("BOT_BANKROLL", "50")
    monkeypatch.setattr(wallet_state.polymarket_client, "WALLET_ADDRESS", None)
    monkeypatch.setattr(wallet_state, "Web3", BrokenWeb3)
    monkeypatch.setattr(wallet_state, "_CACHE", {"state": None, "fetched_monotonic": None})

    state = wallet_state.fetch_wallet_state(storage=_RaisingStorage(), force_refresh=True)

    assert state["wallet_address"] == "0xdef"
    assert state["usdc_e_balance"] is None
    assert state["pol_balance"] is None
    assert state["reserved_exposure_usdc"] == 5.0
    assert state["free_usdc"] == 45.0
    assert state["effective_bankroll"] == 45.0
    assert state["bankroll_source"] == "env_fallback"
    assert state["fetch_failed"] is True
    assert "rpc down" in state["error"]
