"""Runtime wallet balance helpers for sizing and observability."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Optional

from . import storage as storage_module
from . import polymarket_client

try:
    from web3 import Web3
except Exception:  # pragma: no cover - optional dependency in test/dev environments
    Web3 = None


DEFAULT_POLYGON_RPC = "https://polygon-rpc.com"
DEFAULT_USDC_E_TOKEN_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
DEFAULT_USDC_E_TOKEN_DECIMALS = 6
WALLET_STATE_CACHE_TTL_SEC = float(os.getenv("WALLET_STATE_CACHE_TTL_SEC", "5"))

_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

_CACHE: dict[str, Any] = {"state": None, "fetched_monotonic": None}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fallback_bankroll() -> float:
    return max(0.0, float(os.getenv("BOT_BANKROLL", "1000")))


def _get_reserved_exposure(storage: Any = None) -> float:
    storage_ref = storage or storage_module
    getter = getattr(storage_ref, "get_inflight_exposure", None)
    if getter is None:
        return 0.0
    try:
        return max(0.0, float(getter()))
    except Exception:
        return 0.0


def _wallet_address() -> Optional[str]:
    return polymarket_client.WALLET_ADDRESS or os.getenv("POLY_WALLET_ADDRESS")


def _checksum_address(address: str) -> str:
    if Web3 is None:
        return address
    if hasattr(Web3, "to_checksum_address"):
        return Web3.to_checksum_address(address)
    return Web3.toChecksumAddress(address)


def _from_wei(value: int) -> float:
    if Web3 is not None and hasattr(Web3, "from_wei"):
        return float(Web3.from_wei(value, "ether"))
    return float(value) / 1e18


def _build_state(
    *,
    wallet_address: Optional[str],
    usdc_e_balance: Optional[float],
    pol_balance: Optional[float],
    reserved_exposure_usdc: float,
    bankroll_source: str,
    fetch_failed: bool,
    error: Optional[str],
    fallback_bankroll: Optional[float] = None,
) -> dict:
    free_usdc = None
    effective_bankroll = None
    if usdc_e_balance is not None:
        free_usdc = max(0.0, float(usdc_e_balance) - reserved_exposure_usdc)
        effective_bankroll = free_usdc
    else:
        fallback = _fallback_bankroll() if fallback_bankroll is None else max(0.0, float(fallback_bankroll))
        free_usdc = max(0.0, fallback - reserved_exposure_usdc)
        effective_bankroll = free_usdc
    return {
        "wallet_address": wallet_address,
        "usdc_e_balance": usdc_e_balance,
        "pol_balance": pol_balance,
        "reserved_exposure_usdc": reserved_exposure_usdc,
        "free_usdc": free_usdc,
        "effective_bankroll": effective_bankroll,
        "bankroll_source": bankroll_source,
        "fetch_failed": fetch_failed,
        "error": error,
        "fetched_at": _utc_now_iso(),
    }


def _read_onchain_balances(wallet_address: str) -> tuple[float, float]:
    if Web3 is None:
        raise RuntimeError("web3 is not installed")
    rpc_url = os.getenv("POLYGON_RPC") or polymarket_client.POLYGON_RPC or DEFAULT_POLYGON_RPC
    timeout_sec = float(os.getenv("POLYGON_RPC_TIMEOUT_SEC", "10"))
    provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout_sec})
    web3 = polymarket_client._inject_poa_middleware(Web3(provider))
    checksum_wallet = _checksum_address(wallet_address)
    pol_wei = web3.eth.get_balance(checksum_wallet)
    pol_balance = _from_wei(pol_wei)

    token_address = _checksum_address(os.getenv("USDC_E_TOKEN_ADDRESS", DEFAULT_USDC_E_TOKEN_ADDRESS))
    token = web3.eth.contract(address=token_address, abi=_ERC20_ABI)
    raw_balance = token.functions.balanceOf(checksum_wallet).call()
    decimals = int(os.getenv("USDC_E_TOKEN_DECIMALS", str(DEFAULT_USDC_E_TOKEN_DECIMALS)))
    try:
        decimals = int(token.functions.decimals().call())
    except Exception:
        pass
    usdc_balance = float(raw_balance) / (10 ** decimals)
    return usdc_balance, pol_balance


def fetch_wallet_state(storage=None, force_refresh: bool = False) -> dict:
    reserved_exposure = _get_reserved_exposure(storage)
    fallback_bankroll = _fallback_bankroll()
    cached = _CACHE.get("state")
    cached_ts = _CACHE.get("fetched_monotonic")
    if not force_refresh and cached is not None and cached_ts is not None:
        age_seconds = (datetime.now(timezone.utc).timestamp() - float(cached_ts))
        if age_seconds < WALLET_STATE_CACHE_TTL_SEC:
            return _build_state(
                wallet_address=cached.get("wallet_address"),
                usdc_e_balance=cached.get("usdc_e_balance"),
                pol_balance=cached.get("pol_balance"),
                reserved_exposure_usdc=reserved_exposure,
                bankroll_source=cached.get("bankroll_source", "env_fallback"),
                fetch_failed=bool(cached.get("fetch_failed")),
                error=cached.get("error"),
                fallback_bankroll=fallback_bankroll,
            )

    wallet_address = _wallet_address()
    if not wallet_address:
        state = _build_state(
            wallet_address=None,
            usdc_e_balance=None,
            pol_balance=None,
            reserved_exposure_usdc=reserved_exposure,
            bankroll_source="env_fallback",
            fetch_failed=True,
            error="wallet address not configured",
            fallback_bankroll=fallback_bankroll,
        )
        _CACHE.update({"state": state, "fetched_monotonic": datetime.now(timezone.utc).timestamp()})
        return dict(state)

    try:
        usdc_balance, pol_balance = _read_onchain_balances(wallet_address)
        state = _build_state(
            wallet_address=wallet_address,
            usdc_e_balance=usdc_balance,
            pol_balance=pol_balance,
            reserved_exposure_usdc=reserved_exposure,
            bankroll_source="wallet_live",
            fetch_failed=False,
            error=None,
        )
    except Exception as exc:
        state = _build_state(
            wallet_address=wallet_address,
            usdc_e_balance=None,
            pol_balance=None,
            reserved_exposure_usdc=reserved_exposure,
            bankroll_source="env_fallback",
            fetch_failed=True,
            error=str(exc),
            fallback_bankroll=fallback_bankroll,
        )
    _CACHE.update({"state": state, "fetched_monotonic": datetime.now(timezone.utc).timestamp()})
    return dict(state)


def get_effective_bankroll(wallet_state: Optional[dict] = None, fallback_bankroll: Optional[float] = None) -> float:
    state = wallet_state or {}
    effective = state.get("effective_bankroll")
    if effective is not None:
        return max(0.0, float(effective))
    fallback = _fallback_bankroll() if fallback_bankroll is None else float(fallback_bankroll)
    return max(0.0, fallback)
