"""Minimal Binance feed stub: backfill helper (REST) and placeholder for WS.
This is a stub to be expanded for production use.
"""
import pandas as pd
import requests
from typing import Optional


def backfill_klines(symbol: str = 'BTCUSDT', interval: str = '1m', limit: int = 1000, end_ts: Optional[int] = None) -> pd.Series:
    """Fetch recent klines from Binance REST and return a price series (close prices) indexed by UTC timestamps in minutes."""
    base = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if end_ts is not None:
        params['endTime'] = int(end_ts)
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # kline format: [open_time, open, high, low, close, ...]
    idx = [pd.to_datetime(int(d[0]), unit='ms', utc=True).floor('min') for d in data]
    closes = [float(d[4]) for d in data]
    return pd.Series(closes, index=idx)


def get_1h_open_for_timestamp(ts: pd.Timestamp, symbol: str = 'BTCUSDT') -> Optional[float]:
    """Return the 1H kline open price for the hour starting at `ts` (UTC).

    `ts` should be a pandas.Timestamp (timezone-aware or naive UTC).
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    start_ms = int(ts.floor('h').value // 1_000_000)
    # fetch a couple of klines around the time
    base = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': '1h', 'limit': 5, 'endTime': start_ms + 3600 * 1000}
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for d in data:
            open_time = pd.to_datetime(int(d[0]), unit='ms', utc=True)
            if open_time == ts.floor('h'):
                return float(d[1])
    except Exception:
        return None
    return None


def get_1h_close_for_timestamp(ts: pd.Timestamp, symbol: str = 'BTCUSDT') -> Optional[float]:
    """Return the 1H kline close price for the hour starting at `ts` (UTC)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    start_ms = int(ts.floor('h').value // 1_000_000)
    base = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': '1h', 'limit': 5, 'endTime': start_ms + 3600 * 1000}
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for d in data:
            open_time = pd.to_datetime(int(d[0]), unit='ms', utc=True)
            if open_time == ts.floor('h'):
                return float(d[4])
    except Exception:
        return None
    return None


async def live_kline_1m(symbol: str = 'btcusdt'):
    """Async generator that yields 1m kline dicts from Binance WebSocket.

    Yields the raw kline dict as provided by Binance's stream when a kline message appears.
    """
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    import websockets, json
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            # message contains a 'k' field with kline data
            k = data.get('k')
            if k is not None:
                yield k
