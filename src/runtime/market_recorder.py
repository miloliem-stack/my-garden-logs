from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.market_router import resolve_active_market_bundle
from src.polymarket_client import POLY_WS_URL
from src.polymarket_feed import (
    _build_quote_snapshot_from_book,
    classify_quote_snapshot,
    get_quote_snapshot,
)

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None


WS_RETRY_DELAY_SEC = max(1.0, float(os.getenv("POLY_WS_RETRY_DELAY_SEC", "5")))
WS_OPEN_TIMEOUT_SEC = max(1.0, float(os.getenv("POLY_WS_OPEN_TIMEOUT_SEC", "10")))


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _parse_json_env(name: str) -> list[dict[str, Any]]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _compact_log(message: str) -> None:
    print(message, flush=True)


def _extract_token_id(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("asset_id", "assetId", "token_id", "tokenId", "market", "market_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("data", "book", "orderbook", "event", "payload"):
            value = payload.get(key)
            token_id = _extract_token_id(value)
            if token_id:
                return token_id
    elif isinstance(payload, list):
        for item in payload:
            token_id = _extract_token_id(item)
            if token_id:
                return token_id
    return None


def _extract_book_payload(payload: Any) -> Optional[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("bids"), list) or isinstance(payload.get("asks"), list):
            return payload
        for key in ("book", "orderbook", "data", "event", "payload"):
            nested = payload.get(key)
            book = _extract_book_payload(nested)
            if book is not None:
                return book
    elif isinstance(payload, list):
        for item in payload:
            book = _extract_book_payload(item)
            if book is not None:
                return book
    return None


def _candidate_subscribe_payloads(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    configured = _parse_json_env("POLY_WS_SUBSCRIBE_PAYLOADS")
    if configured:
        return configured
    token_ids = [token for token in (bundle.get("token_yes"), bundle.get("token_no")) if token]
    market_id = bundle.get("market_id")
    payloads = []
    if token_ids:
        payloads.extend(
            [
                {"type": "subscribe", "channel": "market", "assets_ids": token_ids},
                {"type": "subscribe", "channel": "book", "assets_ids": token_ids},
                {"event": "subscribe", "channel": "market", "assets_ids": token_ids},
            ]
        )
    if market_id:
        payloads.extend(
            [
                {"type": "subscribe", "channel": "market", "market_id": market_id},
                {"event": "subscribe", "channel": "market", "market_id": market_id},
            ]
        )
    deduped = []
    seen = set()
    for payload in payloads:
        marker = json.dumps(payload, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(payload)
    return deduped


def _normalize_ws_message_to_quote(
    message: Any,
    *,
    token_ids: set[str],
    received_at: Optional[pd.Timestamp] = None,
) -> Optional[dict[str, Any]]:
    token_id = _extract_token_id(message)
    if token_id not in token_ids:
        return None
    book = _extract_book_payload(message)
    if book is None:
        return None
    received = received_at or _utc_now()
    snapshot = _build_quote_snapshot_from_book(
        token_id,
        book,
        source="polymarket_ws_orderbook",
        fetched_at=received.timestamp(),
    )
    snapshot["age_seconds"] = 0.0
    snapshot["raw"] = message
    return {
        "token_id": token_id,
        "quote": snapshot,
        "received_at": received.isoformat(),
        "message_type": str(message.get("type") or message.get("event") or message.get("channel") or "unknown") if isinstance(message, dict) else "unknown",
    }


@dataclass
class _RecorderState:
    bundle: Optional[dict[str, Any]] = None
    yes_quote: Optional[dict[str, Any]] = None
    no_quote: Optional[dict[str, Any]] = None
    ws_message_count: int = 0
    rest_snapshot_count: int = 0
    records_written: int = 0


class MarketRecorder:
    def __init__(
        self,
        *,
        series_id: str,
        output_dir: str,
        poll_seconds: float = 2.0,
        snapshot_interval_seconds: Optional[float] = None,
        flush_every_n: int = 10,
        use_websocket: bool = True,
        log_mode: str = "compact",
    ) -> None:
        self.series_id = series_id
        self.output_dir = Path(output_dir)
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.snapshot_interval_seconds = (
            max(self.poll_seconds, float(snapshot_interval_seconds))
            if snapshot_interval_seconds is not None
            else max(5.0, self.poll_seconds)
        )
        self.flush_every_n = max(1, int(flush_every_n))
        self.use_websocket = bool(use_websocket)
        self.log_mode = log_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quotes_path = self.output_dir / "market_quotes.jsonl"
        self.events_path = self.output_dir / "market_events.jsonl"

    def run(self, duration_seconds: Optional[float] = None) -> dict[str, Any]:
        return asyncio.run(self._run_async(duration_seconds=duration_seconds))

    async def _run_async(self, duration_seconds: Optional[float]) -> dict[str, Any]:
        started = time.monotonic()
        end_at = started + float(duration_seconds) if duration_seconds is not None else None
        state = _RecorderState()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        writer = self.quotes_path.open("a", encoding="utf-8")
        event_writer = self.events_path.open("a", encoding="utf-8")
        last_snapshot_write = 0.0
        ws_task: Optional[asyncio.Task] = None
        try:
            while end_at is None or time.monotonic() < end_at:
                bundle = await asyncio.to_thread(resolve_active_market_bundle, self.series_id)
                if bundle is not None:
                    if state.bundle is None or bundle.get("market_id") != state.bundle.get("market_id"):
                        state.bundle = bundle
                        state.yes_quote = bundle.get("yes_quote")
                        state.no_quote = bundle.get("no_quote")
                        await self._write_event(
                            event_writer,
                            {
                                "ts": _utc_now().isoformat(),
                                "event_type": "market_bundle",
                                "series_id": self.series_id,
                                "market_id": bundle.get("market_id"),
                                "token_yes": bundle.get("token_yes"),
                                "token_no": bundle.get("token_no"),
                                "raw": bundle,
                            },
                        )
                        if self.log_mode == "compact":
                            _compact_log(
                                f"market_recorder market={bundle.get('market_id')} tokens=({bundle.get('token_yes')},{bundle.get('token_no')})"
                            )
                        if ws_task is not None:
                            ws_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await ws_task
                            ws_task = None
                    if self.use_websocket and POLY_WS_URL and websockets is not None and ws_task is None:
                        ws_task = asyncio.create_task(self._ws_loop(bundle, queue))

                while True:
                    try:
                        update = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    state.ws_message_count += 1
                    quote = update.get("quote")
                    if state.bundle is not None and quote is not None:
                        if update["token_id"] == state.bundle.get("token_yes"):
                            state.yes_quote = quote
                        elif update["token_id"] == state.bundle.get("token_no"):
                            state.no_quote = quote
                    await self._write_event(
                        event_writer,
                        {
                            "ts": update.get("received_at"),
                            "event_type": "ws_quote",
                            "series_id": self.series_id,
                            "market_id": state.bundle.get("market_id") if state.bundle else None,
                            "token_id": update.get("token_id"),
                            "message_type": update.get("message_type"),
                            "raw": quote.get("raw") if isinstance(quote, dict) else None,
                        },
                    )
                    await self._write_bundle_snapshot(writer, state, source="websocket", update_ts=update.get("received_at"))
                    last_snapshot_write = time.monotonic()

                if state.bundle is not None and time.monotonic() - last_snapshot_write >= self.snapshot_interval_seconds:
                    state.yes_quote = await asyncio.to_thread(get_quote_snapshot, state.bundle.get("token_yes"), True)
                    state.no_quote = await asyncio.to_thread(get_quote_snapshot, state.bundle.get("token_no"), True)
                    state.rest_snapshot_count += 1
                    await self._write_bundle_snapshot(writer, state, source="poll", update_ts=_utc_now().isoformat())
                    last_snapshot_write = time.monotonic()

                await asyncio.sleep(self.poll_seconds)
        finally:
            if ws_task is not None:
                ws_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ws_task
            writer.close()
            event_writer.close()

        return {
            "series_id": self.series_id,
            "output_dir": str(self.output_dir),
            "quotes_path": str(self.quotes_path),
            "events_path": str(self.events_path),
            "records_written": state.records_written,
            "ws_message_count": state.ws_message_count,
            "rest_snapshot_count": state.rest_snapshot_count,
            "websocket_enabled": bool(self.use_websocket and POLY_WS_URL and websockets is not None),
            "market_id": state.bundle.get("market_id") if state.bundle else None,
        }

    async def _write_event(self, handle, payload: dict[str, Any]) -> None:
        handle.write(json.dumps(payload, default=_json_default) + "\n")
        handle.flush()

    async def _write_bundle_snapshot(self, handle, state: _RecorderState, *, source: str, update_ts: str) -> None:
        if state.bundle is None:
            return
        yes_quote = dict(state.yes_quote or {})
        no_quote = dict(state.no_quote or {})
        payload = {
            "ts": update_ts,
            "series_id": self.series_id,
            "source": source,
            "market_id": state.bundle.get("market_id"),
            "condition_id": state.bundle.get("condition_id"),
            "token_yes": state.bundle.get("token_yes"),
            "token_no": state.bundle.get("token_no"),
            "start_time": state.bundle.get("start_time"),
            "end_time": state.bundle.get("end_time"),
            "yes_quote": yes_quote,
            "no_quote": no_quote,
            "yes_quote_state": classify_quote_snapshot(yes_quote) if yes_quote else {"tradable": False, "reason": "missing_quote"},
            "no_quote_state": classify_quote_snapshot(no_quote) if no_quote else {"tradable": False, "reason": "missing_quote"},
        }
        handle.write(json.dumps(payload, default=_json_default) + "\n")
        state.records_written += 1
        if state.records_written % self.flush_every_n == 0:
            handle.flush()

    async def _ws_loop(self, bundle: dict[str, Any], queue: asyncio.Queue[dict[str, Any]]) -> None:
        token_ids = {token for token in (bundle.get("token_yes"), bundle.get("token_no")) if token}
        if not token_ids:
            return
        payloads = _candidate_subscribe_payloads(bundle)
        while True:
            try:
                async with websockets.connect(POLY_WS_URL, open_timeout=WS_OPEN_TIMEOUT_SEC) as ws:
                    for payload in payloads:
                        await ws.send(json.dumps(payload))
                    while True:
                        raw = await ws.recv()
                        received_at = _utc_now()
                        try:
                            message = json.loads(raw)
                        except Exception:
                            continue
                        normalized = _normalize_ws_message_to_quote(message, token_ids=token_ids, received_at=received_at)
                        if normalized is not None:
                            await queue.put(normalized)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(WS_RETRY_DELAY_SEC)


import contextlib

