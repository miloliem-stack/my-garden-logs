#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import polymarket_client, storage

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _classify_probe_result(result: dict) -> str:
    if not result:
        return "unexpected_schema"
    if result.get("error_text"):
        lowered = str(result["error_text"]).lower()
        if "timeout" in lowered:
            return "timeout"
        if any(token in lowered for token in ("name resolution", "connection", "network", "dns", "refused")):
            return "network_error"
        return "bad_payload" if result.get("http_status") else "network_error"
    status = result.get("http_status")
    parsed = result.get("response_json")
    if result.get("ok"):
        return "success"
    if status in (401, 403):
        return "auth_error"
    if status == 404:
        return "endpoint_missing"
    if status in (400, 422):
        return "bad_payload"
    if parsed is None and result.get("response_text"):
        return "unexpected_schema"
    return "unexpected_schema"


def _store_probe_exchange(run_id: int, step_name: str, transport: str, result: dict, *, direction: str = "response", channel: str | None = None) -> None:
    classification = _classify_probe_result(result)
    storage.append_probe_event(
        run_id,
        step_name=step_name,
        transport=transport,
        direction="request" if direction != "handshake" else "handshake",
        method=result.get("method"),
        path=result.get("path"),
        url=result.get("url"),
        channel=channel,
        request_headers=result.get("request_headers"),
        request_body={"body": result.get("request_body"), "params": result.get("params")},
        classification=classification,
    )
    if direction == "handshake":
        return
    response_direction = "error" if result.get("error_text") else direction
    storage.append_probe_event(
        run_id,
        step_name=step_name,
        transport=transport,
        direction=response_direction,
        method=result.get("method"),
        path=result.get("path"),
        url=result.get("url"),
        channel=channel,
        http_status=result.get("http_status"),
        ok=result.get("ok"),
        latency_ms=result.get("latency_ms"),
        request_headers=result.get("request_headers"),
        request_body={"body": result.get("request_body"), "params": result.get("params")},
        response_headers=result.get("response_headers"),
        response_body={"json": result.get("response_json"), "text": result.get("response_text")},
        error_text=result.get("error_text"),
        classification=classification,
    )


def _probe_payload_from_normalized(result: dict) -> dict:
    probe = result.get("raw_probe") if isinstance(result, dict) else None
    if isinstance(probe, dict):
        enriched = dict(probe)
        enriched.setdefault("ok", result.get("ok"))
        enriched.setdefault("http_status", result.get("http_status"))
        enriched.setdefault("error_text", result.get("error_message"))
        raw = result.get("raw")
        if isinstance(raw, dict):
            enriched.setdefault("response_json", raw)
        elif raw is not None:
            enriched.setdefault("response_text", str(raw))
        return enriched
    raw = result.get("raw") if isinstance(result, dict) else None
    return {
        "method": None,
        "path": None,
        "url": None,
        "params": None,
        "request_headers": None,
        "request_body": None,
        "http_status": result.get("http_status") if isinstance(result, dict) else None,
        "ok": result.get("ok") if isinstance(result, dict) else False,
        "latency_ms": None,
        "response_headers": None,
        "response_text": None if isinstance(raw, dict) else raw,
        "response_json": raw if isinstance(raw, dict) else None,
        "error_text": result.get("error_message") if isinstance(result, dict) else None,
    }


async def _ws_probe(url: str) -> dict:
    if websockets is None:
        return {"ok": False, "error_text": "websockets dependency unavailable", "latency_ms": 0.0}
    started = datetime.now(timezone.utc)
    try:
        async with websockets.connect(url) as ws:
            latency_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
            return {
                "method": "CONNECT",
                "path": None,
                "url": url,
                "params": None,
                "request_headers": None,
                "request_body": None,
                "http_status": None,
                "ok": True,
                "latency_ms": latency_ms,
                "response_headers": None,
                "response_text": "connected",
                "response_json": None,
                "error_text": None,
            }
    except Exception as exc:
        latency_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "method": "CONNECT",
            "path": None,
            "url": url,
            "params": None,
            "request_headers": None,
            "request_body": None,
            "http_status": None,
            "ok": False,
            "latency_ms": latency_ms,
            "response_headers": None,
            "response_text": None,
            "response_json": None,
            "error_text": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Polymarket venue wiring and persist raw evidence to SQLite.")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--ws-url", default=os.getenv("POLY_WS_URL"))
    parser.add_argument("--skip-ws", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage.ensure_db()
    live_mode = bool(polymarket_client.LIVE)
    write_enabled = live_mode and _env_flag("PROBE_ENABLE_WRITE", False)
    host = socket.gethostname()
    run_id = storage.create_probe_run(
        host=host,
        api_base=None,
        wallet_address=polymarket_client.WALLET_ADDRESS,
        live_mode=live_mode,
        write_enabled=write_enabled,
    )
    summary = {
        "run_id": run_id,
        "steps": [],
        "live_mode": live_mode,
        "write_enabled": write_enabled,
        "gamma_base": polymarket_client.POLY_GAMMA_BASE,
        "clob_base": polymarket_client.POLY_CLOB_BASE,
    }
    try:
        public_steps = [
            ("public_markets", "GET", "/markets", None, None, False),
        ]
        for step_name, method, path, body, params, auth in public_steps:
            result = polymarket_client.raw_http_probe(
                method,
                path,
                body=body,
                params=params,
                auth=auth,
                timeout=args.timeout,
                base_url=polymarket_client.POLY_GAMMA_BASE,
            )
            _store_probe_exchange(run_id, step_name, "http", result)
            summary["steps"].append({"step_name": step_name, "classification": _classify_probe_result(result)})

        auth_steps = [
            ("auth_order_status_bogus", lambda: polymarket_client.get_order_status(order_id="probe-bogus-order", dry_run=False)),
            ("auth_cancel_bogus", lambda: polymarket_client.cancel_order(order_id="probe-bogus-order", dry_run=False)),
            ("auth_heartbeat", lambda: polymarket_client.post_heartbeat("", dry_run=False)),
        ]
        for step_name, runner in auth_steps:
            normalized = runner()
            result = _probe_payload_from_normalized(normalized)
            _store_probe_exchange(run_id, step_name, "clob_sdk", result)
            summary["steps"].append({"step_name": step_name, "classification": _classify_probe_result(result)})

        if write_enabled:
            max_notional = float(os.getenv("PROBE_MAX_NOTIONAL_USDC", "0"))
            token_id = os.getenv("PROBE_TOKEN_ID")
            market_id = os.getenv("PROBE_MARKET_ID")
            outcome_side = os.getenv("PROBE_OUTCOME_SIDE")
            if max_notional <= 0 or not token_id or not market_id or not outcome_side:
                storage.append_probe_event(
                    run_id,
                    step_name="live_write_probe",
                    transport="http",
                    direction="error",
                    error_text="missing required write-probe env vars",
                    classification="bad_payload",
                )
                summary["steps"].append({"step_name": "live_write_probe", "classification": "bad_payload"})
            else:
                qty = round(max_notional / 0.99, 8)
                submit_normalized = polymarket_client.place_marketable_order(
                    token_id=token_id,
                    side="buy",
                    qty=qty,
                    limit_price=0.99,
                    order_type="FAK",
                    dry_run=False,
                )
                submit = _probe_payload_from_normalized(submit_normalized)
                _store_probe_exchange(run_id, "live_write_submit", "clob_sdk", submit)
                summary["steps"].append({"step_name": "live_write_submit", "classification": _classify_probe_result(submit)})
                order_id = submit_normalized.get("order_id")
                client_order_id = submit_normalized.get("client_order_id")
                tx_hash = submit_normalized.get("tx_hash")
                if order_id:
                    status_normalized = polymarket_client.get_order_status(order_id=order_id, dry_run=False)
                    _store_probe_exchange(run_id, "live_write_status", "clob_sdk", _probe_payload_from_normalized(status_normalized))
                if order_id or client_order_id:
                    cancel_normalized = polymarket_client.cancel_order(order_id=order_id, client_order_id=client_order_id, dry_run=False)
                    _store_probe_exchange(run_id, "live_write_cancel", "clob_sdk", _probe_payload_from_normalized(cancel_normalized))
                if tx_hash:
                    rpc_result = polymarket_client.raw_rpc_probe("eth_getTransactionReceipt", [tx_hash], timeout=args.timeout)
                    _store_probe_exchange(run_id, "live_write_receipt", "rpc", rpc_result)
        else:
            storage.append_probe_event(
                run_id,
                step_name="live_write_probe",
                transport="http",
                direction="error",
                error_text="write probe disabled",
                classification="skipped",
            )
            summary["steps"].append({"step_name": "live_write_probe", "classification": "skipped"})

        if args.ws_url and not args.skip_ws:
            import asyncio

            ws_result = asyncio.run(_ws_probe(args.ws_url))
            _store_probe_exchange(
                run_id,
                "ws_handshake",
                "ws",
                ws_result,
                direction="handshake" if ws_result.get("ok") else "error",
                channel=args.ws_url,
            )
            summary["steps"].append({"step_name": "ws_handshake", "classification": "success" if ws_result.get("ok") else "handshake_error"})

        summary["event_count"] = len(storage.list_probe_events(run_id))
        storage.finish_probe_run(run_id, summary=summary)
        print(json.dumps({"run_id": run_id, "summary": summary}, indent=2))
        return 0
    except Exception as exc:
        summary["fatal_error"] = str(exc)
        storage.finish_probe_run(run_id, summary=summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
