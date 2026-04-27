import os
from pathlib import Path

import src.storage as storage


def setup_function(_fn):
    db_path = storage.get_db_path()
    try:
        os.remove(db_path)
    except Exception:
        pass
    storage.ensure_db()


def test_probe_tables_are_created_and_run_roundtrip() -> None:
    run_id = storage.create_probe_run(
        host="host1",
        api_base="https://api.polymarket.com",
        wallet_address="0xabc",
        live_mode=False,
        write_enabled=False,
        summary={"phase": "started"},
    )
    storage.append_probe_event(
        run_id,
        step_name="public_markets",
        transport="http",
        direction="request",
        method="GET",
        path="/markets",
        url="https://api.polymarket.com/markets",
        request_headers={"Content-Type": "application/json"},
        request_body={"params": None},
        classification="success",
    )
    storage.append_probe_event(
        run_id,
        step_name="public_markets",
        transport="http",
        direction="response",
        method="GET",
        path="/markets",
        url="https://api.polymarket.com/markets",
        http_status=200,
        ok=True,
        latency_ms=10.5,
        response_body={"json": {"ok": True}},
        classification="success",
    )
    storage.finish_probe_run(run_id, summary={"phase": "finished"})

    run = storage.get_probe_run(run_id)
    events = storage.list_probe_events(run_id)

    assert run is not None
    assert run["host"] == "host1"
    assert run["summary"]["phase"] == "finished"
    assert len(events) == 2
    assert events[0]["step_name"] == "public_markets"
    assert events[1]["ok"] is True

