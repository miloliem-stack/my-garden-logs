from datetime import datetime, timezone

import pandas as pd

from src import run_bot
from src.live_heartbeat import (
    format_console_action_line,
    format_console_status_line,
    format_heartbeat,
    format_night_heartbeat_summary,
    get_log_mode,
)


def test_log_mode_prefers_new_env_and_night_defaults_hourly(monkeypatch):
    monkeypatch.setenv("LOG_MODE", "night")
    monkeypatch.setenv("LIVE_CONSOLE_MODE", "normal")
    monkeypatch.delenv("LIVE_HEARTBEAT_SEC", raising=False)
    now = pd.Timestamp("2026-03-24T20:00:00Z")
    last = now - pd.Timedelta(seconds=60)

    assert get_log_mode() == "night"
    assert run_bot.get_live_console_mode() == "night"
    assert run_bot.get_live_heartbeat_sec("night") == 3600
    assert run_bot.should_emit_console_heartbeat(last, now, "night") is False
    assert run_bot.should_emit_console_heartbeat(now - pd.Timedelta(seconds=3600), now, "night") is True


def test_log_mode_falls_back_to_legacy_normal_mapping(monkeypatch):
    monkeypatch.delenv("LOG_MODE", raising=False)
    monkeypatch.setenv("LIVE_CONSOLE_MODE", "normal")
    monkeypatch.delenv("LIVE_HEARTBEAT_SEC", raising=False)
    now = pd.Timestamp("2026-03-24T20:00:00Z")
    last = now - pd.Timedelta(seconds=60)

    assert get_log_mode() == "info"
    assert run_bot.get_live_console_mode() == "info"
    assert run_bot.get_live_heartbeat_sec("info") == 60
    assert run_bot.should_emit_console_heartbeat(last, now, "info") is True


def test_blank_live_heartbeat_env_falls_back_to_policy_default(monkeypatch):
    monkeypatch.setenv("LIVE_HEARTBEAT_SEC", "")

    assert run_bot.get_live_heartbeat_sec("info") == 60
    assert run_bot.get_live_heartbeat_sec("night") == 3600


def test_invalid_live_heartbeat_env_falls_back_to_policy_default(monkeypatch):
    monkeypatch.setenv("LIVE_HEARTBEAT_SEC", "not-a-number")

    assert run_bot.get_live_heartbeat_sec("debug") == 60


def test_night_mode_compact_summary_renders_two_lines_max() -> None:
    now = datetime(2026, 3, 24, 20, 0, tzinfo=timezone.utc)
    rendered = format_night_heartbeat_summary(
        {
            "market_id": "M1",
            "wallet_free_usdc": 120.5,
            "wallet_reserved_exposure": 20.0,
            "disabled_reason": "quote_stale",
        },
        {
            "entries_submitted": 2,
            "accepted_orders": 2,
            "rejected_orders": 1,
            "fills": 3,
            "partial_fills": 1,
            "exits_submitted": 1,
            "redeems_attempted": 2,
            "redeems_succeeded": 1,
            "redeems_failed": 1,
            "highlight": False,
        },
        {
            "open_lot_count": 4,
            "resolved_pending_count": 1,
            "realized_pnl_total": 3.25,
        },
        now=now,
        ansi_enabled=False,
        mode="night",
    )

    assert len(rendered.text.splitlines()) == 2
    assert "[HB 20:00Z] M1 | free $120.50 | reserved $20.00 | lots 4 | resolved 1 | realized $3.25" in rendered.text
    assert "entries 2 | accepted 2 | rejected 1 | fills 3 | partials 1 | exits 1 | redeem a/s/f 2/1/1 | blocked quote_stale" in rendered.text


def test_info_mode_hides_ids_hashes_and_raw_resp_payloads() -> None:
    line = format_console_action_line(
        {
            "side": "buy_no",
            "market_id": "M2",
            "qty": 5.0,
            "price": 0.41,
            "quantized_notional": 2.05,
            "edge": 0.08,
            "order_id": "ord-1",
            "client_order_id": "cid-1",
            "venue_order_id": "vid-1",
            "tx_hash": "0xabc",
            "resp": {
                "status": "submitted",
                "order_id": "ord-1",
                "tx_hash": "0xabc",
                "raw": {"deep": "payload"},
            },
        },
        mode="info",
        ansi_enabled=False,
    )

    assert line.startswith("BUY_NO")
    assert "mkt=M2" in line
    assert "px=0.410" in line
    assert "qty=5.00" in line
    assert "notional=$2.05" in line
    assert "order_id=" not in line
    assert "client_order_id=" not in line
    assert "venue_order_id=" not in line
    assert "tx_hash=" not in line
    assert "raw" not in line
    assert "resp={" not in line


def test_debug_mode_includes_ids_hashes_and_selected_payload_fields() -> None:
    line = format_console_action_line(
        {
            "side": "buy_yes",
            "market_id": "M4",
            "qty": 3.0,
            "price": 0.62,
            "order_id": "ord-9",
            "client_order_id": "cid-9",
            "venue_order_id": "vid-9",
            "tx_hash": "0xdef",
            "routing_debug": {"attempts": ["a", "b"]},
            "resp": {"status": "rejected", "reason": "price_too_far", "error_message": "venue rejected"},
        },
        mode="debug",
        ansi_enabled=False,
    )

    assert "order_id=ord-9" in line
    assert "client_order_id=cid-9" in line
    assert "venue_order_id=vid-9" in line
    assert "tx_hash=0xdef" in line
    assert "resp=[" in line
    assert "status=rejected" in line
    assert "reason=price_too_far" in line


def test_console_action_line_shows_regime_live_block_reason_and_policy() -> None:
    line = format_console_action_line(
        {
            "action": "skipped_regime_entry_guard",
            "market_id": "M5",
            "reason": "veto_extreme_minority_side",
            "guard_mode": "live",
            "guard_details": {"minority_side_quote": 0.01},
            "decision_state": {
                "regime_guard_mode": "live",
                "regime_guard_blocked": True,
                "regime_guard_reason": "veto_extreme_minority_side",
                "minority_side_quote": 0.01,
                "same_side_existing_qty": 0.0,
                "same_side_existing_filled_entry_count": 0,
            },
        },
        mode="info",
        ansi_enabled=False,
    )

    assert "reason=[veto_extreme_minority_side]" in line
    assert "regime_block" in line
    assert "mode=live" in line
    assert "policy=extreme_minority_side" in line
    assert "quote=0.010" in line


def test_console_action_line_shows_regime_shadow_would_skip_policy() -> None:
    line = format_console_action_line(
        {
            "side": "buy_no",
            "market_id": "M6",
            "qty": 1.0,
            "price": 0.08,
            "decision_state": {
                "regime_guard_mode": "shadow",
                "regime_guard_blocked": False,
                "regime_guard_reason": "veto_regime_polarized_tail_minority_side",
                "minority_side_quote": 0.08,
                "same_side_existing_qty": 0.0,
                "same_side_existing_filled_entry_count": 0,
                "would_block_in_shadow": True,
            },
        },
        mode="info",
        ansi_enabled=False,
    )

    assert "regime_shadow_skip" in line
    assert "mode=shadow" in line
    assert "policy=polarized_tail_minority_side" in line
    assert "quote=0.080" in line


def test_response_field_highlighting_helper_applies_to_all_modes() -> None:
    for mode in ("info", "night", "debug"):
        line = format_console_status_line(
            "error",
            mode=mode,
            tone="error",
            market="M3",
            reason="quote_fetch_failed",
        )
        assert "reason=[quote_fetch_failed]" in line


def test_emit_console_status_keeps_critical_errors_in_night_mode(monkeypatch):
    lines = []
    monkeypatch.setattr(run_bot, "print_console_event_line", lambda message: lines.append(message))

    run_bot.emit_console_status("error", console_mode="night", tone="error", market="M3", reason="quote_fetch_failed")

    assert len(lines) == 1
    assert "ERROR" in lines[0]
    assert "reason=[quote_fetch_failed]" in lines[0]


def test_format_heartbeat_info_mode_stays_within_two_lines() -> None:
    rendered = format_heartbeat(
        {
            "market_id": "M1",
            "market_window": {"start": "2026-03-24T19:00:00Z", "end": "2026-03-24T20:00:00Z"},
            "strike_price": 70000,
            "q_yes": 0.52,
            "q_no": 0.48,
            "p_yes": 0.57,
            "p_no": 0.43,
            "edge_yes": 0.05,
            "edge_no": -0.05,
            "chosen_action": "buy_yes",
            "policy_bucket": "near",
            "wallet_free_usdc": 90.0,
            "effective_bankroll": 120.0,
            "exposure": 10.0,
            "disabled_reason": "quote_stale",
        },
        {
            "window_seconds": 60,
            "order_attempts": 2,
            "accepted_orders": 1,
            "fills": 1,
            "cancels_errors": 1,
            "highlight": False,
        },
        now="2026-03-24T19:30:00Z",
        mode="info",
        ansi_enabled=False,
    )

    assert len(rendered.text.splitlines()) == 2
    assert "blocked quote_stale" in rendered.text
