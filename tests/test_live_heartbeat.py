from datetime import datetime, timedelta, timezone

import pandas as pd

from src.live_heartbeat import RollingEventBuffer, format_live_heartbeat_summary


def test_live_heartbeat_format_handles_missing_values() -> None:
    now = datetime(2026, 3, 23, 20, 14, tzinfo=timezone.utc)
    heartbeat = {
        "market_id": "M1",
        "market_window": {"start": None, "end": None},
        "strike_price": None,
        "q_yes": None,
        "q_no": None,
        "p_yes": None,
        "p_no": None,
        "edge_yes": None,
        "edge_no": None,
        "chosen_action": None,
        "policy_bucket": None,
        "wallet_free_usdc": None,
        "effective_bankroll": None,
        "exposure": None,
        "disabled_reason": None,
    }
    event_summary = {
        "window_seconds": 60,
        "order_attempts": 0,
        "accepted_orders": 0,
        "fills": 0,
        "cancels_errors": 0,
        "latest_event": None,
        "highlight": False,
    }

    rendered = format_live_heartbeat_summary(heartbeat, event_summary, now=now, ansi_enabled=False, mode="debug")

    assert rendered.highlight is False
    assert rendered.ansi_enabled is False
    assert "[HB 20:14Z] M1 | n/a-n/a UTC | strike n/a | bucket n/a | action none" in rendered.text
    assert "mkt/model/edge yes n/a/n/a/n/a no n/a/n/a/n/a" in rendered.text
    assert "wallet n/a | exposure n/a | counts a/0 f/0 e/0" in rendered.text
    assert "bankroll n/a | last 60s: attempts 0 accepted 0 fills 0 cancels/errors 0" in rendered.text


def test_live_heartbeat_highlight_tracks_real_execution_since_last_emit() -> None:
    now = pd.Timestamp("2026-03-23T20:14:00Z")
    buffer = RollingEventBuffer(window_seconds=60)
    buffer.record("order_attempt", ts=now - pd.Timedelta(seconds=30), action="buy_yes")
    buffer.record("order_accept", ts=now - pd.Timedelta(seconds=25), status="accepted", real_execution=True)

    summary = buffer.summarize(now=now, since=now - pd.Timedelta(seconds=60))
    rendered = format_live_heartbeat_summary(
        {
            "market_id": "M1",
            "market_window": {"start": now, "end": now + pd.Timedelta(hours=1)},
            "strike_price": 70915.59,
            "q_yes": 0.595,
            "q_no": 0.405,
            "p_yes": 0.565,
            "p_no": 0.435,
            "edge_yes": -0.03,
            "edge_no": 0.03,
            "chosen_action": "buy_no",
            "policy_bucket": "far",
            "wallet_free_usdc": 10.59,
            "effective_bankroll": 10.59,
            "exposure": 0.0,
            "disabled_reason": None,
        },
        summary,
        now=now,
        ansi_enabled=True,
        mode="debug",
    )

    assert summary["accepted_orders"] == 1
    assert rendered.highlight is True
    assert rendered.ansi_enabled is True
    assert rendered.styled_text().startswith("\033[97m")


def test_live_heartbeat_rolling_aggregation_counts_and_latest_event() -> None:
    now = pd.Timestamp("2026-03-23T20:14:00Z")
    buffer = RollingEventBuffer(window_seconds=60)
    buffer.record("market_switch", ts=now - pd.Timedelta(seconds=58), new_market_id="M2")
    buffer.record("order_attempt", ts=now - pd.Timedelta(seconds=40), action="buy_no")
    buffer.record("order_accept", ts=now - pd.Timedelta(seconds=39), status="submitted", real_execution=True)
    buffer.record("fill", ts=now - pd.Timedelta(seconds=20), side="buy_no", qty=5.0, price=0.41, real_execution=True)
    buffer.record("cancel", ts=now - pd.Timedelta(seconds=10), status="canceled", real_execution=True)
    buffer.record("error", ts=now - pd.Timedelta(seconds=5), reason="quote_fetch_failed")
    buffer.record("order_attempt", ts=now - pd.Timedelta(seconds=61), action="old_event")

    summary = buffer.summarize(now=now, since=now - pd.Timedelta(seconds=60))

    assert summary["market_switches"] == 1
    assert summary["order_attempts"] == 1
    assert summary["accepted_orders"] == 1
    assert summary["fills"] == 1
    assert summary["cancels_errors"] == 2
    assert summary["highlight"] is True
    assert summary["latest_event"]["event_type"] == "error"


def test_live_heartbeat_shows_regime_shadow_skip_summary() -> None:
    now = datetime(2026, 3, 23, 20, 14, tzinfo=timezone.utc)
    heartbeat = {
        "market_id": "M1",
        "market_window": {"start": None, "end": None},
        "strike_price": 70000,
        "q_yes": 0.92,
        "q_no": 0.08,
        "p_yes": 0.10,
        "p_no": 0.90,
        "edge_yes": -0.82,
        "edge_no": 0.82,
        "chosen_action": "buy_no",
        "policy_bucket": "far",
        "wallet_free_usdc": 10.0,
        "effective_bankroll": 10.0,
        "exposure": 0.0,
        "disabled_reason": None,
        "regime_guard_mode": "shadow",
        "regime_guard_blocked": False,
        "regime_guard_reason": "veto_regime_polarized_tail_minority_side",
        "minority_side_quote": 0.08,
        "same_side_existing_qty": 0.0,
        "same_side_existing_filled_entry_count": 0,
        "would_block_in_shadow": True,
    }
    event_summary = {
        "window_seconds": 60,
        "order_attempts": 0,
        "accepted_orders": 0,
        "fills": 0,
        "cancels_errors": 0,
        "latest_event": None,
        "highlight": False,
    }

    rendered = format_live_heartbeat_summary(heartbeat, event_summary, now=now, ansi_enabled=False, mode="debug")

    assert "regime_shadow_skip" in rendered.text
    assert "policy=polarized_tail_minority_side" in rendered.text
    assert "reason=veto_regime_polarized_tail_minority_side" in rendered.text


def test_live_heartbeat_shows_microstructure_summary_when_available() -> None:
    now = datetime(2026, 3, 23, 20, 14, tzinfo=timezone.utc)
    heartbeat = {
        "market_id": "M1",
        "market_window": {"start": None, "end": None},
        "strike_price": 70000,
        "q_yes": 0.52,
        "q_no": 0.48,
        "p_yes": 0.56,
        "p_no": 0.44,
        "edge_yes": 0.04,
        "edge_no": -0.04,
        "chosen_action": "buy_yes",
        "policy_bucket": "mid",
        "wallet_free_usdc": 10.0,
        "effective_bankroll": 10.0,
        "exposure": 0.0,
        "disabled_reason": None,
        "microstructure_regime": "smooth",
        "smoothness_score": 0.81,
    }
    event_summary = {
        "window_seconds": 60,
        "order_attempts": 0,
        "accepted_orders": 0,
        "fills": 0,
        "cancels_errors": 0,
        "latest_event": None,
        "highlight": False,
    }

    rendered = format_live_heartbeat_summary(heartbeat, event_summary, now=now, ansi_enabled=False, mode="info")

    assert "micro smooth s=0.81" in rendered.text


def test_live_heartbeat_includes_shadow_summary_only_in_debug() -> None:
    now = datetime(2026, 3, 23, 20, 14, tzinfo=timezone.utc)
    heartbeat = {
        "market_id": "M1",
        "market_window": {"start": None, "end": None},
        "strike_price": 70000,
        "q_yes": 0.52,
        "q_no": 0.48,
        "p_yes": 0.56,
        "p_no": 0.44,
        "edge_yes": 0.04,
        "edge_no": -0.04,
        "chosen_action": "buy_yes",
        "policy_bucket": "mid",
        "wallet_free_usdc": 10.0,
        "effective_bankroll": 10.0,
        "exposure": 0.0,
        "disabled_reason": None,
        "shadow_probability_models": {
            "kalman_blended_sigma_v1_cfg1": {
                "agrees_with_live_entry": True,
                "would_veto_live_entry": False,
                "would_flip_live_side": False,
                "shadow_only_entry": False,
                "trade_allowed": True,
                "p_yes": 0.54,
                "edge_yes": 0.03,
                "shadow_entry_side": "YES",
            },
            "gaussian_pde_diffusion_kalman_v1_cfg1": {
                "agrees_with_live_entry": False,
                "would_veto_live_entry": True,
                "would_flip_live_side": False,
                "shadow_only_entry": False,
                "trade_allowed": False,
                "p_yes": 0.49,
                "edge_yes": -0.01,
                "shadow_entry_side": None,
            },
        },
    }
    event_summary = {
        "window_seconds": 60,
        "order_attempts": 0,
        "accepted_orders": 0,
        "fills": 0,
        "cancels_errors": 0,
        "latest_event": None,
        "highlight": False,
    }

    debug_rendered = format_live_heartbeat_summary(heartbeat, event_summary, now=now, ansi_enabled=False, mode="debug")
    info_rendered = format_live_heartbeat_summary(heartbeat, event_summary, now=now, ansi_enabled=False, mode="info")

    assert "shadow kb=agree p=0.540 edge=0.030 | pde=veto p=0.490 edge=-0.010" in debug_rendered.text
    assert "shadow kb=agree" not in info_rendered.text
