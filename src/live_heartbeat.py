from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import json
import sys
from typing import Any, Deque, Optional

import pandas as pd

ANSI_DIM = "\033[2m"
ANSI_BRIGHT = "\033[97m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"
VENUE_RESPONSE_FIELDS = {"reason", "resp", "error_message", "skip_reason", "response_status"}
IDENTIFIER_FIELDS = {"order_id", "client_order_id", "venue_order_id", "tx_hash"}
VALID_LOG_MODES = {"info", "night", "debug"}


@dataclass(frozen=True)
class LogPolicy:
    mode: str
    default_heartbeat_sec: int
    allow_ids: bool
    allow_payloads: bool
    compact_payloads: bool
    suppress_action_chatter: bool
    heartbeat_line_limit: Optional[int]
    quiet_tones: frozenset[str]


def _parse_ts(value) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _fmt_prob(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_int(value) -> str:
    if value is None:
        return "0"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _polarization_summary(source: dict) -> Optional[str]:
    zone = source.get("polarization_zone")
    if zone in {None, "", "normal", "unknown"}:
        return None
    chosen_quote = source.get("chosen_side_quote")
    raw_p = source.get("raw_p_yes_decision") if source.get("chosen_side") == "YES" else source.get("raw_p_no_decision")
    adjusted_p = source.get("adjusted_p_yes") if source.get("chosen_side") == "YES" else source.get("adjusted_p_no")
    discounted_p = source.get("discounted_p_yes") if source.get("chosen_side") == "YES" else source.get("discounted_p_no")
    raw_edge = source.get("raw_edge_yes") if source.get("chosen_side") == "YES" else source.get("raw_edge_no")
    adjusted_edge = source.get("adjusted_edge_yes") if source.get("chosen_side") == "YES" else source.get("adjusted_edge_no")
    admission_edge = source.get("admission_edge_yes") if source.get("chosen_side") == "YES" else source.get("admission_edge_no")
    credibility_weight = source.get("credibility_weight_yes") if source.get("chosen_side") == "YES" else source.get("credibility_weight_no")
    same_side_present = (float(source.get("same_side_existing_qty") or 0.0) > 0.0) or int(source.get("same_side_existing_filled_entry_count") or 0) > 0
    reversal_side = ((source.get("reversal_evidence_by_side") or {}).get(source.get("chosen_side")) or {})
    reversal_present = bool(reversal_side) and reversal_side.get("passes_min_score") is True
    block_reason = source.get("credibility_block_reason") or source.get("same_side_reentry_reason")
    return (
        f"pol action={source.get('chosen_action') or 'none'} mode={source.get('polarization_credibility_mode') or 'off'} "
        f"zone={zone} q={_fmt_prob(chosen_quote)} raw/adj/disc={_fmt_prob(raw_p)}/{_fmt_prob(adjusted_p)}/{_fmt_prob(discounted_p)} "
        f"edge={_fmt_prob(raw_edge)}/{_fmt_prob(adjusted_edge)}/{_fmt_prob(admission_edge)} "
        f"w={_fmt_prob(credibility_weight)} same={'yes' if same_side_present else 'no'} "
        f"rev={'yes' if reversal_present else 'no'} block={block_reason or 'none'}"
    )


def _fmt_window(value) -> str:
    ts = pd.to_datetime(value, utc=True) if value is not None else None
    if ts is None or pd.isna(ts):
        return "n/a"
    return ts.strftime("%H:%M")


def supports_ansi(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def get_log_mode() -> str:
    primary = os.getenv("LOG_MODE")
    fallback = os.getenv("LIVE_CONSOLE_MODE")
    raw_mode = primary if primary is not None else fallback
    mode = str(raw_mode or "info").strip().lower() or "info"
    if mode == "normal":
        mode = "info"
    return mode if mode in VALID_LOG_MODES else "info"


def get_log_policy(mode: Optional[str] = None) -> LogPolicy:
    resolved = get_log_mode() if mode is None else str(mode).strip().lower() or "info"
    if resolved == "normal":
        resolved = "info"
    if resolved not in VALID_LOG_MODES:
        resolved = "info"
    if resolved == "night":
        return LogPolicy(
            mode="night",
            default_heartbeat_sec=3600,
            allow_ids=False,
            allow_payloads=False,
            compact_payloads=True,
            suppress_action_chatter=True,
            heartbeat_line_limit=2,
            quiet_tones=frozenset({"info", "success", "warning", "dim"}),
        )
    if resolved == "debug":
        return LogPolicy(
            mode="debug",
            default_heartbeat_sec=60,
            allow_ids=True,
            allow_payloads=True,
            compact_payloads=False,
            suppress_action_chatter=False,
            heartbeat_line_limit=None,
            quiet_tones=frozenset(),
        )
    return LogPolicy(
        mode="info",
        default_heartbeat_sec=60,
        allow_ids=False,
        allow_payloads=False,
        compact_payloads=True,
        suppress_action_chatter=False,
        heartbeat_line_limit=2,
        quiet_tones=frozenset({"info", "success", "warning", "dim"}),
    )


def style_heartbeat_text(text: str, *, highlight: bool, ansi_enabled: bool) -> str:
    if not ansi_enabled:
        return text
    prefix = ANSI_BRIGHT if highlight else ANSI_DIM
    return f"{prefix}{text}{ANSI_RESET}"


def style_console_event_text(text: str, *, tone: str, ansi_enabled: bool) -> str:
    if not ansi_enabled:
        return text
    color = {
        "info": ANSI_CYAN,
        "success": ANSI_GREEN,
        "warning": ANSI_YELLOW,
        "error": ANSI_RED,
        "dim": ANSI_DIM,
    }.get(tone, ANSI_BRIGHT)
    return f"{color}{text}{ANSI_RESET}"


def _trim_text(text: str, limit: int = 80) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _regime_policy_name(veto_reason: Any) -> Optional[str]:
    mapping = {
        "veto_extreme_minority_side": "extreme_minority_side",
        "veto_regime_polarized_tail_minority_side": "polarized_tail_minority_side",
        "veto_same_side_reentry_cap": "same_side_reentry_cap",
    }
    key = None if veto_reason in (None, "") else str(veto_reason)
    return mapping.get(key, key)


def _regime_guard_summary(source: dict) -> Optional[str]:
    if not isinstance(source, dict):
        return None
    mode = source.get("regime_guard_mode")
    reason = source.get("regime_guard_reason")
    policy_name = _regime_policy_name(reason)
    blocked = bool(source.get("regime_guard_blocked"))
    would_block = bool(source.get("would_block_in_shadow"))
    if not blocked and not would_block:
        return None
    prefix = "regime_block" if blocked else "regime_shadow_skip"
    parts = [prefix]
    if mode not in (None, ""):
        parts.append(f"mode={mode}")
    if policy_name not in (None, ""):
        parts.append(f"policy={policy_name}")
    if reason not in (None, ""):
        parts.append(f"reason={reason}")
    minority_side_quote = source.get("minority_side_quote")
    if minority_side_quote is not None:
        parts.append(f"quote={_fmt_prob(minority_side_quote)}")
    same_side_existing_qty = source.get("same_side_existing_qty")
    if same_side_existing_qty is not None:
        parts.append(f"same_qty={_fmt_num(same_side_existing_qty)}")
    same_side_existing_filled_entry_count = source.get("same_side_existing_filled_entry_count")
    if same_side_existing_filled_entry_count is not None:
        parts.append(f"same_entries={_fmt_int(same_side_existing_filled_entry_count)}")
    return " ".join(parts)


def _microstructure_summary(source: dict) -> Optional[str]:
    if not isinstance(source, dict):
        return None
    regime = source.get("microstructure_regime")
    score = source.get("smoothness_score")
    if regime in (None, "", "unknown", "disabled") and score is None:
        return None
    parts = []
    if regime not in (None, "", "unknown", "disabled"):
        parts.append(f"micro {regime}")
    if score is not None:
        parts.append(f"s={_fmt_num(score)}")
    return " ".join(parts) or None


def _compact_payload_fragment(value: Any, *, max_len: int = 80) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        preferred_keys = ("status", "reason", "error_message", "skip_reason", "message", "code")
        parts = []
        for key in preferred_keys:
            item = value.get(key)
            if item in (None, "", [], {}):
                continue
            parts.append(f"{key}={_trim_text(item, 24)}")
        if not parts:
            keys = ",".join(sorted(str(key) for key in value.keys())[:4]) or "dict"
            parts.append(f"keys={keys}")
        return _trim_text("; ".join(parts), max_len)
    if isinstance(value, (list, tuple)):
        return _trim_text(json.dumps(value, default=str), max_len)
    return _trim_text(value, max_len)


def format_highlighted_response_field(
    key: str,
    value: Any,
    *,
    mode: Optional[str] = None,
    ansi_enabled: Optional[bool] = None,
) -> Optional[str]:
    if value in (None, "", [], {}):
        return None
    policy = get_log_policy(mode)
    ansi = supports_ansi() if ansi_enabled is None else bool(ansi_enabled)
    rendered = _compact_payload_fragment(value, max_len=120 if policy.allow_payloads else 56)
    token = f"{key}=[{rendered}]"
    if not ansi:
        return token
    return f"{ANSI_YELLOW}{token}{ANSI_RESET}"


def _render_status_value(key: str, value: Any, *, policy: LogPolicy) -> Optional[str]:
    if value is None:
        return None
    if key in IDENTIFIER_FIELDS and not policy.allow_ids:
        return None
    if key == "resp" and not policy.allow_payloads:
        return None
    if key in VENUE_RESPONSE_FIELDS:
        return format_highlighted_response_field(key, value, mode=policy.mode, ansi_enabled=False)
    if key in {"notional", "free", "reserved", "pnl", "wallet", "bankroll", "exposure", "realized"}:
        return _fmt_money(value)
    if key in {"px", "price", "edge", "q_yes", "q_no", "p_yes", "p_no"}:
        return _fmt_prob(value)
    if key in {"qty"}:
        return _fmt_num(value)
    if isinstance(value, dict):
        return _compact_payload_fragment(value, max_len=120 if policy.allow_payloads else 56) if policy.allow_payloads else None
    if isinstance(value, (list, tuple)):
        return _compact_payload_fragment(value, max_len=120 if policy.allow_payloads else 56) if policy.allow_payloads else None
    return str(value)


def should_print_console_event(*, mode: Optional[str] = None, tone: str = "info", critical: bool = False) -> bool:
    policy = get_log_policy(mode)
    if policy.mode == "night":
        return critical or tone not in policy.quiet_tones
    return True


def is_debug_mode(mode: Optional[str] = None) -> bool:
    return get_log_policy(mode).mode == "debug"


@dataclass
class HeartbeatRenderResult:
    text: str
    highlight: bool
    ansi_enabled: bool

    def styled_text(self) -> str:
        return style_heartbeat_text(self.text, highlight=self.highlight, ansi_enabled=self.ansi_enabled)


class RollingEventBuffer:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = int(window_seconds)
        self._events: Deque[dict] = deque()

    def record(self, event_type: str, *, ts=None, real_execution: bool = False, **fields) -> None:
        event_ts = _parse_ts(ts)
        self._events.append(
            {
                "event_type": event_type,
                "ts": event_ts,
                "real_execution": bool(real_execution),
                **fields,
            }
        )
        self._prune(event_ts)

    def _prune(self, now=None) -> None:
        cutoff = _parse_ts(now) - pd.Timedelta(seconds=self.window_seconds)
        while self._events and self._events[0]["ts"] < cutoff:
            self._events.popleft()

    def summarize(self, now=None, since=None) -> dict:
        now_ts = _parse_ts(now)
        self._prune(now_ts)
        recent = [event for event in self._events if event["ts"] >= now_ts - pd.Timedelta(seconds=self.window_seconds)]
        since_ts = _parse_ts(since) if since is not None else None
        recent_since = [event for event in recent if since_ts is None or event["ts"] > since_ts]
        fills = sum(1 for event in recent if event["event_type"] == "fill")
        partial_fills = sum(1 for event in recent if event["event_type"] == "fill" and event.get("status") == "partially_filled")
        accepted = sum(1 for event in recent if event["event_type"] == "order_accept")
        attempts = sum(1 for event in recent if event["event_type"] == "order_attempt")
        cancels_errors = sum(1 for event in recent if event["event_type"] in {"cancel", "error"})
        entry_attempts = sum(1 for event in recent if event["event_type"] == "order_attempt" and str(event.get("action") or "").startswith("buy_"))
        exit_attempts = sum(1 for event in recent if event["event_type"] == "order_attempt" and str(event.get("action") or "").startswith("sell_"))
        rejected = sum(1 for event in recent if event["event_type"] == "error" and event.get("status") == "rejected")
        redeem_attempted = sum(1 for event in recent if event["event_type"] == "redeem")
        redeem_succeeded = sum(1 for event in recent if event["event_type"] == "redeem" and event.get("status") in {"ok", "success", "redeemed"})
        redeem_failed = sum(1 for event in recent if event["event_type"] == "redeem" and event.get("status") not in {"ok", "success", "redeemed"})
        notable = recent[-1] if recent else None
        return {
            "window_seconds": self.window_seconds,
            "market_switches": sum(1 for event in recent if event["event_type"] == "market_switch"),
            "order_attempts": attempts,
            "entries_submitted": entry_attempts,
            "exits_submitted": exit_attempts,
            "accepted_orders": accepted,
            "rejected_orders": rejected,
            "fills": fills,
            "partial_fills": partial_fills,
            "cancels_errors": cancels_errors,
            "redeem_events": redeem_attempted,
            "redeems_attempted": redeem_attempted,
            "redeems_succeeded": redeem_succeeded,
            "redeems_failed": redeem_failed,
            "latest_event": notable,
            "highlight": any(event.get("real_execution") for event in recent_since),
        }


def describe_event(event: Optional[dict]) -> Optional[str]:
    if not event:
        return None
    event_type = event.get("event_type")
    if event_type == "fill":
        side = event.get("side") or event.get("action") or event.get("outcome_side")
        qty = _fmt_num(event.get("qty"), digits=2)
        price = _fmt_prob(event.get("price"))
        return f"fill {side or 'order'} qty={qty} px={price}"
    if event_type == "order_accept":
        side = event.get("side") or event.get("action")
        status = event.get("status") or "accepted"
        return f"accepted {side or 'order'} status={status}"
    if event_type == "order_attempt":
        side = event.get("side") or event.get("action")
        return f"attempted {side or 'order'}"
    if event_type == "cancel":
        reason = event.get("reason") or event.get("status") or "cancel"
        return f"cancel {reason}"
    if event_type == "error":
        reason = event.get("reason") or event.get("status") or "error"
        return f"error {reason}"
    if event_type == "market_switch":
        market_id = event.get("new_market_id") or event.get("market_id")
        return f"market_switch {market_id or 'unknown'}"
    if event_type == "redeem":
        status = event.get("status") or "attempted"
        return f"redeem {status}"
    return str(event_type)


def _shadow_status(payload: dict) -> str:
    if payload.get("error"):
        return "error"
    if payload.get("agrees_with_live_entry"):
        return "agree"
    if payload.get("would_veto_live_entry"):
        return "veto"
    if payload.get("would_flip_live_side"):
        return "flip"
    if payload.get("shadow_only_entry"):
        return "shadow"
    if payload.get("trade_allowed"):
        return "entry"
    return "none"


def _shadow_engine_label(engine_name: str) -> str:
    aliases = {
        "kalman_blended_sigma_v1_cfg1": "kb",
        "gaussian_pde_diffusion_kalman_v1_cfg1": "pde",
    }
    return aliases.get(engine_name, engine_name)


def _shadow_summary(heartbeat: dict) -> Optional[str]:
    shadow_models = heartbeat.get("shadow_probability_models") or {}
    if not shadow_models:
        return None
    parts = []
    for engine_name, payload in shadow_models.items():
        if not isinstance(payload, dict):
            continue
        label = _shadow_engine_label(str(engine_name))
        status = _shadow_status(payload)
        if payload.get("error"):
            parts.append(f"{label}=error")
            continue
        edge = payload.get("edge_yes")
        if payload.get("shadow_entry_side") == "NO":
            edge = payload.get("edge_no")
        parts.append(f"{label}={status} p={_fmt_prob(payload.get('p_yes'))} edge={_fmt_prob(edge)}")
    return None if not parts else "shadow " + " | ".join(parts)


def format_live_heartbeat_summary(
    heartbeat: dict,
    event_summary: dict,
    *,
    now=None,
    ansi_enabled: Optional[bool] = None,
    mode: Optional[str] = None,
) -> HeartbeatRenderResult:
    now_ts = _parse_ts(now)
    ansi = supports_ansi() if ansi_enabled is None else bool(ansi_enabled)
    policy = get_log_policy(mode)
    market_window = heartbeat.get("market_window") or {}
    market_label = heartbeat.get("series_id") or heartbeat.get("active_market_id") or heartbeat.get("market_id") or "market=n/a"
    action = heartbeat.get("chosen_action") or "none"
    bucket = heartbeat.get("policy_bucket") or "n/a"
    first = (
        f"[HB {now_ts.strftime('%H:%MZ')}] {market_label} | "
        f"{_fmt_window(market_window.get('start'))}-{_fmt_window(market_window.get('end'))} UTC | "
        f"strike {_fmt_num(heartbeat.get('strike_price'), digits=2)} | "
        f"bucket {bucket} | action {action}"
    )
    second = (
        f"mkt/model/edge yes {_fmt_prob(heartbeat.get('q_yes'))}/{_fmt_prob(heartbeat.get('p_yes'))}/{_fmt_prob(heartbeat.get('edge_yes'))} "
        f"no {_fmt_prob(heartbeat.get('q_no'))}/{_fmt_prob(heartbeat.get('p_no'))}/{_fmt_prob(heartbeat.get('edge_no'))} | "
        f"wallet {_fmt_money(heartbeat.get('wallet_free_usdc'))} | "
        f"exposure {_fmt_money(heartbeat.get('exposure'))} | "
        f"counts a/{event_summary.get('accepted_orders', 0)} f/{event_summary.get('fills', 0)} e/{event_summary.get('cancels_errors', 0)}"
    )
    blocked = heartbeat.get("disabled_reason")
    if blocked:
        second += f" | blocked {blocked}"
    regime_guard = _regime_guard_summary(heartbeat)
    if regime_guard:
        second += f" | {regime_guard}"
    microstructure = _microstructure_summary(heartbeat)
    if microstructure:
        second += f" | {microstructure}"
    lines = [first, second]
    polarization = _polarization_summary(heartbeat)
    if polarization:
        lines.append(polarization)
    if policy.mode == "debug":
        third = (
            f"bankroll {_fmt_money(heartbeat.get('effective_bankroll'))} | "
            f"last {event_summary.get('window_seconds', 60)}s: attempts {event_summary.get('order_attempts', 0)} "
            f"accepted {event_summary.get('accepted_orders', 0)} fills {event_summary.get('fills', 0)} "
            f"cancels/errors {event_summary.get('cancels_errors', 0)}"
        )
        lines.append(third)
        if (
            heartbeat.get("expected_log_growth_entry") is not None
            or heartbeat.get("position_reeval_shadow_best_action") is not None
        ):
            lines.append(
                "growth "
                f"entry {_fmt_num(heartbeat.get('expected_log_growth_entry'), digits=6)}/"
                f"{_fmt_num(heartbeat.get('expected_log_growth_entry_conservative'), digits=6)} "
                f"old={_fmt_num(heartbeat.get('expected_log_growth_entry_conservative_old'), digits=6)} "
                f"disc={_fmt_num(heartbeat.get('expected_log_growth_entry_conservative_discounted'), digits=6)} "
                f"pass={heartbeat.get('expected_log_growth_pass_shadow')} "
                f"reeval={heartbeat.get('position_reeval_shadow_best_action') or 'n/a'} "
                f"gain={_fmt_num(heartbeat.get('position_reeval_shadow_best_growth_gain'), digits=6)} "
                f"exec={heartbeat.get('position_reeval_shadow_best_executable')}"
            )
        latest = describe_event(event_summary.get("latest_event"))
        if latest:
            lines.append(f"last event: {latest}")
        shadow = _shadow_summary(heartbeat)
        if shadow:
            lines.append(shadow)
    text = "\n".join(lines)
    return HeartbeatRenderResult(text=text, highlight=bool(event_summary.get("highlight")), ansi_enabled=ansi)


def format_night_heartbeat_summary(
    heartbeat: dict,
    event_summary: dict,
    inventory_summary: dict,
    *,
    now=None,
    ansi_enabled: Optional[bool] = None,
    mode: Optional[str] = None,
) -> HeartbeatRenderResult:
    now_ts = _parse_ts(now)
    ansi = supports_ansi() if ansi_enabled is None else bool(ansi_enabled)
    market_label = heartbeat.get("series_id") or heartbeat.get("active_market_id") or heartbeat.get("market_id") or "market=n/a"
    line_one = (
        f"[HB {now_ts.strftime('%H:%MZ')}] {market_label} | "
        f"free {_fmt_money(heartbeat.get('wallet_free_usdc'))} | "
        f"reserved {_fmt_money(heartbeat.get('wallet_reserved_exposure'))} | "
        f"lots {_fmt_int(inventory_summary.get('open_lot_count'))} | "
        f"resolved {_fmt_int(inventory_summary.get('resolved_pending_count'))} | "
        f"realized {_fmt_money(inventory_summary.get('realized_pnl_total'))}"
    )
    line_two = (
        f"entries {_fmt_int(event_summary.get('entries_submitted'))} | "
        f"accepted {_fmt_int(event_summary.get('accepted_orders'))} | "
        f"rejected {_fmt_int(event_summary.get('rejected_orders'))} | "
        f"fills {_fmt_int(event_summary.get('fills'))} | "
        f"partials {_fmt_int(event_summary.get('partial_fills'))} | "
        f"exits {_fmt_int(event_summary.get('exits_submitted'))} | "
        f"redeem a/s/f {_fmt_int(event_summary.get('redeems_attempted'))}/{_fmt_int(event_summary.get('redeems_succeeded'))}/{_fmt_int(event_summary.get('redeems_failed'))}"
    )
    blocked = heartbeat.get("disabled_reason")
    if blocked:
        line_two += f" | blocked {blocked}"
    regime_guard = _regime_guard_summary(heartbeat)
    if regime_guard:
        line_two += f" | {regime_guard}"
    return HeartbeatRenderResult(text="\n".join([line_one, line_two]), highlight=bool(event_summary.get("highlight")), ansi_enabled=ansi)


def format_console_action_line(action: dict, *, mode: Optional[str] = None, ansi_enabled: Optional[bool] = None) -> str:
    ansi = supports_ansi() if ansi_enabled is None else bool(ansi_enabled)
    policy = get_log_policy(mode)
    action_name = str(action.get("side") or action.get("action") or "").upper()
    market_id = action.get("market_id") or (action.get("decision_state") or {}).get("market_id") or "n/a"
    qty = action.get("qty")
    if qty is None:
        qty = action.get("submitted_qty")
    if qty is None:
        qty = action.get("filled_qty")
    price = action.get("price")
    if price is None:
        price = action.get("avg_fill_price")
    if price is None:
        price = action.get("executable_exit_price")
    notional = action.get("quantized_notional")
    if notional is None and qty is not None and price is not None:
        try:
            notional = float(qty) * float(price)
        except (TypeError, ValueError):
            notional = None
    status = ((action.get("resp") or {}).get("status")) or action.get("status") or "ok"
    response_value = action.get("reason") or action.get("error_message") or action.get("skip_reason") or action.get("response_status") or action.get("resp")
    tone = "info"
    label = action_name or "EVENT"
    extra = []
    if label.startswith("SELL_"):
        label = "EXIT"
        tone = "success" if action.get("expected_profit_total", 0.0) >= 0 else "warning"
        extra.append(f"side={action.get('held_side')}")
        extra.append(f"pnl={_fmt_money(action.get('expected_profit_total'))}")
    elif label.startswith("BUY_"):
        tone = "info"
        extra.append(f"edge={_fmt_prob(action.get('edge'))}")
    if status in {"filled"}:
        tone = "success"
    elif status in {"partially_filled"}:
        tone = "warning"
        label = "FILL"
    elif status in {"rejected", "error", "failed"}:
        tone = "error"
    text = f"{label:<6} | mkt={market_id} px={_fmt_prob(price)} qty={_fmt_num(qty)} notional={_fmt_money(notional)} status={status}"
    if extra:
        text += " | " + " ".join(extra)
    if policy.allow_ids:
        for key in ("order_id", "client_order_id", "venue_order_id", "tx_hash"):
            value = action.get(key) or (action.get("resp") or {}).get(key)
            if value not in (None, ""):
                text += f" | {key}={value}"
    highlighted = format_highlighted_response_field(
        "resp" if isinstance(response_value, dict) else ("reason" if action.get("reason") else "response_status"),
        response_value,
        mode=policy.mode,
        ansi_enabled=False,
    )
    if highlighted:
        text += f" | {highlighted}"
    elif policy.allow_payloads and action.get("routing_debug"):
        text += f" | routing_debug={_compact_payload_fragment(action.get('routing_debug'), max_len=120)}"
    guard_source = action if action.get("regime_guard_mode") is not None else (action.get("decision_state") or {})
    regime_guard = _regime_guard_summary(guard_source)
    if regime_guard:
        text += f" | {regime_guard}"
    return style_console_event_text(text, tone=tone, ansi_enabled=ansi)


def format_console_status_line(label: str, *, mode: Optional[str] = None, tone: str = "info", ansi_enabled: Optional[bool] = None, **fields) -> str:
    ansi = supports_ansi() if ansi_enabled is None else bool(ansi_enabled)
    policy = get_log_policy(mode)
    parts = [f"{str(label).upper():<6}"]
    for key, value in fields.items():
        rendered = _render_status_value(key, value, policy=policy)
        if rendered is None:
            continue
        parts.append(f"{key}={rendered}")
    rendered_line = " | ".join(parts)
    if ansi and any(key in VENUE_RESPONSE_FIELDS for key in fields):
        for key in VENUE_RESPONSE_FIELDS:
            value = fields.get(key)
            token = format_highlighted_response_field(key, value, mode=policy.mode, ansi_enabled=False)
            colored = format_highlighted_response_field(key, value, mode=policy.mode, ansi_enabled=True)
            if token and colored:
                rendered_line = rendered_line.replace(token, colored)
    return style_console_event_text(rendered_line, tone=tone, ansi_enabled=ansi)


def format_heartbeat(
    heartbeat: dict,
    event_summary: dict,
    *,
    inventory_summary: Optional[dict] = None,
    now=None,
    mode: Optional[str] = None,
    ansi_enabled: Optional[bool] = None,
) -> HeartbeatRenderResult:
    policy = get_log_policy(mode)
    if policy.mode == "night":
        return format_night_heartbeat_summary(
            heartbeat,
            event_summary,
            inventory_summary or {},
            now=now,
            mode=policy.mode,
            ansi_enabled=ansi_enabled,
        )
    return format_live_heartbeat_summary(
        heartbeat,
        event_summary,
        now=now,
        mode=policy.mode,
        ansi_enabled=ansi_enabled,
    )
