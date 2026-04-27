#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


DEFAULT_DB_PATH = Path("state/canary_wallet.db")
DEFAULT_DECISION_LOG_PATH = Path("decision_state.jsonl")
DEFAULT_MARKET_DATA_DIR = Path("data/markets/btc-1h")
DEFAULT_BINANCE_DIR = Path("data/binance/btcusdt_1m")
OUTPUT_DIR = Path("artifacts/loss_forensics")
TAPES_DIR = OUTPUT_DIR / "tapes"
DETAIL_CSV_PATH = OUTPUT_DIR / "recent_entry_losses.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "recent_entry_loss_summary.csv"
REPORT_MD_PATH = OUTPUT_DIR / "recent_entry_loss_report.md"

RESOLVED_STATUSES = {"resolved", "redeemed", "archived"}
DECISION_FIELDS = [
    "timestamp",
    "market_id",
    "strike_price",
    "spot_now",
    "tau_minutes",
    "p_yes",
    "p_no",
    "q_yes",
    "q_no",
    "edge_yes",
    "edge_no",
    "policy_bucket",
    "edge_threshold_yes",
    "edge_threshold_no",
    "kelly_multiplier",
    "max_trade_notional_multiplier",
    "allow_new_entries",
    "trade_allowed",
    "action",
    "reason",
    "engine_name",
    "engine_version",
    "sigma_per_sqrt_min",
    "z_score",
    "raw_p_yes",
    "calibrated_p_yes",
    "routing_reason",
    "routing_runtime_status",
    "routing_detection_source",
    "effective_bankroll",
    "bankroll_source",
    "wallet_free_usdc",
    "regime_state",
]
ORDER_FIELDS = [
    "order_id",
    "client_order_id",
    "market_id",
    "slug",
    "entered_side",
    "winning_outcome",
    "result",
    "requested_qty",
    "filled_qty",
    "limit_price",
    "created_ts",
    "final_status",
]


@dataclass
class DecisionCandidate:
    dt: datetime
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect recent live buy-entry losses using local DB, decision logs, and recorded market data.")
    parser.add_argument("--limit", type=int, default=20, help="Number of selected entries to analyze.")
    parser.add_argument("--include-open", action="store_true", help="Include unresolved/open entries in the selected set.")
    parser.add_argument("--side", choices=("buy_yes", "buy_no", "all"), default="all", help="Filter analyzed entries by entered side.")
    parser.add_argument("--tolerance-seconds", type=int, default=120, help="Decision-log match tolerance in seconds.")
    parser.add_argument("--db-path", default=None, help="Override DB path. Defaults to BOT_DB_PATH or state/canary_wallet.db.")
    parser.add_argument("--decision-log-path", default=None, help="Override decision log path. Defaults to DECISION_LOG_PATH or decision_state.jsonl.")
    parser.add_argument("--market-data-dir", default=str(DEFAULT_MARKET_DATA_DIR), help="Recorded market data directory.")
    parser.add_argument("--binance-dir", default=str(DEFAULT_BINANCE_DIR), help="Optional local Binance 1m directory.")
    return parser.parse_args()


def _env_or_default_path(env_name: str, default: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser()
    return default


def _parse_ts(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        if text.isdigit():
            raw = int(text)
            if raw >= 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc)
            if raw >= 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(raw, tz=timezone.utc)
    except Exception:
        return None
    return None


def _ts_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _decision_field(decision: Optional[dict[str, Any]], *names: str) -> Any:
    if not isinstance(decision, dict):
        return None
    for name in names:
        if decision.get(name) is not None:
            return decision.get(name)
    policy = decision.get("policy")
    if isinstance(policy, dict):
        for name in names:
            if policy.get(name) is not None:
                return policy.get(name)
    return None


def _entered_side_from_outcome(outcome_side: Any) -> str:
    return "buy_yes" if str(outcome_side or "").upper() == "YES" else "buy_no"


def _classify_result(market_status: Any, winning_outcome: Any, outcome_side: Any) -> str:
    status = str(market_status or "").lower()
    winning = str(winning_outcome or "").upper()
    side = str(outcome_side or "").upper()
    if status in RESOLVED_STATUSES and winning in {"YES", "NO"} and side in {"YES", "NO"}:
        return "win" if winning == side else "loss"
    return "open"


def load_recent_buy_orders(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB path not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                o.id AS order_id,
                o.client_order_id,
                o.market_id,
                m.slug,
                o.outcome_side,
                o.requested_qty,
                o.filled_qty,
                o.limit_price,
                o.created_ts,
                o.status AS final_status,
                m.status AS market_status,
                m.winning_outcome
            FROM bot_orders o
            LEFT JOIN markets m ON m.market_id = o.market_id
            WHERE lower(coalesce(o.side, '')) = 'buy'
            ORDER BY o.created_ts DESC, o.id DESC
            """
        ).fetchall()
    finally:
        conn.close()
    orders: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["entered_side"] = _entered_side_from_outcome(item.get("outcome_side"))
        item["result"] = _classify_result(item.get("market_status"), item.get("winning_outcome"), item.get("outcome_side"))
        orders.append(item)
    return orders


def filter_selected_orders(orders: list[dict[str, Any]], *, limit: int, include_open: bool, side: str) -> list[dict[str, Any]]:
    filtered = []
    for order in orders:
        if side != "all" and order.get("entered_side") != side:
            continue
        if order.get("result") == "loss":
            filtered.append(order)
            continue
        if include_open and order.get("result") == "open":
            filtered.append(order)
    return filtered[: max(0, int(limit))]


def load_decision_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    decisions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                decisions.append(payload)
    return decisions


def index_decisions(decisions: Iterable[dict[str, Any]]) -> dict[str, list[DecisionCandidate]]:
    index: dict[str, list[DecisionCandidate]] = defaultdict(list)
    for row in decisions:
        market_id = row.get("market_id")
        dt = _parse_ts(row.get("timestamp") or row.get("ts") or row.get("decision_ts"))
        if not market_id or dt is None:
            continue
        index[str(market_id)].append(DecisionCandidate(dt=dt, row=row))
    for market_id in list(index):
        index[market_id].sort(key=lambda item: item.dt)
    return index


def _decision_action_side(decision: dict[str, Any]) -> Optional[str]:
    action = str(decision.get("action") or "").lower()
    if action in {"buy_yes", "buy_no"}:
        return action
    chosen = str(decision.get("chosen_side") or "").upper()
    if chosen in {"YES", "NO"}:
        return _entered_side_from_outcome(chosen)
    return None


def match_order_to_decision(order: dict[str, Any], decisions_index: dict[str, list[DecisionCandidate]], tolerance_seconds: int) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    market_id = str(order.get("market_id") or "")
    order_dt = _parse_ts(order.get("created_ts"))
    candidates = decisions_index.get(market_id) or []
    if not market_id or order_dt is None or not candidates:
        return None, None
    timestamps = [candidate.dt for candidate in candidates]
    pos = bisect_left(timestamps, order_dt)
    considered: list[tuple[int, int, int, dict[str, Any]]] = []
    desired_side = order.get("entered_side")
    for idx in range(max(0, pos - 4), min(len(candidates), pos + 5)):
        candidate = candidates[idx]
        delta = abs(int((candidate.dt - order_dt).total_seconds()))
        if delta > int(tolerance_seconds):
            continue
        action_side = _decision_action_side(candidate.row)
        side_penalty = 0 if desired_side and action_side == desired_side else 1
        future_penalty = 0 if candidate.dt <= order_dt else 1
        considered.append((side_penalty, delta, future_penalty, candidate.row))
    if not considered:
        return None, None
    considered.sort(key=lambda item: (item[0], item[1], item[2]))
    best = considered[0]
    return best[3], best[1]


def _selected_edge(decision: Optional[dict[str, Any]], entered_side: str) -> Optional[float]:
    if entered_side == "buy_yes":
        return _safe_float(_decision_field(decision, "edge_yes", "adjusted_edge_yes"))
    return _safe_float(_decision_field(decision, "edge_no", "adjusted_edge_no"))


def _selected_quote(decision: Optional[dict[str, Any]], entered_side: str) -> Optional[float]:
    if entered_side == "buy_yes":
        return _safe_float(_decision_field(decision, "q_yes"))
    return _safe_float(_decision_field(decision, "q_no"))


def _selected_probability(decision: Optional[dict[str, Any]], entered_side: str) -> Optional[float]:
    if entered_side == "buy_yes":
        return _safe_float(_decision_field(decision, "p_yes", "adjusted_p_yes", "calibrated_p_yes"))
    return _safe_float(_decision_field(decision, "p_no", "adjusted_p_no"))


def compute_forensic_flags(decision: Optional[dict[str, Any]], entered_side: str) -> dict[str, Optional[bool]]:
    policy_bucket = str(_decision_field(decision, "policy_bucket") or "").lower()
    edge = _selected_edge(decision, entered_side)
    threshold = _safe_float(_decision_field(decision, "edge_threshold_yes" if entered_side == "buy_yes" else "edge_threshold_no"))
    q_yes = _safe_float(_decision_field(decision, "q_yes"))
    q_no = _safe_float(_decision_field(decision, "q_no"))
    p_side = _selected_probability(decision, entered_side)
    q_side = _selected_quote(decision, entered_side)
    minority_side = None
    if q_yes is not None and q_no is not None:
        minority_side = "buy_yes" if q_yes < q_no else "buy_no"
    highly_polarized = False
    if q_yes is not None and q_no is not None:
        highly_polarized = min(q_yes, q_no) <= 0.10 or max(q_yes, q_no) >= 0.90
    return {
        "entered_late_bucket": policy_bucket == "late" if policy_bucket else None,
        "entered_final_bucket": policy_bucket == "final" if policy_bucket else None,
        "edge_barely_cleared_threshold": None if edge is None or threshold is None else edge >= threshold and (edge - threshold) <= 0.01,
        "entered_against_market_mid": None if q_yes is None else ((entered_side == "buy_yes" and q_yes < 0.5) or (entered_side == "buy_no" and q_yes > 0.5)),
        "model_vs_market_gap_small": None if p_side is None or q_side is None else abs(p_side - q_side) <= 0.01,
        "contrarian_tail_candidate": None if minority_side is None else highly_polarized and entered_side == minority_side,
    }


def _flatten_regime_state(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        label = value.get("regime_label")
        if label:
            return str(label)
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_detail_row(order: dict[str, Any], decision: Optional[dict[str, Any]], decision_delta_seconds: Optional[int], tape_row_count: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in ORDER_FIELDS:
        row[field] = order.get(field)
    for field in DECISION_FIELDS:
        if field == "policy_bucket":
            row[field] = _decision_field(decision, "policy_bucket")
        elif field == "regime_state":
            row[field] = _flatten_regime_state(_decision_field(decision, "regime_state"))
        else:
            row[field] = _decision_field(decision, field)
    row["decision_match_delta_seconds"] = decision_delta_seconds
    row["decision_matched"] = decision is not None
    row["market_status"] = order.get("market_status")
    row["market_data_tape_rows"] = tape_row_count
    row["market_data_tape_path"] = str(TAPES_DIR / f"{order.get('order_id')}.csv")
    row.update(compute_forensic_flags(decision, str(order.get("entered_side") or "")))
    return row


def _maybe_load_market_frames(market_data_dir: Path) -> list[Any]:
    if not market_data_dir.exists():
        return []
    frames: list[Any] = []
    candidate_paths: list[Path] = []
    direct_quotes = market_data_dir / "market_quotes.jsonl"
    if direct_quotes.exists():
        candidate_paths.append(direct_quotes)
    else:
        candidate_paths.extend(sorted(market_data_dir.glob("*.jsonl")))
        candidate_paths.extend(sorted(market_data_dir.glob("*.csv")))
        if not candidate_paths:
            candidate_paths.extend(sorted(market_data_dir.rglob("market_quotes.jsonl")))
            candidate_paths.extend(sorted(market_data_dir.rglob("*.csv")))
    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        suffixes = set(path.suffixes)
        try:
            if path.name.endswith(".jsonl"):
                rows = []
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            rows.append(item)
                if rows and pd is not None:
                    frames.append(pd.DataFrame(rows))
            elif ".csv" in suffixes or path.suffix.lower() == ".csv":
                if pd is not None:
                    frames.append(pd.read_csv(path))
        except Exception:
            continue
    return frames


def _normalize_market_frame(frame: Any) -> Optional[Any]:
    if pd is None or frame is None or frame.empty:
        return None
    ts_column = None
    for candidate in ("ts", "timestamp", "received_at", "created_ts"):
        if candidate in frame.columns:
            ts_column = candidate
            break
    if ts_column is None:
        return None
    normalized = frame.copy()
    normalized["tape_ts"] = pd.to_datetime(normalized[ts_column], utc=True, errors="coerce")
    normalized = normalized[normalized["tape_ts"].notna()].copy()
    if normalized.empty:
        return None

    def nested(source: dict[str, Any], key: str) -> Any:
        value = source.get(key)
        if isinstance(value, dict):
            return value
        return {}

    yes_quote_series = normalized["yes_quote"] if "yes_quote" in normalized.columns else None
    no_quote_series = normalized["no_quote"] if "no_quote" in normalized.columns else None
    normalized["yes_mid"] = normalized.get("yes_mid")
    normalized["no_mid"] = normalized.get("no_mid")
    normalized["yes_best_bid"] = normalized.get("yes_best_bid")
    normalized["yes_best_ask"] = normalized.get("yes_best_ask")
    normalized["no_best_bid"] = normalized.get("no_best_bid")
    normalized["no_best_ask"] = normalized.get("no_best_ask")
    if yes_quote_series is not None:
        normalized["yes_mid"] = normalized["yes_mid"].where(normalized["yes_mid"].notna(), yes_quote_series.apply(lambda item: nested({"v": item}, "v").get("mid")))
        normalized["yes_best_bid"] = normalized["yes_best_bid"].where(normalized["yes_best_bid"].notna(), yes_quote_series.apply(lambda item: nested({"v": item}, "v").get("best_bid")))
        normalized["yes_best_ask"] = normalized["yes_best_ask"].where(normalized["yes_best_ask"].notna(), yes_quote_series.apply(lambda item: nested({"v": item}, "v").get("best_ask")))
    if no_quote_series is not None:
        normalized["no_mid"] = normalized["no_mid"].where(normalized["no_mid"].notna(), no_quote_series.apply(lambda item: nested({"v": item}, "v").get("mid")))
        normalized["no_best_bid"] = normalized["no_best_bid"].where(normalized["no_best_bid"].notna(), no_quote_series.apply(lambda item: nested({"v": item}, "v").get("best_bid")))
        normalized["no_best_ask"] = normalized["no_best_ask"].where(normalized["no_best_ask"].notna(), no_quote_series.apply(lambda item: nested({"v": item}, "v").get("best_ask")))
    if "market_id" not in normalized.columns:
        normalized["market_id"] = None
    if "source" not in normalized.columns:
        normalized["source"] = None
    keep = ["tape_ts", "market_id", "source", "yes_mid", "no_mid", "yes_best_bid", "yes_best_ask", "no_best_bid", "no_best_ask"]
    return normalized[keep].sort_values("tape_ts").reset_index(drop=True)


def _load_market_tape_store(market_data_dir: Path) -> dict[str, Any]:
    if pd is None:
        return {}
    frames = []
    for frame in _maybe_load_market_frames(market_data_dir):
        normalized = _normalize_market_frame(frame)
        if normalized is not None and not normalized.empty:
            frames.append(normalized)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    store: dict[str, Any] = {}
    for market_id, bucket in combined.groupby(combined["market_id"].fillna("__missing__"), dropna=False):
        rows = bucket.sort_values("tape_ts").reset_index(drop=True)
        if market_id == "__missing__":
            store["*"] = rows
        else:
            store[str(market_id)] = rows
    return store


@lru_cache(maxsize=32)
def _load_binance_day_records(binance_dir_str: str, day_iso: str) -> tuple[dict[str, Any], ...]:
    binance_dir = Path(binance_dir_str)
    if pd is None or not binance_dir.exists():
        return ()
    from src.historical_data import load_binance_1m_dataframe

    try:
        df = load_binance_1m_dataframe(
            input_dir=binance_dir,
            start_date=day_iso,
            end_date=day_iso,
        )
    except Exception:
        return ()
    if df.empty:
        return ()
    rows = []
    for ts, record in df.iterrows():
        rows.append(
            {
                "binance_ts": ts.isoformat(),
                "binance_open": _safe_float(record.get("open")),
                "binance_high": _safe_float(record.get("high")),
                "binance_low": _safe_float(record.get("low")),
                "binance_close": _safe_float(record.get("close")),
                "binance_volume": _safe_float(record.get("volume")),
            }
        )
    return tuple(rows)


def _load_binance_window(binance_dir: Path, center_dt: datetime) -> list[dict[str, Any]]:
    if pd is None or not binance_dir.exists():
        return []
    start_ts = center_dt.astimezone(timezone.utc) - timedelta(minutes=10)
    end_ts = center_dt.astimezone(timezone.utc) + timedelta(minutes=10)
    days = {
        start_ts.date().isoformat(),
        center_dt.astimezone(timezone.utc).date().isoformat(),
        end_ts.date().isoformat(),
    }
    rows: list[dict[str, Any]] = []
    for day_iso in sorted(days):
        rows.extend(_load_binance_day_records(str(binance_dir), day_iso))
    if not rows:
        return []
    filtered = []
    for row in rows:
        ts = _parse_ts(row.get("binance_ts"))
        if ts is None:
            continue
        if start_ts <= ts <= end_ts:
            filtered.append(row)
    return filtered


def build_order_tape(order: dict[str, Any], market_tape_store: dict[str, Any], binance_dir: Path) -> list[dict[str, Any]]:
    order_dt = _parse_ts(order.get("created_ts"))
    if order_dt is None:
        return []
    rows: list[dict[str, Any]] = []
    market_id = str(order.get("market_id") or "")
    market_frame = market_tape_store.get(market_id)
    if market_frame is None:
        market_frame = market_tape_store.get("*")
    if pd is not None and market_frame is not None and not market_frame.empty:
        start_ts = pd.Timestamp(order_dt.astimezone(timezone.utc) - timedelta(minutes=10))
        end_ts = pd.Timestamp(order_dt.astimezone(timezone.utc) + timedelta(minutes=10))
        window = market_frame[(market_frame["tape_ts"] >= start_ts) & (market_frame["tape_ts"] <= end_ts)].copy()
        for record in window.to_dict(orient="records"):
            rows.append(
                {
                    "tape_ts": record.get("tape_ts").isoformat() if record.get("tape_ts") is not None else None,
                    "market_id": record.get("market_id"),
                    "source": record.get("source"),
                    "yes_mid": _safe_float(record.get("yes_mid")),
                    "no_mid": _safe_float(record.get("no_mid")),
                    "yes_best_bid": _safe_float(record.get("yes_best_bid")),
                    "yes_best_ask": _safe_float(record.get("yes_best_ask")),
                    "no_best_bid": _safe_float(record.get("no_best_bid")),
                    "no_best_ask": _safe_float(record.get("no_best_ask")),
                }
            )
    binance_rows = {item["binance_ts"]: item for item in _load_binance_window(binance_dir, order_dt)}
    merged = []
    for row in rows:
        item = dict(row)
        item.update(binance_rows.pop(item.get("tape_ts"), {}))
        merged.append(item)
    for _, item in sorted(binance_rows.items()):
        merged.append({"tape_ts": item.get("binance_ts"), **item})
    merged.sort(key=lambda item: str(item.get("tape_ts") or ""))
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in columns})


def build_summary_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_counts = Counter(str(row.get("result") or "unknown") for row in detail_rows)
    policy_counts = Counter(str(row.get("policy_bucket") or "unknown") for row in detail_rows)
    side_counts = Counter(str(row.get("entered_side") or "unknown") for row in detail_rows)
    for key, value in sorted(result_counts.items()):
        rows.append({"group": "result", "key": key, "value": value})
    for key, value in sorted(policy_counts.items()):
        rows.append({"group": "policy_bucket", "key": key, "value": value})
    for key, value in sorted(side_counts.items()):
        rows.append({"group": "entered_side", "key": key, "value": value})
    for flag in (
        "entered_late_bucket",
        "entered_final_bucket",
        "edge_barely_cleared_threshold",
        "entered_against_market_mid",
        "model_vs_market_gap_small",
        "contrarian_tail_candidate",
    ):
        true_count = sum(1 for row in detail_rows if row.get(flag) is True)
        rows.append({"group": "forensic_flag_true_count", "key": flag, "value": true_count})
    return rows


def _avg(rows: list[dict[str, Any]], field: str) -> Optional[float]:
    values = [_safe_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _worst_recent_losses(detail_rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    losses = [row for row in detail_rows if row.get("result") == "loss"]
    def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        margin = _selected_edge(row, str(row.get("entered_side") or "")) if False else None
        del margin
        side = row.get("entered_side")
        edge = _safe_float(row.get("edge_yes") if side == "buy_yes" else row.get("edge_no"))
        qty = _safe_float(row.get("filled_qty")) or _safe_float(row.get("requested_qty")) or 0.0
        price = _safe_float(row.get("limit_price")) or 0.0
        return (edge if edge is not None else 1e9, -(qty * price), str(row.get("created_ts") or ""))
    return sorted(losses, key=sort_key)[:limit]


def build_patterns(detail_rows: list[dict[str, Any]]) -> list[str]:
    if not detail_rows:
        return ["No matching entries were available, so no pattern readout is possible."]
    losses = [row for row in detail_rows if row.get("result") == "loss"]
    if not losses:
        return ["The selected sample contains no resolved losses after filtering."]
    patterns: list[str] = []
    late_losses = sum(1 for row in losses if row.get("entered_late_bucket") is True)
    final_losses = sum(1 for row in losses if row.get("entered_final_bucket") is True)
    contrarian_losses = sum(1 for row in losses if row.get("contrarian_tail_candidate") is True)
    threshold_losses = sum(1 for row in losses if row.get("edge_barely_cleared_threshold") is True)
    small_gap_losses = sum(1 for row in losses if row.get("model_vs_market_gap_small") is True)
    against_mid_losses = sum(1 for row in losses if row.get("entered_against_market_mid") is True)
    if late_losses:
        patterns.append(f"{late_losses}/{len(losses)} losses were entered in the `late` bucket.")
    if final_losses:
        patterns.append(f"{final_losses}/{len(losses)} losses were entered in the `final` bucket.")
    if threshold_losses:
        patterns.append(f"{threshold_losses}/{len(losses)} losses only barely cleared the configured edge threshold.")
    if small_gap_losses:
        patterns.append(f"{small_gap_losses}/{len(losses)} losses had model-vs-market gaps of 1 point or less.")
    if against_mid_losses:
        patterns.append(f"{against_mid_losses}/{len(losses)} losses were on the side the market mid treated as the minority view.")
    if contrarian_losses:
        patterns.append(f"{contrarian_losses}/{len(losses)} losses look like contrarian tail entries in already-polarized markets.")
    if not patterns:
        patterns.append("No single heuristic dominated the current loss sample.")
    return patterns


def render_report(detail_rows: list[dict[str, Any]], selected_orders: list[dict[str, Any]], config: dict[str, Any]) -> str:
    result_counts = Counter(str(row.get("result") or "unknown") for row in detail_rows)
    policy_counts = Counter(str(row.get("policy_bucket") or "unknown") for row in detail_rows)
    side_counts = Counter(str(row.get("entered_side") or "unknown") for row in detail_rows)
    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[str(row.get("result") or "unknown")].append(row)

    lines = [
        "# Recent Entry Loss Forensics",
        "",
        f"- Generated at: {_ts_iso(datetime.now(timezone.utc))}",
        f"- DB path: `{config['db_path']}`",
        f"- Decision log path: `{config['decision_log_path']}`",
        f"- Recorded market data dir: `{config['market_data_dir']}`",
        f"- Binance dir: `{config['binance_dir']}`",
        f"- Selection: `{len(selected_orders)}` entries, side filter `{config['side']}`, include open `{config['include_open']}`, tolerance `{config['tolerance_seconds']}s`",
        "",
        "## Counts By Result",
        "",
    ]
    if result_counts:
        for key, value in sorted(result_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No entries matched the selection.")

    lines.extend(["", "## Counts By Policy Bucket", ""])
    if policy_counts:
        for key, value in sorted(policy_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No matched decision rows.")

    lines.extend(["", "## Counts By Entered Side", ""])
    if side_counts:
        for key, value in sorted(side_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No entries.")

    lines.extend(["", "## Wins Vs Losses Averages", ""])
    for result_name in ("win", "loss"):
        bucket = grouped.get(result_name) or []
        lines.append(
            f"- {result_name}: "
            f"avg p_yes={_fmt_float(_avg(bucket, 'p_yes'))}, "
            f"avg q_yes={_fmt_float(_avg(bucket, 'q_yes'))}, "
            f"avg edge_yes={_fmt_float(_avg(bucket, 'edge_yes'))}, "
            f"avg edge_no={_fmt_float(_avg(bucket, 'edge_no'))}"
        )

    lines.extend(["", "## Worst Recent Losses", ""])
    worst = _worst_recent_losses(detail_rows)
    if worst:
        for row in worst:
            lines.append(
                f"- order `{row.get('order_id')}` market `{row.get('market_id')}` side `{row.get('entered_side')}` "
                f"created `{row.get('created_ts')}` edge_yes={_fmt_float(_safe_float(row.get('edge_yes')))} "
                f"edge_no={_fmt_float(_safe_float(row.get('edge_no')))} policy_bucket=`{row.get('policy_bucket') or 'unknown'}`"
            )
    else:
        lines.append("- No resolved losses in the selected sample.")

    lines.extend(["", "## Suspected Patterns", ""])
    for item in build_patterns(detail_rows):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _fmt_float(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> int:
    args = parse_args()
    db_path = _env_or_default_path("BOT_DB_PATH", DEFAULT_DB_PATH, args.db_path)
    decision_log_path = _env_or_default_path("DECISION_LOG_PATH", DEFAULT_DECISION_LOG_PATH, args.decision_log_path)
    market_data_dir = Path(args.market_data_dir).expanduser()
    binance_dir = Path(args.binance_dir).expanduser()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TAPES_DIR.mkdir(parents=True, exist_ok=True)

    orders = load_recent_buy_orders(db_path)
    selected_orders = filter_selected_orders(
        orders,
        limit=args.limit,
        include_open=args.include_open,
        side=args.side,
    )
    decisions = load_decision_log(decision_log_path)
    decisions_index = index_decisions(decisions)
    market_tape_store = _load_market_tape_store(market_data_dir)

    detail_rows: list[dict[str, Any]] = []
    for order in selected_orders:
        decision, delta_seconds = match_order_to_decision(order, decisions_index, args.tolerance_seconds)
        tape_rows = build_order_tape(order, market_tape_store, binance_dir)
        tape_path = TAPES_DIR / f"{order.get('order_id')}.csv"
        write_csv(tape_path, tape_rows)
        detail_rows.append(build_detail_row(order, decision, delta_seconds, len(tape_rows)))

    detail_fieldnames = ORDER_FIELDS + DECISION_FIELDS + [
        "decision_match_delta_seconds",
        "decision_matched",
        "market_status",
        "market_data_tape_rows",
        "market_data_tape_path",
        "entered_late_bucket",
        "entered_final_bucket",
        "edge_barely_cleared_threshold",
        "entered_against_market_mid",
        "model_vs_market_gap_small",
        "contrarian_tail_candidate",
    ]
    write_csv(DETAIL_CSV_PATH, detail_rows, fieldnames=detail_fieldnames)
    write_csv(SUMMARY_CSV_PATH, build_summary_rows(detail_rows), fieldnames=["group", "key", "value"])

    report = render_report(
        detail_rows,
        selected_orders,
        {
            "db_path": str(db_path),
            "decision_log_path": str(decision_log_path),
            "market_data_dir": str(market_data_dir),
            "binance_dir": str(binance_dir),
            "side": args.side,
            "include_open": args.include_open,
            "tolerance_seconds": args.tolerance_seconds,
        },
    )
    REPORT_MD_PATH.write_text(report, encoding="utf-8")

    print(f"wrote {DETAIL_CSV_PATH}")
    print(f"wrote {SUMMARY_CSV_PATH}")
    print(f"wrote {REPORT_MD_PATH}")
    print(f"selected_entries={len(selected_orders)} decision_rows={len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
