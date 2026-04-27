from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported file format: {data_path}")


def _write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
        return
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
        return
    raise ValueError(f"Unsupported file format: {out_path}")


def _flatten_quote_record(record: dict[str, Any]) -> dict[str, Any]:
    yes_quote = record.get("yes_quote") or {}
    no_quote = record.get("no_quote") or {}
    yes_state = record.get("yes_quote_state") or {}
    no_state = record.get("no_quote_state") or {}
    return {
        "market_quotes_ts": pd.to_datetime(record.get("ts"), utc=True),
        "market_quotes_source": record.get("source"),
        "market_id": record.get("market_id"),
        "token_yes": record.get("token_yes"),
        "token_no": record.get("token_no"),
        "market_yes_mid": yes_quote.get("mid"),
        "market_yes_bid": yes_quote.get("best_bid"),
        "market_yes_ask": yes_quote.get("best_ask"),
        "market_yes_spread": yes_quote.get("spread"),
        "market_yes_bid_size": yes_quote.get("bid_size"),
        "market_yes_ask_size": yes_quote.get("ask_size"),
        "market_yes_tradable": yes_state.get("tradable"),
        "market_yes_reason": yes_state.get("reason"),
        "market_no_mid": no_quote.get("mid"),
        "market_no_bid": no_quote.get("best_bid"),
        "market_no_ask": no_quote.get("best_ask"),
        "market_no_spread": no_quote.get("spread"),
        "market_no_bid_size": no_quote.get("bid_size"),
        "market_no_ask_size": no_quote.get("ask_size"),
        "market_no_tradable": no_state.get("tradable"),
        "market_no_reason": no_state.get("reason"),
    }


def enrich_event_file_with_market_quotes(
    *,
    events_path: str | Path,
    output_path: str | Path,
    quotes_path: str | Path,
    quote_tolerance_seconds: int = 300,
    drop_unmatched: bool = False,
    write_summary_json: bool = True,
) -> dict[str, Any]:
    events = _read_dataframe(events_path).copy()
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True)
    quote_rows = []
    with Path(quotes_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if "yes_quote" not in record or "no_quote" not in record:
                continue
            quote_rows.append(_flatten_quote_record(record))
    quotes = pd.DataFrame(quote_rows)
    if quotes.empty:
        enriched = events.copy()
        enriched["market_quotes_ts"] = pd.NaT
        enriched["market_quote_lag_seconds"] = pd.NA
    else:
        quotes = quotes.sort_values("market_quotes_ts").reset_index(drop=True)
        enriched = pd.merge_asof(
            events.sort_values("decision_ts").reset_index(drop=True),
            quotes,
            left_on="decision_ts",
            right_on="market_quotes_ts",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=int(quote_tolerance_seconds)),
        )
        enriched["market_quote_lag_seconds"] = (
            enriched["decision_ts"] - enriched["market_quotes_ts"]
        ).abs().dt.total_seconds()
    matched_mask = enriched["market_quotes_ts"].notna()
    if drop_unmatched:
        enriched = enriched[matched_mask].reset_index(drop=True)
        matched_mask = enriched["market_quotes_ts"].notna()
    _write_dataframe(enriched, output_path)
    report = {
        "total_rows": int(len(events)),
        "matched_rows": int(matched_mask.sum()),
        "unmatched_rows": int((~matched_mask).sum()),
        "coverage_rate": float(matched_mask.mean()) if len(enriched) else 0.0,
        "output_path": str(output_path),
    }
    if write_summary_json:
        summary_path = f"{output_path}.summary.json"
        Path(summary_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["summary_path"] = summary_path
    return report
