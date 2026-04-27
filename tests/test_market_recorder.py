import json

import pandas as pd

from src.runtime.market_recorder import _normalize_ws_message_to_quote
from src.market_event_enrichment import enrich_event_file_with_market_quotes


def test_normalize_ws_message_to_quote_extracts_orderbook():
    message = {
        "event": "book",
        "asset_id": "YES123",
        "book": {
            "bids": [{"price": "0.41", "size": "12"}],
            "asks": [{"price": "0.43", "size": "8"}],
        },
    }
    out = _normalize_ws_message_to_quote(message, token_ids={"YES123", "NO123"}, received_at=pd.Timestamp("2026-04-03T12:00:00Z"))
    assert out is not None
    assert out["token_id"] == "YES123"
    assert out["quote"]["best_bid"] == 0.41
    assert out["quote"]["best_ask"] == 0.43
    assert out["quote"]["source"] == "polymarket_ws_orderbook"
    assert out["quote"]["age_seconds"] == 0.0


def test_enrich_event_file_with_market_quotes_matches_nearest_snapshot(tmp_path):
    events = pd.DataFrame(
        [
            {"decision_ts": "2026-04-03T12:00:10Z", "tau_minutes": 60, "realized_yes": 1},
            {"decision_ts": "2026-04-03T12:05:10Z", "tau_minutes": 55, "realized_yes": 0},
        ]
    )
    events_path = tmp_path / "events.csv"
    events.to_csv(events_path, index=False)
    quotes_path = tmp_path / "quotes.jsonl"
    quote_rows = [
        {
            "ts": "2026-04-03T12:00:08Z",
            "source": "websocket",
            "market_id": "M1",
            "token_yes": "YES1",
            "token_no": "NO1",
            "yes_quote": {"mid": 0.44, "best_bid": 0.43, "best_ask": 0.45, "spread": 0.02, "bid_size": 10, "ask_size": 12},
            "no_quote": {"mid": 0.56, "best_bid": 0.55, "best_ask": 0.57, "spread": 0.02, "bid_size": 9, "ask_size": 11},
            "yes_quote_state": {"tradable": True, "reason": None},
            "no_quote_state": {"tradable": True, "reason": None},
        },
        {
            "ts": "2026-04-03T12:05:12Z",
            "source": "poll",
            "market_id": "M1",
            "token_yes": "YES1",
            "token_no": "NO1",
            "yes_quote": {"mid": 0.47, "best_bid": 0.46, "best_ask": 0.48, "spread": 0.02, "bid_size": 8, "ask_size": 7},
            "no_quote": {"mid": 0.53, "best_bid": 0.52, "best_ask": 0.54, "spread": 0.02, "bid_size": 6, "ask_size": 6},
            "yes_quote_state": {"tradable": True, "reason": None},
            "no_quote_state": {"tradable": True, "reason": None},
        },
    ]
    with quotes_path.open("w", encoding="utf-8") as handle:
        for row in quote_rows:
            handle.write(json.dumps(row) + "\n")
    output_path = tmp_path / "enriched.csv"
    report = enrich_event_file_with_market_quotes(
        events_path=events_path,
        output_path=output_path,
        quotes_path=quotes_path,
        quote_tolerance_seconds=10,
    )
    enriched = pd.read_csv(output_path)
    assert report["matched_rows"] == 2
    assert list(enriched["market_yes_mid"]) == [0.44, 0.47]
    assert list(enriched["market_quotes_source"]) == ["websocket", "poll"]
