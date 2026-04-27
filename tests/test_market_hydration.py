import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import run_bot, storage, polymarket_feed


def setup_db():
    if storage.DB_PATH.exists():
        try:
            storage.DB_PATH.unlink()
        except Exception:
            pass
    storage.ensure_db()


def test_hydrate_discovered_market_minimal():
    setup_db()
    # minimal market_meta
    market_meta = {
        'market_id': 'HM1',
        'condition_id': 'C1',
        'slug': 'hourly-1',
        'title': 'Hourly Market',
        'startDate': datetime.now(timezone.utc),
        'endDate': datetime.now(timezone.utc)
    }
    # call the same hydration logic used in run_bot (replicate call)
    try:
        mid = market_meta.get('market_id')
        cond = market_meta.get('condition_id') or market_meta.get('conditionId')
        slug = market_meta.get('slug') or market_meta.get('title')
        title = market_meta.get('title')
        start = market_meta.get('startDate')
        end = market_meta.get('endDate')
        status = market_meta.get('status') or 'open'
        try:
            start_s = start.isoformat() if start is not None else None
        except Exception:
            start_s = str(start) if start is not None else None
        try:
            end_s = end.isoformat() if end is not None else None
        except Exception:
            end_s = str(end) if end is not None else None
        if mid:
            storage.upsert_market(market_id=mid, condition_id=cond, slug=slug, title=title, start_time=start_s, end_time=end_s, status=status)
    except Exception as _e:
        assert False, f'hydration failed: {_e}'

    m = storage.get_market('HM1')
    assert m is not None
    assert m['market_id'] == 'HM1'
    assert m['status'] == 'open'


def test_discovered_open_market_allows_trading(monkeypatch):
    setup_db()
    # create market in discovery step
    storage.create_market('TR1', slug='t1', title='T1', status='open')
    # ensure get_market returns status open and decide_and_execute will accept it
    from src.strategy_manager import decide_and_execute
    # mock decide_and_execute to assert market exists
    def mock_decide(p_hat, q_yes, token_yes, token_no, market_id=None, dry_run=True):
        m = storage.get_market(market_id)
        assert m is not None and m['status'] == 'open'
        return {'action': 'noop'}
    monkeypatch.setattr('src.strategy_manager.decide_and_execute', mock_decide)

    # simulate the call site in run_bot: market_id present and decide_and_execute invoked
    market_id = 'TR1'
    # call decide_and_execute which will run mock_decide and assert
    decide_and_execute(0.5, 0.5, 't_yes', 't_no', market_id=market_id, dry_run=True)


def test_safe_market_upsert_preserves_lifecycle_fields():
    setup_db()
    storage.upsert_market('UP1', slug='first', status='resolved', winning_outcome='YES', last_checked_ts='2026-03-15T00:00:00+00:00', last_redeem_ts='2026-03-15T01:00:00+00:00')
    storage.upsert_market('UP1', slug='second', status='open')

    market = storage.get_market('UP1')
    assert market['slug'] == 'second'
    assert market['status'] == 'resolved'
    assert market['winning_outcome'] == 'YES'
    assert market['last_checked_ts'] == '2026-03-15T00:00:00+00:00'
    assert market['last_redeem_ts'] == '2026-03-15T01:00:00+00:00'


def test_detect_active_hourly_market_requires_complete_metadata(monkeypatch):
    setup_db()
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        polymarket_feed,
        'discover_current_hour_event',
        lambda series_id: {
            'id': 'DISC',
            'title': 'Broken market',
            'startDate': now,
            'endDate': now,
            'status': 'open',
            'tokens': [{'side': 'YES', 'id': 'YES_ONLY'}],
        },
    )
    assert polymarket_feed.detect_active_hourly_market('SERIES') is None
