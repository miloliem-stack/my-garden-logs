import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import polymarket_feed


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'http {self.status_code}')

    def json(self):
        return self._payload


def test_build_btc_hourly_event_slug_matches_et_format_pm():
    ts = pd.Timestamp('2026-03-22T21:15:00Z')
    assert polymarket_feed.build_btc_hourly_event_slug(ts) == 'bitcoin-up-or-down-march-22-2026-5pm-et'


def test_build_btc_hourly_event_slug_matches_et_format_am():
    ts = pd.Timestamp('2026-01-05T05:15:00Z')
    assert polymarket_feed.build_btc_hourly_event_slug(ts) == 'bitcoin-up-or-down-january-5-2026-12am-et'


def test_candidate_btc_hourly_event_slugs_prioritize_current_prev_next_et_hour():
    now = pd.Timestamp('2026-03-22T21:15:00Z')
    assert polymarket_feed.candidate_btc_hourly_event_slugs(now) == [
        'bitcoin-up-or-down-march-22-2026-5pm-et',
        'bitcoin-up-or-down-march-22-2026-4pm-et',
        'bitcoin-up-or-down-march-22-2026-6pm-et',
    ]


def test_fetch_event_by_slug_uses_public_gamma_endpoint(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _Response({'slug': 'bitcoin-up-or-down-march-22-2026-5pm-et'})

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)

    payload = polymarket_feed.fetch_event_by_slug('bitcoin-up-or-down-march-22-2026-5pm-et')

    assert payload['slug'] == 'bitcoin-up-or-down-march-22-2026-5pm-et'
    assert calls[0][0].endswith('/events/slug/bitcoin-up-or-down-march-22-2026-5pm-et')


def test_normalize_event_payload_with_embedded_market_tokens_via_locator(monkeypatch):
    now = pd.Timestamp('2026-03-22T21:15:00Z')
    slug = 'bitcoin-up-or-down-march-22-2026-5pm-et'

    def fake_get(url, params=None, timeout=None):
        if url.endswith(f'/events/slug/{slug}'):
            return _Response({
                'slug': slug,
                'title': 'Bitcoin Up Or Down March 22, 2026 5PM ET',
                'markets': [
                    {
                        'id': 'M-BTC-1',
                        'conditionId': 'COND-BTC-1',
                        'slug': slug,
                        'startDate': '2026-03-22T21:00:00Z',
                        'endDate': '2026-03-22T22:00:00Z',
                        'status': 'open',
                        'outcomes': [
                            {'tokenId': 'YES-TOKEN', 'outcome': 'UP'},
                            {'tokenId': 'NO-TOKEN', 'outcome': 'DOWN'},
                        ],
                    }
                ],
            })
        return _Response({}, status_code=404)

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)

    result = polymarket_feed.locate_active_btc_hourly_market_with_debug(now=now)

    assert result['reason'] == 'ok'
    assert result['market']['market_id'] == 'M-BTC-1'
    assert result['market']['condition_id'] == 'COND-BTC-1'
    assert result['market']['token_yes'] == 'YES-TOKEN'
    assert result['market']['token_no'] == 'NO-TOKEN'
    assert result['market']['status'] == 'open'
    assert result['market']['detection_source'] == 'gamma_event_slug'


def test_normalize_market_payload_with_direct_tokens_via_locator(monkeypatch):
    now = pd.Timestamp('2026-03-22T21:15:00Z')
    slug = 'bitcoin-up-or-down-march-22-2026-5pm-et'

    def fake_get(url, params=None, timeout=None):
        if url.endswith(f'/events/slug/{slug}'):
            return _Response({}, status_code=404)
        if url.endswith(f'/events') and params == {'slug': slug}:
            return _Response([], status_code=200)
        if url.endswith(f'/markets/slug/{slug}'):
            return _Response({
                'id': 'M-BTC-2',
                'conditionId': 'COND-BTC-2',
                'slug': slug,
                'title': 'Bitcoin Up Or Down March 22, 2026 5PM ET',
                'startDate': '2026-03-22T21:00:00Z',
                'endDate': '2026-03-22T22:00:00Z',
                'status': 'active',
                'yesTokenId': 'YES-DIRECT',
                'noTokenId': 'NO-DIRECT',
            })
        return _Response({}, status_code=404)

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)

    result = polymarket_feed.locate_active_btc_hourly_market_with_debug(now=now)

    assert result['reason'] == 'ok'
    assert result['market']['market_id'] == 'M-BTC-2'
    assert result['market']['token_yes'] == 'YES-DIRECT'
    assert result['market']['token_no'] == 'NO-DIRECT'
    assert result['market']['detection_source'] == 'gamma_market_slug'


def test_extract_tokens_accepts_stringified_clob_token_ids_and_outcomes():
    tokens = polymarket_feed._extract_tokens({
        'clobTokenIds': '["YES-STRING", "NO-STRING"]',
        'outcomes': '["Up", "Down"]',
    })

    assert tokens['token_yes'] == 'YES-STRING'
    assert tokens['token_no'] == 'NO-STRING'


def test_extract_tokens_accepts_list_clob_token_ids_and_outcomes():
    tokens = polymarket_feed._extract_tokens({
        'clobTokenIds': ['YES-LIVE', 'NO-LIVE'],
        'outcomes': ['Up', 'Down'],
    })

    assert tokens['token_yes'] == 'YES-LIVE'
    assert tokens['token_no'] == 'NO-LIVE'


def test_normalize_event_payload_with_stringified_up_down_tokens_via_locator(monkeypatch):
    now = pd.Timestamp('2026-03-22T21:15:00Z')
    slug = 'bitcoin-up-or-down-march-22-2026-5pm-et'

    def fake_get(url, params=None, timeout=None):
        if url.endswith(f'/events/slug/{slug}'):
            return _Response({
                'slug': slug,
                'title': 'Bitcoin Up Or Down March 22, 2026 5PM ET',
                'markets': [
                    {
                        'id': 'M-BTC-STRING',
                        'conditionId': 'COND-BTC-STRING',
                        'slug': slug,
                        'startDate': '2026-03-22T21:00:00Z',
                        'endDate': '2026-03-22T22:00:00Z',
                        'status': 'open',
                        'clobTokenIds': '["YES-UP", "NO-DOWN"]',
                        'outcomes': '["Up", "Down"]',
                    }
                ],
            })
        return _Response({}, status_code=404)

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)

    result = polymarket_feed.locate_active_btc_hourly_market_with_debug(now=now)

    assert result['reason'] == 'ok'
    assert result['market']['market_id'] == 'M-BTC-STRING'
    assert result['market']['condition_id'] == 'COND-BTC-STRING'
    assert result['market']['token_yes'] == 'YES-UP'
    assert result['market']['token_no'] == 'NO-DOWN'
    assert result['market']['status'] == 'open'
    assert result['market']['detection_source'] == 'gamma_event_slug'


def test_normalize_market_bundle_derives_btc_hourly_window_from_slug():
    market = polymarket_feed._normalize_market_bundle(
        {
            'id': '1666708',
            'slug': 'bitcoin-up-or-down-march-23-2026-7am-et',
            'title': 'Bitcoin Up Or Down March 23, 2026 7AM ET',
            'startDate': '2026-03-21T11:02:35.527542Z',
            'endDate': '2026-03-23T12:00:00Z',
            'acceptingOrders': True,
            'enableOrderBook': True,
            'active': True,
            'closed': False,
            'outcomes': ['Up', 'Down'],
            'clobTokenIds': ['YES-LIVE', 'NO-LIVE'],
        },
        detection_source='gamma_market_slug',
        now=pd.Timestamp('2026-03-23T11:15:00Z'),
    )

    assert market['token_yes'] == 'YES-LIVE'
    assert market['token_no'] == 'NO-LIVE'
    assert market['startDate'] == pd.Timestamp('2026-03-23T11:00:00Z')
    assert market['endDate'] == pd.Timestamp('2026-03-23T12:00:00Z')
    assert market['status'] == 'open'
    assert polymarket_feed._validate_market_bundle(market, now=pd.Timestamp('2026-03-23T11:15:00Z')) == 'ok'


def test_normalize_market_bundle_falls_back_to_end_date_minus_one_hour():
    market = polymarket_feed._normalize_market_bundle(
        {
            'id': '1666709',
            'slug': 'not-a-btc-hourly-slug',
            'title': 'Unparseable Title',
            'startDate': '2026-03-21T11:02:35.527542Z',
            'endDate': '2026-03-23T12:00:00Z',
            'acceptingOrders': True,
            'enableOrderBook': True,
            'active': True,
            'closed': False,
            'outcomes': '["Up", "Down"]',
            'clobTokenIds': '["YES-FALLBACK", "NO-FALLBACK"]',
        },
        detection_source='gamma_market_slug',
        now=pd.Timestamp('2026-03-23T11:15:00Z'),
    )

    assert market['startDate'] == pd.Timestamp('2026-03-23T11:00:00Z')
    assert market['endDate'] == pd.Timestamp('2026-03-23T12:00:00Z')
    assert market['token_yes'] == 'YES-FALLBACK'
    assert market['token_no'] == 'NO-FALLBACK'
    assert polymarket_feed._validate_market_bundle(market, now=pd.Timestamp('2026-03-23T11:15:00Z')) == 'ok'


def test_locate_active_btc_hourly_market_with_debug_returns_structured_failure(monkeypatch):
    now = pd.Timestamp('2026-03-22T21:15:00Z')
    slug = 'bitcoin-up-or-down-march-22-2026-5pm-et'

    def fake_get(url, params=None, timeout=None):
        if url.endswith(f'/events/slug/{slug}'):
            return _Response({}, status_code=404)
        if url.endswith('/events') and params == {'slug': slug}:
            return _Response([], status_code=200)
        if url.endswith(f'/markets/slug/{slug}'):
            return _Response({
                'id': 'M-BTC-1',
                'conditionId': 'COND-BTC-1',
                'slug': slug,
                'startDate': '2026-03-22T21:00:00Z',
                'endDate': '2026-03-22T22:00:00Z',
                'status': 'open',
            })
        return _Response({}, status_code=404)

    monkeypatch.setattr(polymarket_feed.requests, 'get', fake_get)

    result = polymarket_feed.locate_active_btc_hourly_market_with_debug(now=now)

    assert result['market'] is None
    assert result['reason'] == 'missing_tokens'
    assert result['candidate_slugs'][0] == slug
    assert result['attempts'][0]['event_reason'] == 'event_fetch_failed'
    assert result['attempts'][0]['market_reason'] == 'missing_tokens'


def test_detect_active_hourly_market_aliases_btc_series(monkeypatch):
    market = {'market_id': 'M1'}
    monkeypatch.setattr(polymarket_feed, 'locate_active_btc_hourly_market_with_debug', lambda now=None: {
        'market': market,
        'reason': 'ok',
        'candidate_slugs': ['slug-1'],
        'attempts': [{'slug': 'slug-1', 'event_reason': 'ok'}],
    })

    assert polymarket_feed.detect_active_hourly_market('btc-hourly') == market


def test_detect_active_hourly_market_with_debug_for_other_series_is_structured(monkeypatch):
    monkeypatch.setattr(polymarket_feed, '_fetch_generic_events', lambda series_id: [])

    result = polymarket_feed.detect_active_hourly_market_with_debug('other-series', now=pd.Timestamp('2026-03-22T21:15:00Z'))

    assert result['market'] is None
    assert result['reason'] == 'unsupported_series_or_no_active_hourly_market'
    assert result['attempts'][0]['series_id'] == 'other-series'
