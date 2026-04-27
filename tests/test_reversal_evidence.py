import pandas as pd

from src.reversal_evidence import compute_reversal_evidence


def test_upward_reversal_evidence_passes_for_yes(monkeypatch):
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')
    monkeypatch.setenv('REVERSAL_EVIDENCE_MIN_SCORE', '3')
    monkeypatch.setenv('REVERSAL_EVIDENCE_USE_STRIKE', 'true')

    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    evidence = compute_reversal_evidence(prices, side='YES', strike_price=104.0, spot_now=103.0)

    assert evidence['passes_min_score'] is True
    assert evidence['score'] >= 4
    assert evidence['last_1m_return'] > 0
    assert evidence['last_3m_return'] > 0
    assert evidence['short_slope'] > 0


def test_downward_reversal_evidence_passes_for_no(monkeypatch):
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')
    monkeypatch.setenv('REVERSAL_EVIDENCE_MIN_SCORE', '3')

    prices = pd.Series([103.0, 102.0, 101.0, 100.0])
    evidence = compute_reversal_evidence(prices, side='NO', strike_price=99.0, spot_now=100.0)

    assert evidence['passes_min_score'] is True
    assert evidence['score'] >= 4
    assert evidence['last_1m_return'] < 0
    assert evidence['last_3m_return'] < 0
    assert evidence['short_slope'] < 0


def test_insufficient_history_fails_cleanly(monkeypatch):
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')

    evidence = compute_reversal_evidence(pd.Series([100.0, 100.5, 101.0]), side='YES')

    assert evidence['passes_min_score'] is False
    assert evidence['reason'] == 'insufficient_price_history'


def test_strike_aware_moved_toward_strike_signal(monkeypatch):
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')
    monkeypatch.setenv('REVERSAL_EVIDENCE_MIN_SCORE', '1')
    monkeypatch.setenv('REVERSAL_EVIDENCE_USE_STRIKE', 'true')

    prices = pd.Series([98.0, 99.0, 100.0, 102.0])
    evidence = compute_reversal_evidence(prices, side='YES', strike_price=103.0, spot_now=102.0)

    assert evidence['moved_toward_strike'] is True
    assert evidence['passes_min_score'] is True
