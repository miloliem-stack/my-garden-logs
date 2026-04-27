import math
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.decision_overlay import apply_polarized_tail_overlay
from src import run_bot


def _bundle(*, q_yes: float, q_no: float) -> dict:
    now = datetime.now(timezone.utc)
    return {
        'series_id': 'BTC-HOURLY',
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'strike_price': 100.0,
        'start_time': now.isoformat(),
        'end_time': (now + timedelta(minutes=20)).isoformat(),
        'yes_quote': {'mid': q_yes},
        'no_quote': {'mid': q_no},
    }


def _probability_state(*, p_yes: float, spot: float = 101.0, strike: float = 100.0, tau_minutes: int = 20, horizon_sigma: float = 0.04, z_score: float = 0.25) -> dict:
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'series_id': 'BTC-HOURLY',
        'market_id': 'M1',
        'spot_now': spot,
        'strike_price': strike,
        'tau_minutes': tau_minutes,
        'p_yes': p_yes,
        'p_no': 1.0 - p_yes,
        'blocked': False,
        'reason': None,
        'sigma_per_sqrt_min': horizon_sigma / max(math.sqrt(float(tau_minutes)), 1e-12),
        'raw_model_output': {
            'raw_output': {
                'horizon_sigma': horizon_sigma,
                'z_score': z_score,
            },
            'horizon_sigma': horizon_sigma,
            'z_score': z_score,
        },
    }


def test_overlay_leaves_normal_non_polarized_state_nearly_unchanged(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.52, q_no=0.48)
    probability_state = _probability_state(p_yes=0.55, z_score=0.4, horizon_sigma=0.03)

    overlay = apply_polarized_tail_overlay(bundle, probability_state)

    assert overlay['enabled'] is True
    assert abs(overlay['adj_p_yes'] - overlay['raw_p_yes']) < 1e-12
    assert abs(overlay['adj_p_no'] - overlay['raw_p_no']) < 1e-12
    assert overlay['tail_penalty_score'] == 0.0
    assert overlay['tail_hard_block'] is False


def test_overlay_reduces_contrarian_probability_in_moderately_polarized_state(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.95, q_no=0.05)
    probability_state = _probability_state(p_yes=0.10, spot=92.0, z_score=-2.1, horizon_sigma=0.04)

    overlay = apply_polarized_tail_overlay(bundle, probability_state)

    assert overlay['contrarian_side'] == 'YES'
    assert overlay['adj_p_yes'] < overlay['raw_p_yes']
    assert overlay['adj_p_yes'] > overlay['q_tail']
    assert overlay['adj_p_no'] > overlay['raw_p_no']
    assert overlay['tail_penalty_score'] > 0.0
    assert overlay['tail_hard_block'] is False


def test_overlay_extreme_state_turns_on_hard_block(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.99, q_no=0.01)
    probability_state = _probability_state(p_yes=0.02, spot=88.0, z_score=-3.0, horizon_sigma=0.04)

    overlay = apply_polarized_tail_overlay(bundle, probability_state)

    assert overlay['tail_hard_block'] is True
    assert overlay['contrarian_side'] == 'YES'
    assert overlay['favored_side'] == 'NO'


def test_overlay_does_not_penalize_favored_side_the_wrong_way(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.97, q_no=0.03)
    probability_state = _probability_state(p_yes=0.08, spot=90.0, z_score=-2.0, horizon_sigma=0.04)

    overlay = apply_polarized_tail_overlay(bundle, probability_state)

    assert overlay['favored_side'] == 'NO'
    assert overlay['adj_p_no'] >= overlay['raw_p_no']
    assert overlay['adj_p_yes'] <= overlay['raw_p_yes']


def test_decision_state_blocks_extreme_contrarian_trade(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.01, q_no=0.99)
    probability_state = _probability_state(p_yes=0.30, spot=88.0, z_score=-3.0, horizon_sigma=0.04)

    decision_state = run_bot.build_market_decision_state(bundle, probability_state)

    assert decision_state['trade_allowed'] is False
    assert decision_state['reason'] == 'tail_contrarian_hard_block'
    assert decision_state['contrarian_side'] == 'YES'
    assert decision_state['chosen_side'] == 'YES'
    assert decision_state['tail_hard_block'] is True


def test_decision_state_allows_normal_edge_trade(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'true')
    bundle = _bundle(q_yes=0.42, q_no=0.58)
    probability_state = _probability_state(p_yes=0.60, spot=101.0, z_score=0.5, horizon_sigma=0.04)

    decision_state = run_bot.build_market_decision_state(bundle, probability_state)

    assert decision_state['trade_allowed'] is True
    assert decision_state['reason'] == 'ok'
    assert decision_state['action'] == 'buy_yes'


def test_tail_guard_disabled_keeps_decision_behavior_unchanged(monkeypatch):
    monkeypatch.setenv('TAIL_GUARD_ENABLED', 'false')
    bundle = _bundle(q_yes=0.42, q_no=0.58)
    probability_state = _probability_state(p_yes=0.60, spot=101.0, z_score=2.8, horizon_sigma=0.04)

    overlay = apply_polarized_tail_overlay(bundle, probability_state)
    decision_state = run_bot.build_market_decision_state(bundle, probability_state)

    assert overlay['enabled'] is False
    assert overlay['adj_p_yes'] == overlay['raw_p_yes']
    assert decision_state['p_yes'] == probability_state['p_yes']
    assert decision_state['reason'] == 'ok'
