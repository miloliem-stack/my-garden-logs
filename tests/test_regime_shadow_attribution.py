import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import storage, strategy_manager
from src.regime_detector import compute_microstructure_regime, detect_regime


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_detector(history, prev_state, decision_inputs):
    previous = dict(prev_state or {})
    previous['recent_observations'] = list(history)
    out = detect_regime(decision_inputs, previous_state=previous)
    history.append({'source': out['source']})
    return out


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def test_regime_detector_is_deterministic_and_transitions_with_persistence():
    history = []
    prev = None
    normal = {'tau_minutes': 50, 'q_yes': 0.55, 'p_yes': 0.57, 'abs_z_score': 0.5, 'spot_now': 100.0}
    prev = _step_detector(history, prev, normal)
    assert prev['regime_label'] == 'normal'
    assert prev['shadow_only'] is True
    assert prev['used_market_quote_proxy'] is False

    trend_inputs = [
        {'tau_minutes': 45, 'q_yes': 0.60, 'p_yes': 0.63, 'abs_z_score': 1.0, 'spot_now': 101.0},
        {'tau_minutes': 43, 'q_yes': 0.62, 'p_yes': 0.64, 'abs_z_score': 1.1, 'spot_now': 102.0},
        {'tau_minutes': 41, 'q_yes': 0.64, 'p_yes': 0.66, 'abs_z_score': 1.2, 'spot_now': 103.0},
        {'tau_minutes': 39, 'q_yes': 0.66, 'p_yes': 0.68, 'abs_z_score': 1.3, 'spot_now': 104.0},
        {'tau_minutes': 37, 'q_yes': 0.68, 'p_yes': 0.70, 'abs_z_score': 1.4, 'spot_now': 105.0},
    ]
    for item in trend_inputs:
        prev = _step_detector(history, prev, item)
    assert prev['regime_label'] == 'trend'
    assert prev['trend_score'] >= 0.65

    tail_state = {'tau_minutes': 12, 'q_yes': 0.95, 'p_yes': 0.94, 'abs_z_score': 2.2, 'spot_now': 104.0}
    prev = _step_detector(history, prev, tail_state)
    prev = _step_detector(history, prev, tail_state)
    assert prev['regime_label'] == 'polarized_tail'
    assert prev['market_polarization'] >= 0.88

    repeat = detect_regime(tail_state, previous_state={**prev, 'recent_observations': list(history)})
    assert repeat['regime_label'] == detect_regime(tail_state, previous_state={**prev, 'recent_observations': list(history)})['regime_label']


def test_regime_shadow_mode_attaches_state_without_changing_entry_decision(monkeypatch):
    ts = _ts()
    storage.create_market('RM1', status='open')
    calls = []

    def fake_buy(*args, **kwargs):
        calls.append(kwargs)
        return {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.55}

    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', fake_buy)
    monkeypatch.setattr(strategy_manager, 'BOT_BANKROLL', 1000.0)
    monkeypatch.setattr(strategy_manager, 'PER_TRADE_CAP_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'TOTAL_EXPOSURE_CAP', 10.0)
    decision_state = {
        'ts': ts,
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.45,
        'edge_yes': 0.07,
        'edge_no': -0.07,
        'tau_minutes': 30,
        'abs_z_score': 1.1,
        'spot_now': 100.0,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': {'edge_threshold_yes': 0.02, 'edge_threshold_no': 0.02, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0, 'allow_new_entries': True},
    }

    monkeypatch.setattr(strategy_manager, 'REGIME_SHADOW_MODE', False)
    baseline = strategy_manager.build_trade_action(dict(decision_state), 'YES1', 'NO1', 'RM1', dry_run=True)
    monkeypatch.setattr(strategy_manager, 'REGIME_SHADOW_MODE', True)
    monkeypatch.setattr(strategy_manager, 'REGIME_DECISION_ACTIVE', False)
    shadow_state = dict(decision_state)
    shadow = strategy_manager.build_trade_action(shadow_state, 'YES1', 'NO1', 'RM1', dry_run=True)

    assert baseline['side'] == shadow['side'] == 'buy_yes'
    assert baseline['price'] == shadow['price'] == 0.55
    assert 'regime_state' in shadow_state
    assert shadow_state['regime_state']['shadow_only'] is True
    assert calls[-1]['decision_context']['regime_state']['regime_label'] in {'normal', 'trend', 'polarized_tail', 'reversal_attempt'}


def test_microstructure_regime_returns_unknown_when_history_is_insufficient(monkeypatch):
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_ENABLED', 'true')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_WINDOW', '32')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_MIN_OBS', '24')
    prices = pd.Series([100.0 + 0.1 * i for i in range(20)])

    result = compute_microstructure_regime(prices)

    assert result['microstructure_regime'] == 'unknown'
    assert result['spectral_ready'] is False
    assert result['spectral_reason'] == 'insufficient_history'
    assert result['spectral_observation_count'] == 19
    assert result['spectral_window_minutes'] == 32


def test_microstructure_regime_classifies_smooth_low_frequency_series(monkeypatch):
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_ENABLED', 'true')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_WINDOW', '32')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_MIN_OBS', '24')
    points = 80
    index = pd.date_range('2026-01-01T00:00:00Z', periods=points, freq='min')
    phase = np.arange(points, dtype=float)
    prices = pd.Series(100.0 + 2.5 * np.sin((2.0 * np.pi * phase) / points), index=index, dtype=float)

    result = compute_microstructure_regime(prices)

    assert result['spectral_ready'] is True
    assert result['microstructure_regime'] == 'smooth'
    assert result['smoothness_score'] >= 0.67
    assert result['low_freq_power_ratio'] > result['high_freq_power_ratio']


def test_microstructure_regime_classifies_noisy_high_frequency_series(monkeypatch):
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_ENABLED', 'true')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_WINDOW', '32')
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_MIN_OBS', '24')
    points = 80
    index = pd.date_range('2026-01-01T00:00:00Z', periods=points, freq='min')
    values = [100.0 + (0.8 if i % 2 == 0 else -0.8) for i in range(points)]
    prices = pd.Series(values, index=index, dtype=float)

    result = compute_microstructure_regime(prices)

    assert result['spectral_ready'] is True
    assert result['microstructure_regime'] == 'noisy'
    assert result['smoothness_score'] <= 0.40
    assert result['high_freq_power_ratio'] > result['low_freq_power_ratio']


def test_microstructure_regime_returns_non_ready_state_when_env_off(monkeypatch):
    monkeypatch.setenv('MICROSTRUCTURE_SPECTRAL_ENABLED', 'false')

    result = compute_microstructure_regime(pd.Series([100.0, 100.1, 100.2]))

    assert result['microstructure_regime'] == 'unknown'
    assert result['spectral_ready'] is False
    assert result['spectral_reason'] in {'disabled', 'insufficient_history'}


def test_regime_observation_persists_microstructure_fields(monkeypatch):
    ts = _ts()
    storage.create_market('RM2', status='open')

    def fake_buy(*args, **kwargs):
        return {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.55}

    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', fake_buy)
    monkeypatch.setattr(strategy_manager, 'REGIME_SHADOW_MODE', True)
    monkeypatch.setattr(strategy_manager, 'REGIME_DECISION_ACTIVE', False)
    decision_state = {
        'ts': ts,
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.45,
        'edge_yes': 0.07,
        'edge_no': -0.07,
        'tau_minutes': 30,
        'abs_z_score': 1.1,
        'spot_now': 100.0,
        'microstructure_regime': 'smooth',
        'spectral_entropy': 0.1,
        'low_freq_power_ratio': 0.85,
        'high_freq_power_ratio': 0.03,
        'smoothness_score': 0.9,
        'spectral_observation_count': 32,
        'spectral_window_minutes': 32,
        'spectral_ready': True,
        'spectral_reason': 'ok',
        'trade_allowed': True,
        'reason': 'ok',
        'policy': {'edge_threshold_yes': 0.02, 'edge_threshold_no': 0.02, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0, 'allow_new_entries': True},
    }

    strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'RM2', dry_run=True)

    latest = storage.get_latest_regime_observation('RM2')
    assert latest is not None
    assert latest['microstructure_regime'] == 'smooth'
    assert latest['smoothness_score'] == 0.9
    assert latest['decision_state']['microstructure_regime'] == 'smooth'
    assert latest['source']['microstructure_regime'] == 'smooth'


def test_buy_fill_creates_lot_regime_attribution_and_sell_realized_event():
    ts = _ts()
    storage.create_market('ATTR1', status='open')
    buy_order = storage.create_order(
        'buy-attr-1',
        'ATTR1',
        'TOKYES',
        'YES',
        'buy',
        2.0,
        0.40,
        'open',
        ts,
        decision_context={'regime_state': {'regime_label': 'trend', 'trend_score': 0.8, 'tail_score': 0.2, 'reversal_score': 0.1, 'market_polarization': 0.6}},
    )
    storage.apply_incremental_order_fill(buy_order['id'], 2.0, fill_ts=ts, price=0.40, tx_hash='tx-buy-attr')
    open_lot = storage.get_open_lots(market_id='ATTR1')[0]
    attribution = storage.get_lot_regime_attribution(open_lot['id'])
    assert attribution is not None
    assert attribution['entry_regime_label'] == 'trend'

    sell_order = storage.create_order(
        'sell-attr-1',
        'ATTR1',
        'TOKYES',
        'YES',
        'sell',
        2.0,
        0.55,
        'open',
        ts,
        decision_context={'regime_state': {'regime_label': 'reversal_attempt', 'trend_score': 0.3, 'tail_score': 0.2, 'reversal_score': 0.8, 'market_polarization': 0.54}},
    )
    storage.apply_incremental_order_fill(sell_order['id'], 2.0, fill_ts=ts, price=0.55, tx_hash='tx-sell-attr')

    events = storage.list_realized_pnl_events('ATTR1')
    assert len(events) == 1
    assert events[0]['entry_regime_label'] == 'trend'
    assert events[0]['exit_regime_label'] == 'reversal_attempt'
    assert abs(events[0]['net_pnl'] - 0.30) < 1e-9


def test_redeem_creates_realized_event_with_entry_regime_preserved():
    ts = _ts()
    storage.create_market('ATTR2', status='open')
    storage.update_market_status('ATTR2', 'resolved', winning_outcome='YES')
    buy_order = storage.create_order(
        'buy-attr-2',
        'ATTR2',
        'TOKYES2',
        'YES',
        'buy',
        1.5,
        0.33,
        'open',
        ts,
        decision_context={'regime_state': {'regime_label': 'polarized_tail', 'trend_score': 0.4, 'tail_score': 0.9, 'reversal_score': 0.1, 'market_polarization': 0.94}},
    )
    storage.apply_incremental_order_fill(buy_order['id'], 1.5, fill_ts=ts, price=0.33, tx_hash='tx-buy-attr-2')

    storage.redeem_market('ATTR2', 'YES', redeem_tx_hash='tx-redeem-attr-2', ts=ts, exit_regime_state={'regime_label': 'normal', 'trend_score': 0.2, 'tail_score': 0.1, 'reversal_score': 0.1, 'market_polarization': 0.55})
    events = storage.list_realized_pnl_events('ATTR2')
    assert len(events) == 1
    assert events[0]['disposition_type'] == 'redeem'
    assert events[0]['entry_regime_label'] == 'polarized_tail'
    assert events[0]['exit_regime_label'] == 'normal'


def test_realized_pnl_reporting_helpers_group_by_regime():
    ts = _ts()
    storage.record_realized_pnl_event(
        ts=ts,
        market_id='REP1',
        token_id='TOK1',
        outcome_side='YES',
        qty=1.0,
        disposition_type='sell',
        entry_price=0.40,
        exit_price=0.50,
        gross_pnl=0.10,
        net_pnl=0.10,
        entry_regime_label='trend',
        exit_regime_label='normal',
    )
    storage.record_realized_pnl_event(
        ts=ts,
        market_id='REP1',
        token_id='TOK2',
        outcome_side='NO',
        qty=2.0,
        disposition_type='sell',
        entry_price=0.60,
        exit_price=0.45,
        gross_pnl=0.30,
        net_pnl=0.30,
        entry_regime_label='trend',
        exit_regime_label='reversal_attempt',
    )
    storage.record_realized_pnl_event(
        ts=ts,
        market_id='REP1',
        token_id='TOK3',
        outcome_side='YES',
        qty=1.0,
        disposition_type='sell',
        entry_price=0.70,
        exit_price=0.60,
        gross_pnl=-0.10,
        net_pnl=-0.10,
        entry_regime_label='normal',
        exit_regime_label='normal',
    )

    summary = storage.get_realized_pnl_summary('REP1')
    by_entry = {row['entry_regime_label']: row for row in storage.get_realized_pnl_by_entry_regime('REP1')}
    by_exit = {row['exit_regime_label']: row for row in storage.get_realized_pnl_by_exit_regime('REP1')}
    pairs = {(row['entry_regime_label'], row['exit_regime_label']): row for row in storage.get_realized_pnl_by_regime_pair('REP1')}
    counts = {row['entry_regime_label']: row for row in storage.get_regime_trade_counts('REP1')}

    assert abs(summary['realized_pnl_total'] - 0.30) < 1e-9
    assert summary['count_realized_events'] == 3
    assert by_entry['trend']['count_realized_events'] == 2
    assert abs(by_entry['trend']['realized_pnl_total'] - 0.40) < 1e-9
    assert by_exit['normal']['count_realized_events'] == 2
    assert pairs[('trend', 'reversal_attempt')]['count_realized_events'] == 1
    assert counts['trend']['qty_realized'] == 3.0
