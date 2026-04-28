import os

import pandas as pd

from src import storage, strategy_manager
from src.position_reevaluation import REEVAL_POLICY_VERSION, evaluate_position_reevaluation


def setup_function(fn):
    os.environ['BOT_DB_PATH'] = f'/tmp/btc_1h_reeval_{fn.__name__}.db'
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def _decision_state(**overrides):
    base = {
        'market_id': 'M-REEVAL',
        'token_yes': 'TOKY',
        'token_no': 'TOKN',
        'p_yes': 0.60,
        'p_no': 0.40,
        'q_yes': 0.50,
        'q_no': 0.50,
        'edge_yes': 0.10,
        'edge_no': -0.10,
        'chosen_side': 'YES',
        'trade_allowed': True,
        'policy': {'policy_bucket': 'mid', 'allow_new_entries': True, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
    }
    base.update(overrides)
    return base


def _trade_context(**overrides):
    base = {
        'market': {'market_id': 'M-REEVAL'},
        'position_summary': {
            'available_inventory': {'YES': 3.0, 'NO': 0.0},
            'avg_entry_price_yes': 0.45,
            'avg_entry_price_no': None,
        },
        'position_management_state': {'add_count': 0, 'reduce_count': 0, 'flip_count': 0, 'persistence_target_action': None, 'persistence_count': 0, 'last_action_ts': None},
        'open_orders': [],
    }
    base.update(overrides)
    return base


def test_position_reevaluation_is_disabled_noop_even_when_env_enabled(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_ALLOW_ADD', 'true')
    monkeypatch.setenv('POSITION_REEVAL_ALLOW_REDUCE', 'true')
    monkeypatch.setenv('POSITION_REEVAL_ALLOW_FLIP', 'true')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(position_reeval_action='add_same_side'),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 101.0, 102.0]),
    )

    assert result['enabled'] is False
    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_disabled'
    assert result['reeval_policy_version'] == REEVAL_POLICY_VERSION
    assert result['reevaluation_growth_candidates'] == []


def test_position_reevaluation_noop_preserves_basic_inventory_diagnostics():
    result = evaluate_position_reevaluation(
        decision_state=_decision_state(chosen_side='YES'),
        trade_context=_trade_context(position_summary={'available_inventory': {'YES': 2.0, 'NO': 1.0}}),
        price_history=pd.Series([100.0, 101.0, 102.0]),
    )

    assert result['current_position_side'] == 'YES'
    assert result['current_position_qty'] == 2.0
    assert result['chosen_side'] == 'YES'
    assert result['same_side_as_position'] is True
    assert result['persistence_target_action'] == 'hold'
    assert result['persistence_next_count'] == 0


def test_strategy_manager_ignores_legacy_reeval_action_and_uses_first_entry_path(monkeypatch):
    storage.create_market('M-REEVAL', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'off')
    calls = []
    monkeypatch.setattr(
        strategy_manager,
        'place_marketable_buy',
        lambda token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', **kwargs: (
            calls.append({'token_id': token_id, 'qty': qty, 'limit_price': limit_price, 'outcome_side': outcome_side})
            or {'status': 'dry_run', 'submitted_qty': qty, 'submitted_notional': qty * limit_price}
        ),
    )

    action = strategy_manager.build_trade_action(
        _decision_state(position_reeval_action='add_same_side', position_reeval_reason='legacy_reeval'),
        'TOKY',
        'TOKN',
        'M-REEVAL',
        dry_run=True,
        wallet_state={'effective_bankroll': 1000.0},
    )

    assert action['side'] == 'buy_yes'
    assert calls and calls[0]['outcome_side'] == 'YES'
