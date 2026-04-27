from datetime import datetime, timedelta, timezone

import pandas as pd

from src import run_bot, storage, strategy_manager
from src.position_reevaluation import evaluate_position_reevaluation


def setup_function(fn):
    import os

    os.environ['BOT_DB_PATH'] = f'/tmp/btc_1h_growth_phase1_{fn.__name__}.db'
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass
    storage.ensure_db()


def _bundle(*, q_yes: float, q_no: float) -> dict:
    now = datetime.now(timezone.utc)
    return {
        'series_id': 'BTC-HOURLY',
        'market_id': 'M-GROWTH',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'strike_price': 100000.0,
        'yes_quote': {'mid': q_yes},
        'no_quote': {'mid': q_no},
        'start_time': now.isoformat(),
        'end_time': (now + timedelta(minutes=20)).isoformat(),
    }


def _probability_state(*, p_yes: float, spot: float = 101000.0, strike: float = 100000.0, tau_minutes: int = 20, z_score: float = 0.4) -> dict:
    return {
        'blocked': False,
        'reason': None,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'series_id': 'BTC-HOURLY',
        'market_id': 'M-GROWTH',
        'spot_now': spot,
        'strike_price': strike,
        'tau_minutes': tau_minutes,
        'p_yes': p_yes,
        'p_no': 1.0 - p_yes,
        'sigma_per_sqrt_min': 0.001,
        'raw_model_output': {
            'horizon_sigma': 0.04,
            'z_score': z_score,
            'raw_output': {'horizon_sigma': 0.04, 'z_score': z_score},
        },
    }


def _decision_state(**overrides):
    base = {
        'market_id': 'M-GROWTH',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.45,
        'edge_yes': 0.07,
        'edge_no': -0.07,
        'chosen_side': 'YES',
        'trade_allowed': True,
        'policy': {
            'policy_bucket': 'mid',
            'allow_new_entries': True,
            'edge_threshold_yes': 0.01,
            'edge_threshold_no': 0.01,
            'kelly_multiplier': 1.0,
            'max_trade_notional_multiplier': 1.0,
        },
        'favored_side': 'YES',
        'contrarian_side': 'NO',
        'spot_now': 101000.0,
        'strike_price': 100000.0,
        'wallet_state': {'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    }
    base.update(overrides)
    return base


def _trade_context(**overrides):
    base = {
        'market': {'market_id': 'M-GROWTH'},
        'wallet_state': {'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
        'quotes': {
            'yes': {'best_bid': 0.55, 'best_ask': 0.56},
            'no': {'best_bid': 0.44, 'best_ask': 0.45},
        },
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


def test_entry_growth_metric_computes_finite_values(monkeypatch):
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.42, q_no=0.58),
        _probability_state(p_yes=0.60),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )

    assert decision['expected_log_growth_entry'] is not None
    assert decision['expected_log_growth_entry_conservative'] is not None
    assert decision['entry_growth_eval_mode'] == 'shadow'


def test_conservative_growth_metric_is_not_above_naive_in_tail_setup(monkeypatch):
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.01, q_no=0.99),
        _probability_state(p_yes=0.30, spot=88000.0, z_score=-3.0),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )

    assert decision['expected_log_growth_entry_conservative'] <= decision['expected_log_growth_entry']


def test_polarized_minority_entry_gets_worse_growth_than_balanced_entry(monkeypatch):
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')

    balanced = run_bot.build_market_decision_state(
        _bundle(q_yes=0.42, q_no=0.58),
        _probability_state(p_yes=0.60, z_score=0.4),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )
    tail = run_bot.build_market_decision_state(
        _bundle(q_yes=0.01, q_no=0.99),
        _probability_state(p_yes=0.30, spot=88000.0, z_score=-3.0),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )

    assert tail['expected_log_growth_entry_conservative'] < balanced['expected_log_growth_entry_conservative']


def test_reevaluation_optimizer_can_prefer_hold(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_GROWTH_OPTIMIZER_MODE', 'shadow')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(p_yes=0.56, edge_yes=0.01),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 100.2, 100.1, 100.0]),
    )

    assert result['reevaluation_shadow_best_action'] == 'hold'
    assert result['reevaluation_shadow_keep_current_position'] is True


def test_reevaluation_optimizer_can_prefer_add(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_GROWTH_OPTIMIZER_MODE', 'shadow')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(p_yes=0.78, edge_yes=0.23),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 100.5, 101.0, 101.5]),
    )

    assert result['reevaluation_shadow_best_action'] in {'add_small', 'add_medium'}
    assert result['reevaluation_shadow_best_growth_gain'] > 0.0


def test_non_executable_tiny_best_actions_are_logged_not_sent(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_GROWTH_OPTIMIZER_MODE', 'shadow')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('BOT_BANKROLL', '3.0')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(
            p_yes=0.90,
            edge_yes=0.35,
            wallet_state={'effective_bankroll': 3.0, 'free_usdc': 1.5},
        ),
        trade_context=_trade_context(
            wallet_state={'effective_bankroll': 3.0, 'free_usdc': 1.5},
            position_summary={
                'available_inventory': {'YES': 3.0, 'NO': 0.0},
                'avg_entry_price_yes': 0.45,
                'avg_entry_price_no': None,
            },
        ),
        price_history=pd.Series([100.0, 100.5, 101.0, 101.5]),
    )

    assert result['reevaluation_shadow_best_action'] in {'add_small', 'add_medium'}
    assert result['reevaluation_shadow_best_executable'] is False
    assert any(
        candidate['reeval_candidate_action'] == result['reevaluation_shadow_best_action']
        and candidate['reeval_candidate_executable_now'] is False
        for candidate in result['reevaluation_growth_candidates']
    )


def test_shadow_metrics_do_not_change_live_action_selection(monkeypatch):
    storage.create_market('M-GROWTH', status='open')
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.55})

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.55, q_no=0.45),
        _probability_state(p_yes=0.62),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )
    action = strategy_manager.build_trade_action(decision, 'YES1', 'NO1', 'M-GROWTH', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision['expected_log_growth_entry'] is not None
    assert decision['expected_log_growth_entry_conservative'] is not None


def test_discounted_conservative_growth_is_used_in_polarized_zone(monkeypatch):
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'shadow')
    monkeypatch.setenv('POLARIZATION_CAUTION_BASE_WEIGHT', '0.50')

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.18, q_no=0.82),
        _probability_state(p_yes=0.40, z_score=-1.8),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )

    assert decision['polarization_zone'] == 'caution'
    assert decision['expected_log_growth_entry_conservative_discounted'] is not None
    assert decision['expected_log_growth_entry_conservative_old'] is not None
    assert decision['expected_log_growth_entry_conservative_discounted'] < decision['expected_log_growth_entry_conservative_old']
    assert decision['expected_log_growth_entry_conservative'] == decision['expected_log_growth_entry_conservative_discounted']
