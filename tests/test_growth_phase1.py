from datetime import datetime, timedelta, timezone

import pandas as pd

from src import run_bot, storage, strategy_manager
from src.position_reevaluation import evaluate_position_reevaluation
from src.research.decision_contract import (
    DecisionInput,
    ExpectedGrowthSnapshot,
    HMMPolicyState,
    ProbabilitySnapshot,
    QuoteSnapshot,
    SafetyVetoSnapshot,
    TauPolicySnapshot,
    evaluate_replay_decision,
)


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


def test_tail_blocked_legacy_state_can_leave_growth_shadow_metrics_unset(monkeypatch):
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.01, q_no=0.99),
        _probability_state(p_yes=0.30, spot=88000.0, z_score=-3.0),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )

    assert decision['trade_allowed'] is False
    assert decision['expected_log_growth_entry'] is None
    assert decision['expected_log_growth_entry_conservative'] is None


def test_expected_growth_veto_is_owned_by_offline_decision_contract():
    out = evaluate_replay_decision(
        DecisionInput(
            timestamp='2026-04-30T12:00:00Z',
            market_id='M-GROWTH',
            probability=ProbabilitySnapshot(p_yes=0.60, p_no=0.40, engine_name='gaussian_vol'),
            quote=QuoteSnapshot(q_yes=0.50, q_no=0.50),
            tau_policy=TauPolicySnapshot(
                tau_minutes=20,
                tau_bucket='mid',
                edge_threshold_yes=0.03,
                edge_threshold_no=0.03,
                allow_new_entries=True,
            ),
            hmm_policy_state=HMMPolicyState(
                map_state=1,
                posterior_confidence=0.90,
                next_same_state_confidence=0.85,
                persistence_count=3,
                policy_state='state_1_confident',
            ),
            safety_veto=SafetyVetoSnapshot(
                tail_veto=False,
                polarization_veto=False,
                reversal_veto=False,
                quote_quality_pass=True,
            ),
            expected_growth=ExpectedGrowthSnapshot(
                expected_log_growth=0.01,
                conservative_expected_log_growth=-0.01,
                passes=False,
            ),
        )
    )

    assert out.allowed is False
    assert out.reason == 'expected_growth_veto'


def test_reevaluation_optimizer_is_disabled_noop(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_GROWTH_OPTIMIZER_MODE', 'shadow')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(p_yes=0.56, edge_yes=0.01),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 100.2, 100.1, 100.0]),
    )

    assert result['enabled'] is False
    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_disabled'
    assert result['reevaluation_shadow_best_action'] == 'hold'
    assert result['reevaluation_shadow_keep_current_position'] is True


def test_reevaluation_optimizer_does_not_recommend_add(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_GROWTH_OPTIMIZER_MODE', 'shadow')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(p_yes=0.78, edge_yes=0.23),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 100.5, 101.0, 101.5]),
    )

    assert result['enabled'] is False
    assert result['action'] == 'hold'
    assert result['reevaluation_shadow_best_action'] == 'hold'
    assert result['reevaluation_shadow_best_growth_gain'] == 0.0


def test_reevaluation_growth_candidates_are_not_generated(monkeypatch):
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

    assert result['enabled'] is False
    assert result['action'] == 'hold'
    assert result['reevaluation_shadow_best_action'] == 'hold'
    assert result['reevaluation_shadow_best_executable'] is False
    assert result['reevaluation_growth_candidates'] == []


def test_shadow_metrics_remain_diagnostic_when_enabled(monkeypatch):
    storage.create_market('M-GROWTH', status='open')
    monkeypatch.setenv('EXPECTED_GROWTH_SHADOW_ENABLED', 'true')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.55})

    decision = run_bot.build_market_decision_state(
        _bundle(q_yes=0.55, q_no=0.45),
        _probability_state(p_yes=0.62),
        wallet_state={'effective_bankroll': 1000.0, 'free_usdc': 1000.0},
    )
    action = strategy_manager.build_trade_action(decision, 'YES1', 'NO1', 'M-GROWTH', dry_run=True)

    assert isinstance(action, dict)
    assert action.get('side') == 'buy_yes' or action.get('action', '').startswith('skipped_')
    assert decision['p_yes'] == 0.62
    assert 'expected_log_growth_entry' in decision
    assert 'expected_log_growth_entry_conservative' in decision


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
