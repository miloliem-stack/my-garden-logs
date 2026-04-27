import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from src import run_bot, storage, strategy_manager
from src.position_reevaluation import evaluate_position_reevaluation


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
        'favored_side': 'YES',
        'contrarian_side': 'NO',
        'spot_now': 101000.0,
        'strike_price': 100000.0,
    }
    base.update(overrides)
    return base


def _trade_context(**overrides):
    base = {
        'market': {'market_id': 'M-REEVAL'},
        'quotes': {
            'yes': {'best_bid': 0.55, 'best_ask': 0.56},
            'no': {'best_bid': 0.44, 'best_ask': 0.45},
        },
        'position_summary': {'available_inventory': {'YES': 0.0, 'NO': 0.0}},
        'position_management_state': {'add_count': 0, 'reduce_count': 0, 'flip_count': 0, 'persistence_target_action': None, 'persistence_count': 0, 'last_action_ts': None},
        'open_orders': [],
    }
    base.update(overrides)
    return base


def test_no_inventory_holds(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )

    assert result['action'] == 'hold'
    assert result['current_position_qty'] == 0.0


def test_same_side_strengthening_adds(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}}),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )

    assert result['action'] == 'add_same_side'
    assert result['reason'] == 'position_reeval_add_same_side'
    assert result['candidate_side_entry_ev_per_share'] > 0


def test_negative_hold_ev_reduces(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(p_yes=0.48, edge_yes=-0.08, chosen_side=None, trade_allowed=False),
        trade_context=_trade_context(
            quotes={'yes': {'best_bid': 0.54, 'best_ask': 0.55}, 'no': {'best_bid': 0.45, 'best_ask': 0.46}},
            position_summary={'available_inventory': {'YES': 2.0, 'NO': 0.0}},
        ),
        price_history=pd.Series([100.0, 100.1, 100.0, 99.9]),
    )

    assert result['action'] == 'reduce_position'
    assert result['held_side_hold_ev_per_share'] <= -0.03


def test_opposite_side_signal_without_reversal_evidence_holds(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_ALLOW_FLIP', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')
    monkeypatch.setenv('REVERSAL_EVIDENCE_MIN_SCORE', '999')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(
            p_yes=0.35,
            p_no=0.65,
            edge_yes=-0.15,
            edge_no=0.20,
            chosen_side='NO',
            favored_side='YES',
            contrarian_side='NO',
        ),
        trade_context=_trade_context(position_summary={'available_inventory': {'YES': 2.0, 'NO': 0.0}}),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )

    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_reversal_evidence_failed'


def test_opposite_side_signal_with_reversal_and_persistence_flips(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_ALLOW_FLIP', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('REVERSAL_EVIDENCE_ENABLED', 'true')
    monkeypatch.setenv('REVERSAL_EVIDENCE_MIN_SCORE', '3')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(
            p_yes=0.35,
            p_no=0.65,
            edge_yes=-0.15,
            edge_no=0.20,
            chosen_side='NO',
            favored_side='YES',
            contrarian_side='NO',
        ),
        trade_context=_trade_context(position_summary={'available_inventory': {'YES': 2.0, 'NO': 0.0}}),
        price_history=pd.Series([103.0, 102.0, 101.0, 100.0]),
    )

    assert result['action'] == 'flip_position'
    assert result['reason'] == 'position_reeval_flip_advantage'
    assert result['reversal_evidence']['passes_min_score'] is True


def test_cooldown_suppresses_repeated_action(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    now = datetime.now(timezone.utc)

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(
            position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}},
            position_management_state={'add_count': 0, 'reduce_count': 0, 'flip_count': 0, 'persistence_target_action': 'add_same_side', 'persistence_count': 1, 'last_action_ts': (now - timedelta(seconds=30)).isoformat()},
        ),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
        now_ts=now.isoformat(),
    )

    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_cooldown_active'
    assert result['persistence_target_action'] == 'add_same_side'


def test_action_caps_suppress_repeated_actions(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('POSITION_REEVAL_MAX_ADDS_PER_MARKET', '1')

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(
            position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}},
            position_management_state={'add_count': 1, 'reduce_count': 0, 'flip_count': 0, 'persistence_target_action': None, 'persistence_count': 0, 'last_action_ts': None},
        ),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )

    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_action_cap_reached'


def test_final_bucket_blocks_add_but_allows_reduce(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('POSITION_REEVAL_DISABLE_IN_FINAL_BUCKET', 'true')

    add_result = evaluate_position_reevaluation(
        decision_state=_decision_state(policy={'policy_bucket': 'final', 'allow_new_entries': False}),
        trade_context=_trade_context(position_summary={'available_inventory': {'YES': 2.0, 'NO': 0.0}}),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )
    reduce_result = evaluate_position_reevaluation(
        decision_state=_decision_state(
            policy={'policy_bucket': 'final', 'allow_new_entries': False},
            p_yes=0.45,
            edge_yes=-0.10,
            chosen_side=None,
            trade_allowed=False,
        ),
        trade_context=_trade_context(
            quotes={'yes': {'best_bid': 0.54, 'best_ask': 0.55}, 'no': {'best_bid': 0.45, 'best_ask': 0.46}},
            position_summary={'available_inventory': {'YES': 2.0, 'NO': 0.0}},
        ),
        price_history=pd.Series([100.0, 100.0, 100.0, 100.0]),
    )

    assert add_result['action'] == 'hold'
    assert add_result['reason'] == 'position_reeval_disabled_final_bucket'
    assert reduce_result['action'] == 'reduce_position'


def test_run_bot_decision_state_includes_reeval_fields():
    now = datetime.now(timezone.utc)
    bundle = {
        'series_id': 'SER',
        'market_id': 'M-REEVAL',
        'token_yes': 'TOKY',
        'token_no': 'TOKN',
        'strike_price': 100000.0,
        'yes_quote': {'mid': 0.50},
        'no_quote': {'mid': 0.49},
        'start_time': (now - timedelta(minutes=5)).isoformat(),
        'end_time': (now + timedelta(minutes=20)).isoformat(),
    }
    probability_state = {
        'blocked': False,
        'reason': None,
        'timestamp': now.isoformat(),
        'series_id': 'SER',
        'market_id': 'M-REEVAL',
        'spot_now': 101000.0,
        'strike_price': 100000.0,
        'tau_minutes': 20,
        'p_yes': 0.60,
        'p_no': 0.40,
        'sigma_per_sqrt_min': 0.001,
    }

    decision = run_bot.build_market_decision_state(bundle, probability_state)

    assert 'position_reeval_action' in decision
    assert 'position_reeval_reason' in decision
    assert 'position_reeval' in decision


def test_strategy_manager_uses_explicit_reeval_action(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL', status='open')
    storage.create_open_lot('M-REEVAL', 'TOKY', 'YES', 3.0, 0.42, ts)
    monkeypatch.setattr(strategy_manager, 'place_marketable_sell', lambda *args, **kwargs: {'status': 'dry_run', 'order_id': 17})

    decision_state = _decision_state(
        market_id='M-REEVAL',
        chosen_side=None,
        trade_allowed=False,
        position_reeval_action='reduce_position',
        position_reeval_reason='position_reeval_reduce_negative_hold_ev',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 3.0, 'executable_exit_price': 0.54, 'action': 'reduce_position'},
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-REEVAL', dry_run=True)

    assert action['action'] == 'reduce_position'
    assert action['reason'] == 'position_reeval_reduce_negative_hold_ev'
    state = storage.get_position_management_state('M-REEVAL')
    assert state['reduce_count'] == 1


def test_strategy_manager_reeval_error_does_not_consume_action_limit(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-ERR', status='open')
    storage.create_open_lot('M-REEVAL-ERR', 'TOKY', 'YES', 3.0, 0.42, ts)
    monkeypatch.setattr(
        strategy_manager,
        'place_marketable_sell',
        lambda *args, **kwargs: {'status': 'error', 'reason': 'temporary_venue_error', 'error_message': 'temporary_venue_error'},
    )

    decision_state = _decision_state(
        market_id='M-REEVAL-ERR',
        chosen_side=None,
        trade_allowed=False,
        position_reeval_action='reduce_position',
        position_reeval_reason='position_reeval_reduce_negative_hold_ev',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 3.0, 'executable_exit_price': 0.54, 'action': 'reduce_position'},
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-REEVAL-ERR', dry_run=False)

    assert action['action'] == 'reduce_position'
    assert action['resp']['status'] == 'error'
    state = storage.get_position_management_state('M-REEVAL-ERR')
    assert state['reduce_count'] == 0
    assert state['last_action'] == 'reeval_attempt_failed'
    assert state['last_action_reason'] == 'temporary_venue_error'


def test_first_entry_buy_calls_shared_gate(monkeypatch):
    storage.create_market('M-FIRST', status='open')
    calls = []
    original = strategy_manager.evaluate_buy_admission

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get('action_origin'))
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy_manager, 'evaluate_buy_admission', wrapped)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.50})

    action = strategy_manager.build_trade_action(_decision_state(market_id='M-FIRST'), 'TOKY', 'TOKN', 'M-FIRST', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert calls == ['first_entry']


def test_reeval_add_buy_calls_shared_gate(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-ADD', status='open')
    storage.create_open_lot('M-REEVAL-ADD', 'TOKY', 'YES', 3.0, 0.42, ts)
    calls = []
    original = strategy_manager.evaluate_buy_admission

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get('action_origin'))
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy_manager, 'evaluate_buy_admission', wrapped)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'off')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.52})

    action = strategy_manager.build_trade_action(
        _decision_state(
            market_id='M-REEVAL-ADD',
            position_reeval_action='add_same_side',
            position_reeval_reason='position_reeval_add_same_side',
            position_reeval={'current_position_side': 'YES', 'current_position_qty': 3.0, 'executable_entry_price': 0.52, 'action': 'add_same_side'},
        ),
        'TOKY',
        'TOKN',
        'M-REEVAL-ADD',
        dry_run=True,
    )

    assert action['action'] == 'add_same_side'
    assert calls == ['reeval_add']


def test_reeval_add_blocked_by_regime_entry_guard_when_extreme_minority(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-RISK', status='open')
    storage.create_open_lot('M-REEVAL-RISK', 'TOKY', 'YES', 2.0, 0.30, ts)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    decision_state = _decision_state(
        market_id='M-REEVAL-RISK',
        p_yes=0.30,
        p_no=0.70,
        q_yes=0.04,
        q_no=0.96,
        edge_yes=0.26,
        edge_no=-0.26,
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.04, 'action': 'add_same_side'},
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-REEVAL-RISK', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_extreme_minority_side'
    assert decision_state['action_origin'] == 'reeval_add'
    assert decision_state['shared_buy_gate_evaluated'] is True
    assert decision_state['shared_buy_gate_passed'] is False
    assert decision_state['regime_guard_evaluated'] is True
    assert decision_state['regime_guard_blocked'] is True


def test_reeval_add_blocked_when_policy_disallows_new_entries(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-POLICY', status='open')
    storage.create_open_lot('M-REEVAL-POLICY', 'TOKY', 'YES', 2.0, 0.42, ts)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    decision_state = _decision_state(
        market_id='M-REEVAL-POLICY',
        policy={'policy_bucket': 'final', 'allow_new_entries': False, 'edge_threshold_yes': 0.01, 'edge_threshold_no': 0.01, 'kelly_multiplier': 1.0, 'max_trade_notional_multiplier': 1.0},
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.52, 'action': 'add_same_side'},
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-REEVAL-POLICY', dry_run=True)

    assert action['action'] == 'skipped_policy_blocked_entries'
    assert action['reason'] == 'policy_blocks_new_entries'
    assert decision_state['shared_guard_policy_checked'] is True
    assert decision_state['shared_buy_gate_passed'] is False


def test_reeval_add_blocked_on_invalid_or_stale_quote(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-QUOTE', status='open')
    storage.create_open_lot('M-REEVAL-QUOTE', 'TOKY', 'YES', 2.0, 0.42, ts)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    stale_state = _decision_state(
        market_id='M-REEVAL-QUOTE',
        trade_allowed=False,
        reason='quote_stale',
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.52, 'action': 'add_same_side'},
    )
    stale_action = strategy_manager.build_trade_action(stale_state, 'TOKY', 'TOKN', 'M-REEVAL-QUOTE', dry_run=True)
    assert stale_action['action'] == 'skipped_shared_buy_gate'
    assert stale_action['reason'] == 'quote_stale'

    invalid_state = _decision_state(
        market_id='M-REEVAL-QUOTE',
        q_yes=None,
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.52, 'action': 'add_same_side'},
    )
    invalid_action = strategy_manager.build_trade_action(invalid_state, 'TOKY', 'TOKN', 'M-REEVAL-QUOTE', dry_run=True)
    assert invalid_action['action'] == 'skipped_invalid_quote'
    assert invalid_action['reason'] == 'missing_or_invalid_yes_quote'


def test_reeval_add_preserves_reeval_specific_upstream_checks_before_shared_gate(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('POSITION_REEVAL_MAX_ADDS_PER_MARKET', '1')
    calls = []
    original = strategy_manager.evaluate_buy_admission

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get('action_origin'))
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy_manager, 'evaluate_buy_admission', wrapped)

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(
            position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}},
            position_management_state={'add_count': 1, 'reduce_count': 0, 'flip_count': 0, 'persistence_target_action': None, 'persistence_count': 0, 'last_action_ts': None},
        ),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
    )

    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_action_cap_reached'
    assert calls == []


def test_executed_buy_decision_context_contains_action_origin_and_shared_gate_fields(monkeypatch):
    storage.create_market('M-FIRST-CONTEXT', status='open')
    captured = {}

    def fake_buy(*args, **kwargs):
        captured['decision_context'] = kwargs.get('decision_context') or {}
        return {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.50}

    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', fake_buy)

    action = strategy_manager.build_trade_action(_decision_state(market_id='M-FIRST-CONTEXT'), 'TOKY', 'TOKN', 'M-FIRST-CONTEXT', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert captured['decision_context']['action_origin'] == 'first_entry'
    assert captured['decision_context']['shared_buy_gate_evaluated'] is True
    assert captured['decision_context']['shared_buy_gate_passed'] is True
    assert captured['decision_context']['shared_guard_trade_allowed_checked'] is True
    assert isinstance(captured['decision_context']['shared_guard_result_json'], dict)
    assert captured['decision_context']['blocker_telemetry']['candidate_stage'] == 'submitted'
    assert captured['decision_context']['blocker_telemetry']['candidate_snapshot']['chosen_side_candidate'] == 'YES'


def test_microstructure_blocker_preserves_candidate_snapshot_fields(monkeypatch):
    storage.create_market('M-MICRO', status='open')
    monkeypatch.setattr(strategy_manager, 'microstructure_spectral_mode', lambda: 'live')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    decision_state = _decision_state(
        market_id='M-MICRO',
        microstructure_regime='noisy',
        spectral_ready=True,
        spectral_reason='ok',
        smoothness_score=0.12,
        spectral_entropy=0.91,
        low_freq_power_ratio=0.08,
        high_freq_power_ratio=0.72,
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-MICRO', dry_run=True)

    assert action['action'] == 'skipped_microstructure_noisy'
    assert decision_state['reason'] == 'microstructure_noisy_block'
    telemetry = decision_state['blocker_telemetry']
    assert telemetry['candidate_stage'] == 'post_candidate_blocked'
    assert telemetry['candidate_snapshot']['chosen_side_candidate'] == 'YES'
    assert telemetry['candidate_snapshot']['q_yes'] == 0.50
    assert telemetry['candidate_snapshot']['microstructure_regime'] == 'noisy'
    assert telemetry['blockers']['microstructure_noisy_block']['blocked'] is True


def test_first_entry_and_reeval_add_emit_consistent_regime_guard_diagnostics(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-CONSISTENT', status='open')
    storage.create_open_lot('M-CONSISTENT', 'TOKY', 'YES', 2.0, 0.30, ts)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    first_entry_state = _decision_state(
        market_id='M-CONSISTENT',
        p_yes=0.30,
        p_no=0.70,
        q_yes=0.04,
        q_no=0.96,
        edge_yes=0.26,
        edge_no=-0.26,
    )
    first_entry_action = strategy_manager.build_trade_action(first_entry_state, 'TOKY', 'TOKN', 'M-CONSISTENT', dry_run=True)

    reeval_state = _decision_state(
        market_id='M-CONSISTENT',
        p_yes=0.30,
        p_no=0.70,
        q_yes=0.04,
        q_no=0.96,
        edge_yes=0.26,
        edge_no=-0.26,
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.04, 'action': 'add_same_side'},
    )
    reeval_action = strategy_manager.build_trade_action(reeval_state, 'TOKY', 'TOKN', 'M-CONSISTENT', dry_run=True)

    assert first_entry_action['action'] == 'skipped_regime_entry_guard'
    assert reeval_action['action'] == 'skipped_regime_entry_guard'
    assert first_entry_state['regime_guard_reason'] == reeval_state['regime_guard_reason'] == 'veto_extreme_minority_side'
    assert first_entry_state['regime_guard_evaluated'] is True
    assert reeval_state['regime_guard_evaluated'] is True
    assert first_entry_state['blocker_telemetry']['candidate_snapshot']['chosen_side_candidate'] == 'YES'
    assert reeval_state['blocker_telemetry']['candidate_snapshot']['chosen_side_candidate'] == 'YES'


def test_reduce_position_path_not_forced_through_buy_gate(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-REDUCE', status='open')
    storage.create_open_lot('M-REEVAL-REDUCE', 'TOKY', 'YES', 3.0, 0.42, ts)
    monkeypatch.setattr(strategy_manager, 'place_marketable_sell', lambda *args, **kwargs: {'status': 'dry_run', 'order_id': 17})

    def fail_if_called(*args, **kwargs):
        raise AssertionError('shared buy gate should not run for reduce_position')

    monkeypatch.setattr(strategy_manager, 'evaluate_buy_admission', fail_if_called)

    action = strategy_manager.build_trade_action(
        _decision_state(
            market_id='M-REEVAL-REDUCE',
            chosen_side=None,
            trade_allowed=False,
            position_reeval_action='reduce_position',
            position_reeval_reason='position_reeval_reduce_negative_hold_ev',
            position_reeval={'current_position_side': 'YES', 'current_position_qty': 3.0, 'executable_exit_price': 0.54, 'action': 'reduce_position'},
        ),
        'TOKY',
        'TOKN',
        'M-REEVAL-REDUCE',
        dry_run=True,
    )

    assert action['action'] == 'reduce_position'


def test_reeval_add_path_includes_action_origin_and_blocker_telemetry(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M-REEVAL-TELEM', status='open')
    storage.create_open_lot('M-REEVAL-TELEM', 'TOKY', 'YES', 2.0, 0.42, ts)
    monkeypatch.setattr(strategy_manager, 'microstructure_spectral_mode', lambda: 'live')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})

    decision_state = _decision_state(
        market_id='M-REEVAL-TELEM',
        microstructure_regime='noisy',
        spectral_ready=True,
        smoothness_score=0.10,
        spectral_entropy=0.95,
        position_reeval_action='add_same_side',
        position_reeval_reason='position_reeval_add_same_side',
        position_reeval={'current_position_side': 'YES', 'current_position_qty': 2.0, 'executable_entry_price': 0.50, 'action': 'add_same_side'},
    )

    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-REEVAL-TELEM', dry_run=True)

    assert action['action'] == 'skipped_microstructure_noisy'
    assert decision_state['action_origin'] == 'reeval_add'
    assert decision_state['blocked_by'] == 'microstructure_noisy_block'
    assert decision_state['blocker_telemetry']['action_origin'] == 'reeval_add'
    assert decision_state['blocker_telemetry']['candidate_snapshot']['position_reeval_action'] == 'add_same_side'


def test_no_behavior_regression_for_first_entry_happy_path(monkeypatch):
    storage.create_market('M-FIRST-HAPPY', status='open')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.50})

    decision_state = _decision_state(market_id='M-FIRST-HAPPY')
    action = strategy_manager.build_trade_action(decision_state, 'TOKY', 'TOKN', 'M-FIRST-HAPPY', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision_state['action_origin'] == 'first_entry'
    assert decision_state['shared_buy_gate_passed'] is True
    assert decision_state['regime_guard_evaluated'] is True


def test_failed_reeval_attempt_has_short_retry_cooldown(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('POSITION_REEVAL_FAILED_RETRY_SEC', '5')
    now = datetime.now(timezone.utc)

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(
            position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}},
            position_management_state={
                'add_count': 0,
                'reduce_count': 0,
                'flip_count': 0,
                'last_action': 'reeval_attempt_failed',
                'last_action_ts': (now - timedelta(seconds=3)).isoformat(),
                'last_action_reason': 'temporary_venue_error',
                'persistence_target_action': 'add_same_side',
                'persistence_count': 1,
            },
        ),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
        now_ts=now.isoformat(),
    )

    assert result['action'] == 'hold'
    assert result['reason'] == 'position_reeval_failed_retry_cooldown'


def test_failed_reeval_attempt_reevaluates_after_short_retry_cooldown(monkeypatch):
    monkeypatch.setenv('POSITION_REEVAL_ENABLED', 'true')
    monkeypatch.setenv('POSITION_REEVAL_PERSISTENCE_REQUIRED', '1')
    monkeypatch.setenv('POSITION_REEVAL_FAILED_RETRY_SEC', '5')
    now = datetime.now(timezone.utc)

    result = evaluate_position_reevaluation(
        decision_state=_decision_state(),
        trade_context=_trade_context(
            position_summary={'available_inventory': {'YES': 3.0, 'NO': 0.0}},
            position_management_state={
                'add_count': 0,
                'reduce_count': 0,
                'flip_count': 0,
                'last_action': 'reeval_attempt_failed',
                'last_action_ts': (now - timedelta(seconds=6)).isoformat(),
                'last_action_reason': 'temporary_venue_error',
                'persistence_target_action': 'add_same_side',
                'persistence_count': 1,
            },
        ),
        price_history=pd.Series([100.0, 101.0, 102.0, 103.0]),
        now_ts=now.isoformat(),
    )

    assert result['action'] == 'add_same_side'
    assert result['reason'] == 'position_reeval_add_same_side'
