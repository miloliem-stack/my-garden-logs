import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src import run_bot, storage, strategy_manager
from src.time_policy import build_time_policy


def _entry_policy(edge_threshold_yes: float = 0.01, edge_threshold_no: float = 0.01, kelly_multiplier: float = 1.0) -> dict:
    return {
        'policy_bucket': 'mid',
        'allow_new_entries': True,
        'edge_threshold_yes': edge_threshold_yes,
        'edge_threshold_no': edge_threshold_no,
        'kelly_multiplier': kelly_multiplier,
        'max_trade_notional_multiplier': 1.0,
    }


class _FakeModel:
    def __init__(self, last_price=64123.0, p_hat=0.63):
        self.last_price = last_price
        self._p_hat = p_hat
        self.calls = []

    def simulate_probability(self, target_price, tau_minutes, n_sims=2000, seed=None):
        self.calls.append({
            'target_price': target_price,
            'tau_minutes': tau_minutes,
            'n_sims': n_sims,
            'seed': seed,
        })
        return {'p_hat': self._p_hat, 'n_sims': n_sims, 'target_price': float(target_price)}


def _bundle(strike_price=64000.0, start=None, end=None, yes_mid=0.55, no_mid=0.41):
    start = start or datetime.now(timezone.utc)
    end = end or (start + timedelta(minutes=37))
    return {
        'series_id': 'SERIES1',
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'start_time': start.isoformat(),
        'end_time': end.isoformat(),
        'strike_price': strike_price,
        'strike_source': 'binance_1h_open',
        'yes_quote': {'mid': yes_mid},
        'no_quote': {'mid': no_mid},
    }


def setup_function(_fn):
    try:
        storage.get_db_path().unlink()
    except Exception:
        pass
    storage.ensure_db()
    if run_bot.DECISION_LOG_PATH.exists():
        run_bot.DECISION_LOG_PATH.unlink()


def test_probability_state_uses_bundle_strike_not_ad_hoc_target():
    now = datetime.now(timezone.utc)
    bundle = _bundle(strike_price=64500.0, start=now - timedelta(minutes=10), end=now + timedelta(minutes=20))
    model = _FakeModel(last_price=65000.0, p_hat=0.7)

    state = run_bot.compute_market_probabilities(bundle, model, now=now, n_sims=123)

    assert state['strike_price'] == 64500.0
    assert model.calls[0]['target_price'] == 64500.0
    assert model.calls[0]['n_sims'] == 123


def test_policy_bucket_selection_across_multiple_tau_ranges():
    assert build_time_policy({'tau_minutes': 45})['policy_bucket'] == 'far'
    assert build_time_policy({'tau_minutes': 20})['policy_bucket'] == 'mid'
    assert build_time_policy({'tau_minutes': 8})['policy_bucket'] == 'late'
    assert build_time_policy({'tau_minutes': 3})['policy_bucket'] == 'final'


def test_required_edge_threshold_decreases_according_to_schedule():
    far = build_time_policy({'tau_minutes': 45})
    mid = build_time_policy({'tau_minutes': 20})
    late = build_time_policy({'tau_minutes': 8})
    assert far['edge_threshold_yes'] > mid['edge_threshold_yes'] > late['edge_threshold_yes']
    assert far['edge_threshold_no'] > mid['edge_threshold_no'] > late['edge_threshold_no']


def test_kelly_multiplier_changes_by_tau_bucket():
    far = build_time_policy({'tau_minutes': 45})
    late = build_time_policy({'tau_minutes': 8})
    final = build_time_policy({'tau_minutes': 3})
    assert far['kelly_multiplier'] > late['kelly_multiplier']
    assert final['kelly_multiplier'] == 0.0


def test_final_late_expiry_bucket_blocks_new_entries():
    final = build_time_policy({'tau_minutes': 2})
    assert final['policy_bucket'] == 'final'


def test_blank_policy_multiplier_envs_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv('POLICY_FAR_KELLY_MULTIPLIER', '')
    monkeypatch.setenv('POLICY_FAR_MAX_TRADE_NOTIONAL_MULTIPLIER', '')

    far = build_time_policy({'tau_minutes': 45})

    assert far['kelly_multiplier'] == 1.0
    assert far['max_trade_notional_multiplier'] == 1.0


def test_hour_switch_changes_strike_and_decision_state():
    now = datetime.now(timezone.utc)
    model = _FakeModel(p_hat=0.55)
    bundle1 = _bundle(strike_price=64000.0, start=now - timedelta(minutes=5), end=now + timedelta(minutes=30), yes_mid=0.50, no_mid=0.48)
    bundle2 = _bundle(strike_price=64600.0, start=now + timedelta(hours=1), end=now + timedelta(hours=1, minutes=30), yes_mid=0.50, no_mid=0.48)
    bundle2['market_id'] = 'M2'
    bundle2['token_yes'] = 'YES2'
    bundle2['token_no'] = 'NO2'

    state1 = run_bot.build_market_decision_state(bundle1, run_bot.compute_market_probabilities(bundle1, model, now=now))
    state2 = run_bot.build_market_decision_state(bundle2, run_bot.compute_market_probabilities(bundle2, model, now=now + timedelta(hours=1)))

    assert state1['strike_price'] == 64000.0
    assert state2['strike_price'] == 64600.0
    assert state1['market_id'] == 'M1'
    assert state2['market_id'] == 'M2'


def test_tau_minutes_is_derived_from_bundle_end_time():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=42))
    model = _FakeModel()

    state = run_bot.compute_market_probabilities(bundle, model, now=now)

    assert state['tau_minutes'] == 42
    assert model.calls[0]['tau_minutes'] == 42


def test_missing_strike_blocks_trading_cleanly():
    now = datetime.now(timezone.utc)
    bundle = _bundle(strike_price=None, start=now - timedelta(minutes=5), end=now + timedelta(minutes=10))
    model = _FakeModel()

    prob = run_bot.compute_market_probabilities(bundle, model, now=now)
    decision = run_bot.build_market_decision_state(bundle, prob)

    assert prob['blocked'] is True
    assert prob['reason'] == 'missing_strike_price'
    assert decision['trade_allowed'] is False
    assert decision['reason'] == 'missing_strike_price'


def test_missing_yes_or_no_quote_produces_explicit_skip_state():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=10))
    model = _FakeModel()
    prob = run_bot.compute_market_probabilities(bundle, model, now=now)

    missing_yes = dict(bundle)
    missing_yes['yes_quote'] = {}
    yes_state = run_bot.build_market_decision_state(missing_yes, prob)
    assert yes_state['trade_allowed'] is False
    assert yes_state['reason'] == 'missing_yes_quote'

    missing_no = dict(bundle)
    missing_no['no_quote'] = {}
    no_state = run_bot.build_market_decision_state(missing_no, prob)
    assert no_state['trade_allowed'] is False
    assert no_state['reason'] == 'missing_no_quote'


def test_decision_path_can_choose_buy_yes_when_edge_yes_is_best(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.67,
        'p_no': 0.33,
        'q_yes': 0.50,
        'q_no': 0.40,
        'edge_yes': 0.17,
        'edge_no': -0.07,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_yes'


def test_decision_path_can_choose_buy_no_when_edge_no_is_best(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.44,
        'p_no': 0.56,
        'q_yes': 0.48,
        'q_no': 0.40,
        'edge_yes': -0.04,
        'edge_no': 0.16,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_no'


def test_decision_path_allows_opposite_side_entry_when_price_is_below_complement(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 2.0, 0.45, ts)
    monkeypatch.setattr(strategy_manager, 'ALLOW_OPPOSITE_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.44,
        'p_no': 0.56,
        'q_yes': 0.48,
        'q_no': 0.40,
        'edge_yes': -0.04,
        'edge_no': 0.16,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_no'


def test_decision_path_blocks_opposite_side_entry_above_complement_price(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 2.0, 0.45, ts)
    monkeypatch.setattr(strategy_manager, 'ALLOW_OPPOSITE_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.25,
        'p_no': 0.75,
        'q_yes': 0.48,
        'q_no': 0.60,
        'edge_yes': -0.23,
        'edge_no': 0.15,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_existing_market_side_exposure'
    assert action['reason'] == 'opposite_side_above_complement_price'
    assert action['blocking_outcome_side'] == 'YES'
    assert action['blocking_qty'] == 2.0
    assert action['avg_entry_price'] == 0.45
    assert action['max_opposite_entry_price'] == 0.55


def test_decision_path_allows_opposite_side_entry_when_inventory_is_reserved_in_sell(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 2.0, 0.45, ts)
    sell_order = storage.create_order('exit-yes', 'M1', 'YES1', 'YES', 'sell', 2.0, 0.52, 'open', ts)
    storage.create_reservation(sell_order['id'], 'M1', 'YES1', 'YES', 'inventory', 2.0, ts)
    monkeypatch.setattr(strategy_manager, 'ALLOW_OPPOSITE_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.44,
        'p_no': 0.56,
        'q_yes': 0.48,
        'q_no': 0.40,
        'edge_yes': -0.04,
        'edge_no': 0.16,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_no'


def test_decision_path_blocks_opposite_side_entry_when_opposite_buy_order_is_open(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_order('open-yes-buy', 'M1', 'YES1', 'YES', 'buy', 2.0, 0.45, 'open', ts)
    monkeypatch.setattr(strategy_manager, 'ALLOW_OPPOSITE_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.25,
        'p_no': 0.75,
        'q_yes': 0.48,
        'q_no': 0.40,
        'edge_yes': -0.23,
        'edge_no': 0.35,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_existing_market_side_exposure'
    assert action['reason'] == 'opposite_side_buy_order_open'
    assert action['blocking_buy_order_ids']


def test_decision_path_blocks_opposite_side_entry_when_flag_disabled(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 2.0, 0.45, ts)
    monkeypatch.setattr(strategy_manager, 'ALLOW_OPPOSITE_SIDE_ENTRY', False)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.44,
        'p_no': 0.56,
        'q_yes': 0.48,
        'q_no': 0.40,
        'edge_yes': -0.04,
        'edge_no': 0.16,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': build_time_policy({'tau_minutes': 20}),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_existing_market_side_exposure'
    assert action['reason'] == 'opposite_side_entry_disabled'
    assert action['blocking_outcome_side'] == 'YES'


def test_decision_logs_include_strike_tau_probability_and_edge_fields(tmp_path, monkeypatch):
    log_path = tmp_path / 'decision_state.jsonl'
    monkeypatch.setattr(run_bot, 'DECISION_LOG_PATH', log_path)
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'series_id': 'SERIES1',
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'strike_price': 64000.0,
        'spot_now': 64123.0,
        'tau_minutes': 12,
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.41,
        'q_tail': 0.41,
        'tail_side': 'NO',
        'chosen_side': 'YES',
        'polarized_tail_penalty': 1.0,
        'polarized_tail_blocked': False,
        'z_distance_to_strike': 0.8,
        'edge_yes': 0.07,
        'edge_no': -0.03,
        'policy_bucket': 'mid',
        'edge_threshold_yes': 0.03,
        'edge_threshold_no': 0.03,
        'kelly_multiplier': 0.8,
        'max_trade_notional_multiplier': 0.8,
        'allow_new_entries': True,
        'trade_allowed': True,
        'action': 'buy_yes',
        'reason': 'ok',
        'wallet_address': '0xabc',
        'wallet_usdc_e': 10.5,
        'wallet_pol': 1.0,
        'wallet_reserved_exposure': 0.5,
        'wallet_free_usdc': 10.0,
        'effective_bankroll': 10.0,
        'bankroll_source': 'wallet_live',
        'wallet_fetch_failed': False,
        'regime_guard_mode': 'shadow',
        'regime_guard_evaluated': True,
        'regime_guard_blocked': False,
        'regime_guard_reason': 'veto_extreme_minority_side',
        'regime_guard_details': {'minority_side_quote': 0.01},
        'minority_side_quote': 0.01,
        'same_side_existing_qty': 2.0,
        'same_side_existing_filled_entry_count': 1,
        'microstructure_regime': 'smooth',
        'spectral_entropy': 0.15,
        'low_freq_power_ratio': 0.84,
        'high_freq_power_ratio': 0.04,
        'smoothness_score': 0.88,
        'spectral_observation_count': 32,
        'spectral_window_minutes': 32,
        'spectral_ready': True,
        'spectral_reason': 'ok',
        'regime_state': {'regime_label': 'polarized_tail'},
        'would_block_in_shadow': True,
    }

    run_bot.append_decision_log(record)

    payload = json.loads(log_path.read_text(encoding='utf-8').strip())
    assert payload['strike_price'] == 64000.0
    assert payload['tau_minutes'] == 12
    assert payload['p_yes'] == 0.62
    assert payload['edge_yes'] == 0.07
    assert payload['policy_bucket'] == 'mid'
    assert payload['kelly_multiplier'] == 0.8
    assert payload['q_tail'] == 0.41
    assert payload['tail_side'] == 'NO'
    assert payload['chosen_side'] == 'YES'
    assert payload['wallet_usdc_e'] == 10.5
    assert payload['effective_bankroll'] == 10.0
    assert payload['bankroll_source'] == 'wallet_live'
    assert payload['regime_guard_mode'] == 'shadow'
    assert payload['regime_guard_evaluated'] is True
    assert payload['regime_guard_blocked'] is False
    assert payload['regime_guard_reason'] == 'veto_extreme_minority_side'
    assert payload['minority_side_quote'] == 0.01
    assert payload['same_side_existing_qty'] == 2.0
    assert payload['same_side_existing_filled_entry_count'] == 1
    assert payload['microstructure_regime'] == 'smooth'
    assert payload['smoothness_score'] == 0.88
    assert payload['spectral_ready'] is True
    assert payload['regime_state']['regime_label'] == 'polarized_tail'
    assert payload['would_block_in_shadow'] is True


def test_decision_log_includes_shadow_probability_models(tmp_path, monkeypatch):
    log_path = tmp_path / 'decision_state.jsonl'
    monkeypatch.setattr(run_bot, 'DECISION_LOG_PATH', log_path)
    now = datetime.now(timezone.utc).isoformat()
    decision_state = {
        'timestamp': now,
        'series_id': 'SERIES1',
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'strike_price': 64000.0,
        'spot_now': 64123.0,
        'tau_minutes': 12,
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.41,
        'edge_yes': 0.07,
        'edge_no': -0.03,
        'trade_allowed': True,
        'action': 'buy_yes',
        'reason': 'ok',
        'policy': _entry_policy(kelly_multiplier=0.8),
    }
    shadow_probability_models = {
        'kalman_blended_sigma_v1_cfg1': {
            'engine_name': 'kalman_blended_sigma_v1_cfg1',
            'p_yes': 0.61,
            'p_no': 0.39,
            'q_yes': 0.55,
            'q_no': 0.41,
            'edge_yes': 0.06,
            'edge_no': -0.02,
            'chosen_side': 'YES',
            'chosen_action': 'buy_yes',
            'trade_allowed': True,
            'disabled_reason': None,
            'policy_bucket': 'mid',
            'polarization_zone': 'balanced',
            'regime_label': None,
            'tail_penalty_score': 0.0,
            'tail_hard_block': False,
            'same_side_existing_qty': 0.0,
            'same_side_existing_filled_entry_count': 0,
            'agrees_with_live_entry': True,
            'would_veto_live_entry': False,
            'would_flip_live_side': False,
            'shadow_only_entry': False,
            'live_entry_side': 'YES',
            'shadow_entry_side': 'YES',
        },
        'gaussian_pde_diffusion_kalman_v1_cfg1': {
            'engine_name': 'gaussian_pde_diffusion_kalman_v1_cfg1',
            'p_yes': 0.49,
            'p_no': 0.51,
            'q_yes': 0.55,
            'q_no': 0.41,
            'edge_yes': -0.06,
            'edge_no': 0.10,
            'chosen_side': 'NO',
            'chosen_action': None,
            'trade_allowed': False,
            'disabled_reason': 'no_edge_above_threshold',
            'policy_bucket': 'mid',
            'polarization_zone': None,
            'regime_label': None,
            'tail_penalty_score': 0.0,
            'tail_hard_block': False,
            'same_side_existing_qty': 0.0,
            'same_side_existing_filled_entry_count': 0,
            'agrees_with_live_entry': False,
            'would_veto_live_entry': True,
            'would_flip_live_side': False,
            'shadow_only_entry': False,
            'live_entry_side': 'YES',
            'shadow_entry_side': None,
        },
    }

    record = run_bot.build_decision_log_record(
        decision_state=decision_state,
        trading_allowed=True,
        disabled_reason='ok',
        wallet_snapshot={},
        routing_debug={},
        shadow_probability_models=shadow_probability_models,
    )
    run_bot.append_decision_log(record)

    payload = json.loads(log_path.read_text(encoding='utf-8').strip())
    assert set(payload['shadow_probability_models']) == {
        'kalman_blended_sigma_v1_cfg1',
        'gaussian_pde_diffusion_kalman_v1_cfg1',
    }
    assert payload['shadow_probability_models']['kalman_blended_sigma_v1_cfg1']['agrees_with_live_entry'] is True
    assert payload['shadow_probability_models']['gaussian_pde_diffusion_kalman_v1_cfg1']['would_veto_live_entry'] is True
    assert 'shadow_entry_side' in payload['shadow_probability_models']['gaussian_pde_diffusion_kalman_v1_cfg1']


def test_decision_log_includes_shared_buy_gate_fields(tmp_path, monkeypatch):
    log_path = tmp_path / 'decision_state.jsonl'
    monkeypatch.setattr(run_bot, 'DECISION_LOG_PATH', log_path)
    now = datetime.now(timezone.utc).isoformat()
    decision_state = {
        'timestamp': now,
        'series_id': 'SERIES1',
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'strike_price': 64000.0,
        'spot_now': 64123.0,
        'tau_minutes': 12,
        'p_yes': 0.62,
        'p_no': 0.38,
        'q_yes': 0.55,
        'q_no': 0.41,
        'edge_yes': 0.07,
        'edge_no': -0.03,
        'trade_allowed': True,
        'action': 'buy_yes',
        'reason': 'ok',
        'policy': _entry_policy(kelly_multiplier=0.8),
        'action_origin': 'first_entry',
        'shared_buy_gate_evaluated': True,
        'shared_buy_gate_passed': True,
        'shared_buy_gate_reason': 'ok',
        'shared_guard_trade_allowed_checked': True,
        'shared_guard_quote_checked': True,
        'shared_guard_market_open_checked': True,
        'shared_guard_policy_checked': True,
        'shared_guard_tail_checked': True,
        'shared_guard_regime_checked': True,
        'shared_guard_exposure_checked': True,
        'shared_guard_active_order_checked': True,
        'shared_guard_result_json': {
            'action_origin': 'first_entry',
            'evaluated': True,
            'passed': True,
            'reason': 'ok',
            'checks': {
                'trade_allowed': {'checked': True, 'applicable': True},
            },
            'details': {'requested_notional': 1.23},
        },
    }

    record = run_bot.build_decision_log_record(
        decision_state=decision_state,
        trading_allowed=True,
        disabled_reason='ok',
        wallet_snapshot={},
        routing_debug={},
    )
    run_bot.append_decision_log(record)

    payload = json.loads(log_path.read_text(encoding='utf-8').strip())
    assert payload['action_origin'] == 'first_entry'
    assert payload['shared_buy_gate_evaluated'] is True
    assert payload['shared_buy_gate_passed'] is True
    assert payload['shared_guard_regime_checked'] is True
    assert payload['shared_guard_result_json']['details']['requested_notional'] == 1.23


def test_decision_log_includes_blocker_telemetry(tmp_path, monkeypatch):
    log_path = tmp_path / 'decision_state.jsonl'
    monkeypatch.setattr(run_bot, 'DECISION_LOG_PATH', log_path)
    decision_state = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'market_id': 'M1',
        'token_yes': 'YES1',
        'token_no': 'NO1',
        'q_yes': 0.51,
        'q_no': 0.47,
        'p_yes': 0.56,
        'p_no': 0.44,
        'action_origin': 'first_entry',
        'candidate_stage': 'post_candidate_blocked',
        'terminal_reason': 'microstructure_noisy_block',
        'terminal_reason_family': 'microstructure',
        'blocked_by': 'microstructure_noisy_block',
        'blocked_by_stage': 'post_candidate_blocked',
        'first_blocking_guard': 'microstructure_noisy_block',
        'all_triggered_blockers': ['microstructure_noisy_block'],
        'blocker_telemetry': {
            'candidate_stage': 'post_candidate_blocked',
            'candidate_available': True,
            'action_origin': 'first_entry',
            'terminal_reason': 'microstructure_noisy_block',
            'terminal_reason_family': 'microstructure',
            'blocked_by': 'microstructure_noisy_block',
            'blocked_by_stage': 'post_candidate_blocked',
            'first_blocking_guard': 'microstructure_noisy_block',
            'all_triggered_blockers': ['microstructure_noisy_block'],
            'chosen_side_snapshot': {'chosen_side': 'YES', 'chosen_action_candidate': 'buy_yes', 'chosen_side_quote': 0.51},
            'candidate_snapshot': {'q_yes': 0.51, 'q_no': 0.47, 'chosen_side_candidate': 'YES'},
            'blockers': {
                'microstructure_noisy_block': {
                    'evaluated': True,
                    'mode': 'live',
                    'would_block': True,
                    'blocked': True,
                    'reason': 'microstructure_noisy_block',
                    'inputs': {'microstructure_regime': 'noisy'},
                },
            },
        },
    }

    record = run_bot.build_decision_log_record(
        decision_state=decision_state,
        trading_allowed=False,
        disabled_reason='microstructure_noisy_block',
        wallet_snapshot={},
        routing_debug={},
    )
    run_bot.append_decision_log(record)

    payload = json.loads(log_path.read_text(encoding='utf-8').strip())
    assert payload['blocked_by'] == 'microstructure_noisy_block'
    assert payload['blocker_telemetry']['candidate_snapshot']['chosen_side_candidate'] == 'YES'
    assert payload['blocker_telemetry']['blockers']['microstructure_noisy_block']['blocked'] is True


def test_no_edge_above_threshold_is_classified_post_candidate():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20), yes_mid=0.55, no_mid=0.45)
    probability_state = {
        'blocked': False,
        'reason': None,
        'timestamp': now.isoformat(),
        'market_id': 'M1',
        'series_id': 'SERIES1',
        'spot_now': 64010.0,
        'strike_price': 64000.0,
        'tau_minutes': 20,
        'p_yes': 0.56,
        'p_no': 0.44,
        'sigma_per_sqrt_min': 0.001,
    }

    state = run_bot.build_market_decision_state(bundle, probability_state)

    assert state['reason'] == 'no_edge_above_threshold'
    assert state['candidate_stage'] == 'candidate_built'
    assert state['terminal_reason'] == 'no_edge_above_threshold'
    assert state['blocker_telemetry']['candidate_snapshot']['q_yes'] == 0.55
    assert state['blocker_telemetry']['blockers']['no_edge_above_threshold']['blocked'] is True


def test_missing_quote_is_classified_pre_candidate():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20), yes_mid=None, no_mid=0.45)
    probability_state = {
        'blocked': False,
        'reason': None,
        'timestamp': now.isoformat(),
        'market_id': 'M1',
        'series_id': 'SERIES1',
        'spot_now': 64010.0,
        'strike_price': 64000.0,
        'tau_minutes': 20,
        'p_yes': 0.60,
        'p_no': 0.40,
        'sigma_per_sqrt_min': 0.001,
    }

    state = run_bot.build_market_decision_state(bundle, probability_state)

    assert state['reason'] == 'missing_yes_quote'
    assert state['candidate_stage'] == 'pre_candidate'
    assert state['blocked_by'] == 'quote_invalid'


def test_entry_growth_veto_preserves_candidate_snapshot_fields(monkeypatch):
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20), yes_mid=0.50, no_mid=0.48)
    probability_state = {
        'blocked': False,
        'reason': None,
        'timestamp': now.isoformat(),
        'market_id': 'M1',
        'series_id': 'SERIES1',
        'spot_now': 64100.0,
        'strike_price': 64000.0,
        'tau_minutes': 20,
        'p_yes': 0.65,
        'p_no': 0.35,
        'sigma_per_sqrt_min': 0.001,
    }
    monkeypatch.setattr(run_bot, 'expected_growth_shadow_enabled', lambda: True)
    monkeypatch.setattr(run_bot, 'entry_growth_mode', lambda: 'live')
    monkeypatch.setattr(
        run_bot,
        'evaluate_entry_shadow',
        lambda **kwargs: {
            'expected_log_growth_entry': -0.1,
            'expected_log_growth_entry_conservative': -0.2,
            'expected_log_growth_entry_conservative_old': -0.2,
            'expected_log_growth_entry_conservative_discounted': -0.2,
            'expected_log_growth_pass_shadow': False,
            'expected_log_growth_reason_shadow': 'negative_expected_growth',
            'growth_gate_pass_shadow': False,
            'growth_gate_reason_shadow': 'negative_expected_growth',
            'expected_terminal_wealth_if_yes': 0.9,
            'expected_terminal_wealth_if_no': 1.0,
            'entry_growth_eval_mode': 'live',
            'entry_growth_candidate_side': kwargs['outcome_side'],
            'entry_growth_qty': 1.0,
            'entry_growth_trade_notional': 1.0,
            'entry_growth_probability_conservative': 0.60,
            'entry_growth_fragility_score': 0.2,
        },
    )

    state = run_bot.build_market_decision_state(bundle, probability_state, wallet_state={'effective_bankroll': 100.0, 'free_usdc': 100.0})

    assert state['reason'] == 'entry_growth_optimizer_veto'
    snapshot = state['blocker_telemetry']['candidate_snapshot']
    assert snapshot['chosen_side_candidate'] == 'YES'
    assert snapshot['q_yes'] == 0.50
    assert snapshot['adjusted_p_yes'] is not None
    assert snapshot['expected_log_growth_entry'] == -0.1
    assert state['blocked_by'] == 'entry_growth_optimizer_veto'


def test_multiple_blockers_preserve_order_and_first_blocking_guard(monkeypatch):
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20), yes_mid=0.40, no_mid=0.60)
    probability_state = {
        'blocked': False,
        'reason': None,
        'timestamp': now.isoformat(),
        'market_id': 'M1',
        'series_id': 'SERIES1',
        'spot_now': 64100.0,
        'strike_price': 64000.0,
        'tau_minutes': 20,
        'p_yes': 0.70,
        'p_no': 0.30,
        'sigma_per_sqrt_min': 0.001,
    }

    def fake_credibility_discount(**kwargs):
        outcome_side = kwargs['outcome_side']
        blocked = outcome_side == 'YES'
        return {
            'discounted_probability': kwargs['adjusted_probability'],
            'discounted_edge': kwargs['adjusted_probability'] - kwargs['chosen_side_quote'],
            'credibility_weight': 1.0,
            'polarization_zone': 'minority' if blocked else 'balanced',
            'reason': 'hard_block_yes' if blocked else 'ok',
            'hard_block': blocked,
        }

    monkeypatch.setattr(run_bot, 'polarization_credibility_mode', lambda: 'live')
    monkeypatch.setattr(run_bot, 'compute_credibility_discount', fake_credibility_discount)
    monkeypatch.setattr(
        run_bot,
        'evaluate_same_side_reentry_veto',
        lambda **kwargs: {'would_block': True, 'blocked': True, 'reason': 'veto_same_side_reentry_polarized_zone', 'mode': 'live'},
    )

    state = run_bot.build_market_decision_state(bundle, probability_state)

    assert state['reason'] == 'veto_polarization_hard_block'
    assert state['all_triggered_blockers'][:2] == ['veto_same_side_reentry_polarized_zone', 'veto_polarization_hard_block']
    assert state['first_blocking_guard'] == 'veto_polarization_hard_block'
    assert state['blocker_telemetry']['blockers']['veto_same_side_reentry_polarized_zone']['blocked'] is True


def test_extreme_minority_buy_yes_veto_blocks_in_live_mode(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'p_yes': 0.747,
        'p_no': 0.253,
        'q_yes': 0.01,
        'q_no': 0.99,
        'edge_yes': 0.737,
        'edge_no': -0.737,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_extreme_minority_side'
    assert decision_state['regime_guard_blocked'] is True
    assert decision_state['minority_side_quote'] == 0.01
    assert decision_state['trade_allowed'] is False


def test_extreme_minority_buy_no_veto_blocks_in_live_mode(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'p_yes': 0.269,
        'p_no': 0.731,
        'q_yes': 0.973,
        'q_no': 0.02,
        'edge_yes': -0.704,
        'edge_no': 0.711,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_extreme_minority_side'
    assert decision_state['regime_guard_blocked'] is True
    assert decision_state['minority_side_quote'] == 0.02


def test_polarized_tail_regime_veto_blocks_in_live_mode(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setenv('REGIME_POLARIZED_TAIL_MINORITY_QUOTE_MAX', '0.10')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'p_yes': 0.12,
        'p_no': 0.88,
        'q_yes': 0.92,
        'q_no': 0.08,
        'edge_yes': -0.80,
        'edge_no': 0.80,
        'trade_allowed': True,
        'reason': 'ok',
        'regime_state': {'regime_label': 'polarized_tail'},
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_regime_polarized_tail_minority_side'
    assert decision_state['regime_guard_blocked'] is True


def test_polarized_tail_regime_veto_is_shadow_only_when_configured(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'shadow')
    monkeypatch.setenv('REGIME_EXTREME_MINORITY_QUOTE_MAX', '0.05')
    monkeypatch.setenv('REGIME_POLARIZED_TAIL_MINORITY_QUOTE_MAX', '0.10')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.08})
    decision_state = {
        'p_yes': 0.12,
        'p_no': 0.88,
        'q_yes': 0.92,
        'q_no': 0.08,
        'edge_yes': -0.80,
        'edge_no': 0.80,
        'trade_allowed': True,
        'reason': 'ok',
        'regime_state': {'regime_label': 'polarized_tail'},
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_no'
    assert decision_state['regime_guard_blocked'] is False
    assert decision_state['regime_guard_reason'] == 'veto_regime_polarized_tail_minority_side'
    assert decision_state['would_block_in_shadow'] is True


def test_same_side_reentry_cap_blocks_existing_meaningful_yes_exposure(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 2.0, 0.42, ts)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_SAME_SIDE_MAX_ENTRIES_PER_MARKET_SIDE', '1')
    monkeypatch.setenv('REGIME_SAME_SIDE_MIN_FILLED_QTY', '0.01')
    monkeypatch.setattr(strategy_manager, 'ALLOW_SAME_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'p_yes': 0.60,
        'p_no': 0.40,
        'q_yes': 0.40,
        'q_no': 0.60,
        'edge_yes': 0.20,
        'edge_no': -0.20,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_same_side_reentry_cap'
    assert decision_state['same_side_existing_qty'] == 2.0
    assert decision_state['same_side_existing_filled_entry_count'] == 0


def test_same_side_reentry_cap_blocks_prior_materially_filled_buy_order(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    order = storage.create_order('filled-buy-1', 'M1', 'YES1', 'YES', 'buy', 1.0, 0.42, 'filled', ts)
    storage.update_order(order['id'], status='filled', filled_qty=0.5, remaining_qty=0.5, updated_ts=ts)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_SAME_SIDE_MAX_ENTRIES_PER_MARKET_SIDE', '1')
    monkeypatch.setenv('REGIME_SAME_SIDE_MIN_FILLED_QTY', '0.01')
    monkeypatch.setattr(strategy_manager, 'ALLOW_SAME_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'should_not_place'})
    decision_state = {
        'p_yes': 0.60,
        'p_no': 0.40,
        'q_yes': 0.40,
        'q_no': 0.60,
        'edge_yes': 0.20,
        'edge_no': -0.20,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_regime_entry_guard'
    assert action['reason'] == 'veto_same_side_reentry_cap'
    assert decision_state['same_side_existing_filled_entry_count'] == 1


def test_zero_fill_attempts_do_not_trigger_same_side_reentry_cap(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_order('attempt-1', 'M1', 'YES1', 'YES', 'buy', 1.0, 0.40, 'canceled', ts)
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setenv('REGIME_SAME_SIDE_MAX_ENTRIES_PER_MARKET_SIDE', '1')
    monkeypatch.setenv('REGIME_SAME_SIDE_MIN_FILLED_QTY', '0.01')
    monkeypatch.setattr(strategy_manager, 'ALLOW_SAME_SIDE_ENTRY', True)
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.40})
    decision_state = {
        'p_yes': 0.60,
        'p_no': 0.40,
        'q_yes': 0.40,
        'q_no': 0.60,
        'edge_yes': 0.20,
        'edge_no': -0.20,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision_state['same_side_existing_filled_entry_count'] == 0


def test_off_mode_disables_regime_entry_guard(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'off')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.01})
    decision_state = {
        'p_yes': 0.747,
        'p_no': 0.253,
        'q_yes': 0.01,
        'q_no': 0.99,
        'edge_yes': 0.737,
        'edge_no': -0.737,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision_state['regime_guard_mode'] == 'off'
    assert decision_state['regime_guard_evaluated'] is False
    assert decision_state['regime_guard_blocked'] is False


def test_normal_regime_with_non_extreme_quotes_is_unchanged(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'live')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run', 'submitted_qty': 1.0, 'submitted_notional': 0.40})
    decision_state = {
        'p_yes': 0.60,
        'p_no': 0.40,
        'q_yes': 0.40,
        'q_no': 0.60,
        'edge_yes': 0.20,
        'edge_no': -0.20,
        'trade_allowed': True,
        'reason': 'ok',
        'regime_state': {'regime_label': 'trend'},
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision_state['regime_guard_evaluated'] is True
    assert decision_state['regime_guard_blocked'] is False


def test_polarized_tail_guard_blocks_extreme_tail_trade(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.0015,
        'p_no': 0.9985,
        'q_yes': 0.999,
        'q_no': 0.001,
        'edge_yes': -0.9975,
        'edge_no': 0.9975,
        'spot_now': 74.08182206817179,
        'strike_price': 100.0,
        'tau_minutes': 1,
        'sigma_per_sqrt_min': 0.1,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['action'] == 'skipped_polarized_tail_block'
    assert action['q_tail'] == 0.001
    assert action['chosen_side'] == 'NO'
    assert abs(action['z'] - 3.0) < 1e-9
    assert decision_state['polarized_tail_blocked'] is True
    assert decision_state['tail_side'] == 'NO'
    assert decision_state['chosen_side'] == 'NO'


def test_polarized_tail_guard_penalizes_tail_trade_without_blocking(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setenv('REGIME_ENTRY_GUARD_MODE', 'off')
    buy_calls = []

    def fake_buy(token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None, **kwargs):
        buy_calls.append({'qty': qty, 'limit_price': limit_price, 'outcome_side': outcome_side})
        return {'status': 'dry_run', 'submitted_qty': qty, 'quantized_qty': qty, 'submitted_notional': qty * limit_price}

    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', fake_buy)
    monkeypatch.setattr(strategy_manager, 'BOT_BANKROLL', 1000.0)
    monkeypatch.setattr(strategy_manager, 'KELLY_K', 0.1)
    monkeypatch.setattr(strategy_manager, 'PER_TRADE_CAP_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'TOTAL_EXPOSURE_CAP', 10.0)
    decision_state = {
        'p_yes': 0.01,
        'p_no': 0.99,
        'q_yes': 0.97,
        'q_no': 0.03,
        'edge_yes': -0.96,
        'edge_no': 0.96,
        'spot_now': 81.87307530779819,
        'strike_price': 100.0,
        'tau_minutes': 1,
        'sigma_per_sqrt_min': 0.1,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(),
    }

    raw_fraction = strategy_manager.fractional_kelly(0.99, 0.03, k=strategy_manager.KELLY_K)
    expected_penalty = math.exp(-1.25 * (2.0 - 1.5))
    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_no'
    assert decision_state['polarized_tail_blocked'] is False
    assert decision_state['tail_side'] == 'NO'
    assert decision_state['chosen_side'] == 'NO'
    assert abs(decision_state['z_distance_to_strike'] - 2.0) < 1e-9
    assert abs(decision_state['polarized_tail_penalty'] - expected_penalty) < 1e-9
    assert buy_calls
    expected_notional = 1000.0 * raw_fraction * decision_state['policy']['kelly_multiplier'] * expected_penalty
    assert abs(action['raw_requested_notional'] - expected_notional) < 1e-9


def test_polarized_tail_guard_does_not_penalize_non_tail_trade(monkeypatch):
    storage.create_market('M1', status='open')
    monkeypatch.setattr(strategy_manager, 'place_marketable_buy', lambda *args, **kwargs: {'status': 'dry_run'})
    decision_state = {
        'p_yes': 0.99,
        'p_no': 0.01,
        'q_yes': 0.97,
        'q_no': 0.03,
        'edge_yes': 0.02,
        'edge_no': -0.02,
        'spot_now': 150.0,
        'strike_price': 100.0,
        'tau_minutes': 1,
        'sigma_per_sqrt_min': 0.1,
        'trade_allowed': True,
        'reason': 'ok',
        'policy': _entry_policy(edge_threshold_yes=0.01),
    }

    action = strategy_manager.build_trade_action(decision_state, 'YES1', 'NO1', 'M1', dry_run=True)

    assert action['side'] == 'buy_yes'
    assert decision_state['tail_side'] == 'NO'
    assert decision_state['chosen_side'] == 'YES'
    assert decision_state['polarized_tail_penalty'] == 1.0
    assert decision_state['polarized_tail_blocked'] is False


def test_inventory_exit_sells_profitable_fifo_inventory_and_ignores_entry_policy(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('EXIT1', status='open')
    storage.create_open_lot('EXIT1', 'YES1', 'YES', 3.0, 0.40, ts)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_VENUE_QTY_HAIRCUT_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'INVENTORY_EXIT_MIN_PROFIT_PER_SHARE', 0.01)

    sell_calls = []
    monkeypatch.setattr(
        strategy_manager,
        'place_marketable_sell',
        lambda token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None, **kwargs: (
            sell_calls.append((token_id, qty, limit_price, market_id, outcome_side))
            or {'status': 'filled', 'submitted_qty': qty, 'filled_qty': qty, 'price': limit_price}
        ),
    )

    action = strategy_manager.build_inventory_exit_action(
        'EXIT1',
        'YES1',
        'NO1',
        {'best_bid': 0.56, 'best_ask': 0.90},
        {'best_bid': 0.01, 'best_ask': 0.02},
        dry_run=True,
    )

    assert action['action'] == 'sell_yes'
    assert action['exit_decision_source'] == 'recycle_open_market_inventory'
    assert action['held_side'] == 'YES'
    assert abs(action['entry_price'] - 0.40) < 1e-9
    assert action['executable_exit_price'] == 0.56
    assert action['expected_profit_per_share'] > 0
    assert action['submitted_qty'] == 3.0
    assert action['filled_qty'] == 3.0
    assert action['avg_fill_price'] == 0.56
    assert action['residual_qty'] == 0.0
    assert sell_calls == [('YES1', 3.0, 0.56, 'EXIT1', 'YES')]


def test_inventory_exit_is_not_blocked_by_entry_exposure_or_final_bucket(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('EXIT2', status='open')
    storage.create_open_lot('EXIT2', 'NO2', 'NO', 2.0, 0.30, ts)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_VENUE_QTY_HAIRCUT_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'INVENTORY_EXIT_MIN_PROFIT_PER_SHARE', 0.01)
    monkeypatch.setattr(
        strategy_manager,
        'place_marketable_sell',
        lambda *args, **kwargs: {'status': 'filled', 'submitted_qty': 2.0, 'filled_qty': 2.0, 'price': 0.45},
    )

    action = strategy_manager.build_inventory_exit_action(
        'EXIT2',
        'YES2',
        'NO2',
        {'best_bid': 0.05, 'best_ask': 0.06},
        {'best_bid': 0.45, 'best_ask': 0.46},
        dry_run=True,
    )

    assert action['action'] == 'sell_no'
    assert action['filled_qty'] == 2.0


def test_recycler_two_sided_lock_sells_both_sides(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('PAIR1', status='open')
    storage.create_open_lot('PAIR1', 'YESP', 'YES', 2.0, 0.41, ts)
    storage.create_open_lot('PAIR1', 'NOP', 'NO', 3.0, 0.39, ts)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_VENUE_QTY_HAIRCUT_PCT', 1.0)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_PAIR_MAX_DISCOUNT_TO_PAYOUT', 0.03)
    monkeypatch.setattr(strategy_manager.polymarket_client, 'get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash} if tx_hash else None)

    calls = []

    def fake_sell(token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None, **kwargs):
        calls.append((token_id, qty, limit_price, outcome_side))
        return {'status': 'filled', 'submitted_qty': qty, 'filled_qty': qty, 'price': limit_price, 'tx_hash': f'tx-{token_id}'}

    monkeypatch.setattr(strategy_manager, 'place_marketable_sell', fake_sell)

    action = strategy_manager.recycle_open_market_inventory(
        'PAIR1',
        'YESP',
        'NOP',
        {'best_bid': 0.51},
        {'best_bid': 0.48},
        dry_run=False,
    )

    assert action['action'] == 'recycle_pair'
    assert action['trigger'] == 'two_sided_lock'
    assert action['pair_qty'] == 2.0
    assert [leg['action'] for leg in action['legs']] == ['sell_yes', 'sell_no']
    assert calls == [('YESP', 2.0, 0.51, 'YES'), ('NOP', 2.0, 0.48, 'NO')]
    disposals = storage.list_inventory_disposals('PAIR1')
    assert any(item['classification'] == 'recycle_pair' and item['action'] == 'recycle_pair' for item in disposals)


def test_recycler_pair_skips_when_existing_active_exit_order_present(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('PAIRBLOCK', status='open')
    storage.create_open_lot('PAIRBLOCK', 'YESPB', 'YES', 2.0, 0.41, ts)
    storage.create_open_lot('PAIRBLOCK', 'NOPB', 'NO', 2.0, 0.39, ts)
    storage.create_order('pairblock-sell', 'PAIRBLOCK', 'YESPB', 'YES', 'sell', 2.0, 0.51, 'open', ts)

    calls = []

    def fake_sell(token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None):
        calls.append((token_id, qty, limit_price, outcome_side))
        return {'status': 'filled', 'submitted_qty': qty, 'filled_qty': qty, 'price': limit_price}

    monkeypatch.setattr(strategy_manager, 'place_marketable_sell', fake_sell)

    action = strategy_manager.recycle_open_market_inventory(
        'PAIRBLOCK',
        'YESPB',
        'NOPB',
        {'best_bid': 0.51},
        {'best_bid': 0.48},
        dry_run=False,
    )

    assert action['action'] == 'hold'
    assert action['skip_reason'] == 'existing_active_exit_order'
    assert action['trigger'] == 'two_sided_lock'
    assert calls == []


def test_recycler_shrinks_sell_qty_once_on_balance_shortfall(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('SHRINK1', status='open')
    storage.create_open_lot('SHRINK1', 'YESS', 'YES', 5.0, 0.40, ts)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_VENUE_QTY_HAIRCUT_PCT', 0.995)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_SHORTFALL_RETRY_SHRINK_PCT', 0.5)
    monkeypatch.setattr(strategy_manager, 'INVENTORY_EXIT_MIN_PROFIT_PER_SHARE', 0.01)
    monkeypatch.setattr(strategy_manager.polymarket_client, 'get_tx_receipt', lambda tx_hash: {'transactionHash': tx_hash} if tx_hash else None)

    calls = []

    def fake_sell(token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None, **kwargs):
        calls.append(qty)
        if len(calls) == 1:
            return {
                'status': 'error',
                'raw_response_json': {
                    'error_message': 'not enough balance / allowance: the balance is not enough -> balance: 1254420, order amount: 1260000',
                },
            }
        return {'status': 'filled', 'submitted_qty': qty, 'filled_qty': qty, 'price': limit_price, 'tx_hash': '0xshrink'}

    monkeypatch.setattr(strategy_manager, 'place_marketable_sell', fake_sell)

    action = strategy_manager.build_inventory_exit_action(
        'SHRINK1',
        'YESS',
        'NOS',
        {'best_bid': 0.56},
        {'best_bid': 0.02},
        dry_run=False,
    )

    assert action['action'] == 'sell_yes'
    assert calls == [4.975, 1.2481]
    assert action['submitted_qty'] == 1.2481
    assert action['filled_qty'] == 1.2481
    assert action['shortfall_details']['available_qty'] == 1.25442
    assert action['shortfall_details']['attempted_qty'] == 1.26
    disposals = storage.list_inventory_disposals('SHRINK1')
    sell_disposals = [item for item in disposals if item['policy_type'] == 'recycler' and item['action'] == 'sell_yes']
    assert sell_disposals[-1]['tx_hash'] == '0xshrink'
    assert len(sell_disposals[-1]['response']['attempts']) == 2


def test_recycler_marks_dust_inventory_ineligible(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('DUST1', status='open')
    storage.create_open_lot('DUST1', 'YD1', 'YES', 0.005, 0.40, ts)
    monkeypatch.setattr(strategy_manager, 'RECYCLER_DUST_QTY_THRESHOLD', 0.01)

    action = strategy_manager.build_inventory_exit_action(
        'DUST1',
        'YD1',
        'ND1',
        {'best_bid': 0.56},
        {'best_bid': 0.02},
        dry_run=False,
    )

    assert action['action'] == 'hold'
    assert action['skip_reason'] == 'dust_inventory'
    assert action['residual_recycler_ineligible'] is True
    assert action['residual_classification'] == 'dust'


def test_no_stale_fallback_path_recomputes_target_from_current_price():
    now = datetime.now(timezone.utc)
    bundle = _bundle(strike_price=65000.0, start=now - timedelta(minutes=5), end=now + timedelta(minutes=15))
    model = _FakeModel(last_price=63000.0, p_hat=0.3)

    state = run_bot.compute_market_probabilities(bundle, model, now=now)

    assert state['strike_price'] == 65000.0
    assert model.calls[0]['target_price'] != model.last_price


def test_quote_freshness_and_spread_policy_tighten_near_expiry():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=3))
    model = _FakeModel()
    prob = run_bot.compute_market_probabilities(bundle, model, now=now)
    decision = run_bot.build_market_decision_state(bundle, prob)
    ctx = run_bot.build_trade_context(
        {
            'market_id': bundle['market_id'],
            'token_yes': bundle['token_yes'],
            'token_no': bundle['token_no'],
            'status': 'open',
            'startDate': pd.to_datetime(bundle['start_time'], utc=True),
            'endDate': pd.to_datetime(bundle['end_time'], utc=True),
        },
        {'mid': 0.55, 'age_seconds': 3.5, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.04},
        {'mid': 0.41, 'age_seconds': 3.5, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.04},
        now=now,
        routing_bundle=bundle,
    )
    ctx['decision_state'] = decision
    ctx['policy'] = decision['policy']

    ok, reason = run_bot.can_trade_context(ctx)

    assert decision['policy']['policy_bucket'] == 'final'
    assert ok is False
    assert reason in ('policy_blocks_new_entries', 'quote_stale', 'quote_too_wide', 'quote_empty')


def test_old_compatibility_wrapper_is_not_primary_tuned_path():
    decision = run_bot.build_market_decision_state(
        _bundle(),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.62,
            'p_no': 0.38,
        },
    )
    assert decision['policy'] is not None
    assert decision['policy']['policy_bucket'] == 'mid'


def test_heartbeat_includes_policy_fields():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20))
    model = _FakeModel()
    prob = run_bot.compute_market_probabilities(bundle, model, now=now)
    decision = run_bot.build_market_decision_state(bundle, prob)
    ctx = run_bot.build_trade_context(
        {
            'market_id': bundle['market_id'],
            'token_yes': bundle['token_yes'],
            'token_no': bundle['token_no'],
            'status': 'open',
            'startDate': pd.to_datetime(bundle['start_time'], utc=True),
            'endDate': pd.to_datetime(bundle['end_time'], utc=True),
        },
        {'mid': 0.55, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        {'mid': 0.41, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        now=now,
        routing_bundle=bundle,
    )
    ctx['probability_state'] = prob
    ctx['decision_state'] = decision
    ctx['policy'] = decision['policy']

    ok, reason = run_bot.can_trade_context(ctx)
    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, ok, reason)

    assert heartbeat['policy_bucket'] == decision['policy']['policy_bucket']
    assert heartbeat['edge_threshold_yes'] == decision['policy']['edge_threshold_yes']
    assert heartbeat['kelly_multiplier'] == decision['policy']['kelly_multiplier']


def test_heartbeat_includes_regime_guard_fields():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20))
    ctx = run_bot.build_trade_context(
        {
            'market_id': bundle['market_id'],
            'token_yes': bundle['token_yes'],
            'token_no': bundle['token_no'],
            'status': 'open',
            'startDate': pd.to_datetime(bundle['start_time'], utc=True),
            'endDate': pd.to_datetime(bundle['end_time'], utc=True),
        },
        {'mid': 0.55, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        {'mid': 0.41, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        now=now,
        routing_bundle=bundle,
    )
    ctx['decision_state'] = {
        'regime_state': {'regime_label': 'polarized_tail'},
        'regime_guard_mode': 'shadow',
        'regime_guard_evaluated': True,
        'regime_guard_blocked': False,
        'regime_guard_reason': 'veto_regime_polarized_tail_minority_side',
        'regime_guard_details': {'minority_side_quote': 0.08},
        'minority_side_quote': 0.08,
        'same_side_existing_qty': 2.0,
        'same_side_existing_filled_entry_count': 1,
        'would_block_in_shadow': True,
    }

    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, True, None)

    assert heartbeat['regime_state']['regime_label'] == 'polarized_tail'
    assert heartbeat['regime_guard_mode'] == 'shadow'
    assert heartbeat['regime_guard_evaluated'] is True
    assert heartbeat['regime_guard_blocked'] is False
    assert heartbeat['regime_guard_reason'] == 'veto_regime_polarized_tail_minority_side'
    assert heartbeat['minority_side_quote'] == 0.08
    assert heartbeat['same_side_existing_qty'] == 2.0
    assert heartbeat['same_side_existing_filled_entry_count'] == 1
    assert heartbeat['would_block_in_shadow'] is True


def test_heartbeat_includes_microstructure_fields():
    now = datetime.now(timezone.utc)
    bundle = _bundle(start=now - timedelta(minutes=5), end=now + timedelta(minutes=20))
    ctx = run_bot.build_trade_context(
        {
            'market_id': bundle['market_id'],
            'token_yes': bundle['token_yes'],
            'token_no': bundle['token_no'],
            'status': 'open',
            'startDate': pd.to_datetime(bundle['start_time'], utc=True),
            'endDate': pd.to_datetime(bundle['end_time'], utc=True),
        },
        {'mid': 0.55, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        {'mid': 0.41, 'age_seconds': 1, 'fetch_failed': False, 'is_empty': False, 'is_crossed': False, 'spread': 0.02},
        now=now,
        routing_bundle=bundle,
    )
    ctx['decision_state'] = {
        'microstructure_regime': 'mixed',
        'spectral_entropy': 0.42,
        'low_freq_power_ratio': 0.49,
        'high_freq_power_ratio': 0.22,
        'smoothness_score': 0.53,
        'spectral_observation_count': 32,
        'spectral_window_minutes': 32,
        'spectral_ready': True,
        'spectral_reason': 'ok',
    }

    heartbeat = run_bot.build_diagnostic_heartbeat(ctx, True, None)

    assert heartbeat['microstructure_regime'] == 'mixed'
    assert heartbeat['smoothness_score'] == 0.53
    assert heartbeat['spectral_ready'] is True


def test_polarization_normal_zone_leaves_admission_unchanged(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.35, no_mid=0.65),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.55,
            'p_no': 0.45,
        },
    )

    assert decision['polarization_zone'] == 'normal'
    assert decision['admission_edge_yes'] == decision['adjusted_edge_yes']
    assert decision['trade_allowed'] is True
    assert decision['action'] == 'buy_yes'


def test_polarization_caution_zone_reduces_admission_edge_materially(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')
    monkeypatch.setenv('POLARIZATION_CAUTION_BASE_WEIGHT', '0.50')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )

    assert decision['polarization_zone'] == 'caution'
    assert decision['discounted_edge_yes'] < decision['adjusted_edge_yes']
    assert decision['credibility_weight_yes'] <= 0.50


def test_polarization_strict_zone_reduces_admission_edge_more_than_caution(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')
    monkeypatch.setenv('POLARIZATION_CAUTION_BASE_WEIGHT', '0.60')
    monkeypatch.setenv('POLARIZATION_STRICT_BASE_WEIGHT', '0.20')

    caution = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )
    strict = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.08, no_mid=0.92),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )

    assert strict['polarization_zone'] == 'strict'
    assert strict['credibility_weight_yes'] < caution['credibility_weight_yes']
    assert strict['discounted_edge_yes'] < caution['discounted_edge_yes']


def test_polarization_hard_block_blocks_trade(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.04, no_mid=0.96),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )

    assert decision['polarization_zone'] == 'hard_block'
    assert decision['trade_allowed'] is False
    assert decision['reason'] == 'veto_polarization_hard_block'


def test_large_probability_divergence_causes_stronger_discount(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')
    monkeypatch.setenv('POLARIZATION_CAUTION_BASE_WEIGHT', '0.70')
    monkeypatch.setenv('POLARIZATION_DIVERGENCE_FULL_DISCOUNT_AT', '0.20')

    low_div = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
            'raw_model_output': {'raw_p_yes': 0.38, 'raw_p_no': 0.62, 'calibrated_p_yes': 0.40, 'calibrated_p_no': 0.60},
        },
    )
    hi_div = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
            'raw_model_output': {'raw_p_yes': 0.10, 'raw_p_no': 0.90, 'calibrated_p_yes': 0.40, 'calibrated_p_no': 0.60},
        },
    )

    assert hi_div['credibility_weight_yes'] < low_div['credibility_weight_yes']
    assert hi_div['discounted_edge_yes'] < low_div['discounted_edge_yes']


def test_same_side_reentry_in_polarized_zone_is_blocked_in_live_mode(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 1.0, 0.12, ts)
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'live')
    monkeypatch.setenv('POLARIZATION_SAME_SIDE_REENTRY_MODE', 'live')
    monkeypatch.setenv('POLARIZATION_SAME_SIDE_REENTRY_QUOTE_MAX', '0.20')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': ts,
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )

    assert decision['trade_allowed'] is False
    assert decision['reason'] == 'veto_same_side_reentry_polarized_zone'
    assert decision['same_side_reentry_live_blocked'] is True


def test_shadow_mode_logs_would_block_without_changing_action(monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    storage.create_market('M1', status='open')
    storage.create_open_lot('M1', 'YES1', 'YES', 1.0, 0.12, ts)
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'shadow')
    monkeypatch.setenv('POLARIZATION_SAME_SIDE_REENTRY_MODE', 'shadow')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': ts,
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
        },
    )

    assert decision['trade_allowed'] is True
    assert decision['action'] == 'buy_yes'
    assert decision['same_side_reentry_shadow_blocked'] is True
    assert decision['credibility_block_reason'] == 'veto_same_side_reentry_polarized_zone'


def test_lineage_fields_are_finite_when_decision_is_valid(monkeypatch):
    monkeypatch.setenv('POLARIZATION_CREDIBILITY_MODE', 'shadow')

    decision = run_bot.build_market_decision_state(
        _bundle(yes_mid=0.18, no_mid=0.82),
        {
            'blocked': False,
            'reason': None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'series_id': 'SERIES1',
            'market_id': 'M1',
            'spot_now': 64123.0,
            'strike_price': 64000.0,
            'tau_minutes': 20,
            'p_yes': 0.40,
            'p_no': 0.60,
            'raw_model_output': {'raw_p_yes': 0.35, 'raw_p_no': 0.65, 'calibrated_p_yes': 0.40, 'calibrated_p_no': 0.60},
        },
    )

    for key in (
        'raw_p_yes',
        'raw_p_no',
        'calibrated_p_yes',
        'calibrated_p_no',
        'discounted_p_yes',
        'discounted_p_no',
        'raw_edge_yes',
        'raw_edge_no',
        'adjusted_edge_yes',
        'adjusted_edge_no',
        'discounted_edge_yes',
        'discounted_edge_no',
        'admission_edge_yes',
        'admission_edge_no',
        'credibility_weight_yes',
        'credibility_weight_no',
    ):
        assert math.isfinite(float(decision[key])), key
