from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np


REGIME_NORMAL = 'normal'
REGIME_TREND = 'trend'
REGIME_POLARIZED_TAIL = 'polarized_tail'
REGIME_REVERSAL_ATTEMPT = 'reversal_attempt'
REGIME_LABELS = {
    REGIME_NORMAL,
    REGIME_TREND,
    REGIME_POLARIZED_TAIL,
    REGIME_REVERSAL_ATTEMPT,
}

TAIL_ENTER_TAU_MAX = 20.0
TAIL_ENTER_POLARIZATION = 0.88
TAIL_ENTER_ABS_Z = 2.0
TAIL_EXIT_POLARIZATION = 0.82
TAIL_EXIT_ABS_Z = 1.6
TAIL_ENTER_PERSISTENCE = 2
TAIL_EXIT_PERSISTENCE = 3

TREND_ENTER_SCORE = 0.65
TREND_EXIT_SCORE = 0.45
TREND_ENTER_SIGN_CONSISTENCY = 0.75
TREND_EXIT_SIGN_CONSISTENCY = 0.55
TREND_ENTER_MOVE_MAGNITUDE = 0.003
TREND_EXIT_PERSISTENCE = 3
TREND_ENTER_PERSISTENCE = 2

REVERSAL_ENTER_PERSISTENCE = 3
REVERSAL_COUNTER_MOVE_MIN = 0.0015
REVERSAL_QUOTE_RECENTER_MIN = 0.05
REVERSAL_TREND_WEAKENING_DELTA = 0.10
REVERSAL_SCORE_THRESHOLD = 0.60
REVERSAL_FADE_PERSISTENCE = 2

RECENT_HISTORY_LIMIT = 6
MICROSTRUCTURE_FIELD_NAMES = (
    'microstructure_regime',
    'spectral_entropy',
    'low_freq_power_ratio',
    'high_freq_power_ratio',
    'smoothness_score',
    'spectral_observation_count',
    'spectral_window_minutes',
    'spectral_ready',
    'spectral_reason',
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_mode(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value not in ('off', 'shadow', 'live'):
        return default
    return value


def microstructure_spectral_mode() -> str:
    """Return the microstructure spectral mode: 'off', 'shadow', or 'live'."""
    return _env_mode('MICROSTRUCTURE_SPECTRAL_MODE', 'shadow')


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _get_nested(mapping: dict, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _microstructure_result(
    *,
    regime: str,
    spectral_entropy: Optional[float],
    low_freq_power_ratio: Optional[float],
    high_freq_power_ratio: Optional[float],
    smoothness_score: Optional[float],
    spectral_observation_count: int,
    spectral_window_minutes: int,
    spectral_ready: bool,
    spectral_reason: str,
) -> dict:
    return {
        'microstructure_regime': regime,
        'spectral_entropy': spectral_entropy,
        'low_freq_power_ratio': low_freq_power_ratio,
        'high_freq_power_ratio': high_freq_power_ratio,
        'smoothness_score': smoothness_score,
        'spectral_observation_count': int(max(0, spectral_observation_count)),
        'spectral_window_minutes': int(max(1, spectral_window_minutes)),
        'spectral_ready': bool(spectral_ready),
        'spectral_reason': spectral_reason,
    }


def compute_microstructure_regime(price_history) -> dict:
    mode = microstructure_spectral_mode()
    spectral_window = max(2, _env_int('MICROSTRUCTURE_SPECTRAL_WINDOW', 32))
    min_obs = max(2, _env_int('MICROSTRUCTURE_SPECTRAL_MIN_OBS', 24))
    smooth_threshold = _env_float('MICROSTRUCTURE_SMOOTH_THRESHOLD', 0.67)
    noisy_threshold = _env_float('MICROSTRUCTURE_NOISY_THRESHOLD', 0.40)
    if mode == 'off':
        return _microstructure_result(
            regime='disabled',
            spectral_entropy=None,
            low_freq_power_ratio=None,
            high_freq_power_ratio=None,
            smoothness_score=None,
            spectral_observation_count=0,
            spectral_window_minutes=spectral_window,
            spectral_ready=False,
            spectral_reason='disabled',
        )

    if price_history is None:
        prices = np.asarray([], dtype=float)
    else:
        values = getattr(price_history, 'values', price_history)
        try:
            prices = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            prices = np.asarray([], dtype=float)
    if prices.ndim == 0:
        prices = prices.reshape(1)
    prices = prices[np.isfinite(prices)]
    prices = prices[prices > 0.0]
    if prices.size < 2:
        return _microstructure_result(
            regime='unknown',
            spectral_entropy=None,
            low_freq_power_ratio=None,
            high_freq_power_ratio=None,
            smoothness_score=None,
            spectral_observation_count=0,
            spectral_window_minutes=spectral_window,
            spectral_ready=False,
            spectral_reason='insufficient_history',
        )

    returns = np.diff(np.log(prices))
    observation_count = int(min(int(returns.size), spectral_window))
    if observation_count < min_obs:
        return _microstructure_result(
            regime='unknown',
            spectral_entropy=None,
            low_freq_power_ratio=None,
            high_freq_power_ratio=None,
            smoothness_score=None,
            spectral_observation_count=observation_count,
            spectral_window_minutes=spectral_window,
            spectral_ready=False,
            spectral_reason='insufficient_history',
        )

    windowed_returns = returns[-observation_count:]
    demeaned_returns = windowed_returns - float(np.mean(windowed_returns))
    power = np.abs(np.fft.rfft(demeaned_returns)) ** 2
    retained_power = power[1:] if power.size > 1 else np.asarray([], dtype=float)
    bin_count = int(retained_power.size)
    epsilon = 1e-12
    total_power = float(np.sum(retained_power)) if bin_count > 0 else 0.0
    if bin_count <= 1 or total_power <= epsilon:
        normalized_power = np.zeros(bin_count, dtype=float)
        spectral_entropy = 0.0
    else:
        normalized_power = retained_power / total_power
        safe_power = normalized_power[normalized_power > epsilon]
        spectral_entropy = 0.0 if safe_power.size == 0 else float(-np.sum(safe_power * np.log(safe_power)) / math.log(bin_count))

    if bin_count <= 0:
        low_freq_power_ratio = 0.0
        high_freq_power_ratio = 0.0
    else:
        third = max(1, bin_count // 3)
        low_freq_power_ratio = float(np.sum(normalized_power[:third]))
        high_freq_power_ratio = float(np.sum(normalized_power[-third:]))
    smoothness_score = float(0.65 * low_freq_power_ratio + 0.35 * (1.0 - spectral_entropy))
    if smoothness_score >= smooth_threshold:
        regime = 'smooth'
    elif smoothness_score <= noisy_threshold:
        regime = 'noisy'
    else:
        regime = 'mixed'
    return _microstructure_result(
        regime=regime,
        spectral_entropy=float(spectral_entropy),
        low_freq_power_ratio=float(low_freq_power_ratio),
        high_freq_power_ratio=float(high_freq_power_ratio),
        smoothness_score=smoothness_score,
        spectral_observation_count=observation_count,
        spectral_window_minutes=spectral_window,
        spectral_ready=True,
        spectral_reason='ok',
    )


def _extract_state_features(state: dict) -> dict:
    q_yes = _safe_float(state.get('q_yes'))
    p_yes = _safe_float(state.get('p_yes'))
    used_market_quote_proxy = False
    if q_yes is None:
        q_yes = p_yes
        used_market_quote_proxy = q_yes is not None
    abs_z_score = _safe_float(state.get('abs_z_score'))
    if abs_z_score is None:
        abs_z_score = abs(_safe_float(state.get('z_score')) or 0.0)
    market_polarization = None if q_yes is None else max(q_yes, 1.0 - q_yes)
    tau_minutes = _safe_float(state.get('tau_minutes'))
    spot_now = _safe_float(state.get('spot_now'))
    return {
        'q_yes': q_yes,
        'p_yes': p_yes,
        'tau_minutes': tau_minutes,
        'abs_z_score': abs_z_score,
        'spot_now': spot_now,
        'market_polarization': market_polarization,
        'used_market_quote_proxy': used_market_quote_proxy,
        'edge_yes': _safe_float(state.get('edge_yes')),
        'edge_no': _safe_float(state.get('edge_no')),
    }


def _normalized_tau_pressure(tau_minutes: Optional[float]) -> float:
    if tau_minutes is None:
        return 0.0
    return _clip01((60.0 - max(0.0, tau_minutes)) / 60.0)


def _build_recent_features(previous_state: Optional[dict], current_features: dict) -> dict:
    previous_state = previous_state or {}
    recent = list(previous_state.get('recent_observations') or [])
    observations = recent[-(RECENT_HISTORY_LIMIT - 1):] + [{'source': current_features}]
    feature_rows = []
    for item in observations:
        source = item.get('source') if isinstance(item, dict) else {}
        if not isinstance(source, dict):
            source = {}
        row = _extract_state_features(source)
        feature_rows.append(row)

    valid_spots = [row['spot_now'] for row in feature_rows if row.get('spot_now') is not None and row['spot_now'] > 0]
    spot_returns = []
    if len(valid_spots) >= 2:
        for left, right in zip(valid_spots[:-1], valid_spots[1:]):
            if left and right:
                spot_returns.append((right - left) / left)

    if spot_returns:
        sign_sum = sum(1 if value > 0 else -1 if value < 0 else 0 for value in spot_returns)
        sign_consistency = abs(sign_sum) / len(spot_returns)
        recent_move_direction = 1 if sign_sum > 0 else -1 if sign_sum < 0 else 0
        cumulative_move = sum(spot_returns)
        counter_move_last = spot_returns[-1]
    else:
        sign_consistency = 0.0
        recent_move_direction = 0
        cumulative_move = 0.0
        counter_move_last = 0.0

    current_abs_z = current_features.get('abs_z_score')
    current_polarization = current_features.get('market_polarization')
    trend_score = _clip01(
        0.5 * sign_consistency
        + 0.35 * _clip01(abs(cumulative_move) / 0.01)
        + 0.15 * _clip01((current_abs_z or 0.0) / 2.5)
    )
    tail_score = _clip01(
        0.45 * _clip01((current_abs_z or 0.0) / 2.5)
        + 0.35 * _clip01(((current_polarization or 0.5) - 0.5) / 0.5)
        + 0.20 * _normalized_tau_pressure(current_features.get('tau_minutes'))
    )
    return {
        'recent_observations': observations,
        'spot_returns': spot_returns,
        'sign_consistency': sign_consistency,
        'recent_move_direction': recent_move_direction,
        'cumulative_move': cumulative_move,
        'counter_move_last': counter_move_last,
        'trend_score': trend_score,
        'tail_score': tail_score,
    }


def detect_regime(decision_inputs: dict, previous_state: Optional[dict] = None) -> dict:
    previous_state = previous_state or {}
    prev_label = str(previous_state.get('regime_label') or REGIME_NORMAL)
    if prev_label not in REGIME_LABELS:
        prev_label = REGIME_NORMAL
    prev_source = previous_state.get('source') or {}
    prev_machine = prev_source.get('state_machine') or {}

    current_features = _extract_state_features(decision_inputs)
    recent = _build_recent_features(previous_state, current_features)
    market_polarization = current_features.get('market_polarization')
    abs_z_score = current_features.get('abs_z_score') or 0.0
    tau_minutes = current_features.get('tau_minutes')
    trend_score = recent['trend_score']
    tail_score = recent['tail_score']
    recent_move_direction = recent['recent_move_direction']
    sign_consistency = recent['sign_consistency']
    cumulative_move = recent['cumulative_move']

    insufficient_history = len(recent['spot_returns']) < 2
    tail_enter_cond = (
        tau_minutes is not None
        and tau_minutes <= TAIL_ENTER_TAU_MAX
        and (
            (market_polarization is not None and market_polarization >= TAIL_ENTER_POLARIZATION)
            or abs_z_score >= TAIL_ENTER_ABS_Z
        )
    )
    tail_exit_weak_cond = (
        market_polarization is not None
        and market_polarization < TAIL_EXIT_POLARIZATION
        and abs_z_score < TAIL_EXIT_ABS_Z
    )
    trend_enter_cond = (
        not tail_enter_cond
        and not insufficient_history
        and trend_score >= TREND_ENTER_SCORE
        and sign_consistency >= TREND_ENTER_SIGN_CONSISTENCY
        and abs(cumulative_move) >= TREND_ENTER_MOVE_MAGNITUDE
        and recent_move_direction != 0
    )
    trend_exit_weak_cond = (
        insufficient_history
        or trend_score < TREND_EXIT_SCORE
        or sign_consistency < TREND_EXIT_SIGN_CONSISTENCY
        or recent_move_direction == 0
    )

    dominant_regime_label = prev_machine.get('dominant_regime_label') or prev_label
    dominant_direction = prev_machine.get('dominant_direction')
    if dominant_direction not in (-1, 1):
        dominant_direction = prev_source.get('trend_direction') if prev_source.get('trend_direction') in (-1, 1) else None
    prior_extreme_q_yes = _safe_float(prev_machine.get('prior_extreme_q_yes'))
    if prior_extreme_q_yes is None:
        prior_extreme_q_yes = _safe_float(prev_source.get('prior_extreme_q_yes'))
    if prior_extreme_q_yes is None:
        prior_extreme_q_yes = _safe_float(prev_source.get('q_yes'))

    allowed_reversal_origin = dominant_regime_label in {REGIME_TREND, REGIME_POLARIZED_TAIL}
    counter_move = (
        dominant_direction in (-1, 1)
        and recent_move_direction in (-1, 1)
        and recent_move_direction == -dominant_direction
        and abs(recent['counter_move_last']) >= REVERSAL_COUNTER_MOVE_MIN
    )
    prior_pressure_score = max(
        _safe_float(prev_source.get('dominant_trend_score')) or 0.0,
        _safe_float(previous_state.get('trend_score')) or 0.0,
        _safe_float(previous_state.get('tail_score')) or 0.0 if dominant_regime_label == REGIME_POLARIZED_TAIL else 0.0,
    )
    weakening_prior = trend_score <= max(0.0, prior_pressure_score - REVERSAL_TREND_WEAKENING_DELTA)
    quote_recenter = False
    q_yes = current_features.get('q_yes')
    if q_yes is not None and prior_extreme_q_yes is not None:
        quote_recenter = (
            abs(q_yes - prior_extreme_q_yes) >= REVERSAL_QUOTE_RECENTER_MIN
            and abs(q_yes - 0.5) < abs(prior_extreme_q_yes - 0.5)
        )
    reversal_score = _clip01(
        0.4 * (1.0 if counter_move else 0.0)
        + 0.3 * (1.0 if weakening_prior else 0.0)
        + 0.3 * (1.0 if quote_recenter else 0.0)
    )
    reversal_cond = allowed_reversal_origin and counter_move and weakening_prior and quote_recenter and reversal_score >= REVERSAL_SCORE_THRESHOLD

    counters = {
        'tail_enter_count': (int(prev_machine.get('tail_enter_count') or 0) + 1) if tail_enter_cond else 0,
        'tail_exit_count': (int(prev_machine.get('tail_exit_count') or 0) + 1) if tail_exit_weak_cond else 0,
        'trend_enter_count': (int(prev_machine.get('trend_enter_count') or 0) + 1) if trend_enter_cond else 0,
        'trend_exit_count': (int(prev_machine.get('trend_exit_count') or 0) + 1) if trend_exit_weak_cond else 0,
        'reversal_enter_count': (int(prev_machine.get('reversal_enter_count') or 0) + 1) if reversal_cond else 0,
        'reversal_fade_count': (int(prev_machine.get('reversal_fade_count') or 0) + 1) if (prev_label == REGIME_REVERSAL_ATTEMPT and not reversal_cond) else 0,
    }

    next_label = prev_label
    reason = 'hold_previous_regime'

    if prev_label == REGIME_POLARIZED_TAIL:
        if counters['tail_enter_count'] >= TAIL_ENTER_PERSISTENCE and tail_enter_cond:
            next_label = REGIME_POLARIZED_TAIL
            reason = 'polarized_tail_persistent'
        elif counters['reversal_enter_count'] >= REVERSAL_ENTER_PERSISTENCE:
            next_label = REGIME_REVERSAL_ATTEMPT
            reason = 'polarized_tail_weakening_with_reversal_evidence'
        elif counters['tail_exit_count'] >= TAIL_EXIT_PERSISTENCE:
            next_label = REGIME_TREND if counters['trend_enter_count'] >= TREND_ENTER_PERSISTENCE else REGIME_NORMAL
            reason = 'polarized_tail_exit_persistence_satisfied'
        else:
            next_label = REGIME_POLARIZED_TAIL
            reason = 'polarized_tail_hysteresis_hold'
    elif prev_label == REGIME_TREND:
        if counters['tail_enter_count'] >= TAIL_ENTER_PERSISTENCE:
            next_label = REGIME_POLARIZED_TAIL
            reason = 'tail_precedence_over_trend'
        elif counters['reversal_enter_count'] >= REVERSAL_ENTER_PERSISTENCE:
            next_label = REGIME_REVERSAL_ATTEMPT
            reason = 'trend_weakening_with_counter_move'
        elif trend_enter_cond:
            next_label = REGIME_TREND
            reason = 'trend_persistent'
        elif counters['trend_exit_count'] >= TREND_EXIT_PERSISTENCE:
            next_label = REGIME_NORMAL
            reason = 'trend_exit_persistence_satisfied'
        else:
            next_label = REGIME_TREND
            reason = 'trend_hysteresis_hold'
    elif prev_label == REGIME_REVERSAL_ATTEMPT:
        if counters['tail_enter_count'] >= TAIL_ENTER_PERSISTENCE:
            next_label = REGIME_POLARIZED_TAIL
            reason = 'tail_reasserted_during_reversal_attempt'
        elif trend_enter_cond and dominant_direction in (-1, 1) and recent_move_direction == -dominant_direction and counters['trend_enter_count'] >= TREND_ENTER_PERSISTENCE:
            next_label = REGIME_TREND
            reason = 'opposite_trend_persistent_after_reversal_attempt'
        elif reversal_cond:
            next_label = REGIME_REVERSAL_ATTEMPT
            reason = 'reversal_attempt_persistent'
        elif counters['reversal_fade_count'] >= REVERSAL_FADE_PERSISTENCE:
            if dominant_regime_label == REGIME_POLARIZED_TAIL and not tail_exit_weak_cond:
                next_label = REGIME_POLARIZED_TAIL
                reason = 'reversal_attempt_faded_back_to_tail'
            elif dominant_regime_label == REGIME_TREND and not trend_exit_weak_cond:
                next_label = REGIME_TREND
                reason = 'reversal_attempt_faded_back_to_trend'
            else:
                next_label = REGIME_NORMAL
                reason = 'reversal_attempt_faded_to_normal'
        else:
            next_label = REGIME_REVERSAL_ATTEMPT
            reason = 'reversal_attempt_hysteresis_hold'
    else:
        if counters['tail_enter_count'] >= TAIL_ENTER_PERSISTENCE:
            next_label = REGIME_POLARIZED_TAIL
            reason = 'tail_entry_persistence_satisfied'
        elif counters['trend_enter_count'] >= TREND_ENTER_PERSISTENCE:
            next_label = REGIME_TREND
            reason = 'trend_entry_persistence_satisfied'
        else:
            next_label = REGIME_NORMAL
            reason = 'normal_default_fallback'

    if insufficient_history and next_label in {REGIME_TREND, REGIME_REVERSAL_ATTEMPT}:
        next_label = REGIME_NORMAL
        reason = 'insufficient_history_fallback_to_normal'

    if next_label == REGIME_REVERSAL_ATTEMPT and dominant_regime_label not in {REGIME_TREND, REGIME_POLARIZED_TAIL}:
        next_label = REGIME_NORMAL
        reason = 'reversal_not_allowed_without_prior_trend_or_tail'

    if next_label == REGIME_REVERSAL_ATTEMPT:
        resolved_dominant_regime = dominant_regime_label
        resolved_dominant_direction = dominant_direction
    elif next_label in {REGIME_TREND, REGIME_POLARIZED_TAIL}:
        resolved_dominant_regime = next_label
        resolved_dominant_direction = recent_move_direction if recent_move_direction in (-1, 1) else dominant_direction
    else:
        resolved_dominant_regime = REGIME_NORMAL
        resolved_dominant_direction = None

    if next_label != prev_label:
        transition = f'{prev_label}->{next_label}'
        persistence_count = 1
    else:
        transition = 'stay'
        persistence_count = int(previous_state.get('persistence_count') or 0) + 1

    if prior_extreme_q_yes is None and q_yes is not None:
        prior_extreme_q_yes = q_yes
    if next_label in {REGIME_TREND, REGIME_POLARIZED_TAIL} and q_yes is not None:
        if prior_extreme_q_yes is None:
            prior_extreme_q_yes = q_yes
        elif abs(q_yes - 0.5) > abs(prior_extreme_q_yes - 0.5):
            prior_extreme_q_yes = q_yes

    microstructure_state = {
        key: decision_inputs.get(key)
        for key in MICROSTRUCTURE_FIELD_NAMES
    }

    source = {
        **current_features,
        **microstructure_state,
        'recent_move_direction': recent_move_direction,
        'sign_consistency': sign_consistency,
        'cumulative_move': cumulative_move,
        'trend_direction': recent_move_direction if recent_move_direction in (-1, 1) else None,
        'prior_extreme_q_yes': prior_extreme_q_yes,
        'dominant_trend_score': trend_score if next_label in {REGIME_TREND, REGIME_REVERSAL_ATTEMPT} else previous_state.get('trend_score'),
        'state_machine': {
            **counters,
            'dominant_regime_label': resolved_dominant_regime,
            'dominant_direction': resolved_dominant_direction,
            'prior_extreme_q_yes': prior_extreme_q_yes,
        },
    }

    return {
        'regime_label': next_label,
        'previous_regime': prev_label,
        'trend_score': trend_score,
        'tail_score': tail_score,
        'reversal_score': reversal_score,
        'market_polarization': market_polarization,
        **microstructure_state,
        'regime_reason': reason,
        'regime_transition': transition,
        'persistence_count': persistence_count,
        'used_market_quote_proxy': bool(current_features.get('used_market_quote_proxy')),
        'shadow_only': True,
        'source': source,
    }
