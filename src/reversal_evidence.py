from __future__ import annotations

import os
from typing import Any

import pandas as pd


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_result(side: str, reason: str, max_score: int) -> dict:
    return {
        'side': side,
        'score': 0,
        'max_score': max_score,
        'score_ratio': 0.0,
        'passes_min_score': False,
        'last_1m_return': None,
        'last_3m_return': None,
        'short_slope': None,
        'spot_vs_short_ewma': None,
        'moved_toward_strike': None,
        'reason': reason,
    }


def compute_reversal_evidence(prices, *, side: str, strike_price: float | None = None, spot_now: float | None = None) -> dict:
    normalized_side = str(side or '').upper()
    if normalized_side not in {'YES', 'NO'}:
        raise ValueError(f'unsupported side: {side}')
    enabled = _env_flag('REVERSAL_EVIDENCE_ENABLED', True)
    min_score = max(0, _env_int('REVERSAL_EVIDENCE_MIN_SCORE', 3))
    use_strike = _env_flag('REVERSAL_EVIDENCE_USE_STRIKE', True)
    max_score = 5 if use_strike else 4
    if not enabled:
        result = _empty_result(normalized_side, 'disabled', max_score)
        result['passes_min_score'] = True
        result['reason'] = 'disabled'
        return result

    series = pd.Series(prices).dropna().astype(float)
    if len(series) < 4:
        return _empty_result(normalized_side, 'insufficient_price_history', max_score)

    closes = series.iloc[-4:]
    last_close = float(closes.iloc[-1])
    prior_close = float(closes.iloc[-2])
    close_3m_ago = float(closes.iloc[0])
    baseline = max(abs(prior_close), 1e-12)
    baseline_3m = max(abs(close_3m_ago), 1e-12)
    last_1m_return = (last_close / baseline) - 1.0
    last_3m_return = (last_close / baseline_3m) - 1.0
    short_slope = (last_close - close_3m_ago) / 3.0
    short_ewma = float(closes.ewm(span=3, adjust=False).mean().iloc[-1])
    spot_ref = _safe_float(spot_now)
    if spot_ref is None:
        spot_ref = last_close
    spot_vs_short_ewma = spot_ref - short_ewma

    strike_val = _safe_float(strike_price)
    moved_toward_strike = None
    if use_strike and strike_val is not None:
        prev_distance = abs(prior_close - strike_val)
        current_distance = abs(last_close - strike_val)
        moved_toward_strike = current_distance + 1e-12 < prev_distance

    sign = 1.0 if normalized_side == 'YES' else -1.0
    score = 0
    if sign * last_1m_return > 0:
        score += 1
    if sign * last_3m_return > 0:
        score += 1
    if sign * short_slope > 0:
        score += 1
    if sign * spot_vs_short_ewma > 0:
        score += 1
    if use_strike and moved_toward_strike is True:
        score += 1

    passes = score >= min_score
    reason = 'ok' if passes else 'min_score_not_met'
    return {
        'side': normalized_side,
        'score': score,
        'max_score': max_score,
        'score_ratio': 0.0 if max_score <= 0 else float(score) / float(max_score),
        'passes_min_score': passes,
        'last_1m_return': last_1m_return,
        'last_3m_return': last_3m_return,
        'short_slope': short_slope,
        'spot_vs_short_ewma': spot_vs_short_ewma,
        'moved_toward_strike': moved_toward_strike,
        'reason': reason,
    }
