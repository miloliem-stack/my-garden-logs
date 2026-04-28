import os
import json
from pathlib import Path
from typing import Dict


_SCHEDULES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'policy_schedules.json'

# Map schedule keys to the policy dict keys they override
_SCHEDULE_KEY_MAP = {
    'POLICY_FAR_EDGE_THRESHOLD_YES': ('far', 'edge_threshold_yes'),
    'POLICY_FAR_EDGE_THRESHOLD_NO': ('far', 'edge_threshold_no'),
    'POLICY_MID_EDGE_THRESHOLD_YES': ('mid', 'edge_threshold_yes'),
    'POLICY_MID_EDGE_THRESHOLD_NO': ('mid', 'edge_threshold_no'),
    'POLICY_LATE_EDGE_THRESHOLD_YES': ('late', 'edge_threshold_yes'),
    'POLICY_LATE_EDGE_THRESHOLD_NO': ('late', 'edge_threshold_no'),
    'POLICY_FINAL_EDGE_THRESHOLD_YES': ('final', 'edge_threshold_yes'),
    'POLICY_FINAL_EDGE_THRESHOLD_NO': ('final', 'edge_threshold_no'),
    'POLICY_FAR_KELLY_MULTIPLIER': ('far', 'kelly_multiplier'),
    'POLICY_MID_KELLY_MULTIPLIER': ('mid', 'kelly_multiplier'),
    'POLICY_LATE_KELLY_MULTIPLIER': ('late', 'kelly_multiplier'),
    'POLICY_FINAL_KELLY_MULTIPLIER': ('final', 'kelly_multiplier'),
    'POLICY_FAR_MAX_TRADE_NOTIONAL_MULTIPLIER': ('far', 'max_trade_notional_multiplier'),
    'POLICY_MID_MAX_TRADE_NOTIONAL_MULTIPLIER': ('mid', 'max_trade_notional_multiplier'),
    'POLICY_LATE_MAX_TRADE_NOTIONAL_MULTIPLIER': ('late', 'max_trade_notional_multiplier'),
    'POLICY_FINAL_MAX_TRADE_NOTIONAL_MULTIPLIER': ('final', 'max_trade_notional_multiplier'),
    'POLICY_FINAL_ALLOW_NEW_ENTRIES': ('final', 'allow_new_entries'),
    'POLICY_LATE_ALLOW_NEW_ENTRIES': ('late', 'allow_new_entries'),
}


def _fenv(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _ienv(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def build_time_policy(decision_state: dict) -> Dict:
    tau_minutes = decision_state.get('tau_minutes')
    if tau_minutes is None:
        return {
            'tau_minutes': None,
            'edge_threshold_yes': _fenv('POLICY_DEFAULT_EDGE_THRESHOLD_YES', 0.03),
            'edge_threshold_no': _fenv('POLICY_DEFAULT_EDGE_THRESHOLD_NO', 0.03),
            'kelly_multiplier': _fenv('POLICY_DEFAULT_KELLY_MULTIPLIER', 1.0),
            'max_trade_notional_multiplier': _fenv('POLICY_DEFAULT_MAX_TRADE_NOTIONAL_MULTIPLIER', 1.0),
            'allow_new_entries': False,
            'quote_max_age_sec': _fenv('POLICY_DEFAULT_QUOTE_MAX_AGE_SEC', 5),
            'quote_max_spread': _fenv('POLICY_DEFAULT_QUOTE_MAX_SPREAD', 0.10),
            'cancel_open_orders_after_sec': _ienv('POLICY_DEFAULT_CANCEL_OPEN_AFTER_SEC', 300),
            'policy_bucket': 'unknown',
            'reason': 'missing_tau_minutes',
        }

    tau_minutes = int(tau_minutes)
    if tau_minutes >= _ienv('POLICY_FAR_MIN_TAU', 30):
        return {
            'tau_minutes': tau_minutes,
            'edge_threshold_yes': _fenv('POLICY_FAR_EDGE_THRESHOLD_YES', 0.04),
            'edge_threshold_no': _fenv('POLICY_FAR_EDGE_THRESHOLD_NO', 0.04),
            'kelly_multiplier': _fenv('POLICY_FAR_KELLY_MULTIPLIER', 1.0),
            'max_trade_notional_multiplier': _fenv('POLICY_FAR_MAX_TRADE_NOTIONAL_MULTIPLIER', 1.0),
            'allow_new_entries': True,
            'quote_max_age_sec': _fenv('POLICY_FAR_QUOTE_MAX_AGE_SEC', 5),
            'quote_max_spread': _fenv('POLICY_FAR_QUOTE_MAX_SPREAD', 0.10),
            'cancel_open_orders_after_sec': _ienv('POLICY_FAR_CANCEL_OPEN_AFTER_SEC', 300),
            'policy_bucket': 'far',
            'reason': 'ok',
        }
    if tau_minutes >= _ienv('POLICY_MID_MIN_TAU', 15):
        return {
            'tau_minutes': tau_minutes,
            'edge_threshold_yes': _fenv('POLICY_MID_EDGE_THRESHOLD_YES', 0.03),
            'edge_threshold_no': _fenv('POLICY_MID_EDGE_THRESHOLD_NO', 0.03),
            'kelly_multiplier': _fenv('POLICY_MID_KELLY_MULTIPLIER', 0.8),
            'max_trade_notional_multiplier': _fenv('POLICY_MID_MAX_TRADE_NOTIONAL_MULTIPLIER', 0.8),
            'allow_new_entries': True,
            'quote_max_age_sec': _fenv('POLICY_MID_QUOTE_MAX_AGE_SEC', 4),
            'quote_max_spread': _fenv('POLICY_MID_QUOTE_MAX_SPREAD', 0.08),
            'cancel_open_orders_after_sec': _ienv('POLICY_MID_CANCEL_OPEN_AFTER_SEC', 180),
            'policy_bucket': 'mid',
            'reason': 'ok',
        }
    if tau_minutes >= _ienv('POLICY_LATE_MIN_TAU', 5):
        return {
            'tau_minutes': tau_minutes,
            'edge_threshold_yes': _fenv('POLICY_LATE_EDGE_THRESHOLD_YES', 0.02),
            'edge_threshold_no': _fenv('POLICY_LATE_EDGE_THRESHOLD_NO', 0.02),
            'kelly_multiplier': _fenv('POLICY_LATE_KELLY_MULTIPLIER', 0.5),
            'max_trade_notional_multiplier': _fenv('POLICY_LATE_MAX_TRADE_NOTIONAL_MULTIPLIER', 0.5),
            'allow_new_entries': str(os.getenv('POLICY_LATE_ALLOW_NEW_ENTRIES', 'true')).lower() in ('1', 'true', 'yes', 'on'),
            'quote_max_age_sec': _fenv('POLICY_LATE_QUOTE_MAX_AGE_SEC', 3),
            'quote_max_spread': _fenv('POLICY_LATE_QUOTE_MAX_SPREAD', 0.05),
            'cancel_open_orders_after_sec': _ienv('POLICY_LATE_CANCEL_OPEN_AFTER_SEC', 90),
            'policy_bucket': 'late',
            'reason': 'ok',
        }
    return {
        'tau_minutes': tau_minutes,
        'edge_threshold_yes': _fenv('POLICY_FINAL_EDGE_THRESHOLD_YES', 0.05),
        'edge_threshold_no': _fenv('POLICY_FINAL_EDGE_THRESHOLD_NO', 0.05),
        'kelly_multiplier': _fenv('POLICY_FINAL_KELLY_MULTIPLIER', 0.0),
        'max_trade_notional_multiplier': _fenv('POLICY_FINAL_MAX_TRADE_NOTIONAL_MULTIPLIER', 0.0),
        'allow_new_entries': str(os.getenv('POLICY_FINAL_ALLOW_NEW_ENTRIES', 'false')).lower() in ('1', 'true', 'yes', 'on'),
        'quote_max_age_sec': _fenv('POLICY_FINAL_QUOTE_MAX_AGE_SEC', 2),
        'quote_max_spread': _fenv('POLICY_FINAL_QUOTE_MAX_SPREAD', 0.03),
        'cancel_open_orders_after_sec': _ienv('POLICY_FINAL_CANCEL_OPEN_AFTER_SEC', 45),
        'policy_bucket': 'final',
        'reason': 'late_expiry_restriction',
    }


def policy_schedule_mode() -> str:
    """Return the policy schedule mode: 'off', 'shadow', or 'live'."""
    raw = os.getenv('POLICY_SCHEDULE_MODE')
    if raw is None or not str(raw).strip():
        return 'off'
    value = str(raw).strip().lower()
    if value not in ('off', 'shadow', 'live'):
        return 'off'
    return value


def policy_schedule_name() -> str:
    """Return the selected policy schedule name."""
    return str(os.getenv('POLICY_SCHEDULE_NAME', '')).strip()


def _load_schedules(path: Path = _SCHEDULES_PATH) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _coerce_schedule_value(key: str, value):
    """Coerce a schedule value to the correct type for the policy field."""
    if 'allow_' in key.lower():
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_named_schedule(name: str, path: Path = _SCHEDULES_PATH) -> Dict:
    """Load a named schedule from the policy_schedules.json file.
    Returns a dict mapping policy field names to values for the selected schedule,
    or an empty dict if not found.
    """
    schedules = _load_schedules(path)
    schedule = schedules.get(name)
    if not isinstance(schedule, dict):
        return {}
    return dict(schedule)


def apply_schedule_overlay(policy: Dict, *, mode: str = 'off', schedule_name: str = '') -> Dict:
    """Apply a named schedule overlay to the policy dict.

    mode='off'    → return policy unchanged
    mode='shadow' → attach schedule_shadow_overrides showing what would change
    mode='live'   → merge matching schedule values into the policy
    """
    if mode == 'off' or not schedule_name:
        policy['policy_schedule_mode'] = mode
        policy['policy_schedule_name'] = schedule_name
        policy['policy_schedule_applied'] = False
        return policy

    schedule = load_named_schedule(schedule_name)
    if not schedule:
        policy['policy_schedule_mode'] = mode
        policy['policy_schedule_name'] = schedule_name
        policy['policy_schedule_applied'] = False
        policy['policy_schedule_error'] = 'schedule_not_found'
        return policy

    bucket = str(policy.get('policy_bucket') or 'unknown')
    overrides = {}
    for schedule_key, (target_bucket, field_name) in _SCHEDULE_KEY_MAP.items():
        if target_bucket != bucket:
            continue
        if schedule_key not in schedule:
            continue
        new_value = _coerce_schedule_value(schedule_key, schedule[schedule_key])
        old_value = policy.get(field_name)
        if old_value != new_value:
            overrides[field_name] = {'old': old_value, 'new': new_value}

    policy['policy_schedule_mode'] = mode
    policy['policy_schedule_name'] = schedule_name
    policy['policy_schedule_applied'] = mode == 'live' and bool(overrides)

    if mode == 'shadow':
        policy['policy_schedule_shadow_overrides'] = overrides
    elif mode == 'live' and overrides:
        for field_name, change in overrides.items():
            policy[field_name] = change['new']
        policy['policy_schedule_live_overrides'] = overrides

    return policy
