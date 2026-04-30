import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from . import execution, run_bot, storage, strategy_manager

DEFAULT_SCHEDULES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'policy_schedules.json'


class ReplayModel:
    def __init__(self):
        self.last_price = None
        self._p_yes = None

    def set_step(self, spot_now: float, p_yes: Optional[float]):
        self.last_price = float(spot_now)
        self._p_yes = None if p_yes is None else float(p_yes)

    def simulate_probability(self, target_price, tau_minutes, n_sims=2000, seed=None):
        if self._p_yes is None:
            raise RuntimeError('scenario step missing model_p_yes/precomputed_probability')
        return {'p_hat': self._p_yes, 'n_sims': n_sims, 'target_price': float(target_price), 'seed': seed}


@contextmanager
def policy_overrides_env(overrides: Optional[Dict[str, object]] = None):
    overrides = overrides or {}
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_scenarios(path: Path) -> List[Dict]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict) and 'scenarios' in payload:
        return list(payload['scenarios'])
    if isinstance(payload, list):
        return list(payload)
    return [payload]


def filter_scenarios(scenarios: Iterable[Dict], split: Optional[str] = None) -> List[Dict]:
    scenarios = list(scenarios)
    if split is None:
        return scenarios
    return [scenario for scenario in scenarios if scenario.get('split') == split]


def load_named_schedules(path: Optional[Path] = None) -> Dict[str, Dict]:
    target = path or DEFAULT_SCHEDULES_PATH
    return json.loads(Path(target).read_text(encoding='utf-8'))


def _bundle_from_step(scenario: Dict, step: Dict) -> Dict:
    bundle = dict(scenario.get('bundle') or {})
    bundle.update(step.get('bundle_override') or {})
    quotes = step.get('quotes') or {}
    if 'yes' in quotes:
        bundle['yes_quote'] = quotes['yes']
    if 'no' in quotes:
        bundle['no_quote'] = quotes['no']
    return bundle


def _seed_inventory(items: Iterable[Dict], ts: str):
    for item in items or []:
        storage.create_open_lot(
            item['market_id'],
            item['token_id'],
            item['outcome_side'],
            float(item['qty']),
            float(item.get('avg_price', 0.5)),
            ts,
            tx_hash=item.get('tx_hash'),
        )


def _deterministic_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def apply_execution_realism(step_submit: Dict, decision_state: Dict, step: Dict, qty: float, *, enabled: bool = False, seed: int = 0) -> Dict:
    payload = dict(step_submit)
    if not enabled:
        return payload
    policy = decision_state.get('policy') or {}
    tau_minutes = float(decision_state.get('tau_minutes') or 0.0)
    spread = max(float((step.get('quotes') or {}).get('yes', {}).get('spread') or 0.0), float((step.get('quotes') or {}).get('no', {}).get('spread') or 0.0))
    chosen_action = decision_state.get('action')
    edge = decision_state.get('edge_yes') if chosen_action == 'buy_yes' else decision_state.get('edge_no')
    edge = float(edge or 0.0)
    friction = float(step.get('execution_realism', {}).get('friction', 0.0))
    fill_ratio = max(0.0, min(1.0, 0.85 + edge * 2.0 - spread * 4.0 - friction - (0.15 if tau_minutes < 5 else 0.0)))
    key = f"{step.get('ts')}:{chosen_action}:{seed}:{policy.get('policy_bucket')}"
    partial_cut = float(step.get('execution_realism', {}).get('partial_cut', 0.2))
    open_cut = float(step.get('execution_realism', {}).get('open_cut', 0.05))
    jitter = _deterministic_fraction(key) * 0.05
    fill_ratio = max(0.0, min(1.0, fill_ratio - jitter))
    if fill_ratio <= open_cut:
        payload['status'] = 'open'
        payload['filledQuantity'] = 0.0
    elif fill_ratio < 1.0 - partial_cut:
        payload['status'] = 'partially_filled'
        payload['filledQuantity'] = round(qty * fill_ratio, 6)
        payload['remaining_qty'] = max(0.0, qty - payload['filledQuantity'])
    else:
        payload['status'] = 'filled'
        payload['filledQuantity'] = qty
    if tau_minutes < 5 and payload.get('status') == 'open':
        payload.setdefault('cancel_success_probability', 1.0)
    return payload


def _position_mark_value(snapshot: List[Dict], last_quotes: Dict[str, float]) -> float:
    value = 0.0
    for item in snapshot:
        for side, qty in (item.get('available_inventory') or {}).items():
            if qty <= 0:
                continue
            token_id = None
            if side == 'YES':
                token_id = item.get('token_yes') or (item.get('market_id') + ':YES')
            elif side == 'NO':
                token_id = item.get('token_no') or (item.get('market_id') + ':NO')
            if token_id and token_id in last_quotes:
                value += qty * last_quotes[token_id]
    return value


def _db_counts() -> Dict:
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    counts = {}
    cur.execute('SELECT COUNT(*) FROM bot_orders')
    counts['trades_attempted'] = int(cur.fetchone()[0])
    cur.execute('SELECT COUNT(*) FROM bot_order_fills')
    counts['fills'] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM bot_orders WHERE status = 'partially_filled'")
    counts['partial_fills'] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM bot_orders WHERE status = 'canceled'")
    counts['cancels'] = int(cur.fetchone()[0])
    conn.close()
    return counts


def summarize_replay(decision_traces: List[Dict], max_exposure: float, last_quotes: Dict[str, float], score_weights: Optional[Dict[str, float]] = None) -> Dict:
    counts = _db_counts()
    snapshot = storage.get_position_snapshot()
    blocked_reasons = {}
    entry_edges = []
    late_stale_orders = 0
    for trace in decision_traces:
        reason = trace.get('reason')
        if trace.get('trade_allowed') is False and reason:
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        if trace.get('action') in ('buy_yes', 'buy_no'):
            edge_value = trace.get('edge_yes') if trace.get('action') == 'buy_yes' else trace.get('edge_no')
            if edge_value is not None:
                entry_edges.append(edge_value)
        late_stale_orders += int(trace.get('stale_actions_count', 0))
    marked_value = _position_mark_value(snapshot, last_quotes)
    open_cost = 0.0
    conn = sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute('SELECT COALESCE(SUM(qty * avg_price), 0) FROM open_lots')
    open_cost = float(cur.fetchone()[0] or 0.0)
    conn.close()
    pnl_proxy = marked_value - open_cost
    final_bucket_trades = sum(1 for trace in decision_traces if trace.get('policy_bucket') == 'final' and trace.get('action') in ('buy_yes', 'buy_no'))
    residual_inventory_count = sum(1 for item in snapshot if item.get('resolved_redeemable_qty', 0) > 0)
    blocked_total = sum(blocked_reasons.values())
    score_components = {
        'pnl_proxy': pnl_proxy * (score_weights or {}).get('pnl_proxy', 1.0),
        'average_edge_at_entry': (sum(entry_edges) / len(entry_edges) if entry_edges else 0.0) * (score_weights or {}).get('average_edge_at_entry', 0.25),
        'exposure_penalty': -max_exposure * (score_weights or {}).get('max_exposure', 0.1),
        'stale_order_penalty': -late_stale_orders * (score_weights or {}).get('late_stale_orders', 0.25),
        'unresolved_inventory_penalty': -residual_inventory_count * (score_weights or {}).get('residual_inventory', 0.2),
        'final_bucket_trade_penalty': -final_bucket_trades * (score_weights or {}).get('final_bucket_trades', 0.5),
        'blocked_opportunity_penalty': -blocked_total * (score_weights or {}).get('blocked_opportunities', 0.05),
    }
    total_score = sum(score_components.values())
    return {
        **counts,
        'average_edge_at_entry': sum(entry_edges) / len(entry_edges) if entry_edges else 0.0,
        'max_exposure': max_exposure,
        'terminal_inventory_state': snapshot,
        'pnl_proxy': pnl_proxy,
        'blocked_opportunities_by_reason': blocked_reasons,
        'late_stale_orders': late_stale_orders,
        'score_components': score_components,
        'score': total_score,
    }


def score_summary(summary: Dict, weights: Optional[Dict[str, float]] = None) -> float:
    if 'score_components' in summary:
        return sum(summary['score_components'].values())
    weights = weights or {}
    return (
        summary.get('pnl_proxy', 0.0) * weights.get('pnl_proxy', 1.0)
        + summary.get('average_edge_at_entry', 0.0) * weights.get('average_edge_at_entry', 0.25)
        - summary.get('max_exposure', 0.0) * weights.get('max_exposure', 0.1)
        - summary.get('late_stale_orders', 0) * weights.get('late_stale_orders', 0.25)
        - sum(summary.get('blocked_opportunities_by_reason', {}).values()) * weights.get('blocked_opportunities', 0.05)
    )


def _apply_post_action_updates(step: Dict, order_id: Optional[int], now_ts: str, thresholds: Optional[Dict] = None) -> List[Dict]:
    actions = []
    if not order_id:
        return actions
    for update in step.get('order_updates') or []:
        actions.append({'type': 'order_update', 'result': execution.process_order_update(order_id, update)})
    maintenance = step.get('maintenance') or {}
    if maintenance:
        age_sec = int(maintenance.get('age_sec', 0))
        if age_sec > 0:
            aged_ts = (pd.to_datetime(now_ts, utc=True) - pd.Timedelta(seconds=age_sec)).isoformat()
            storage.update_order(order_id, updated_ts=aged_ts)
        original_get = execution.get_order_status
        original_cancel = execution.cancel_order
        try:
            execution.get_order_status = lambda order_id=None, client_order_id=None, dry_run=False: dict(maintenance.get('refresh_response') or {'status': 'open'})
            execution.cancel_order = lambda order_id=None, client_order_id=None, dry_run=False: dict(maintenance.get('cancel_response') or {'status': 'canceled'})
            actions.extend(execution.manage_stale_orders(now_ts=now_ts, dry_run=False, thresholds=thresholds))
        finally:
            execution.get_order_status = original_get
            execution.cancel_order = original_cancel
    return actions


def run_replay_scenario(scenario: Dict, *, db_path: Path, output_dir: Optional[Path] = None, policy_overrides: Optional[Dict[str, object]] = None, seed: int = 0, execution_realism: Optional[Dict[str, object]] = None, score_weights: Optional[Dict[str, float]] = None) -> Dict:
    previous_db_path = os.environ.get('BOT_DB_PATH')
    os.environ['BOT_DB_PATH'] = str(db_path)
    try:
        if db_path.exists():
            db_path.unlink()
        storage.ensure_db()
        model = ReplayModel()
        decision_traces = []
        last_quotes = {}
        max_exposure = 0.0
        original_place = strategy_manager.place_marketable_buy
        try:
            with policy_overrides_env(policy_overrides):
                initial_ts = scenario.get('steps', [{}])[0].get('ts') if scenario.get('steps') else pd.Timestamp.now(tz='UTC').isoformat()
                _seed_inventory(scenario.get('seed_inventory') or [], initial_ts)
                for step in scenario.get('steps') or []:
                    bundle = _bundle_from_step(scenario, step)
                    market_id = bundle.get('market_id')
                    storage.upsert_market(
                        market_id=market_id,
                        condition_id=bundle.get('condition_id'),
                        slug=bundle.get('slug'),
                        title=bundle.get('title'),
                        start_time=bundle.get('start_time'),
                        end_time=bundle.get('end_time'),
                        status=bundle.get('status', 'open'),
                    )
                    now = pd.to_datetime(step['ts'], utc=True)
                    _seed_inventory(step.get('seed_inventory') or [], now.isoformat())
                    spot_now = float(step['spot_now'])
                    model.set_step(spot_now, step.get('model_p_yes') or (step.get('precomputed_probability') or {}).get('p_yes'))
                    probability_state = run_bot.compute_market_probabilities(bundle, model, now=now, n_sims=int(step.get('n_sims', 200)), seed=seed)
                    decision_state = run_bot.build_market_decision_state(bundle, probability_state)
                    quote_yes = bundle.get('yes_quote') or {}
                    quote_no = bundle.get('no_quote') or {}
                    ctx = run_bot.build_trade_context(
                        {
                            'market_id': market_id,
                            'token_yes': bundle.get('token_yes'),
                            'token_no': bundle.get('token_no'),
                            'status': bundle.get('status', 'open'),
                            'startDate': pd.to_datetime(bundle.get('start_time'), utc=True),
                            'endDate': pd.to_datetime(bundle.get('end_time'), utc=True),
                        },
                        quote_yes,
                        quote_no,
                        now=now,
                        routing_bundle=bundle,
                    )
                    ctx['probability_state'] = probability_state
                    ctx['decision_state'] = decision_state
                    ctx['policy'] = decision_state.get('policy') or {}
                    thresholds = {
                        'max_open_age_sec': int(ctx['policy'].get('cancel_open_orders_after_sec', 300)),
                        'max_pending_submit_age_sec': max(15, int(ctx['policy'].get('cancel_open_orders_after_sec', 300)) // 3),
                        'cancel_retry_sec': max(10, int(ctx['policy'].get('cancel_open_orders_after_sec', 300)) // 4),
                    }
                    trade_allowed, reason = run_bot.can_trade_context(ctx)
                    if trade_allowed and not decision_state.get('trade_allowed'):
                        trade_allowed = False
                        reason = decision_state.get('reason')

                    step_submit = dict((step.get('order_outcomes') or {}).get('submit') or {'status': 'filled', 'filledQuantity': 1.0})

                    def _place_order(token_id, qty, limit_price=None, dry_run=True, market_id=None, outcome_side='YES', client_order_id=None, **kwargs):
                        original = execution.place_marketable_order
                        try:
                            payload = apply_execution_realism(
                                step_submit,
                                decision_state,
                                step,
                                qty,
                                enabled=bool(execution_realism and execution_realism.get('enabled')),
                                seed=seed,
                            )
                            if 'filledQuantity' not in payload and 'filled_qty' not in payload and payload.get('status') == 'filled':
                                payload['filledQuantity'] = qty
                            execution.place_marketable_order = lambda *_args, **_kwargs: payload
                            return execution.place_marketable_buy(
                                token_id,
                                qty,
                                limit_price=limit_price,
                                dry_run=False,
                                market_id=market_id,
                                outcome_side=outcome_side,
                                client_order_id=client_order_id,
                            )
                        finally:
                            execution.place_marketable_order = original

                    strategy_manager.place_marketable_buy = _place_order
                    action = None
                    stale_actions = []
                    if trade_allowed:
                        action = strategy_manager.build_trade_action(
                            decision_state,
                            bundle.get('token_yes'),
                            bundle.get('token_no'),
                            market_id,
                            dry_run=False,
                        )
                        order_id = (((action or {}).get('resp') or {}).get('order_id'))
                        stale_actions = _apply_post_action_updates(step, order_id, now.isoformat(), thresholds=thresholds)

                    max_exposure = max(max_exposure, storage.get_inflight_exposure())
                    if bundle.get('token_yes') and (quote_yes or {}).get('mid') is not None:
                        last_quotes[bundle['token_yes']] = float((quote_yes or {}).get('mid'))
                    if bundle.get('token_no') and (quote_no or {}).get('mid') is not None:
                        last_quotes[bundle['token_no']] = float((quote_no or {}).get('mid'))
                    decision_traces.append({
                        'scenario': scenario.get('name'),
                        'ts': step['ts'],
                        'market_id': market_id,
                        'trade_allowed': trade_allowed,
                        'action': decision_state.get('action') if trade_allowed else None,
                        'reason': reason if not trade_allowed else decision_state.get('reason'),
                        'policy_bucket': (decision_state.get('policy') or {}).get('policy_bucket'),
                        'edge_yes': decision_state.get('edge_yes'),
                        'edge_no': decision_state.get('edge_no'),
                        'stale_actions_count': len(stale_actions),
                        'split': scenario.get('split'),
                    })
        finally:
            strategy_manager.place_marketable_buy = original_place
        summary = summarize_replay(decision_traces, max_exposure, last_quotes, score_weights=score_weights)
        result = {
            'scenario': scenario.get('name'),
            'split': scenario.get('split'),
            'decision_traces': decision_traces,
            'summary': summary,
            'score': score_summary(summary),
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f'{scenario.get("name", "scenario")}_trace.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        if previous_db_path is None:
            os.environ.pop('BOT_DB_PATH', None)
        else:
            os.environ['BOT_DB_PATH'] = previous_db_path


def run_scenario_library(scenarios: Iterable[Dict], *, output_dir: Path, policy_overrides: Optional[Dict[str, object]] = None, seed: int = 0, split: Optional[str] = None, execution_realism: Optional[Dict[str, object]] = None, score_weights: Optional[Dict[str, float]] = None) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    selected = filter_scenarios(scenarios, split=split)
    for idx, scenario in enumerate(selected):
        db_path = output_dir / f'replay_{idx}.db'
        runs.append(
            run_replay_scenario(
                scenario,
                db_path=db_path,
                output_dir=output_dir / 'traces',
                policy_overrides=policy_overrides,
                seed=seed,
                execution_realism=execution_realism,
                score_weights=score_weights,
            )
        )
    aggregate = {
        'split': split,
        'scenario_count': len(runs),
        'score': sum(item['score'] for item in runs),
        'summary': {
            'trades_attempted': sum(item['summary']['trades_attempted'] for item in runs),
            'fills': sum(item['summary']['fills'] for item in runs),
            'partial_fills': sum(item['summary']['partial_fills'] for item in runs),
            'cancels': sum(item['summary']['cancels'] for item in runs),
            'average_edge_at_entry': sum(item['summary']['average_edge_at_entry'] for item in runs) / len(runs) if runs else 0.0,
            'max_exposure': max((item['summary']['max_exposure'] for item in runs), default=0.0),
            'pnl_proxy': sum(item['summary']['pnl_proxy'] for item in runs),
            'blocked_opportunities_by_reason': {},
            'score_components': {},
        },
        'runs': runs,
    }
    blocked = aggregate['summary']['blocked_opportunities_by_reason']
    for item in runs:
        for reason, count in item['summary']['blocked_opportunities_by_reason'].items():
            blocked[reason] = blocked.get(reason, 0) + count
        for key, value in item['summary'].get('score_components', {}).items():
            aggregate['summary']['score_components'][key] = aggregate['summary']['score_components'].get(key, 0.0) + value
    return aggregate
