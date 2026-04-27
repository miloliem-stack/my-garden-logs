"""Low-frequency redeemer for resolved markets."""
from datetime import datetime, timedelta, timezone
import time
import logging
from typing import Optional
import json
from pathlib import Path
import os

from . import storage
from . import polymarket_client
from .live_heartbeat import format_console_status_line, get_log_mode, should_print_console_event

logger = logging.getLogger('redeemer')
logging.basicConfig(level=logging.INFO)
REDEEM_RETRY_COOLDOWN_SEC = int(os.getenv('REDEEM_RETRY_COOLDOWN_SEC', '300'))
REDEEM_INTERVAL_MINUTES = int(os.getenv('REDEEM_INTERVAL_MINUTES', '30'))
SETTLEMENT_END_BUFFER_SEC = int(os.getenv('SETTLEMENT_END_BUFFER_SEC', '60'))


def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _market_missing_metadata(market: Optional[dict]) -> bool:
    if not isinstance(market, dict):
        return True
    required = ('condition_id', 'slug', 'title', 'start_time', 'end_time')
    return any(not market.get(field) for field in required)


def _refresh_market_for_redemption(market_id: str, *, checked_ts: str) -> Optional[dict]:
    minfo = storage.get_market(market_id)
    if minfo and _market_missing_metadata(minfo):
        hydrated = storage.hydrate_market_metadata_by_id(market_id)
        if hydrated is not None:
            minfo = hydrated
    minfo = storage.refresh_market_lifecycle(market_id, checked_ts=checked_ts) or storage.get_market(market_id)
    if minfo and str(minfo.get('status') or '').lower() not in {'resolved', 'redeemed'} and _market_missing_metadata(minfo):
        hydrated = storage.hydrate_market_metadata_by_id(market_id)
        if hydrated is not None:
            minfo = storage.refresh_market_lifecycle(market_id, checked_ts=checked_ts) or storage.get_market(market_id)
    if minfo is not None:
        minfo['missing_market_metadata'] = _market_missing_metadata(minfo)
    return minfo


def settle_resolved_market_inventory(market_id: str, *, dry_run: bool = True, checked_ts: Optional[str] = None) -> dict:
    ts = checked_ts or datetime.now(timezone.utc).isoformat()
    minfo = _refresh_market_for_redemption(market_id, checked_ts=ts)
    lifecycle_diagnostics = (minfo or {}).get('resolution_diagnostics')
    result = {
        'market_id': market_id,
        'policy_type': 'settler',
        'action': 'settle_market',
        'timestamp': ts,
        'market': minfo,
        'status': 'skipped',
        'reason': None,
        'response_status': None,
        'winning_outcome': (minfo or {}).get('winning_outcome') if minfo else None,
        'redeemable_qty': 0.0,
        'redeemed_qty': 0.0,
        'path_used': None,
        'tx_hash': None,
        'receipt': None,
        'dry_run': dry_run,
        'skip_reason': None,
        'classification': None,
        'resolution_diagnostics': lifecycle_diagnostics,
        'inventory': {
            'YES': storage.get_total_qty_by_market_and_side(market_id, 'YES'),
            'NO': storage.get_total_qty_by_market_and_side(market_id, 'NO'),
        },
    }
    if not minfo:
        result.update({
            'reason': 'missing_market_metadata',
            'skip_reason': 'missing_market_metadata',
            'classification': 'missing_market_metadata',
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request={'market_id': market_id},
            response=result,
            classification='missing_market_metadata',
            failure_reason='missing_market_metadata',
            ts=ts,
        )
        return result

    market_status = str(minfo.get('status') or '').lower()
    if market_status == 'redeemed':
        result.update({
            'status': 'skipped',
            'reason': 'already_redeemed',
            'skip_reason': 'already_redeemed',
            'classification': 'already_redeemed',
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request={'market_id': market_id, 'market_status': minfo.get('status')},
            response=result,
            classification='already_redeemed',
            failure_reason=None,
            ts=ts,
        )
        return result

    if market_status != 'resolved':
        result.update({
            'reason': 'market_not_resolved',
            'skip_reason': 'market_not_resolved',
            'classification': 'market_not_resolved',
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request={
                'market_id': market_id,
                'market_status': minfo.get('status'),
                'resolution_diagnostics': minfo.get('resolution_diagnostics'),
            },
            response=result,
            classification='market_not_resolved',
            failure_reason='market_not_resolved',
            ts=ts,
        )
        return result

    winning = minfo.get('winning_outcome')
    if not winning:
        result.update({
            'reason': 'missing_winning_outcome',
            'skip_reason': 'missing_winning_outcome',
            'classification': 'missing_winning_outcome',
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request={'market_id': market_id, 'market_status': minfo.get('status')},
            response=result,
            classification='missing_winning_outcome',
            failure_reason='missing_winning_outcome',
            ts=ts,
        )
        return result

    winning_qty = storage.get_total_qty_by_market_and_side(market_id, winning)
    losing_side = 'NO' if winning == 'YES' else 'YES'
    losing_qty = storage.get_total_qty_by_market_and_side(market_id, losing_side)
    result['redeemable_qty'] = winning_qty
    result['inventory'] = {
        'YES': storage.get_total_qty_by_market_and_side(market_id, 'YES'),
        'NO': storage.get_total_qty_by_market_and_side(market_id, 'NO'),
        'winning_side_qty': winning_qty,
        'losing_side_qty': losing_qty,
    }
    request_payload = {
        'market_id': market_id,
        'condition_id': minfo.get('condition_id'),
        'winning_outcome': winning,
        'redeemable_qty': winning_qty,
        'losing_side_qty': losing_qty,
        'resolution_diagnostics': lifecycle_diagnostics,
    }
    if winning_qty <= 0 and losing_qty > 0:
        finalized_qty = storage.redeem_market(market_id, winning, redeem_tx_hash=None, ts=ts)
        result.update({
            'status': 'finalized_loss',
            'reason': 'no_redeemable_qty',
            'skip_reason': 'no_redeemable_qty',
            'classification': 'settled_loss',
            'redeemed_qty': finalized_qty,
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request=request_payload,
            response=result,
            classification='settled_loss',
            failure_reason='no_redeemable_qty',
            ts=ts,
        )
        return result
    if winning_qty <= 0:
        result.update({
            'reason': 'no_redeemable_qty',
            'skip_reason': 'no_redeemable_qty',
            'classification': 'no_redeemable_qty',
        })
        storage.record_inventory_disposal(
            market_id,
            policy_type='settler',
            action='settle_market',
            request=request_payload,
            response=result,
            classification='no_redeemable_qty',
            failure_reason='no_redeemable_qty',
            ts=ts,
        )
        return result

    resp = polymarket_client.normalize_client_response(
        polymarket_client.redeem_market_onchain(
            market_id,
            dry_run=dry_run,
            condition_id=minfo.get('condition_id'),
            redeemable_qty=winning_qty,
            winning_outcome=winning,
        ),
        default_status='unknown',
    )
    status_resp = resp.get('status')
    tx_hash = resp.get('tx_hash')
    receipt = polymarket_client.get_tx_receipt(tx_hash) if tx_hash else None
    result.update({
        'status': 'attempted',
        'response_status': status_resp,
        'tx_hash': tx_hash,
        'path_used': resp.get('path_used'),
        'redeemed_qty': resp.get('redeemed_qty') or 0.0,
        'receipt': receipt,
        'response': resp,
    })

    classification = 'settle_market'
    failure_reason = resp.get('error_message') or resp.get('reason') or resp.get('error_reason')
    if dry_run:
        result['status'] = 'dry_run'
        result['reason'] = 'dry_run'
        result['classification'] = 'dry_run'
        classification = 'dry_run'
    elif resp.get('ok') and tx_hash and status_resp in ('ok', 'success'):
        redeemed_qty = storage.redeem_market(market_id, winning, redeem_tx_hash=tx_hash, ts=ts)
        result.update({
            'status': 'redeemed',
            'reason': None,
            'redeemed_qty': redeemed_qty,
            'classification': 'settle_market',
        })
        failure_reason = None
    else:
        result.update({
            'status': 'failed' if status_resp not in ('skipped', 'dry_run') else 'skipped',
            'reason': failure_reason or 'status-not-ok-or-missing-tx',
            'classification': 'settle_market_failed',
        })
        classification = 'settle_market_failed'

    storage.record_inventory_disposal(
        market_id,
        policy_type='settler',
        action='settle_market',
        request=request_payload,
        response=result,
        tx_hash=tx_hash,
        receipt=receipt,
        classification=classification,
        failure_reason=failure_reason,
        ts=ts,
    )
    return result


def _inventory_candidate_markets(*, checked_ts: str, end_buffer_sec: int = SETTLEMENT_END_BUFFER_SEC) -> list[str]:
    now_dt = _parse_iso_ts(checked_ts) or datetime.now(timezone.utc)
    cutoff_dt = now_dt - timedelta(seconds=max(0, int(end_buffer_sec)))
    conn = storage.sqlite3.connect(storage.get_db_path())
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT DISTINCT o.market_id, m.end_time
        FROM open_lots o
        LEFT JOIN markets m ON m.market_id = o.market_id
        WHERE o.qty != 0
        ORDER BY o.market_id ASC
        '''
    )
    rows = cur.fetchall()
    conn.close()
    candidates = []
    for market_id, end_time in rows:
        end_dt = _parse_iso_ts(end_time)
        if end_dt is None or end_dt <= cutoff_dt:
            candidates.append(str(market_id))
    return candidates


def settle_inventory_candidates(*, dry_run: bool = True, checked_ts: Optional[str] = None, end_buffer_sec: int = SETTLEMENT_END_BUFFER_SEC) -> list[dict]:
    ts = checked_ts or datetime.now(timezone.utc).isoformat()
    results = []
    for market_id in _inventory_candidate_markets(checked_ts=ts, end_buffer_sec=end_buffer_sec):
        results.append(settle_resolved_market_inventory(market_id, dry_run=dry_run, checked_ts=ts))
    return results


class Redeemer:
    def __init__(self, interval_minutes: int = 30, dry_run: bool = True):
        storage.ensure_db()
        self.interval = max(1, interval_minutes)
        self.dry_run = dry_run
        self.retry_counts = {}
        self.next_retry_after = {}
        self.metrics_path = Path.cwd() / 'redeemer_metrics.jsonl'

    def _append_metrics(self, metrics: dict):
        try:
            with open(self.metrics_path, 'a') as fh:
                fh.write(json.dumps(metrics) + '\n')
        except Exception:
            logger.exception('Failed to write metrics file')

    def _cooldown_active(self, market_id: str) -> bool:
        return time.time() < float(self.next_retry_after.get(market_id, 0))

    def _print_console(self, tone: str, **fields):
        mode = get_log_mode()
        critical = tone in {'error', 'warning'} or any(fields.get(key) is not None for key in ('reason', 'error_message', 'skip_reason', 'response_status'))
        if not should_print_console_event(mode=mode, tone=tone, critical=critical):
            return
        print(format_console_status_line('redeem', mode=mode, tone=tone, **fields))

    def markets_with_inventory(self):
        ts = datetime.now(timezone.utc).isoformat()
        return _inventory_candidate_markets(checked_ts=ts)

    def redeem_once(self):
        markets = self.markets_with_inventory()
        logger.info('Found markets with inventory: %s', markets)
        for m in markets:
            try:
                ts = datetime.now(timezone.utc).isoformat()
                minfo = _refresh_market_for_redemption(m, checked_ts=ts)
                winning = (minfo or {}).get('winning_outcome') if minfo else None
                redeemable_qty = storage.get_total_qty_by_market_and_side(m, winning) if winning else 0.0
                metrics = {
                    'market_id': m,
                    'condition_id': (minfo or {}).get('condition_id') if minfo else None,
                    'market_status': (minfo or {}).get('status') if minfo else None,
                    'winning_outcome': (minfo or {}).get('winning_outcome') if minfo else None,
                    'redeemable_qty': redeemable_qty or 0.0,
                    'redeemed_qty': 0.0,
                    'dry_run': self.dry_run,
                    'timestamp': ts,
                    'attempt': int(self.retry_counts.get(m, 0)),
                    'success': None,
                    'reason': None,
                    'skip_reason': None,
                    'error_reason': None,
                    'tx_hash': None,
                    'response_status': None,
                    'action': 'skipped',
                    'path_used': None,
                    'resolution_diagnostics': (minfo or {}).get('resolution_diagnostics') if minfo else None,
                }
                if not minfo:
                    metrics.update({'reason': 'missing_market_metadata', 'skip_reason': 'missing_market_metadata'})
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue
                market_status = str(minfo.get('status') or '').lower()
                if market_status == 'redeemed':
                    metrics.update({'reason': 'already_redeemed', 'skip_reason': 'already_redeemed'})
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue
                if market_status != 'resolved':
                    storage.record_inventory_disposal(
                        m,
                        policy_type='settler',
                        action='settle_market',
                        request={
                            'market_id': m,
                            'market_status': minfo.get('status'),
                            'resolution_diagnostics': minfo.get('resolution_diagnostics'),
                        },
                        response={'market_id': m, 'status': 'skipped', 'reason': 'market_not_resolved'},
                        classification='market_not_resolved',
                        failure_reason='market_not_resolved',
                        ts=ts,
                    )
                    metrics.update({'reason': 'market_not_resolved', 'skip_reason': 'market_not_resolved'})
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue
                if not winning:
                    storage.record_inventory_disposal(
                        m,
                        policy_type='settler',
                        action='settle_market',
                        request={'market_id': m, 'market_status': minfo.get('status')},
                        response={'market_id': m, 'status': 'skipped', 'reason': 'missing_winning_outcome'},
                        classification='missing_winning_outcome',
                        failure_reason='missing_winning_outcome',
                        ts=ts,
                    )
                    metrics.update({'reason': 'missing_winning_outcome', 'skip_reason': 'missing_winning_outcome'})
                    logger.warning('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue
                if redeemable_qty <= 0:
                    storage.record_inventory_disposal(
                        m,
                        policy_type='settler',
                        action='settle_market',
                        request={'market_id': m, 'winning_outcome': winning, 'redeemable_qty': redeemable_qty},
                        response={'market_id': m, 'status': 'skipped', 'reason': 'no_redeemable_qty'},
                        classification='no_redeemable_qty',
                        failure_reason='no_redeemable_qty',
                        ts=ts,
                    )
                    metrics.update({'reason': 'no_redeemable_qty', 'skip_reason': 'no_redeemable_qty'})
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue
                if not self.dry_run and self._cooldown_active(m):
                    metrics.update({'success': False, 'reason': 'cooldown_active', 'skip_reason': 'cooldown_active'})
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue

                result = settle_resolved_market_inventory(m, dry_run=self.dry_run, checked_ts=ts)
                metrics.update({
                    'redeemable_qty': result.get('redeemable_qty') or redeemable_qty,
                    'redeemed_qty': result.get('redeemed_qty') or 0.0,
                    'reason': result.get('reason'),
                    'skip_reason': result.get('skip_reason'),
                    'tx_hash': result.get('tx_hash'),
                    'response_status': result.get('response_status'),
                    'path_used': result.get('path_used'),
                })
                if self.dry_run:
                    metrics.update({
                        'reason': 'dry_run',
                        'action': 'skipped',
                        'skip_reason': 'dry_run',
                    })
                    self._print_console('warning', market_id=m, qty=result.get('redeemable_qty'), status='dry_run', path=result.get('path_used'))
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    continue

                self.retry_counts[m] = int(self.retry_counts.get(m, 0)) + 1
                metrics['attempt'] = int(self.retry_counts[m])
                self._print_console('info', market_id=m, qty=result.get('redeemable_qty'), status='submit', path=result.get('path_used'))

                if result.get('status') in {'redeemed', 'finalized_loss'}:
                    metrics.update({'success': True, 'action': result.get('status')})
                    self._print_console('success', market_id=m, qty=result.get('redeemable_qty'), tx_hash=result.get('tx_hash'), status=result.get('response_status') or result.get('status'), path=result.get('path_used'), reason=result.get('reason'))
                    logger.info('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    self.next_retry_after.pop(m, None)
                else:
                    metrics.update({
                        'success': False,
                        'action': 'failed' if result.get('status') not in ('skipped', 'dry_run') else 'skipped',
                        'reason': result.get('reason') or 'status-not-ok-or-missing-tx',
                    })
                    self._print_console(
                        'error' if metrics['action'] == 'failed' else 'warning',
                        market_id=m,
                        qty=result.get('redeemable_qty'),
                        tx_hash=result.get('tx_hash'),
                        status=result.get('response_status'),
                        reason=metrics['reason'],
                        path=result.get('path_used'),
                    )
                    logger.error('redeemer.metrics %s', metrics)
                    self._append_metrics(metrics)
                    if metrics['action'] == 'failed':
                        self.next_retry_after[m] = time.time() + REDEEM_RETRY_COOLDOWN_SEC
            except Exception as e:
                logger.exception('Error processing market %s: %s', m, e)

    def loop(self):
        logger.info('Starting redeemer loop, interval %d minutes, dry_run=%s', self.interval, self.dry_run)
        try:
            while True:
                self.redeem_once()
                time.sleep(self.interval * 60)
        except KeyboardInterrupt:
            logger.info('Redeemer loop stopped')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int, default=REDEEM_INTERVAL_MINUTES, help='Interval minutes between sweeps')
    p.add_argument('--once', action='store_true', help='Run only one sweep and exit')
    p.add_argument('--live', action='store_true', help='Run live (not dry-run)')
    args = p.parse_args()

    r = Redeemer(interval_minutes=args.interval, dry_run=not args.live)
    if args.once:
        r.redeem_once()
    else:
        r.loop()
