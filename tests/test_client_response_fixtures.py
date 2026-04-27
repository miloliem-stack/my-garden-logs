import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import polymarket_client, storage, strategy_manager
from scripts import probe_order_status_shape, probe_cancel_shape


FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures' / 'client_responses'


def setup_function(fn):
    try:
        os.remove(storage.get_db_path())
    except Exception:
        pass


def _load_fixture(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


def test_fixture_driven_status_normalization_for_canonical_shapes():
    for name in [
        'status_accepted.json',
        'status_pending_submit.json',
        'status_open.json',
        'status_partial.json',
        'status_filled.json',
        'status_cancel_requested.json',
        'status_canceled.json',
        'status_expired.json',
        'status_rejected.json',
        'status_error.json',
    ]:
        fixture = _load_fixture(name)
        normalized = polymarket_client.normalize_client_response(fixture['raw'], default_status='unknown')
        for key, value in fixture['expected'].items():
            assert normalized.get(key) == value, f'{name} expected {key}={value}, got {normalized.get(key)}'


def test_fixture_driven_cancel_normalization():
    for name in ['cancel_ok.json', 'cancel_error.json']:
        fixture = _load_fixture(name)
        normalized = polymarket_client.normalize_client_response(fixture['raw'], default_status='unknown')
        for key, value in fixture['expected'].items():
            assert normalized.get(key) == value


def test_malformed_payload_remains_non_terminal():
    normalized = polymarket_client.normalize_client_response('not-a-dict', default_status='unknown')
    assert normalized['status'] == 'invalid_response'
    assert normalized['ok'] is False
    assert normalized['tx_hash'] is None


def test_ambiguous_payload_remains_unknown():
    fixture = _load_fixture('status_ambiguous.json')
    normalized = polymarket_client.normalize_client_response(fixture['raw'], default_status='unknown')
    assert normalized['status'] == 'unknown'
    assert normalized['ok'] is False


def test_pending_submit_cancel_requested_and_rejected_remain_conservative():
    for name in ['status_pending_submit.json', 'status_cancel_requested.json', 'status_rejected.json']:
        fixture = _load_fixture(name)
        normalized = polymarket_client.normalize_client_response(fixture['raw'], default_status='unknown')
        assert normalized['ok'] is False


def test_order_status_payload_normalizes_matched_fields_from_polymarket_shape():
    normalized = polymarket_client.normalize_client_response(
        {
            'status': 'MATCHED',
            'orderID': 'OID-MATCHED-1',
            'transactionsHashes': ['0x' + '3' * 64],
            'size_matched': '2.5',
            'original_size': '2.5',
            'side': 'BUY',
        },
        default_status='unknown',
    )

    assert normalized['status'] == 'matched'
    assert normalized['order_id'] == 'OID-MATCHED-1'
    assert normalized['tx_hash'] == '0x' + '3' * 64
    assert normalized['filled_qty'] == '2.5'
    assert normalized['qty'] == '2.5'


def test_probe_scripts_do_not_mutate_storage(monkeypatch, capsys, tmp_path):
    db_path = tmp_path / 'probe.db'
    monkeypatch.setenv('BOT_DB_PATH', str(db_path))
    monkeypatch.setattr(polymarket_client, 'get_order_status', lambda **kwargs: {'status': 'open', 'orderId': 'OID', 'filledQuantity': 0.0})
    monkeypatch.setattr(sys, 'argv', ['probe_order_status_shape.py', '--order-id', 'OID'])
    probe_order_status_shape.main()
    out = capsys.readouterr().out
    assert '"storage_mutated": false' in out.lower()

    monkeypatch.setattr(polymarket_client, 'cancel_order', lambda **kwargs: {'status': 'canceled', 'orderId': 'OID'})
    monkeypatch.setattr(sys, 'argv', ['probe_cancel_shape.py', '--order-id', 'OID'])
    probe_cancel_shape.main()
    out = capsys.readouterr().out
    assert '"storage_mutated": false' in out.lower()


def test_repo_has_no_legacy_midpoint_fallback_path():
    assert not hasattr(strategy_manager, 'get_market_mid')





def test_venue_assumption_report_tracks_fixture_coverage():
    report = polymarket_client.describe_venue_assumptions()
    assert 'fixture_coverage' in report
    assert report['fixture_coverage']['covered_states']['open'] is True
    assert report['fixture_coverage']['covered_states']['filled'] is True
