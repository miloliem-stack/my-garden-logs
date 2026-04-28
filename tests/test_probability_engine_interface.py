import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probability_backtest import run_probability_backtest
from src import storage
from src.probability_engine_factory import build_probability_engine
from src.regime_detector import compute_microstructure_regime
from src.run_bot import (
    apply_completed_kline_to_engine,
    apply_completed_kline_to_engines,
    build_decision_log_record,
    build_diagnostic_heartbeat,
    build_trade_context,
    build_shadow_probability_engines,
    compute_effective_decision_trade_state,
    build_market_decision_state,
    compute_market_probabilities,
    evaluate_shadow_probability_models,
    extract_engine_shadow_diagnostics,
    parse_shadow_probability_engine_names,
)
from src.strategy_manager import build_trade_action


def _make_close_series(n: int = 180) -> pd.Series:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="min", tz="UTC")
    prices = pd.Series([100.0 + 0.02 * i for i in range(n)], index=index)
    return prices


def setup_function(_fn) -> None:
    try:
        storage.get_db_path().unlink()
    except Exception:
        pass
    storage.ensure_db()


def _make_events(close_series: pd.Series) -> pd.DataFrame:
    rows = []
    base = close_series.index[120]
    for offset in [0, 15]:
        decision_ts = base + pd.Timedelta(minutes=offset)
        rows.append(
            {
                "decision_ts": decision_ts,
                "event_hour_start": decision_ts.floor("h"),
                "event_hour_end": decision_ts.floor("h") + pd.Timedelta(hours=1),
                "minute_in_hour": decision_ts.minute,
                "tau_minutes": 60 - decision_ts.minute,
                "strike_price": float(close_series.loc[decision_ts.floor("h")]),
                "spot_now": float(close_series.loc[decision_ts]),
                "settlement_price": float(close_series.iloc[min(len(close_series) - 1, 179)]),
                "realized_yes": 1,
            }
        )
    return pd.DataFrame(rows)


def test_compute_market_probabilities_works_with_both_engines() -> None:
    close_series = _make_close_series()
    now = close_series.index[-1]
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "strike_price": float(close_series.iloc[-10]),
        "end_time": (now + pd.Timedelta(minutes=20)).isoformat(),
    }
    for engine_name in ["ar_egarch", "gaussian_vol"]:
        engine = build_probability_engine(engine_name, fit_window=120, residual_buffer_size=120)
        engine.fit_history(close_series.iloc[:-1])
        engine.observe_bar(float(close_series.iloc[-1]), ts=now, finalized=False)
        state = compute_market_probabilities(bundle, engine, now=now, n_sims=50, seed=7)
        assert state["blocked"] is False
        assert 0.0 <= state["p_yes"] <= 1.0
        assert 0.0 <= state["p_no"] <= 1.0


def test_factory_registers_shadow_probability_engines() -> None:
    close_series = _make_close_series()
    now = close_series.index[-1]
    for engine_name in ["kalman_blended_sigma_v1_cfg1", "gaussian_pde_diffusion_kalman_v1_cfg1"]:
        engine = build_probability_engine(engine_name, fit_window=120)
        engine.fit_history(close_series.iloc[:-1])
        engine.observe_bar(float(close_series.iloc[-1]), ts=now, finalized=False)
        assert engine.engine_name == engine_name


def _make_shadow_ctx(bundle: dict, now, wallet_state: dict | None = None) -> dict:
    return build_trade_context(
        {
            "market_id": bundle["market_id"],
            "token_yes": bundle["token_yes"],
            "token_no": bundle["token_no"],
            "status": "open",
            "startDate": pd.to_datetime(bundle["start_time"], utc=True),
            "endDate": pd.to_datetime(bundle["end_time"], utc=True),
        },
        {"mid": bundle["yes_quote"]["mid"], "age_seconds": 1.0, "fetch_failed": False, "is_empty": False, "is_crossed": False, "spread": 0.02},
        {"mid": bundle["no_quote"]["mid"], "age_seconds": 1.0, "fetch_failed": False, "is_empty": False, "is_crossed": False, "spread": 0.02},
        now=now,
        routing_bundle=bundle,
        wallet_state=wallet_state or {},
    )


def test_shadow_probability_models_do_not_change_live_decision(monkeypatch) -> None:
    close_series = _make_close_series(240)
    now = close_series.index[-1]
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "token_yes": "YES1",
        "token_no": "NO1",
        "yes_quote": {"mid": 0.52},
        "no_quote": {"mid": 0.48},
        "strike_price": float(close_series.iloc[-10]),
        "start_time": (now - pd.Timedelta(minutes=40)).isoformat(),
        "end_time": (now + pd.Timedelta(minutes=20)).isoformat(),
    }
    wallet_state = {"effective_bankroll": 10.0, "free_usdc": 10.0}
    engine = build_probability_engine("gaussian_vol", fit_window=120, residual_buffer_size=120)
    engine.fit_history(close_series.iloc[:-1])
    engine.observe_bar(float(close_series.iloc[-1]), ts=now, finalized=False)
    probability_state = compute_market_probabilities(bundle, engine, now=now, n_sims=50, seed=7)
    live_decision = build_market_decision_state(bundle, probability_state, wallet_state=wallet_state)
    baseline_action = live_decision["action"]
    baseline_trade_allowed = live_decision["trade_allowed"]
    ctx = _make_shadow_ctx(bundle, now, wallet_state)
    ctx["probability_state"] = probability_state
    ctx["decision_state"] = live_decision
    ctx["policy"] = live_decision["policy"]
    live_allowed, _ = compute_effective_decision_trade_state(ctx, live_decision)

    monkeypatch.setenv("SHADOW_KALMAN_BLENDED_SIGMA_V1_CFG1", "true")
    monkeypatch.setenv("SHADOW_GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "true")
    shadow_engines = build_shadow_probability_engines(
        primary_engine_name="gaussian_vol",
        fit_prices=close_series.iloc[:-1],
        engine_kwargs={"fit_window": 120},
    )
    shadow_models = evaluate_shadow_probability_models(
        shadow_engines=shadow_engines,
        bundle=bundle,
        now=now,
        n_sims=50,
        wallet_state=wallet_state,
        trade_context=ctx,
        entry_context=None,
        microstructure_state=None,
        live_decision_state=live_decision,
        live_trade_allowed=live_allowed,
    )

    assert live_decision["action"] == baseline_action
    assert live_decision["trade_allowed"] == baseline_trade_allowed
    assert set(shadow_models) == {
        "kalman_blended_sigma_v1_cfg1",
        "gaussian_pde_diffusion_kalman_v1_cfg1",
    }


def test_shadow_probability_engine_names_parse_from_env_flags(monkeypatch) -> None:
    monkeypatch.delenv("SHADOW_PROBABILITY_ENGINES", raising=False)
    for env_name in (
        "SHADOW_AR_EGARCH",
        "SHADOW_GAUSSIAN_VOL",
        "SHADOW_KALMAN_BLENDED_SIGMA_V1_CFG1",
        "SHADOW_GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1",
        "SHADOW_LGBM",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("SHADOW_KALMAN_BLENDED_SIGMA_V1_CFG1", "true")
    monkeypatch.setenv("SHADOW_GAUSSIAN_PDE_DIFFUSION_KALMAN_V1_CFG1", "1")

    assert parse_shadow_probability_engine_names() == [
        "kalman_blended_sigma_v1_cfg1",
        "gaussian_pde_diffusion_kalman_v1_cfg1",
    ]


def test_shadow_probability_model_failure_is_isolated() -> None:
    now = pd.Timestamp("2026-01-01T00:10:00Z")
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "token_yes": "YES1",
        "token_no": "NO1",
        "yes_quote": {"mid": 0.42},
        "no_quote": {"mid": 0.58},
        "strike_price": 100.0,
        "start_time": (now - pd.Timedelta(minutes=10)).isoformat(),
        "end_time": (now + pd.Timedelta(minutes=20)).isoformat(),
    }
    probability_state = {
        "timestamp": now.isoformat(),
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "spot_now": 101.0,
        "strike_price": 100.0,
        "tau_minutes": 20,
        "p_yes": 0.60,
        "p_no": 0.40,
        "blocked": False,
        "reason": None,
    }
    live_decision = build_market_decision_state(bundle, probability_state)
    ctx = _make_shadow_ctx(bundle, now)
    ctx["probability_state"] = probability_state
    ctx["decision_state"] = live_decision
    ctx["policy"] = live_decision["policy"]
    live_allowed, _ = compute_effective_decision_trade_state(ctx, live_decision)

    class RaisingEngine:
        def current_spot(self):
            return 101.0

        def predict(self, strike_price: float, tau_minutes: int, n_sims=None, seed=None):
            raise RuntimeError("boom")

    shadow_models = evaluate_shadow_probability_models(
        shadow_engines={"kalman_blended_sigma_v1_cfg1": RaisingEngine()},
        bundle=bundle,
        now=now,
        n_sims=50,
        wallet_state={},
        trade_context=ctx,
        entry_context=None,
        microstructure_state=None,
        live_decision_state=live_decision,
        live_trade_allowed=live_allowed,
    )

    assert live_decision["action"] == "buy_yes"
    assert shadow_models["kalman_blended_sigma_v1_cfg1"]["error"] == "boom"
    assert shadow_models["kalman_blended_sigma_v1_cfg1"]["trade_allowed"] is False


def test_apply_completed_kline_to_engines_ignores_shadow_failures() -> None:
    payload = {
        "k": {
            "x": True,
            "c": "101.25",
            "T": 1735689600000,
        }
    }

    class RecordingEngine:
        def __init__(self) -> None:
            self.calls = []

        def observe_bar(self, price: float, ts=None, finalized: bool = False) -> None:
            self.calls.append((price, ts, finalized))

    class RaisingEngine:
        def observe_bar(self, price: float, ts=None, finalized: bool = False) -> None:
            raise RuntimeError("shadow failed")

    healthy = RecordingEngine()

    observed = apply_completed_kline_to_engines([RaisingEngine(), healthy], payload)

    assert observed is True
    assert len(healthy.calls) == 1
    assert healthy.calls[0][0] == 101.25
    assert healthy.calls[0][2] is True


def test_apply_completed_kline_to_engine_preserves_live_fail_fast_behavior() -> None:
    payload = {
        "k": {
            "x": True,
            "c": "101.25",
            "T": 1735689600000,
        }
    }

    class RaisingEngine:
        def observe_bar(self, price: float, ts=None, finalized: bool = False) -> None:
            raise RuntimeError("live failed")

    try:
        apply_completed_kline_to_engine(RaisingEngine(), payload)
    except RuntimeError as exc:
        assert str(exc) == "live failed"
    else:
        raise AssertionError("live engine updates should still fail fast")


def test_decision_state_and_heartbeat_include_microstructure_fields_without_changing_action() -> None:
    close_series = _make_close_series(180)
    now = close_series.index[-1]
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "token_yes": "YES1",
        "token_no": "NO1",
        "yes_quote": {"mid": 0.52},
        "no_quote": {"mid": 0.48},
        "strike_price": float(close_series.iloc[-10]),
        "start_time": (now - pd.Timedelta(minutes=40)).isoformat(),
        "end_time": (now + pd.Timedelta(minutes=20)).isoformat(),
    }
    engine = build_probability_engine("gaussian_vol", fit_window=120, residual_buffer_size=120)
    engine.fit_history(close_series.iloc[:-1])
    engine.observe_bar(float(close_series.iloc[-1]), ts=now, finalized=False)
    probability_state = compute_market_probabilities(bundle, engine, now=now, n_sims=50, seed=7)
    microstructure_state = compute_microstructure_regime(close_series.iloc[-40:])

    baseline = build_market_decision_state(bundle, probability_state)
    decision_state = build_market_decision_state(bundle, probability_state, microstructure_state=microstructure_state)

    assert decision_state["action"] == baseline["action"]
    assert decision_state["trade_allowed"] == baseline["trade_allowed"]
    assert decision_state["reason"] == baseline["reason"]
    assert decision_state["microstructure_regime"] == microstructure_state["microstructure_regime"]
    assert decision_state["smoothness_score"] == microstructure_state["smoothness_score"]
    ctx = {
        "market": {"market_id": "M1", "token_yes": "YES1", "token_no": "NO1", "status": "open"},
        "stored_market": {"status": "open"},
        "quotes": {"yes": {"mid": 0.52}, "no": {"mid": 0.48}},
        "position_summary": {},
        "routing": {"series_id": "BTC-HOURLY", "active_market_id": "M1"},
        "routing_debug": {},
        "policy": decision_state["policy"],
        "wallet_state": {},
        "decision_state": decision_state,
        "probability_state": probability_state,
        "market_window": {"start": now - pd.Timedelta(minutes=40), "end": now + pd.Timedelta(minutes=20)},
    }
    heartbeat = build_diagnostic_heartbeat(ctx, True, "ok")

    assert heartbeat["microstructure_regime"] == microstructure_state["microstructure_regime"]
    assert heartbeat["smoothness_score"] == microstructure_state["smoothness_score"]


def test_probability_backtest_works_with_both_engines() -> None:
    close_series = _make_close_series()
    events = _make_events(close_series)
    for engine_name in ["ar_egarch", "gaussian_vol"]:
        results = run_probability_backtest(
            close_series=close_series,
            events=events,
            fit_window=120,
            residual_buffer_size=120,
            n_sims=20,
            min_history=60,
            refit_every_n_events=1,
            engine_name=engine_name,
        )
        valid = results[results["skip_reason"].isna()]
        assert not valid.empty
        assert valid["p_yes"].between(0.0, 1.0).all()
        assert valid["diagnostics_json"].notna().all()
        for raw in valid["diagnostics_json"]:
            assert isinstance(json.loads(raw), dict)


def test_live_update_path_uses_engine_observe_bar() -> None:
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def observe_bar(self, price: float, ts=None, finalized: bool = True):
            self.calls.append({"price": price, "ts": ts, "finalized": finalized})

    engine = FakeEngine()
    closed_payload = {"k": {"x": True, "c": "43210.5", "T": 1760000000000}}
    open_payload = {"k": {"x": False, "c": "43211.0", "T": 1760000060000}}

    assert apply_completed_kline_to_engine(engine, closed_payload) is True
    assert apply_completed_kline_to_engine(engine, open_payload) is False
    assert len(engine.calls) == 1
    assert engine.calls[0]["price"] == 43210.5
    assert engine.calls[0]["finalized"] is True


def test_fixed_probability_state_keeps_decision_and_trade_contract(monkeypatch) -> None:
    bundle = {
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "token_yes": "YES1",
        "token_no": "NO1",
        "yes_quote": {"mid": 0.42},
        "no_quote": {"mid": 0.58},
    }
    probability_state = {
        "timestamp": pd.Timestamp("2026-01-01T00:10:00Z").isoformat(),
        "series_id": "BTC-HOURLY",
        "market_id": "M1",
        "spot_now": 101.0,
        "strike_price": 100.0,
        "tau_minutes": 20,
        "p_yes": 0.60,
        "p_no": 0.40,
        "blocked": False,
        "reason": None,
    }
    decision_state = build_market_decision_state(bundle, probability_state)
    assert decision_state["action"] == "buy_yes"

    monkeypatch.setattr("src.strategy_manager.get_market", lambda market_id: {"status": "open"})
    monkeypatch.setattr("src.strategy_manager.get_inflight_exposure", lambda: 0.0)
    monkeypatch.setattr(
        "src.strategy_manager.place_marketable_buy",
        lambda token_id, qty, limit_price, dry_run, market_id, outcome_side, decision_context=None: {
            "token_id": token_id,
            "qty": qty,
            "limit_price": limit_price,
            "market_id": market_id,
            "outcome_side": outcome_side,
            "dry_run": dry_run,
            "decision_context": decision_context,
        },
    )

    trade = build_trade_action(decision_state, "YES1", "NO1", "M1", dry_run=True)
    assert trade["side"] == "buy_yes"


def test_engine_shadow_diagnostics_are_generic_and_heartbeat_safe() -> None:
    gaussian_diag = {
        "engine_name": "gaussian_vol",
        "engine_version": "1.0",
        "history_len": 120,
        "sigma_per_sqrt_min": 0.001,
        "fallback_sigma_used": False,
        "vol_window": 120,
        "min_periods": 60,
        "calibration_mode": "none",
        "last_prediction": {
            "horizon_sigma": 0.004,
            "z_score": -0.5,
            "raw_p_yes": 0.69,
            "calibrated_p_yes": 0.69,
        },
    }
    extracted = extract_engine_shadow_diagnostics(gaussian_diag)
    assert extracted["engine_name"] == "gaussian_vol"
    assert extracted["horizon_sigma"] == 0.004
    assert extracted["raw_p_yes"] == 0.69

    non_gaussian = extract_engine_shadow_diagnostics({"engine_name": "ar_egarch"})
    assert non_gaussian["engine_name"] == "ar_egarch"
    assert non_gaussian["raw_p_yes"] is None
    assert non_gaussian["horizon_sigma"] is None

    heartbeat = build_diagnostic_heartbeat(
        {
            "market": {"market_id": "M1", "token_yes": "YES1", "token_no": "NO1", "status": "open"},
            "stored_market": {"status": "open"},
            "wallet_state": {
                "wallet_address": "0xabc",
                "usdc_e_balance": 12.5,
                "pol_balance": 1.25,
                "reserved_exposure_usdc": 2.0,
                "free_usdc": 10.5,
                "effective_bankroll": 10.5,
                "bankroll_source": "wallet_live",
                "fetch_failed": False,
            },
            "quotes": {"yes": {"mid": 0.52, "age_seconds": 1.0}, "no": {"mid": 0.48, "age_seconds": 1.0}},
            "position_summary": {"inflight_exposure": 0.0, "redeemable_qty": 0.0},
            "routing": {"series_id": "BTC-HOURLY", "active_market_id": "M1", "active_token_yes": "YES1", "active_token_no": "NO1"},
            "decision_state": {"q_yes": 0.52, "q_no": 0.48, "edge_yes": 0.03, "edge_no": -0.03, "action": "buy_yes"},
            "probability_state": {"tau_minutes": 20, "p_yes": 0.55, "p_no": 0.45},
            "engine_diagnostics": gaussian_diag,
        },
        True,
        "ok",
    )
    assert heartbeat["engine_name"] == "gaussian_vol"
    assert heartbeat["engine_version"] == "1.0"
    assert heartbeat["history_len"] == 120
    assert heartbeat["sigma_per_sqrt_min"] == 0.001
    assert heartbeat["horizon_sigma"] == 0.004
    assert heartbeat["z_score"] == -0.5
    assert heartbeat["raw_p_yes"] == 0.69
    assert heartbeat["calibrated_p_yes"] == 0.69
    assert heartbeat["wallet_address"] == "0xabc"
    assert heartbeat["wallet_usdc_e"] == 12.5
    assert heartbeat["wallet_pol"] == 1.25
    assert heartbeat["wallet_reserved_exposure"] == 2.0
    assert heartbeat["wallet_free_usdc"] == 10.5
    assert heartbeat["effective_bankroll"] == 10.5
    assert heartbeat["bankroll_source"] == "wallet_live"
    assert heartbeat["wallet_fetch_failed"] is False
