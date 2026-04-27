import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probability_challengers import build_challenger_config_grid, run_challenger_sweep, summarize_config_results


def _make_events() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(8):
        decision_ts = start + pd.Timedelta(minutes=15 * i)
        rows.append(
            {
                "decision_ts": decision_ts,
                "event_hour_start": decision_ts.floor("h"),
                "event_hour_end": decision_ts.floor("h") + pd.Timedelta(hours=1),
                "minute_in_hour": (15 * i) % 60,
                "tau_minutes": [15, 30, 45, 60][i % 4],
                "strike_price": 100.0,
                "spot_now": 100.0 + (i - 3) * 0.25,
                "settlement_price": 100.0 + (1 if i % 2 == 0 else -1),
                "realized_yes": int(i % 2 == 0),
            }
        )
    return pd.DataFrame(rows)


def _fake_backtest(**kwargs) -> pd.DataFrame:
    events = kwargs["events"].copy()
    model_kwargs = kwargs.get("model_kwargs", {})
    use_jump_augmentation = kwargs.get("use_jump_augmentation", False)
    refit_every = kwargs.get("refit_every_n_events", 1)
    signal = 0.72 if model_kwargs.get("ar_lags", 1) == 1 else 0.55
    if model_kwargs.get("egarch_o", 0) == 1:
        signal += 0.05
    if use_jump_augmentation:
        signal += 0.03
    events["p_yes"] = [signal if y == 1 else 1.0 - signal for y in events["realized_yes"]]
    events["p_no"] = 1.0 - events["p_yes"]
    events["sigma_now"] = 1.0
    events["nu"] = 8.0
    events["residual_buffer_len"] = 100
    events["jump_flag"] = False
    events["tail_prob"] = 0.5
    events["training_window_length"] = kwargs.get("fit_window", 1000)
    events["fit_failed"] = False
    events["simulation_failed"] = False
    events["skip_reason"] = None
    events["used_refit"] = [i % refit_every == 0 for i in range(len(events))]
    return events


def test_challenger_grid_size_and_keys() -> None:
    configs = build_challenger_config_grid()
    assert len(configs) == 216
    assert {"config_id", "ar_on", "egarch_asymmetry", "jump_augmentation", "fit_window", "residual_buffer_size", "refit_every_n_events", "model_kwargs"} <= set(configs[0])


def test_challenger_grid_can_build_requested_coarse_24_config_subset() -> None:
    configs = build_challenger_config_grid(
        ar_values=[False, True],
        asymmetry_values=[False, True],
        jump_values=[False],
        fit_windows=[1000, 2000, 4000],
        residual_buffer_sizes=[1000, 2000],
        refit_every_values=[4],
    )
    assert len(configs) == 24
    assert {config["jump_augmentation"] for config in configs} == {False}
    assert {config["refit_every_n_events"] for config in configs} == {4}
    assert {config["fit_window"] for config in configs} == {1000, 2000, 4000}
    assert {config["residual_buffer_size"] for config in configs} == {1000, 2000}


def test_summarize_config_results_compares_against_gaussian_vol() -> None:
    config = build_challenger_config_grid()[0]
    results = _fake_backtest(events=_make_events(), model_kwargs=config["model_kwargs"], use_jump_augmentation=config["jump_augmentation"])
    summary, tau_table, moneyness_table, _ = summarize_config_results(results, config)
    assert "gaussian_vol_brier" in summary
    assert "brier_delta_vs_gaussian_vol" in summary
    assert not tau_table.empty
    assert not moneyness_table.empty


def test_run_challenger_sweep_writes_rankings(tmp_path: Path) -> None:
    events = _make_events()
    configs = build_challenger_config_grid()[:3]
    report = run_challenger_sweep(
        close_series=pd.Series([100.0, 100.1], index=pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="min", tz="UTC")),
        events=events,
        output_dir=tmp_path,
        configs=configs,
        n_sims=50,
        run_backtest_fn=_fake_backtest,
    )
    assert Path(report["summary_path"]).exists()
    assert Path(report["ranking_by_brier_path"]).exists()
    assert Path(report["ranking_by_log_loss_path"]).exists()
    assert Path(report["tau_bucket_path"]).exists()
    assert Path(report["moneyness_bucket_path"]).exists()
