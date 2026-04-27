import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probability_calibration import (
    add_baseline_probabilities,
    add_logistic_calibration_by_tau_bucket,
    build_forecast_summary_table,
    build_reliability_comparison,
    build_tau_bucket_forecast_comparison,
    write_calibration_report,
)


def _make_results() -> pd.DataFrame:
    rows = []
    ts = pd.Timestamp("2026-02-01T00:00:00Z")
    for i in range(80):
        tau = [5, 10, 20, 40][i % 4]
        spot = 100.0 + 0.5 * i
        strike = 100.0 + (0.2 if i % 3 == 0 else -0.2)
        p_yes = min(max(0.05 + (i % 10) * 0.09, 0.01), 0.99)
        realized_yes = int((i % 10) >= 4)
        rows.append(
            {
                "decision_ts": ts + pd.Timedelta(minutes=5 * i),
                "event_hour_start": ts.floor("h"),
                "event_hour_end": ts.floor("h") + pd.Timedelta(hours=1),
                "minute_in_hour": (i * 5) % 60,
                "tau_minutes": tau,
                "strike_price": strike,
                "spot_now": spot,
                "settlement_price": strike + 1.0,
                "p_yes": p_yes,
                "p_no": 1.0 - p_yes,
                "realized_yes": realized_yes,
                "sigma_now": 1.0,
                "nu": 8.0,
                "residual_buffer_len": 100,
                "jump_flag": False,
                "tail_prob": 0.5,
                "training_window_length": 2000,
                "fit_failed": False,
                "simulation_failed": False,
                "skip_reason": None,
                "used_refit": i % 3 == 0,
            }
        )
    return pd.DataFrame(rows)


def test_baselines_are_added_and_bounded() -> None:
    enriched = add_baseline_probabilities(_make_results(), vol_window=12, min_periods=4)
    for column in ["p_yes_constant_0_5", "p_yes_naive_sign", "p_yes_gaussian_vol"]:
        assert column in enriched.columns
        assert enriched[column].between(0.0, 1.0).all()


def test_logistic_calibration_by_tau_bucket_is_bounded() -> None:
    calibrated = add_logistic_calibration_by_tau_bucket(_make_results(), min_bucket_obs=5)
    assert "p_yes_calibrated_logistic_tau" in calibrated.columns
    assert calibrated["p_yes_calibrated_logistic_tau"].between(0.0, 1.0).all()


def test_summary_and_tau_comparisons_include_baselines() -> None:
    summary = build_forecast_summary_table(_make_results())
    tau = build_tau_bucket_forecast_comparison(_make_results())
    reliability = build_reliability_comparison(_make_results())

    assert {"model", "constant_0_5", "naive_sign", "gaussian_vol"} <= set(summary["forecast_name"])
    assert {"model", "constant_0_5", "naive_sign", "gaussian_vol"} <= set(tau["forecast_name"])
    assert {"model", "constant_0_5", "naive_sign", "gaussian_vol"} <= set(reliability["forecast_name"])


def test_write_calibration_report_writes_side_by_side_outputs(tmp_path: Path) -> None:
    report = write_calibration_report(
        _make_results(),
        tmp_path,
        calibration_method="logistic",
        gaussian_vol_window=12,
    )
    assert Path(report["summary_by_forecast_path"]).exists()
    assert Path(report["reliability_comparison_path"]).exists()
    assert Path(report["tau_brier_comparison_path"]).exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "reliability.csv").exists()
    assert (tmp_path / "summary_by_forecast.csv").exists()
