"""Structured challenger-model sweep for the offline probability engine."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .probability_backtest import run_probability_backtest
from .probability_calibration import add_baseline_probabilities


def build_challenger_config_grid(
    ar_values: list[bool] | None = None,
    asymmetry_values: list[bool] | None = None,
    jump_values: list[bool] | None = None,
    fit_windows: list[int] | None = None,
    residual_buffer_sizes: list[int] | None = None,
    refit_every_values: list[int] | None = None,
) -> list[dict]:
    ar_values = [False, True] if ar_values is None else ar_values
    asymmetry_values = [False, True] if asymmetry_values is None else asymmetry_values
    jump_values = [False, True] if jump_values is None else jump_values
    fit_windows = [1000, 2000, 4000] if fit_windows is None else fit_windows
    residual_buffer_sizes = [500, 1000, 2000] if residual_buffer_sizes is None else residual_buffer_sizes
    refit_every_values = [1, 2, 4] if refit_every_values is None else refit_every_values
    configs = []
    for ar_on, egarch_asymmetry, jump_augmentation, fit_window, residual_buffer_size, refit_every_n_events in product(
        ar_values,
        asymmetry_values,
        jump_values,
        fit_windows,
        residual_buffer_sizes,
        refit_every_values,
    ):
        config_id = (
            f"ar{int(ar_on)}_"
            f"egarch_o{1 if egarch_asymmetry else 0}_"
            f"jump{int(jump_augmentation)}_"
            f"fw{fit_window}_"
            f"rb{residual_buffer_size}_"
            f"refit{refit_every_n_events}"
        )
        configs.append(
            {
                "engine_name": "ar_egarch",
                "config_id": config_id,
                "ar_on": ar_on,
                "egarch_asymmetry": egarch_asymmetry,
                "jump_augmentation": jump_augmentation,
                "fit_window": fit_window,
                "residual_buffer_size": residual_buffer_size,
                "refit_every_n_events": refit_every_n_events,
                "engine_kwargs": {},
                "model_kwargs": {"ar_lags": 1 if ar_on else 0, "egarch_o": 1 if egarch_asymmetry else 0},
            }
        )
    return configs


def build_engine_family_comparison_configs(
    fit_window: int = 2000,
    residual_buffer_size: int = 2000,
    refit_every_n_events: int = 4,
) -> list[dict]:
    return [
        {
            "engine_name": "ar_egarch",
            "config_id": "family_ar_egarch_default",
            "ar_on": True,
            "egarch_asymmetry": False,
            "jump_augmentation": False,
            "fit_window": fit_window,
            "residual_buffer_size": residual_buffer_size,
            "refit_every_n_events": refit_every_n_events,
            "model_kwargs": {"ar_lags": 1, "egarch_o": 0},
            "engine_kwargs": {},
        },
        {
            "engine_name": "gaussian_vol",
            "config_id": "family_gaussian_vol",
            "ar_on": None,
            "egarch_asymmetry": None,
            "jump_augmentation": False,
            "fit_window": fit_window,
            "residual_buffer_size": residual_buffer_size,
            "refit_every_n_events": refit_every_n_events,
            "model_kwargs": {},
            "engine_kwargs": {"vol_window": fit_window},
        },
    ]


def _compute_metrics(y: pd.Series, p: pd.Series) -> dict:
    p = p.astype(float).clip(1e-12, 1 - 1e-12)
    y = y.astype(int)
    return {
        "observations": int(len(y)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy_at_0_5": float(np.mean(((p >= 0.5).astype(int)) == y)),
    }


def add_moneyness_buckets(results: pd.DataFrame, near_strike_log_threshold: float = 0.001) -> pd.DataFrame:
    out = results.copy()
    out["log_moneyness"] = np.log(out["spot_now"].astype(float) / out["strike_price"].astype(float))
    threshold = float(near_strike_log_threshold)

    def bucket(value: float) -> str:
        if value <= -0.005:
            return "deep_below"
        if value < -threshold:
            return "below"
        if abs(value) <= threshold:
            return "near_strike"
        if value < 0.005:
            return "above"
        return "deep_above"

    out["moneyness_bucket"] = out["log_moneyness"].apply(bucket)
    out["near_strike"] = out["log_moneyness"].abs() <= threshold
    return out


def summarize_config_results(
    results: pd.DataFrame,
    config: dict,
    gaussian_vol_window: int = 288,
    near_strike_log_threshold: float = 0.001,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    valid = add_baseline_probabilities(results, vol_window=gaussian_vol_window)
    valid = add_moneyness_buckets(valid, near_strike_log_threshold=near_strike_log_threshold)

    model_metrics = _compute_metrics(valid["realized_yes"], valid["p_yes"])
    benchmark_metrics = _compute_metrics(valid["realized_yes"], valid["p_yes_gaussian_vol"])

    tau_rows = []
    for tau_bucket, group in valid.groupby("tau_bucket", observed=False):
        tau_rows.append(
            {
                "config_id": config["config_id"],
                "tau_bucket": tau_bucket,
                "segment": "all",
                **{f"model_{k}": v for k, v in _compute_metrics(group["realized_yes"], group["p_yes"]).items()},
                **{f"gaussian_vol_{k}": v for k, v in _compute_metrics(group["realized_yes"], group["p_yes_gaussian_vol"]).items()},
            }
        )
    tau_table = pd.DataFrame(tau_rows)

    money_rows = []
    for bucket, group in valid.groupby("moneyness_bucket", observed=False):
        money_rows.append(
            {
                "config_id": config["config_id"],
                "moneyness_bucket": bucket,
                **{f"model_{k}": v for k, v in _compute_metrics(group["realized_yes"], group["p_yes"]).items()},
                **{f"gaussian_vol_{k}": v for k, v in _compute_metrics(group["realized_yes"], group["p_yes_gaussian_vol"]).items()},
            }
        )
    moneyness_table = pd.DataFrame(money_rows)

    near_strike = valid[valid["near_strike"]].copy()
    near_strike_metrics = {
        "config_id": config["config_id"],
        "near_strike_observations": int(len(near_strike)),
        "near_strike_model_brier": np.nan,
        "near_strike_model_log_loss": np.nan,
        "near_strike_gaussian_vol_brier": np.nan,
        "near_strike_gaussian_vol_log_loss": np.nan,
    }
    if not near_strike.empty:
        near_model = _compute_metrics(near_strike["realized_yes"], near_strike["p_yes"])
        near_benchmark = _compute_metrics(near_strike["realized_yes"], near_strike["p_yes_gaussian_vol"])
        near_strike_metrics.update(
            {
                "near_strike_model_brier": near_model["brier_score"],
                "near_strike_model_log_loss": near_model["log_loss"],
                "near_strike_gaussian_vol_brier": near_benchmark["brier_score"],
                "near_strike_gaussian_vol_log_loss": near_benchmark["log_loss"],
            }
        )

    summary = {
        "config_id": config["config_id"],
        "ar_on": config["ar_on"],
        "egarch_asymmetry": config["egarch_asymmetry"],
        "jump_augmentation": config["jump_augmentation"],
        "fit_window": config["fit_window"],
        "residual_buffer_size": config["residual_buffer_size"],
        "refit_every_n_events": config["refit_every_n_events"],
        "observations": model_metrics["observations"],
        "model_brier": model_metrics["brier_score"],
        "gaussian_vol_brier": benchmark_metrics["brier_score"],
        "brier_delta_vs_gaussian_vol": model_metrics["brier_score"] - benchmark_metrics["brier_score"],
        "model_log_loss": model_metrics["log_loss"],
        "gaussian_vol_log_loss": benchmark_metrics["log_loss"],
        "log_loss_delta_vs_gaussian_vol": model_metrics["log_loss"] - benchmark_metrics["log_loss"],
        "model_accuracy": model_metrics["accuracy_at_0_5"],
        "gaussian_vol_accuracy": benchmark_metrics["accuracy_at_0_5"],
        "beats_gaussian_vol_brier": bool(model_metrics["brier_score"] < benchmark_metrics["brier_score"]),
        "beats_gaussian_vol_log_loss": bool(model_metrics["log_loss"] < benchmark_metrics["log_loss"]),
        "fit_failed_rows": int(results["fit_failed"].sum()) if "fit_failed" in results.columns else 0,
        "simulation_failed_rows": int(results["simulation_failed"].sum()) if "simulation_failed" in results.columns else 0,
        "skipped_rows": int(results["skip_reason"].notna().sum()) if "skip_reason" in results.columns else 0,
        **near_strike_metrics,
    }
    return summary, tau_table, moneyness_table, {"results": valid}


def run_challenger_sweep(
    close_series: pd.Series,
    events: pd.DataFrame,
    output_dir: str | Path,
    configs: list[dict] | None = None,
    n_sims: int = 500,
    min_history: int | None = None,
    gaussian_vol_window: int = 288,
    near_strike_log_threshold: float = 0.001,
    run_backtest_fn: Callable[..., pd.DataFrame] | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configs = configs or build_challenger_config_grid()
    run_backtest_fn = run_backtest_fn or run_probability_backtest

    summaries = []
    tau_tables = []
    moneyness_tables = []
    winners = []

    for config in configs:
        results = run_backtest_fn(
            close_series=close_series,
            events=events,
            fit_window=config["fit_window"],
            residual_buffer_size=config["residual_buffer_size"],
            n_sims=n_sims,
            min_history=min_history,
            refit_every_n_events=config["refit_every_n_events"],
            engine_name=config.get("engine_name", "ar_egarch"),
            engine_kwargs=config.get("engine_kwargs", {}),
            model_kwargs=config["model_kwargs"],
            use_jump_augmentation=config["jump_augmentation"],
        )
        summary, tau_table, moneyness_table, _details = summarize_config_results(
            results=results,
            config=config,
            gaussian_vol_window=gaussian_vol_window,
            near_strike_log_threshold=near_strike_log_threshold,
        )
        summaries.append(summary)
        tau_tables.append(tau_table)
        moneyness_tables.append(moneyness_table)
        if summary["beats_gaussian_vol_brier"] or summary["beats_gaussian_vol_log_loss"]:
            winners.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(["model_brier", "model_log_loss"]).reset_index(drop=True)
    tau_df = pd.concat(tau_tables, ignore_index=True) if tau_tables else pd.DataFrame()
    moneyness_df = pd.concat(moneyness_tables, ignore_index=True) if moneyness_tables else pd.DataFrame()
    winners_df = pd.DataFrame(winners).sort_values(["brier_delta_vs_gaussian_vol", "log_loss_delta_vs_gaussian_vol"]).reset_index(drop=True) if winners else pd.DataFrame()

    summary_df.to_csv(output_path / "challenger_summary.csv", index=False)
    summary_df.sort_values(["model_brier", "model_log_loss"]).to_csv(output_path / "challenger_ranking_by_brier.csv", index=False)
    summary_df.sort_values(["model_log_loss", "model_brier"]).to_csv(output_path / "challenger_ranking_by_log_loss.csv", index=False)
    tau_df.to_csv(output_path / "challenger_tau_bucket_comparison.csv", index=False)
    moneyness_df.to_csv(output_path / "challenger_moneyness_bucket_comparison.csv", index=False)
    winners_df.to_csv(output_path / "challenger_vs_gaussian_vol_winners.csv", index=False)

    return {
        "summary_path": str(output_path / "challenger_summary.csv"),
        "ranking_by_brier_path": str(output_path / "challenger_ranking_by_brier.csv"),
        "ranking_by_log_loss_path": str(output_path / "challenger_ranking_by_log_loss.csv"),
        "tau_bucket_path": str(output_path / "challenger_tau_bucket_comparison.csv"),
        "moneyness_bucket_path": str(output_path / "challenger_moneyness_bucket_comparison.csv"),
        "winners_path": str(output_path / "challenger_vs_gaussian_vol_winners.csv"),
        "winner_count": int(len(winners_df)),
    }
