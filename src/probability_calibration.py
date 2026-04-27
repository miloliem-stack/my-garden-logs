"""Calibration metrics and comparative reports for offline probability backtests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


TAU_BUCKETS = [
    ("1-5", 1, 5),
    ("6-15", 6, 15),
    ("16-30", 16, 30),
    ("31-60", 31, 60),
]

FORECAST_COLUMNS = {
    "model": "p_yes",
    "constant_0_5": "p_yes_constant_0_5",
    "naive_sign": "p_yes_naive_sign",
    "gaussian_vol": "p_yes_gaussian_vol",
    "model_calibrated_logistic_tau": "p_yes_calibrated_logistic_tau",
}


def _bucket_tau(value: int) -> str | None:
    for label, lower, upper in TAU_BUCKETS:
        if lower <= value <= upper:
            return label
    return None


def add_tau_bucket(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["tau_bucket"] = out["tau_minutes"].apply(lambda value: _bucket_tau(int(value)))
    return out


def get_valid_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Backtest results are empty")
    valid = results[results["p_yes"].notna()].copy()
    if valid.empty:
        raise ValueError("Backtest results contain no successful probability rows")
    return add_tau_bucket(valid)


def add_baseline_probabilities(results: pd.DataFrame, vol_window: int = 288, min_periods: int | None = None) -> pd.DataFrame:
    df = get_valid_results(results).sort_values("decision_ts").reset_index(drop=True)
    min_periods = max(20, vol_window // 4) if min_periods is None else int(min_periods)
    min_periods = min(min_periods, vol_window)

    df["p_yes_constant_0_5"] = 0.5
    df["p_yes_naive_sign"] = np.where(
        df["spot_now"] > df["strike_price"],
        1.0,
        np.where(df["spot_now"] < df["strike_price"], 0.0, 0.5),
    )

    log_spot = np.log(df["spot_now"].astype(float).clip(lower=1e-12))
    delta_minutes = df["decision_ts"].diff().dt.total_seconds().div(60.0)
    scaled_returns = log_spot.diff() / np.sqrt(delta_minutes.clip(lower=1.0))
    rolling_sigma_per_sqrt_min = scaled_returns.rolling(vol_window, min_periods=min_periods).std(ddof=0)
    fallback_sigma = float(scaled_returns.dropna().std(ddof=0)) if scaled_returns.dropna().size else 0.0
    rolling_sigma_per_sqrt_min = rolling_sigma_per_sqrt_min.fillna(fallback_sigma)

    threshold = np.log(df["strike_price"].astype(float) / df["spot_now"].astype(float))
    horizon_sigma = rolling_sigma_per_sqrt_min * np.sqrt(df["tau_minutes"].astype(float).clip(lower=1.0))
    gaussian = np.where(
        horizon_sigma > 0,
        1.0 - stats.norm.cdf(threshold / horizon_sigma),
        np.where(df["spot_now"] > df["strike_price"], 1.0, np.where(df["spot_now"] < df["strike_price"], 0.0, 0.5)),
    )
    df["p_yes_gaussian_vol"] = np.clip(gaussian, 0.0, 1.0)
    df["rolling_sigma_per_sqrt_min"] = rolling_sigma_per_sqrt_min
    return df


def _compute_metrics(y: pd.Series, p: pd.Series) -> dict:
    p = p.astype(float).clip(1e-12, 1 - 1e-12)
    y = y.astype(int)
    return {
        "observations": int(len(y)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy_at_0_5": float(np.mean(((p >= 0.5).astype(int)) == y)),
        "mean_predicted_probability": float(p.mean()),
        "mean_realized_frequency": float(y.mean()),
    }


def compute_calibration_summary(results: pd.DataFrame, bucket_count: int = 10) -> dict:
    valid = get_valid_results(results)
    summary = _compute_metrics(valid["realized_yes"], valid["p_yes"])
    summary["skipped_or_failed_observations"] = int(len(results) - len(valid))
    return summary


def _build_reliability_table_from_column(
    df: pd.DataFrame,
    probability_col: str,
    forecast_name: str,
    bucket_count: int = 10,
) -> pd.DataFrame:
    clipped = df[probability_col].astype(float).clip(0.0, 1.0)
    buckets = np.linspace(0.0, 1.0, bucket_count + 1)
    labels = [f"{buckets[i]:.1f}-{buckets[i+1]:.1f}" for i in range(bucket_count)]
    tmp = df.copy()
    tmp["probability_bucket"] = pd.cut(
        clipped,
        bins=buckets,
        include_lowest=True,
        labels=labels,
        duplicates="drop",
    )
    table = (
        tmp.groupby("probability_bucket", observed=False)
        .agg(
            observations=("realized_yes", "size"),
            mean_predicted=(probability_col, "mean"),
            realized_frequency=("realized_yes", "mean"),
            mean_tau_minutes=("tau_minutes", "mean"),
        )
        .reset_index()
    )
    table["forecast_name"] = forecast_name
    return table


def build_reliability_table(results: pd.DataFrame, bucket_count: int = 10) -> pd.DataFrame:
    valid = get_valid_results(results)
    return _build_reliability_table_from_column(valid, "p_yes", "model", bucket_count=bucket_count).drop(columns=["forecast_name"])


def build_forecast_summary_table(results: pd.DataFrame, forecast_columns: dict[str, str] | None = None) -> pd.DataFrame:
    valid = add_baseline_probabilities(results)
    forecast_columns = forecast_columns or FORECAST_COLUMNS
    rows = []
    for forecast_name, column in forecast_columns.items():
        if column not in valid.columns:
            continue
        rows.append({"forecast_name": forecast_name, **_compute_metrics(valid["realized_yes"], valid[column])})
    return pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)


def build_reliability_comparison(results: pd.DataFrame, bucket_count: int = 10, forecast_columns: dict[str, str] | None = None) -> pd.DataFrame:
    valid = add_baseline_probabilities(results)
    forecast_columns = forecast_columns or FORECAST_COLUMNS
    frames = []
    for forecast_name, column in forecast_columns.items():
        if column not in valid.columns:
            continue
        frames.append(_build_reliability_table_from_column(valid, column, forecast_name, bucket_count=bucket_count))
    return pd.concat(frames, ignore_index=True)


def build_tau_bucket_brier(results: pd.DataFrame) -> pd.DataFrame:
    valid = get_valid_results(results)
    table = (
        valid[valid["tau_bucket"].notna()]
        .groupby("tau_bucket", observed=False)
        .apply(lambda g: pd.Series({"observations": len(g), "brier_score": float(np.mean((g["p_yes"] - g["realized_yes"]) ** 2))}))
        .reset_index()
    )
    return table


def build_tau_bucket_forecast_comparison(results: pd.DataFrame, forecast_columns: dict[str, str] | None = None) -> pd.DataFrame:
    valid = add_baseline_probabilities(results)
    valid = valid[valid["tau_bucket"].notna()].copy()
    forecast_columns = forecast_columns or FORECAST_COLUMNS
    rows: list[dict] = []
    for forecast_name, column in forecast_columns.items():
        if column not in valid.columns:
            continue
        for tau_bucket, group in valid.groupby("tau_bucket", observed=False):
            rows.append(
                {
                    "forecast_name": forecast_name,
                    "tau_bucket": tau_bucket,
                    **_compute_metrics(group["realized_yes"], group[column]),
                }
            )
    return pd.DataFrame(rows).sort_values(["tau_bucket", "brier_score"]).reset_index(drop=True)


def _fit_logistic_parameters(y: pd.Series, p: pd.Series) -> tuple[float, float]:
    p_clip = p.astype(float).clip(1e-6, 1 - 1e-6)
    y_int = y.astype(float).to_numpy()
    x = np.log(p_clip / (1.0 - p_clip)).to_numpy()

    def objective(params: np.ndarray) -> float:
        a, b = params
        logits = a + b * x
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        return float(-np.mean(y_int * np.log(probs) + (1.0 - y_int) * np.log(1.0 - probs)))

    result = optimize.minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    if not result.success:
        return 0.0, 1.0
    return float(result.x[0]), float(result.x[1])


def add_logistic_calibration_by_tau_bucket(
    results: pd.DataFrame,
    source_col: str = "p_yes",
    output_col: str = "p_yes_calibrated_logistic_tau",
    min_bucket_obs: int = 50,
) -> pd.DataFrame:
    df = add_baseline_probabilities(results)
    global_a, global_b = _fit_logistic_parameters(df["realized_yes"], df[source_col])
    calibrated = pd.Series(index=df.index, dtype=float)
    for tau_bucket, group in df.groupby("tau_bucket", observed=False):
        if tau_bucket is None or len(group) < min_bucket_obs:
            a, b = global_a, global_b
        else:
            a, b = _fit_logistic_parameters(group["realized_yes"], group[source_col])
        logits = a + b * np.log(group[source_col].astype(float).clip(1e-6, 1 - 1e-6) / (1.0 - group[source_col].astype(float).clip(1e-6, 1 - 1e-6)))
        calibrated.loc[group.index] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    df[output_col] = calibrated.astype(float).clip(0.0, 1.0)
    return df


def write_calibration_report(
    results: pd.DataFrame,
    output_dir: str | Path,
    bucket_count: int = 10,
    calibration_method: str = "none",
    gaussian_vol_window: int = 288,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    enriched = add_baseline_probabilities(results, vol_window=gaussian_vol_window)
    if calibration_method == "logistic":
        enriched = add_logistic_calibration_by_tau_bucket(enriched)
    elif calibration_method != "none":
        raise ValueError(f"Unsupported calibration method: {calibration_method}")

    forecast_columns = {
        key: value
        for key, value in FORECAST_COLUMNS.items()
        if value in enriched.columns
    }

    summary = compute_calibration_summary(enriched, bucket_count=bucket_count)
    reliability = build_reliability_table(enriched, bucket_count=bucket_count)
    tau_brier = build_tau_bucket_brier(enriched)
    summary_by_forecast = build_forecast_summary_table(enriched, forecast_columns=forecast_columns)
    reliability_by_forecast = build_reliability_comparison(enriched, bucket_count=bucket_count, forecast_columns=forecast_columns)
    tau_by_forecast = build_tau_bucket_forecast_comparison(enriched, forecast_columns=forecast_columns)

    pd.DataFrame([summary]).to_csv(output_path / "summary.csv", index=False)
    reliability.to_csv(output_path / "reliability.csv", index=False)
    tau_brier.to_csv(output_path / "brier_by_tau.csv", index=False)
    summary_by_forecast.to_csv(output_path / "summary_by_forecast.csv", index=False)
    reliability_by_forecast.to_csv(output_path / "reliability_by_forecast.csv", index=False)
    tau_by_forecast.to_csv(output_path / "brier_by_tau_by_forecast.csv", index=False)
    enriched.to_csv(output_path / "forecasts_enriched.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for forecast_name, group in reliability_by_forecast.groupby("forecast_name"):
            ax.plot(group["mean_predicted"], group["realized_frequency"], marker="o", label=forecast_name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Realized frequency")
        ax.set_title("Reliability Comparison")
        ax.legend()
        fig.savefig(output_path / "reliability_comparison.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.hist(enriched["p_yes"].astype(float), bins=20, alpha=0.6, label="model")
        if "p_yes_calibrated_logistic_tau" in enriched.columns:
            ax.hist(enriched["p_yes_calibrated_logistic_tau"].astype(float), bins=20, alpha=0.6, label="calibrated")
        ax.set_xlabel("Predicted p_yes")
        ax.set_ylabel("Count")
        ax.set_title("Forecast Probability Histogram")
        ax.legend()
        fig.savefig(output_path / "p_yes_histogram.png", bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    return {
        "summary": summary,
        "summary_by_forecast_path": str(output_path / "summary_by_forecast.csv"),
        "reliability_path": str(output_path / "reliability.csv"),
        "reliability_comparison_path": str(output_path / "reliability_by_forecast.csv"),
        "tau_brier_path": str(output_path / "brier_by_tau.csv"),
        "tau_brier_comparison_path": str(output_path / "brier_by_tau_by_forecast.csv"),
    }
