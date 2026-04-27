"""Rolling blocked evaluation helpers for probability-engine comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from .feature_builder import build_event_features, get_default_feature_columns
from .probability_backtest import run_probability_backtest
from .probability_engine_factory import build_probability_engine


TAU_BUCKET_ORDER = ["1-5", "6-15", "16-30", "31-60"]


def build_event_hour_splits(
    events: pd.DataFrame,
    train_hours: int,
    calibration_hours: int,
    validation_hours: int,
    step_hours: int | None = None,
) -> list[dict]:
    hours = sorted(pd.to_datetime(events["event_hour_start"], utc=True).drop_duplicates())
    step_hours = validation_hours if step_hours is None else int(step_hours)
    total = train_hours + calibration_hours + validation_hours
    splits = []
    for start_idx in range(0, len(hours) - total + 1, step_hours):
        train_block = hours[start_idx : start_idx + train_hours]
        calib_block = hours[start_idx + train_hours : start_idx + train_hours + calibration_hours]
        val_block = hours[start_idx + train_hours + calibration_hours : start_idx + total]
        splits.append(
            {
                "fold_id": len(splits),
                "train_hours": train_block,
                "calibration_hours": calib_block,
                "validation_hours": val_block,
            }
        )
    return splits


def _fit_logistic_calibrator(y: pd.Series, p: pd.Series) -> tuple[float, float]:
    p = p.astype(float).clip(1e-6, 1 - 1e-6)
    y = y.astype(float).to_numpy()
    x = np.log(p / (1.0 - p)).to_numpy()

    def objective(params):
        a, b = params
        logits = np.clip(a + b * x, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    result = optimize.minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    if not result.success:
        return 0.0, 1.0
    return float(result.x[0]), float(result.x[1])


def apply_logistic_calibrator(p: pd.Series, params: tuple[float, float]) -> pd.Series:
    a, b = params
    p = p.astype(float).clip(1e-6, 1 - 1e-6)
    logits = np.clip(a + b * np.log(p / (1.0 - p)), -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return pd.Series(np.clip(probs, 0.0, 1.0), index=p.index)


def _metrics(y: pd.Series, p: pd.Series) -> dict:
    p = p.astype(float).clip(1e-12, 1 - 1e-12)
    y = y.astype(int)
    return {
        "observations": int(len(y)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean(((p >= 0.5).astype(int)) == y)),
        "mean_predicted_probability": float(p.mean()),
        "mean_realized_frequency": float(y.mean()),
    }


def _tau_bucket(value: int) -> str | None:
    if 1 <= value <= 5:
        return "1-5"
    if 6 <= value <= 15:
        return "6-15"
    if 16 <= value <= 30:
        return "16-30"
    if 31 <= value <= 60:
        return "31-60"
    return None


def _predict_parametric_engine(
    engine_name: str,
    minute_df: pd.DataFrame,
    subset_events: pd.DataFrame,
    engine_kwargs: dict | None = None,
) -> pd.DataFrame:
    results = run_probability_backtest(
        close_series=minute_df["close"],
        events=subset_events,
        fit_window=int((engine_kwargs or {}).get("fit_window", 2000)),
        residual_buffer_size=int((engine_kwargs or {}).get("residual_buffer_size", 2000)),
        n_sims=int((engine_kwargs or {}).get("n_sims", 200)),
        min_history=(engine_kwargs or {}).get("min_history"),
        refit_every_n_events=int((engine_kwargs or {}).get("refit_every_n_events", 4)),
        engine_name=engine_name,
        engine_kwargs=engine_kwargs or {},
        model_kwargs=(engine_kwargs or {}).get("model_kwargs", {}),
    )
    return results[results["skip_reason"].isna()].copy()


def _predict_lgbm_engine(
    feature_df: pd.DataFrame,
    train_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    engine_kwargs: dict | None = None,
) -> pd.DataFrame:
    engine = build_probability_engine("lgbm", **(engine_kwargs or {}))
    feature_columns = (engine_kwargs or {}).get("feature_columns") or get_default_feature_columns(feature_df)
    engine.fit_frame(train_rows[feature_columns + ["realized_yes"]], label_col="realized_yes")
    preds = engine.predict_frame(score_rows[feature_columns])
    out = score_rows[["decision_ts", "event_hour_start", "tau_minutes", "realized_yes", "strike_price", "spot_now"]].copy()
    out["p_yes"] = preds["p_yes"].values
    out["p_no"] = preds["p_no"].values
    out["engine_name"] = "lgbm"
    return out


def evaluate_probability_engines(
    minute_df: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str | Path,
    engine_configs: list[dict] | None = None,
    train_hours: int = 24 * 30,
    calibration_hours: int = 24 * 7,
    validation_hours: int = 24 * 7,
    step_hours: int | None = None,
    write_predictions: bool = True,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    events = events.copy()
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True)
    events["event_hour_start"] = pd.to_datetime(events["event_hour_start"], utc=True)
    splits = build_event_hour_splits(events, train_hours, calibration_hours, validation_hours, step_hours=step_hours)

    engine_configs = engine_configs or [
        {"engine_name": "gaussian_vol", "engine_kwargs": {"fit_window": 2000, "vol_window": 2000, "refit_every_n_events": 4, "n_sims": 0}},
        {"engine_name": "ar_egarch", "engine_kwargs": {"fit_window": 2000, "residual_buffer_size": 1000, "refit_every_n_events": 4, "n_sims": 200}},
        {"engine_name": "lgbm", "engine_kwargs": {}},
    ]

    feature_df = build_event_features(minute_df, events)
    validation_predictions = []

    for split in splits:
        train_hours_set = set(split["train_hours"])
        calib_hours_set = set(split["calibration_hours"])
        val_hours_set = set(split["validation_hours"])
        train_rows = feature_df[feature_df["event_hour_start"].isin(train_hours_set)].copy()
        calib_rows = feature_df[feature_df["event_hour_start"].isin(calib_hours_set)].copy()
        val_rows = feature_df[feature_df["event_hour_start"].isin(val_hours_set)].copy()

        for config in engine_configs:
            engine_name = config["engine_name"]
            engine_kwargs = config.get("engine_kwargs", {})
            config_id = config.get("config_id", engine_name)
            if engine_name == "lgbm":
                calib_preds = _predict_lgbm_engine(feature_df, train_rows, calib_rows, engine_kwargs=engine_kwargs)
                val_preds = _predict_lgbm_engine(feature_df, train_rows, val_rows, engine_kwargs=engine_kwargs)
            else:
                calib_preds = _predict_parametric_engine(engine_name, minute_df, calib_rows, engine_kwargs=engine_kwargs)
                val_preds = _predict_parametric_engine(engine_name, minute_df, val_rows, engine_kwargs=engine_kwargs)

            if calib_preds.empty or val_preds.empty:
                continue

            calibrator = _fit_logistic_calibrator(calib_preds["realized_yes"], calib_preds["p_yes"])
            val_preds = val_preds.copy()
            val_preds["p_yes_raw"] = val_preds["p_yes"]
            val_preds["p_yes"] = apply_logistic_calibrator(val_preds["p_yes"], calibrator)
            val_preds["p_no"] = 1.0 - val_preds["p_yes"]
            val_preds["fold_id"] = split["fold_id"]
            val_preds["engine_name"] = engine_name
            val_preds["config_id"] = config_id
            validation_predictions.append(val_preds)

    validation_df = pd.concat(validation_predictions, ignore_index=True) if validation_predictions else pd.DataFrame()
    if validation_df.empty:
        raise ValueError("No validation predictions were produced")

    validation_df["tau_bucket"] = validation_df["tau_minutes"].apply(lambda value: _tau_bucket(int(value)))
    summary_rows = []
    tau_rows = []
    calibration_rows = []
    for config_id, group in validation_df.groupby("config_id", observed=False):
        engine_name = str(group["engine_name"].iloc[0])
        summary_rows.append({"config_id": config_id, "engine_name": engine_name, **_metrics(group["realized_yes"], group["p_yes"])})
        calibration_rows.append(
            {
                "config_id": config_id,
                "engine_name": engine_name,
                "mean_predicted_probability": float(group["p_yes"].mean()),
                "mean_realized_frequency": float(group["realized_yes"].mean()),
                "calibration_gap": float(group["p_yes"].mean() - group["realized_yes"].mean()),
            }
        )
        for tau_bucket, bucket_group in group.groupby("tau_bucket", observed=False):
            tau_rows.append({"config_id": config_id, "engine_name": engine_name, "tau_bucket": tau_bucket, **_metrics(bucket_group["realized_yes"], bucket_group["p_yes"])})

    summary_df = pd.DataFrame(summary_rows).sort_values("brier_score").reset_index(drop=True)
    tau_df = pd.DataFrame(tau_rows).sort_values(["tau_bucket", "brier_score"]).reset_index(drop=True)
    calibration_df = pd.DataFrame(calibration_rows).sort_values("engine_name").reset_index(drop=True)

    summary_df.to_csv(output_path / "engine_summary.csv", index=False)
    tau_df.to_csv(output_path / "engine_tau_bucket_summary.csv", index=False)
    calibration_df.to_csv(output_path / "engine_calibration_summary.csv", index=False)
    predictions_path = None
    if write_predictions:
        predictions_path = output_path / "engine_validation_predictions.csv"
        validation_df.to_csv(predictions_path, index=False)

    return {
        "summary_path": str(output_path / "engine_summary.csv"),
        "tau_summary_path": str(output_path / "engine_tau_bucket_summary.csv"),
        "calibration_summary_path": str(output_path / "engine_calibration_summary.csv"),
        "predictions_path": None if predictions_path is None else str(predictions_path),
        "fold_count": len(splits),
    }


def evaluate_gaussian_vol_configs(
    minute_df: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str | Path,
    gaussian_configs: list[dict],
    train_hours: int = 24 * 30,
    calibration_hours: int = 24 * 7,
    validation_hours: int = 24 * 7,
    step_hours: int | None = None,
    write_predictions: bool = False,
) -> dict:
    engine_configs = []
    for index, config in enumerate(gaussian_configs):
        calibration_mode = config.get("calibration_mode", "none")
        engine_kwargs = {
            "fit_window": int(config.get("fit_window", config.get("vol_window", 2000))),
            "vol_window": int(config.get("vol_window", 2000)),
            "min_periods": int(config.get("min_periods", max(20, config.get("vol_window", 2000) // 4))),
            "sigma_floor": float(config.get("sigma_floor", 1e-8)),
            "sigma_cap": float(config.get("sigma_cap", 0.25)),
            "fallback_sigma": float(config.get("fallback_sigma", 5e-4)),
            "calibration_mode": calibration_mode,
            "refit_every_n_events": int(config.get("refit_every_n_events", 4)),
            "min_history": int(config.get("min_history", config.get("vol_window", 2000))),
        }
        engine_configs.append(
            {
                "engine_name": "gaussian_vol",
                "engine_kwargs": engine_kwargs,
                "config_id": config.get(
                    "config_id",
                    f"gaussian_vol_vw{engine_kwargs['vol_window']}_mp{engine_kwargs['min_periods']}_{calibration_mode}_{index}",
                ),
            }
        )

    report = evaluate_probability_engines(
        minute_df=minute_df,
        events=events,
        output_dir=output_dir,
        engine_configs=engine_configs,
        train_hours=train_hours,
        calibration_hours=calibration_hours,
        validation_hours=validation_hours,
        step_hours=step_hours,
        write_predictions=write_predictions,
    )

    summary_path = Path(report["summary_path"])
    summary_df = pd.read_csv(summary_path)
    config_df = pd.DataFrame(gaussian_configs)
    ranked_df = summary_df.merge(config_df[["config_id", "vol_window", "min_periods", "calibration_mode"]], on="config_id", how="left")
    ranked_df = ranked_df.rename(
        columns={
            "brier_score": "mean_brier",
            "log_loss": "mean_log_loss",
            "accuracy": "mean_accuracy",
        }
    )
    ranked_df = ranked_df[
        [
            "config_id",
            "vol_window",
            "min_periods",
            "calibration_mode",
            "mean_brier",
            "mean_log_loss",
            "mean_accuracy",
            "observations",
            "mean_predicted_probability",
            "mean_realized_frequency",
        ]
    ].sort_values(["mean_brier", "mean_log_loss", "mean_accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    ranked_path = Path(output_dir) / "gaussian_sweep_ranked_summary.csv"
    ranked_df.to_csv(ranked_path, index=False)
    report["ranked_summary_path"] = str(ranked_path)

    return report
