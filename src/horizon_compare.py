"""Offline horizon comparison lab for 1h/4h/1d BTC directional-event research."""

from __future__ import annotations

from pathlib import Path
import json
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .feature_builder import get_default_feature_columns
from .horizon_event_dataset import HORIZON_TO_MINUTES, build_fixed_horizon_event_dataset, write_dataframe
from .horizon_feature_builder import build_horizon_event_features, get_horizon_feature_columns
from .probability_backtest import run_probability_backtest
from .probability_engine_factory import build_probability_engine


TAU_BUCKETS = [(1, 5, "1-5"), (6, 15, "6-15"), (16, 30, "16-30"), (31, 60, "31-60"), (61, 240, "61-240"), (241, 1440, "241-1440")]
MONTHLY_MIN_OBS = 20


def _duplicate_columns(df: pd.DataFrame) -> list[str]:
    return df.columns[df.columns.duplicated()].tolist()


def _emit_progress(progress_callback: Callable[[dict], None] | None, payload: dict) -> None:
    if progress_callback is None:
        return
    stamped = dict(payload)
    stamped["last_updated"] = datetime.now(timezone.utc).isoformat()
    progress_callback(stamped)


def build_horizon_splits(
    events: pd.DataFrame,
    *,
    train_events: int,
    calibration_events: int,
    validation_events: int,
    step_events: int | None = None,
) -> list[dict]:
    starts = sorted(pd.to_datetime(events["event_start"], utc=True).drop_duplicates())
    step_events = validation_events if step_events is None else int(step_events)
    total = train_events + calibration_events + validation_events
    splits = []
    for start_idx in range(0, len(starts) - total + 1, step_events):
        splits.append(
            {
                "fold_id": len(splits),
                "train_starts": starts[start_idx : start_idx + train_events],
                "calibration_starts": starts[start_idx + train_events : start_idx + train_events + calibration_events],
                "validation_starts": starts[start_idx + train_events + calibration_events : start_idx + total],
            }
        )
    return splits


def select_horizon_splits(
    splits: list[dict],
    *,
    max_windows: int | None = None,
) -> tuple[list[dict], dict]:
    total_candidate_windows = len(splits)
    if max_windows is None:
        selected = list(splits)
        mode = "full"
    else:
        limit = max(1, int(max_windows))
        selected = list(splits[-limit:])
        mode = "single_split" if limit == 1 else "limited_windows"
    return selected, {
        "total_candidate_windows": total_candidate_windows,
        "evaluated_windows": len(selected),
        "selection_policy": "most_recent_windows",
        "mode": mode,
    }


def _window_label(split: dict) -> str:
    validation_starts = split.get("validation_starts") or []
    if not validation_starts:
        return "unknown"
    return f"{pd.Timestamp(validation_starts[0]).isoformat()}..{pd.Timestamp(validation_starts[-1]).isoformat()}"


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


def _apply_logistic_calibrator(p: pd.Series, params: tuple[float, float]) -> pd.Series:
    a, b = params
    p = p.astype(float).clip(1e-6, 1 - 1e-6)
    logits = np.clip(a + b * np.log(p / (1.0 - p)), -30.0, 30.0)
    return pd.Series(1.0 / (1.0 + np.exp(-logits)), index=p.index).clip(0.0, 1.0)


def _tau_bucket(value: int) -> str:
    for lo, hi, label in TAU_BUCKETS:
        if lo <= int(value) <= hi:
            return label
    return f">{TAU_BUCKETS[-1][1]}"


def _moneyness_bucket(spot_now: float, strike_price: float) -> str:
    log_m = float(np.log(max(spot_now, 1e-12) / max(strike_price, 1e-12)))
    if log_m <= -0.01:
        return "deep_below"
    if log_m < -0.002:
        return "below"
    if abs(log_m) <= 0.002:
        return "near_strike"
    if log_m < 0.01:
        return "above"
    return "deep_above"


def _core_metrics(y: pd.Series, p: pd.Series) -> dict:
    y = y.astype(int)
    p = p.astype(float).clip(1e-12, 1 - 1e-12)
    confidence = (p - 0.5).abs()
    pred_side = (p >= 0.5).astype(int)
    pnl_proxy = np.where(pred_side == 1, y - 0.5, (1 - y) - 0.5)
    return {
        "sample_count": int(len(y)),
        "accuracy": float(np.mean(pred_side == y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "mean_confidence": float(confidence.mean()),
        "mean_predicted_probability": float(p.mean()),
        "mean_realized_frequency": float(y.mean()),
        "pnl_proxy": float(np.mean(pnl_proxy)),
    }


def _tradeability_metrics(df: pd.DataFrame, threshold: float) -> dict:
    out = df.copy()
    out["confidence"] = (out["p_yes"] - 0.5).abs()
    tradable = out[out["confidence"] >= threshold].copy()
    result = {
        "edge_threshold": float(threshold),
        "trade_count": int(len(tradable)),
        "trade_fraction": float(len(tradable) / len(out)) if len(out) else 0.0,
        "avg_confidence_edge": float(tradable["confidence"].mean()) if len(tradable) else 0.0,
        "pnl_proxy_thresholded": 0.0,
        "avg_market_implied_edge": None,
    }
    if len(tradable):
        pred_side = (tradable["p_yes"] >= 0.5).astype(int)
        pnl_proxy = np.where(pred_side == 1, tradable["realized_yes"] - 0.5, (1 - tradable["realized_yes"]) - 0.5)
        result["pnl_proxy_thresholded"] = float(np.mean(pnl_proxy))
    if "market_implied_p_yes" in out.columns and out["market_implied_p_yes"].notna().any():
        tradable = tradable.copy()
        tradable["market_implied_edge"] = (tradable["p_yes"] - tradable["market_implied_p_yes"]).abs()
        result["avg_market_implied_edge"] = float(tradable["market_implied_edge"].mean()) if len(tradable) else 0.0
    return result


def _ar_fit_params(history_close: pd.Series) -> tuple[float, float, float]:
    log_close = np.log(history_close.astype(float).clip(lower=1e-12))
    returns = log_close.diff().dropna()
    if len(returns) < 20:
        raise ValueError("not enough history for AR fit")
    x = returns.shift(1).dropna()
    y = returns.loc[x.index]
    X = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    intercept = float(beta[0])
    phi = float(beta[1])
    residuals = y.to_numpy() - (intercept + phi * x.to_numpy())
    sigma = float(np.std(residuals, ddof=0))
    return intercept, phi, max(sigma, 1e-8)


def _predict_ar_only(minute_df: pd.DataFrame, events: pd.DataFrame, fit_window: int = 2000) -> pd.DataFrame:
    close_series = minute_df["close"].astype(float)
    rows = []
    for _, event in events.iterrows():
        decision_ts = pd.to_datetime(event["decision_ts"], utc=True)
        history_close = close_series[close_series.index < decision_ts].iloc[-fit_window:]
        if len(history_close) < 50:
            continue
        intercept, phi, sigma = _ar_fit_params(history_close)
        log_ret_last = float(np.log(history_close.iloc[-1] / history_close.iloc[-2]))
        tau = int(event["tau_minutes"])
        mean_sum = 0.0
        prev = log_ret_last
        for _step in range(tau):
            prev = intercept + phi * prev
            mean_sum += prev
        std_sum = sigma * np.sqrt(tau)
        threshold = np.log(float(event["strike_price"]) / float(event["spot_now"]))
        z = (threshold - mean_sum) / max(std_sum, 1e-8)
        p_yes = float(1.0 - stats.norm.cdf(z))
        rows.append(
            {
                "decision_ts": decision_ts,
                "event_start": pd.to_datetime(event["event_start"], utc=True),
                "event_end": pd.to_datetime(event["event_end"], utc=True),
                "tau_minutes": tau,
                "strike_price": float(event["strike_price"]),
                "spot_now": float(event["spot_now"]),
                "realized_yes": int(event["realized_yes"]),
                "p_yes": p_yes,
                "p_no": 1.0 - p_yes,
            }
        )
    return pd.DataFrame(rows)


def _predict_ar_vol(minute_df: pd.DataFrame, events: pd.DataFrame, fit_window: int = 2000, vol_window: int = 240) -> pd.DataFrame:
    close_series = minute_df["close"].astype(float)
    log_close = np.log(close_series.clip(lower=1e-12))
    returns = log_close.diff()
    rows = []
    for _, event in events.iterrows():
        decision_ts = pd.to_datetime(event["decision_ts"], utc=True)
        history_close = close_series[close_series.index < decision_ts].iloc[-fit_window:]
        history_returns = returns[returns.index < decision_ts].dropna().iloc[-max(vol_window, fit_window):]
        if len(history_close) < 50 or len(history_returns) < 20:
            continue
        intercept, phi, _sigma = _ar_fit_params(history_close)
        rolling_sigma = float(history_returns.iloc[-vol_window:].std(ddof=0)) if len(history_returns) >= vol_window else float(history_returns.std(ddof=0))
        rolling_sigma = max(rolling_sigma, 1e-8)
        log_ret_last = float(np.log(history_close.iloc[-1] / history_close.iloc[-2]))
        tau = int(event["tau_minutes"])
        mean_sum = 0.0
        prev = log_ret_last
        for _step in range(tau):
            prev = intercept + phi * prev
            mean_sum += prev
        std_sum = rolling_sigma * np.sqrt(tau)
        threshold = np.log(float(event["strike_price"]) / float(event["spot_now"]))
        z = (threshold - mean_sum) / max(std_sum, 1e-8)
        p_yes = float(1.0 - stats.norm.cdf(z))
        rows.append(
            {
                "decision_ts": decision_ts,
                "event_start": pd.to_datetime(event["event_start"], utc=True),
                "event_end": pd.to_datetime(event["event_end"], utc=True),
                "tau_minutes": tau,
                "strike_price": float(event["strike_price"]),
                "spot_now": float(event["spot_now"]),
                "realized_yes": int(event["realized_yes"]),
                "p_yes": p_yes,
                "p_no": 1.0 - p_yes,
            }
        )
    return pd.DataFrame(rows)


def _predict_naive(events: pd.DataFrame) -> pd.DataFrame:
    out = events[["decision_ts", "event_start", "event_end", "tau_minutes", "strike_price", "spot_now", "realized_yes"]].copy()
    out["p_yes"] = np.where(out["spot_now"] > out["strike_price"], 1.0, np.where(out["spot_now"] < out["strike_price"], 0.0, 0.5))
    out["p_no"] = 1.0 - out["p_yes"]
    return out


def _predict_market_implied(events: pd.DataFrame) -> pd.DataFrame:
    if "market_implied_p_yes" not in events.columns:
        return pd.DataFrame()
    out = events[["decision_ts", "event_start", "event_end", "tau_minutes", "strike_price", "spot_now", "realized_yes", "market_implied_p_yes"]].copy()
    out = out[out["market_implied_p_yes"].notna()].copy()
    out["p_yes"] = out["market_implied_p_yes"].clip(0.0, 1.0)
    out["p_no"] = 1.0 - out["p_yes"]
    return out


def _predict_parametric_engine(engine_name: str, minute_df: pd.DataFrame, events: pd.DataFrame, engine_kwargs: dict | None = None) -> pd.DataFrame:
    progress_callback = None if engine_kwargs is None else engine_kwargs.pop("progress_callback", None)
    progress_stage = None if engine_kwargs is None else engine_kwargs.pop("progress_stage", None)
    heartbeat_seconds = None if engine_kwargs is None else engine_kwargs.pop("heartbeat_seconds", None)
    parametric_events = events.rename(columns={"event_start": "event_hour_start", "event_end": "event_hour_end", "minute_in_event": "minute_in_hour"})
    duplicated = _duplicate_columns(parametric_events)
    if duplicated:
        raise ValueError(f"Duplicate columns in horizon dataset: {duplicated}")
    results = run_probability_backtest(
        close_series=minute_df["close"],
        events=parametric_events,
        fit_window=int((engine_kwargs or {}).get("fit_window", 2000)),
        residual_buffer_size=int((engine_kwargs or {}).get("residual_buffer_size", 2000)),
        n_sims=int((engine_kwargs or {}).get("n_sims", 300)),
        min_history=(engine_kwargs or {}).get("min_history"),
        refit_every_n_events=int((engine_kwargs or {}).get("refit_every_n_events", 4)),
        engine_name=engine_name,
        engine_kwargs=engine_kwargs or {},
        progress_callback=progress_callback,
        progress_stage=progress_stage,
        heartbeat_seconds=heartbeat_seconds,
    )
    out = results[results["skip_reason"].isna()].copy()
    return out.rename(columns={"event_hour_start": "event_start", "event_hour_end": "event_end"})


def _predict_lgbm(feature_df: pd.DataFrame, train_rows: pd.DataFrame, score_rows: pd.DataFrame, engine_kwargs: dict | None = None) -> pd.DataFrame:
    engine = build_probability_engine("lgbm", **(engine_kwargs or {}))
    feature_columns = (engine_kwargs or {}).get("feature_columns") or get_horizon_feature_columns(feature_df)
    engine.fit_frame(train_rows[feature_columns + ["realized_yes"]], label_col="realized_yes")
    preds = engine.predict_frame(score_rows[feature_columns])
    out = score_rows[["decision_ts", "event_start", "event_end", "tau_minutes", "strike_price", "spot_now", "realized_yes"]].copy()
    out["p_yes"] = preds["p_yes"].values
    out["p_no"] = preds["p_no"].values
    return out


def evaluate_horizon_models(
    minute_df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    output_dir: str | Path,
    model_names: list[str],
    edge_thresholds: list[float],
    train_events: int,
    calibration_events: int,
    validation_events: int,
    step_events: int | None = None,
    horizon: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    heartbeat_seconds: float | None = None,
    max_windows: int | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    events = events.copy()
    duplicated = _duplicate_columns(events)
    if duplicated:
        raise ValueError(f"Duplicate columns in horizon dataset: {duplicated}")
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True)
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True)
    events["event_end"] = pd.to_datetime(events["event_end"], utc=True)
    feature_df = build_horizon_event_features(minute_df, events)
    duplicated = _duplicate_columns(feature_df)
    if duplicated:
        raise ValueError(f"Duplicate columns in horizon dataset: {duplicated}")
    feature_columns = get_horizon_feature_columns(feature_df)
    candidate_splits = build_horizon_splits(
        events,
        train_events=train_events,
        calibration_events=calibration_events,
        validation_events=validation_events,
        step_events=step_events,
    )
    splits, split_meta = select_horizon_splits(candidate_splits, max_windows=max_windows)
    if not splits:
        raise ValueError("No rolling windows available for horizon evaluation")
    model_total = len(model_names)
    total_model_tasks = len(splits) * model_total
    validation_predictions = []
    for split_index, split in enumerate(splits, start=1):
        train_set = set(split["train_starts"])
        calib_set = set(split["calibration_starts"])
        val_set = set(split["validation_starts"])
        train_rows = feature_df[feature_df["event_start"].isin(train_set)].copy()
        calib_rows = feature_df[feature_df["event_start"].isin(calib_set)].copy()
        val_rows = feature_df[feature_df["event_start"].isin(val_set)].copy()
        current_window_label = _window_label(split)
        model_outputs = {}
        for model_index, model_name in enumerate(model_names, start=1):
            started = time.time()
            completed_model_tasks = (split_index - 1) * model_total + (model_index - 1)
            _emit_progress(
                progress_callback,
                {
                    "stage": "model_eval",
                    "mode": split_meta["mode"],
                    "current_horizon": horizon,
                    "current_model": model_name,
                    "total_windows": len(splits),
                    "windows_completed": split_index - 1,
                    "current_window_index": split_index,
                    "current_window_label": current_window_label,
                    "models_completed": model_index - 1,
                    "models_total": model_total,
                    "completed_model_tasks": completed_model_tasks,
                    "total_model_tasks": total_model_tasks,
                    "fold_id": split["fold_id"],
                    "events_completed": 0,
                    "events_total": len(calib_rows) + len(val_rows),
                    "elapsed_seconds": 0.0,
                    "eta_seconds": None,
                },
            )
            if model_name == "naive_directional":
                calib_preds = _predict_naive(calib_rows)
                val_preds = _predict_naive(val_rows)
            elif model_name == "market_implied":
                calib_preds = _predict_market_implied(calib_rows)
                val_preds = _predict_market_implied(val_rows)
            elif model_name == "gaussian_vol":
                calib_preds = _predict_parametric_engine(
                    "gaussian_vol",
                    minute_df,
                    calib_rows,
                    engine_kwargs={
                        "fit_window": 2000,
                        "vol_window": 120,
                        "min_periods": 60,
                        "n_sims": 0,
                        "progress_callback": progress_callback,
                        "progress_stage": "calibration",
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                )
                val_preds = _predict_parametric_engine(
                    "gaussian_vol",
                    minute_df,
                    val_rows,
                    engine_kwargs={
                        "fit_window": 2000,
                        "vol_window": 120,
                        "min_periods": 60,
                        "n_sims": 0,
                        "progress_callback": progress_callback,
                        "progress_stage": "validation",
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                )
            elif model_name == "ar_egarch":
                calib_preds = _predict_parametric_engine(
                    "ar_egarch",
                    minute_df,
                    calib_rows,
                    engine_kwargs={
                        "fit_window": 2000,
                        "residual_buffer_size": 1000,
                        "refit_every_n_events": 4,
                        "n_sims": 300,
                        "progress_callback": progress_callback,
                        "progress_stage": "calibration",
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                )
                val_preds = _predict_parametric_engine(
                    "ar_egarch",
                    minute_df,
                    val_rows,
                    engine_kwargs={
                        "fit_window": 2000,
                        "residual_buffer_size": 1000,
                        "refit_every_n_events": 4,
                        "n_sims": 300,
                        "progress_callback": progress_callback,
                        "progress_stage": "validation",
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                )
            elif model_name == "ar_only":
                calib_preds = _predict_ar_only(minute_df, calib_rows)
                val_preds = _predict_ar_only(minute_df, val_rows)
            elif model_name == "ar_vol":
                calib_preds = _predict_ar_vol(minute_df, calib_rows)
                val_preds = _predict_ar_vol(minute_df, val_rows)
            elif model_name == "lgbm":
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "train",
                        "mode": split_meta["mode"],
                        "current_horizon": horizon,
                        "current_model": model_name,
                        "total_windows": len(splits),
                        "windows_completed": split_index - 1,
                        "current_window_index": split_index,
                        "current_window_label": current_window_label,
                        "models_completed": model_index - 1,
                        "models_total": model_total,
                        "completed_model_tasks": completed_model_tasks,
                        "total_model_tasks": total_model_tasks,
                        "fold_id": split["fold_id"],
                        "events_completed": 0,
                        "events_total": len(train_rows),
                        "elapsed_seconds": time.time() - started,
                        "eta_seconds": None,
                    },
                )
                calib_preds = _predict_lgbm(feature_df, train_rows, calib_rows, engine_kwargs={"feature_columns": feature_columns})
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "calibration",
                        "mode": split_meta["mode"],
                        "current_horizon": horizon,
                        "current_model": model_name,
                        "total_windows": len(splits),
                        "windows_completed": split_index - 1,
                        "current_window_index": split_index,
                        "current_window_label": current_window_label,
                        "models_completed": model_index - 1,
                        "models_total": model_total,
                        "completed_model_tasks": completed_model_tasks,
                        "total_model_tasks": total_model_tasks,
                        "fold_id": split["fold_id"],
                        "events_completed": len(calib_rows),
                        "events_total": len(calib_rows),
                        "elapsed_seconds": time.time() - started,
                        "eta_seconds": None,
                    },
                )
                val_preds = _predict_lgbm(feature_df, train_rows, val_rows, engine_kwargs={"feature_columns": feature_columns})
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "validation",
                        "mode": split_meta["mode"],
                        "current_horizon": horizon,
                        "current_model": model_name,
                        "total_windows": len(splits),
                        "windows_completed": split_index - 1,
                        "current_window_index": split_index,
                        "current_window_label": current_window_label,
                        "models_completed": model_index - 1,
                        "models_total": model_total,
                        "completed_model_tasks": completed_model_tasks,
                        "total_model_tasks": total_model_tasks,
                        "fold_id": split["fold_id"],
                        "events_completed": len(val_rows),
                        "events_total": len(val_rows),
                        "elapsed_seconds": time.time() - started,
                        "eta_seconds": 0.0,
                    },
                )
            elif model_name == "blend_ar_lgbm":
                continue
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            if calib_preds.empty or val_preds.empty:
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "model_eval",
                        "mode": split_meta["mode"],
                        "current_horizon": horizon,
                        "current_model": model_name,
                        "total_windows": len(splits),
                        "windows_completed": split_index - 1,
                        "current_window_index": split_index,
                        "current_window_label": current_window_label,
                        "models_completed": model_index - 1,
                        "models_total": model_total,
                        "completed_model_tasks": completed_model_tasks,
                        "total_model_tasks": total_model_tasks,
                        "fold_id": split["fold_id"],
                        "events_completed": len(calib_rows) + len(val_rows),
                        "events_total": len(calib_rows) + len(val_rows),
                        "elapsed_seconds": time.time() - started,
                        "eta_seconds": 0.0,
                    },
                )
                continue
            model_outputs[model_name] = (calib_preds, val_preds)
            _emit_progress(
                progress_callback,
                {
                    "stage": "model_eval",
                    "mode": split_meta["mode"],
                    "current_horizon": horizon,
                    "current_model": model_name,
                    "total_windows": len(splits),
                    "windows_completed": split_index - 1,
                    "current_window_index": split_index,
                    "current_window_label": current_window_label,
                    "models_completed": model_index,
                    "models_total": model_total,
                    "completed_model_tasks": completed_model_tasks + 1,
                    "total_model_tasks": total_model_tasks,
                    "fold_id": split["fold_id"],
                    "events_completed": len(calib_rows) + len(val_rows),
                    "events_total": len(calib_rows) + len(val_rows),
                    "elapsed_seconds": time.time() - started,
                    "eta_seconds": 0.0,
                },
            )

        if "blend_ar_lgbm" in model_names and "ar_egarch" in model_outputs and "lgbm" in model_outputs:
            calib_ar, val_ar = model_outputs["ar_egarch"]
            calib_lgbm, val_lgbm = model_outputs["lgbm"]
            calib_merge = calib_ar.merge(calib_lgbm[["decision_ts", "p_yes"]], on="decision_ts", suffixes=("_ar", "_lgbm"))
            val_merge = val_ar.merge(val_lgbm[["decision_ts", "p_yes"]], on="decision_ts", suffixes=("_ar", "_lgbm"))
            if not calib_merge.empty and not val_merge.empty:
                blend_calib = calib_merge[["decision_ts", "event_start", "event_end", "tau_minutes", "strike_price", "spot_now", "realized_yes"]].copy()
                blend_calib["p_yes"] = 0.5 * calib_merge["p_yes_ar"] + 0.5 * calib_merge["p_yes_lgbm"]
                blend_calib["p_no"] = 1.0 - blend_calib["p_yes"]
                blend_val = val_merge[["decision_ts", "event_start", "event_end", "tau_minutes", "strike_price", "spot_now", "realized_yes"]].copy()
                blend_val["p_yes"] = 0.5 * val_merge["p_yes_ar"] + 0.5 * val_merge["p_yes_lgbm"]
                blend_val["p_no"] = 1.0 - blend_val["p_yes"]
                model_outputs["blend_ar_lgbm"] = (blend_calib, blend_val)

        for model_name, (calib_preds, val_preds) in model_outputs.items():
            calibrator = _fit_logistic_calibrator(calib_preds["realized_yes"], calib_preds["p_yes"])
            scored = val_preds.copy()
            scored["p_yes_raw"] = scored["p_yes"]
            scored["p_yes"] = _apply_logistic_calibrator(scored["p_yes"], calibrator)
            scored["p_no"] = 1.0 - scored["p_yes"]
            scored["fold_id"] = split["fold_id"]
            scored["model_name"] = model_name
            validation_predictions.append(scored)

    if not validation_predictions:
        raise ValueError("No validation predictions were produced")
    validation_df = pd.concat(validation_predictions, ignore_index=True)
    validation_df["tau_bucket"] = validation_df["tau_minutes"].apply(_tau_bucket)
    validation_df["moneyness_bucket"] = validation_df.apply(lambda row: _moneyness_bucket(row["spot_now"], row["strike_price"]), axis=1)
    validation_df["month"] = pd.to_datetime(validation_df["decision_ts"], utc=True).dt.strftime("%Y-%m")

    summary_rows = []
    tau_rows = []
    month_rows = []
    for model_name, group in validation_df.groupby("model_name", observed=False):
        metrics = _core_metrics(group["realized_yes"], group["p_yes"])
        tradeability = {}
        for threshold in edge_thresholds:
            prefix = f"edge_{threshold:.3f}".replace(".", "p")
            tradeability.update({f"{prefix}_{k}": v for k, v in _tradeability_metrics(group, threshold).items() if k != "edge_threshold"})
        summary_rows.append({"model_name": model_name, **metrics, **tradeability})
        for tau_bucket, bucket_group in group.groupby("tau_bucket", observed=False):
            tau_rows.append({"model_name": model_name, "tau_bucket": tau_bucket, **_core_metrics(bucket_group["realized_yes"], bucket_group["p_yes"])})
        for month, month_group in group.groupby("month", observed=False):
            if len(month_group) >= MONTHLY_MIN_OBS:
                month_rows.append({"model_name": model_name, "month": month, **_core_metrics(month_group["realized_yes"], month_group["p_yes"])})

    summary_df = pd.DataFrame(summary_rows).sort_values(["brier_score", "log_loss"]).reset_index(drop=True)
    tau_df = pd.DataFrame(tau_rows).sort_values(["tau_bucket", "brier_score"]).reset_index(drop=True)
    month_df = pd.DataFrame(month_rows).sort_values(["month", "brier_score"]).reset_index(drop=True) if month_rows else pd.DataFrame()

    summary_df.to_csv(output_path / "summary.csv", index=False)
    tau_df.to_csv(output_path / "by_tau_bucket.csv", index=False)
    if not month_df.empty:
        month_df.to_csv(output_path / "by_month.csv", index=False)
    validation_df.to_csv(output_path / "validation_predictions.csv", index=False)

    notes = [
        "# Horizon Comparison Notes",
        "",
        f"- observations: {len(validation_df)}",
        f"- folds: {len(splits)}",
        f"- total_candidate_windows: {split_meta['total_candidate_windows']}",
        f"- evaluated_windows: {split_meta['evaluated_windows']}",
        f"- evaluation_mode: {split_meta['mode']}",
        f"- selection_policy: {split_meta['selection_policy']}",
        f"- models: {', '.join(sorted(summary_df['model_name'].tolist()))}",
        "",
        "Top models by Brier:",
    ]
    for _, row in summary_df.head(5).iterrows():
        notes.append(f"- {row['model_name']}: brier={row['brier_score']:.6f}, log_loss={row['log_loss']:.6f}, accuracy={row['accuracy']:.4f}")
    (output_path / "notes.md").write_text("\n".join(notes), encoding="utf-8")
    (output_path / "metrics.json").write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
                "tau_bucket": tau_df.to_dict(orient="records"),
                "months": month_df.to_dict(orient="records") if not month_df.empty else [],
                "fold_count": len(splits),
                "window_selection": split_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_df,
        "tau": tau_df,
        "months": month_df,
        "validation_predictions": validation_df,
        "fold_count": len(splits),
        "window_selection": split_meta,
    }


def run_horizon_comparison(
    minute_df: pd.DataFrame,
    *,
    horizons: list[str],
    output_dir: str | Path,
    model_names: list[str],
    decision_step_overrides: dict[str, int] | None = None,
    train_events: int,
    calibration_events: int,
    validation_events: int,
    step_events: int | None = None,
    edge_thresholds: list[float] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    heartbeat_seconds: float | None = None,
    max_windows: int | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    edge_thresholds = edge_thresholds or [0.05, 0.10]
    by_model_by_horizon_rows = []
    all_metrics = {}
    total_horizon_models = len(horizons) * len(model_names)
    horizon_model_offset = 0
    started = time.time()
    for horizon in horizons:
        horizon_dir = output_path / horizon
        step_minutes = (decision_step_overrides or {}).get(horizon)
        events = build_fixed_horizon_event_dataset(minute_df, horizon=horizon, decision_step_minutes=step_minutes)
        _emit_progress(
            progress_callback,
            {
                "stage": "horizon_setup",
                "mode": "full" if max_windows is None else ("single_split" if int(max_windows) == 1 else "limited_windows"),
                "current_horizon": horizon,
                "current_model": None,
                "models_completed": horizon_model_offset,
                "models_total": total_horizon_models,
                "total_windows": None,
                "windows_completed": 0,
                "current_window_index": None,
                "current_window_label": None,
                "completed_model_tasks": horizon_model_offset,
                "total_model_tasks": None,
                "events_completed": 0,
                "events_total": len(events),
                "elapsed_seconds": time.time() - started,
                "eta_seconds": None,
            },
        )
        result = evaluate_horizon_models(
            minute_df=minute_df,
            events=events,
            output_dir=horizon_dir,
            model_names=model_names,
            edge_thresholds=edge_thresholds,
            train_events=train_events,
            calibration_events=calibration_events,
            validation_events=validation_events,
            step_events=step_events,
            horizon=horizon,
            progress_callback=progress_callback,
            heartbeat_seconds=heartbeat_seconds,
            max_windows=max_windows,
        )
        summary = result["summary"].copy()
        summary["horizon"] = horizon
        by_model_by_horizon_rows.append(summary)
        all_metrics[horizon] = {
            "summary_path": str(horizon_dir / "summary.csv"),
            "tau_path": str(horizon_dir / "by_tau_bucket.csv"),
            "notes_path": str(horizon_dir / "notes.md"),
            "fold_count": result["fold_count"],
            "sample_count": int(len(result["validation_predictions"])),
            "window_selection": result["window_selection"],
        }
        write_dataframe(events, horizon_dir / "events.parquet")
        horizon_model_offset += len(model_names)
        _emit_progress(
            progress_callback,
            {
                "stage": "horizon_complete",
                "mode": result["window_selection"]["mode"],
                "current_horizon": horizon,
                "current_model": None,
                "models_completed": horizon_model_offset,
                "models_total": total_horizon_models,
                "total_windows": result["window_selection"]["evaluated_windows"],
                "windows_completed": result["window_selection"]["evaluated_windows"],
                "current_window_index": result["window_selection"]["evaluated_windows"],
                "current_window_label": None,
                "completed_model_tasks": horizon_model_offset,
                "total_model_tasks": None,
                "events_completed": len(result["validation_predictions"]),
                "events_total": len(events),
                "elapsed_seconds": time.time() - started,
                "eta_seconds": 0.0,
            },
        )

    combined = pd.concat(by_model_by_horizon_rows, ignore_index=True).sort_values(["horizon", "brier_score", "log_loss"]).reset_index(drop=True)
    combined.to_csv(output_path / "by_model_by_horizon.csv", index=False)
    summary_ranked = combined.sort_values(["brier_score", "log_loss", "accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    summary_ranked.to_csv(output_path / "summary.csv", index=False)
    notes_lines = [
        "# Horizon Comparison Summary",
        "",
        "Question focus:",
        "- Is 4h better than 1h?",
        "- Is 1d better than 4h?",
        "- Does EGARCH gain relative ground as horizon increases?",
        "- Does LightGBM beat the econometric models after calibration?",
        "- Which horizon has the best balance of edge quality and trade frequency?",
        "",
        "Top rows:",
    ]
    for _, row in summary_ranked.head(10).iterrows():
        notes_lines.append(
            f"- {row['horizon']} / {row['model_name']}: brier={row['brier_score']:.6f}, log_loss={row['log_loss']:.6f}, accuracy={row['accuracy']:.4f}"
        )
    (output_path / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return {
        "summary_path": str(output_path / "summary.csv"),
        "by_model_by_horizon_path": str(output_path / "by_model_by_horizon.csv"),
        "metrics_path": str(output_path / "metrics.json"),
        "notes_path": str(output_path / "notes.md"),
    }
