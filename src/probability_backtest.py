"""Offline walk-forward backtester for hourly BTC settlement probabilities."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .hourly_event_dataset import build_hourly_event_dataset, filter_event_dataset, write_dataframe
from .probability_engine_factory import build_probability_engine


def _load_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format: {table_path}")


def load_event_dataset(path: str | Path) -> pd.DataFrame:
    df = _load_table(path)
    for column in ["decision_ts", "event_hour_start", "event_hour_end"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True)
    return df


def _coerce_close_series(close_series: pd.Series) -> pd.Series:
    if not isinstance(close_series.index, pd.DatetimeIndex):
        raise TypeError("close_series must use a DatetimeIndex")
    if close_series.index.tz is None or str(close_series.index.tz) != "UTC":
        raise ValueError("close_series index must be timezone-aware UTC")
    return close_series.sort_index().dropna().astype(float)


def _fit_engine_from_history(
    engine,
    history_close: pd.Series,
    spot_now: float,
    decision_ts: pd.Timestamp,
):
    if hasattr(engine, "fit_history"):
        engine.fit_history(history_close)
    else:
        engine.update_with_price_series(history_close)
    if hasattr(engine, "observe_bar"):
        engine.observe_bar(float(spot_now), ts=decision_ts, finalized=False)
    else:
        engine.update_on_bar(float(spot_now), ts=decision_ts, include_in_buffer=False)
    return engine


def _normalize_ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _scalar_event_value(event: pd.Series, key: str, default=None):
    if key not in event.index:
        return default
    value = event[key]
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        return value.iloc[0]
    return value


def _base_result_row(event: pd.Series, decision_ts: pd.Timestamp, training_window_length: int, used_refit: bool) -> dict:
    return {
        "decision_ts": decision_ts,
        "event_hour_start": _normalize_ts(_scalar_event_value(event, "event_hour_start")),
        "event_hour_end": _normalize_ts(_scalar_event_value(event, "event_hour_end")),
        "minute_in_hour": int(_scalar_event_value(event, "minute_in_hour", 0)),
        "tau_minutes": int(_scalar_event_value(event, "tau_minutes", 0)),
        "strike_price": float(_scalar_event_value(event, "strike_price", 0.0)),
        "spot_now": float(_scalar_event_value(event, "spot_now", 0.0)),
        "settlement_price": float(_scalar_event_value(event, "settlement_price", 0.0)),
        "p_yes": np.nan,
        "p_no": np.nan,
        "realized_yes": int(_scalar_event_value(event, "realized_yes", 0)),
        "diagnostics_json": None,
        "training_window_length": int(training_window_length),
        "fit_failed": False,
        "simulation_failed": False,
        "skip_reason": None,
        "used_refit": bool(used_refit),
    }


def run_probability_backtest(
    close_series: pd.Series,
    events: pd.DataFrame,
    fit_window: int = 2000,
    residual_buffer_size: int = 2000,
    n_sims: int = 2000,
    seed: int | None = None,
    min_history: int | None = None,
    refit_every_n_events: int = 12,
    engine_name: str = "ar_egarch",
    engine_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    use_jump_augmentation: bool = False,
    jump_augment_weight: int = 10,
    model_factory: Callable[..., object] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    progress_stage: str | None = None,
    heartbeat_seconds: float | None = None,
) -> pd.DataFrame:
    close_series = _coerce_close_series(close_series)
    duplicated = events.columns[events.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate columns passed to probability backtest: {duplicated}")
    events = events.sort_values("decision_ts").reset_index(drop=True)
    factory = model_factory
    min_history = max(fit_window, 1000) if min_history is None else int(min_history)
    refit_every_n_events = max(1, int(refit_every_n_events))
    engine_kwargs = engine_kwargs or {}
    model_kwargs = model_kwargs or {}
    rows: list[dict] = []
    engine = None
    events_since_refit = 0
    last_consumed_close_ts: pd.Timestamp | None = None
    started = time.time()
    last_heartbeat = started
    total_events = len(events)

    def emit_progress(processed_events: int, *, current_decision_ts: pd.Timestamp | None = None) -> None:
        nonlocal last_heartbeat
        if progress_callback is None:
            return
        now = time.time()
        if heartbeat_seconds is not None and processed_events < total_events and (now - last_heartbeat) < heartbeat_seconds:
            return
        elapsed = now - started
        rate = processed_events / elapsed if elapsed > 0 else 0.0
        remaining = max(total_events - processed_events, 0)
        eta = (remaining / rate) if rate > 0 else None
        progress_callback(
            {
                "stage": progress_stage or "probability_backtest",
                "events_completed": int(processed_events),
                "events_total": int(total_events),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "decision_ts": None if current_decision_ts is None else current_decision_ts.isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        last_heartbeat = now

    emit_progress(0)
    for row_number, event in events.iterrows():
        decision_ts = _normalize_ts(event["decision_ts"])
        history_close = close_series[close_series.index < decision_ts]
        should_refit = engine is None or events_since_refit >= refit_every_n_events
        row = _base_result_row(
            event=event,
            decision_ts=decision_ts,
            training_window_length=min(len(history_close), fit_window),
            used_refit=should_refit,
        )
        if len(history_close) < min_history:
            row["skip_reason"] = f"insufficient_history:{len(history_close)}<{min_history}"
            rows.append(row)
            emit_progress(row_number + 1, current_decision_ts=decision_ts)
            continue
        try:
            if should_refit:
                if len(history_close) > fit_window:
                    history_close = history_close.iloc[-fit_window:]
                if factory is not None:
                    engine = factory(
                        residual_buffer_size=residual_buffer_size,
                        fit_window=fit_window,
                        **model_kwargs,
                    )
                else:
                    resolved_engine_kwargs = {
                        "fit_window": fit_window,
                        "residual_buffer_size": residual_buffer_size,
                        **engine_kwargs,
                    }
                    if engine_name == "ar_egarch":
                        resolved_engine_kwargs.update(model_kwargs)
                        resolved_engine_kwargs["use_jump_augmentation"] = use_jump_augmentation
                        resolved_engine_kwargs["jump_augment_weight"] = jump_augment_weight
                    engine = build_probability_engine(engine_name, **resolved_engine_kwargs)
                engine = _fit_engine_from_history(
                    engine=engine,
                    history_close=history_close,
                    spot_now=float(event["spot_now"]),
                    decision_ts=decision_ts,
                )
                events_since_refit = 0
                last_consumed_close_ts = history_close.index[-1]
                row["training_window_length"] = int(len(history_close))
            else:
                assert engine is not None
                incremental_closes = close_series[(close_series.index > last_consumed_close_ts) & (close_series.index < decision_ts)]
                for update_ts, close_price in incremental_closes.items():
                    if hasattr(engine, "observe_bar"):
                        engine.observe_bar(float(close_price), ts=update_ts, finalized=True)
                    else:
                        engine.update_on_bar(float(close_price), ts=update_ts, include_in_buffer=True)
                if not incremental_closes.empty:
                    last_consumed_close_ts = incremental_closes.index[-1]
                if hasattr(engine, "observe_bar"):
                    engine.observe_bar(float(event["spot_now"]), ts=decision_ts, finalized=False)
                else:
                    engine.update_on_bar(float(event["spot_now"]), ts=decision_ts, include_in_buffer=False)
        except Exception as exc:
            row["fit_failed"] = True
            row["skip_reason"] = f"fit_failed:{type(exc).__name__}:{exc}"
            rows.append(row)
            engine = None
            events_since_refit = 0
            last_consumed_close_ts = None
            emit_progress(row_number + 1, current_decision_ts=decision_ts)
            continue

        try:
            assert engine is not None
            if hasattr(engine, "predict"):
                result = engine.predict(
                    strike_price=float(event["strike_price"]),
                    tau_minutes=int(event["tau_minutes"]),
                    n_sims=n_sims,
                    seed=None if seed is None else seed + row_number,
                )
            else:
                if use_jump_augmentation:
                    result = engine.probability_up(
                        float(event["strike_price"]),
                        minutes=int(event["tau_minutes"]),
                        n_sims=n_sims,
                        seed=None if seed is None else seed + row_number,
                        include_jumps=True,
                        jump_augment_weight=jump_augment_weight,
                    )
                else:
                    result = engine.simulate_probability(
                        target_price=float(event["strike_price"]),
                        tau_minutes=int(event["tau_minutes"]),
                        n_sims=n_sims,
                        seed=None if seed is None else seed + row_number,
                    )
        except Exception as exc:
            row["simulation_failed"] = True
            row["skip_reason"] = f"simulation_failed:{type(exc).__name__}:{exc}"
            rows.append(row)
            events_since_refit += 1
            emit_progress(row_number + 1, current_decision_ts=decision_ts)
            continue

        if result.get("failed") or result.get("simulation_failed"):
            row["simulation_failed"] = True
            row["skip_reason"] = f"simulation_failed:{result.get('reason') or result.get('failure_reason', 'unknown')}"
            diagnostics = engine.get_diagnostics() if hasattr(engine, "get_diagnostics") else {}
            row["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)
            rows.append(row)
            events_since_refit += 1
            emit_progress(row_number + 1, current_decision_ts=decision_ts)
            continue

        p_yes = float(result["p_yes"] if result.get("p_yes") is not None else result["p_hat"])
        row["p_yes"] = p_yes
        row["p_no"] = float(1.0 - p_yes)
        diagnostics = engine.get_diagnostics() if hasattr(engine, "get_diagnostics") else {}
        row["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)
        rows.append(row)
        events_since_refit += 1
        emit_progress(row_number + 1, current_decision_ts=decision_ts)
    return pd.DataFrame(rows)


def build_or_load_events(
    close_minute_df: pd.DataFrame,
    events_path: str | Path | None = None,
    decision_step_minutes: int = 5,
    start: str | None = None,
    end: str | None = None,
    max_events: int | None = None,
) -> pd.DataFrame:
    if events_path is not None:
        events = load_event_dataset(events_path)
    else:
        events = build_hourly_event_dataset(close_minute_df, decision_step_minutes=decision_step_minutes)
    return filter_event_dataset(events, start=start, end=end, max_events=max_events)


def save_backtest_results(results: pd.DataFrame, output_path: str | Path) -> None:
    write_dataframe(results, output_path)
