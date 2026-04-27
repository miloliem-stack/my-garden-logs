from __future__ import annotations

from pathlib import Path

import pandas as pd

from .hmm_dataset import read_table


def state_color(state: object) -> str:
    value = str(state).lower()
    if "quiet" in value or ("state_0" in value and "confident" in value):
        return "rgba(80, 180, 100, 0.18)"
    if "trend" in value or "smooth_directional" in value or "state_1" in value:
        return "rgba(70, 140, 220, 0.16)"
    if "whipsaw" in value or "high_entropy" in value:
        return "rgba(220, 80, 80, 0.16)"
    if "polarized_tail" in value:
        return "rgba(160, 80, 180, 0.16)"
    if "transition" in value or "uncertain" in value:
        return "rgba(230, 200, 70, 0.18)"
    return "rgba(150, 150, 150, 0.12)"


def compress_regime_segments(df: pd.DataFrame, *, state_column: str = "regime_policy_state") -> list[dict]:
    if df.empty:
        return []
    segments = []
    current = df.iloc[0][state_column] if state_column in df.columns else "unknown"
    start = df.iloc[0]["timestamp"]
    prev = start
    for _, row in df.iloc[1:].iterrows():
        state = row[state_column] if state_column in row else "unknown"
        if state != current:
            segments.append({"state": current, "start": start, "end": prev, "color": state_color(current)})
            current = state
            start = row["timestamp"]
        prev = row["timestamp"]
    segments.append({"state": current, "start": start, "end": prev, "color": state_color(current)})
    return segments


def build_regime_overlay_html(
    input_path: str | Path,
    output_path: str | Path,
    *,
    png_output: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    title: str = "BTC HMM Regime Overlay",
    max_rows: int | None = None,
    state_column: str = "regime_policy_state",
    confidence_column: str = "hmm_map_confidence",
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = read_table(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:]
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required plot columns: {missing}")
    panel_rows = 1 + sum(col in df.columns for col in [confidence_column, "hmm_entropy", "hmm_next_same_state_confidence"])
    fig = make_subplots(rows=panel_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.62] + [0.12] * (panel_rows - 1))
    hover_cols = [col for col in [state_column, confidence_column, "hmm_entropy", "hmm_next_same_state_confidence", "p_yes", "q_yes_ask", "q_no_ask", "expected_log_growth_conservative", "conservative_expected_log_growth"] if col in df.columns]
    custom = df[hover_cols].astype(str).to_numpy() if hover_cols else None
    hovertemplate = "ts=%{x}<br>close=%{close}" + "".join([f"<br>{col}=%{{customdata[{idx}]}}" for idx, col in enumerate(hover_cols)]) + "<extra></extra>"
    fig.add_trace(
        go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], customdata=custom, hovertemplate=hovertemplate, name="BTC"),
        row=1,
        col=1,
    )
    for segment in compress_regime_segments(df, state_column=state_column):
        fig.add_vrect(x0=segment["start"], x1=segment["end"], fillcolor=segment["color"], line_width=0, layer="below", row=1, col=1)
    if "strike_price" in df.columns and df["strike_price"].notna().any():
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["strike_price"], mode="lines", line={"dash": "dash"}, name="strike"), row=1, col=1)
    if "vanilla_action" in df.columns:
        entries = df[df["vanilla_action"].astype(str).str.startswith("buy", na=False)]
        fig.add_trace(go.Scatter(x=entries["timestamp"], y=entries["close"], mode="markers", marker={"symbol": "triangle-up", "size": 8}, name="entries"), row=1, col=1)
    row = 2
    for col_name in [confidence_column, "hmm_entropy", "hmm_next_same_state_confidence"]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col_name], mode="lines", name=col_name), row=row, col=1)
            row += 1
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_white", height=850)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    if png_output:
        try:
            fig.write_image(str(png_output))
        except Exception:
            pass
    return output_path

