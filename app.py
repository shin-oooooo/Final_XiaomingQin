"""Streamlit entry point.

Workflow (per Exam_question.txt):
  1. User uploads .xlsx with column 'y' (and optional 'date').
  2. App AUTOMATICALLY produces ONE one-step-ahead forecast for y_{N+1}.
  3. App AUTOMATICALLY runs an 80/20 walk-forward backtest.
  4. User downloads the test-segment forecasts as .xlsx.

There is intentionally NO control to pick a forecasting method. The
method (a fixed naive + theta + damped-Holt ensemble) is a project
decision baked into the codebase.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    build_backtest_dataframe,
    forecast_next,
    read_excel_input,
    run_backtest,
    to_excel_bytes,
    validate_series,
)

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon=None,
    layout="wide",
)

st.title("Time Series Forecasting Web App")
st.caption(
    "Upload an Excel file with column **y** (optional **date**). "
    "The app immediately produces a one-step-ahead forecast for y_(N+1) and "
    "an 80/20 walk-forward backtest. No method selection — everything is automatic."
)

uploaded = st.file_uploader(
    "Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False
)

if uploaded is None:
    st.info("Awaiting upload. The file's first sheet will be used.")
    st.stop()

try:
    df_raw = read_excel_input(uploaded)
    series = validate_series(df_raw)
except Exception as exc:  # noqa: BLE001
    st.error(f"Input error: {exc}")
    st.stop()

st.success(
    f"Loaded series: N = {series.n} samples"
    + (" (date column detected)" if series.has_dates else " (no date column)")
)

# ===== Part 1: one-step-ahead, automatic =====
st.subheader("Part 1 — One-step-ahead forecast")

with st.spinner("Forecasting y_(N+1)…"):
    fc = forecast_next(series)

c1, c2 = st.columns([1, 2])
c1.metric(
    label="Forecast y_(N+1)",
    value=f"{fc.y_hat_next:.6f}",
    help=(f"Next date: {fc.next_date.date()}" if fc.next_date is not None else None),
)

with c2:
    last_n = min(60, series.n)
    tail_y = series.y[-last_n:]
    if series.has_dates:
        tail_x = list(series.dates.iloc[-last_n:])
        next_x = fc.next_date if fc.next_date is not None else None
    else:
        tail_x = list(range(series.n - last_n, series.n))
        next_x = series.n
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=tail_x, y=tail_y, mode="lines", name="History (tail)")
    )
    fig1.add_trace(
        go.Scatter(
            x=[next_x],
            y=[fc.y_hat_next],
            mode="markers",
            marker=dict(size=12, symbol="star"),
            name="Forecast y_(N+1)",
        )
    )
    fig1.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        xaxis_title="time",
        yaxis_title="y",
    )
    st.plotly_chart(fig1, use_container_width=True)

# ===== Part 2: walk-forward backtest, automatic =====
st.subheader("Part 2 — Walk-forward backtest (80% train / 20% test)")

with st.spinner("Running walk-forward backtest…"):
    bt = run_backtest(series, train_ratio=0.8)

n_test = len(bt.y_pred)

if series.has_dates:
    train_dates = series.dates.iloc[: bt.train_size]
    test_dates_disp = series.dates.iloc[bt.train_size :]
    train_range = (
        f"{train_dates.iloc[0].date()} → {train_dates.iloc[-1].date()}"
    )
    test_range = (
        f"{test_dates_disp.iloc[0].date()} → {test_dates_disp.iloc[-1].date()}"
    )
else:
    train_range = f"index 0 → {bt.train_size - 1}"
    test_range = f"index {bt.train_size} → {bt.train_size + n_test - 1}"

c_top1, c_top2 = st.columns(2)
c_top1.metric("Train size", f"{bt.train_size}", help=train_range)
c_top2.metric("Test size", f"{n_test}", help=test_range)
st.caption(
    f"**Train period**: {train_range}    |    **Test period**: {test_range}"
)

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{bt.rmse:.6f}")
m2.metric("MAE", f"{bt.mae:.6f}")
m3.metric("MAPE (%)", f"{bt.mape:.4f}")

if bt.test_dates is not None:
    bt_x = list(bt.test_dates)
else:
    bt_x = list(range(bt.train_size, bt.train_size + n_test))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=bt_x, y=bt.y_true, mode="lines", name="Actual"))
fig2.add_trace(
    go.Scatter(x=bt_x, y=bt.y_pred, mode="lines", name="Predicted", line=dict(dash="dash"))
)
fig2.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.0),
    xaxis_title="time (test segment)",
    yaxis_title="y",
)
st.plotly_chart(fig2, use_container_width=True)

# ===== Excel export =====
out_df: pd.DataFrame = build_backtest_dataframe(
    y_pred=bt.y_pred,
    test_dates=bt.test_dates,
)
xlsx_bytes = to_excel_bytes(out_df)

st.download_button(
    label="Download backtest forecast (.xlsx)",
    data=xlsx_bytes,
    file_name="backtest_forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Preview of output Excel (first 20 rows)"):
    st.dataframe(out_df.head(20), use_container_width=True)

st.caption(
    "Anti-data-leakage: every test-segment forecast is computed using only y[:t] "
    "(strict slice; t excluded). See `forecasting/backtest.py`."
)
