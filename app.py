"""
app.py — Streamlit dashboard for PLTA Grindulu inflow prediction.

Run:
    streamlit run app.py

Layout
------
Sidebar   : model parameters display + run-training button
Page 1    : Historical inflow overview (chart)
Page 2    : 7-day forecast input form + q10/q50/q90 Plotly chart + metrics table
"""

import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from charts import (
    chart_fixed_vs_variable_speed,
    chart_observed_vs_predicted,
    chart_power_output,
    chart_prediction_interval,
    chart_quantile_prediction,
    chart_turbine_efficiency,
)
from evaluate import collect_predictions, compute_crps, compute_picp, compute_pinaw
from model import Seq2SeqQuantile
from preprocess import (
    DATA_PATH,
    ENC_LEN,
    PRED_LEN,
    SAVE_DIR,
    load_data,
    make_inference_sequence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH        = os.path.join(SAVE_DIR, "best_model.pt")
FEATURE_SCALER    = os.path.join(SAVE_DIR, "feature_scaler.pkl")
Q_SCALER_PATH     = os.path.join(SAVE_DIR, "q_scaler.pkl")
HISTORY_PATH      = os.path.join(SAVE_DIR, "train_history.csv")

QUANTILE_COLORS = {
    "q90": "#1f77b4",
    "q50": "#2ca02c",
    "q10": "#ff7f0e",
}

st.set_page_config(
    page_title="PLTA Grindulu — Inflow Prediction",
    page_icon="💧",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model() -> Seq2SeqQuantile | None:
    if not os.path.exists(MODEL_PATH):
        return None
    model = Seq2SeqQuantile()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_data(show_spinner="Loading historical data…")
def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_q_scaler():
    if not os.path.exists(Q_SCALER_PATH):
        return None
    with open(Q_SCALER_PATH, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def predict(
    model: Seq2SeqQuantile,
    p_das_history: np.ndarray,
    q_total_history: np.ndarray,
    p_das_forecast: np.ndarray,
    q_scaler,
) -> np.ndarray:
    """Run inference and return predictions in m³/s.

    Returns:
        preds : (pred_len, 3) — [q10, q50, q90] per forecast day (m³/s)
    """
    enc_t, dec_t = make_inference_sequence(
        p_das_history, q_total_history, p_das_forecast
    )
    out_scaled = model(enc_t, dec_t).squeeze(0).numpy()  # (pred_len, 3)

    # De-normalise
    n, q = out_scaled.shape
    out_orig = q_scaler.inverse_transform(out_scaled.reshape(-1, 1)).reshape(n, q)
    out_orig = np.sort(out_orig, axis=-1)  # enforce q10 ≤ q50 ≤ q90
    return out_orig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("💧 PLTA Grindulu")
    st.sidebar.caption("Bi-LSTM Seq2Seq Quantile Regression")
    st.sidebar.divider()

    model_ready = os.path.exists(MODEL_PATH)
    if model_ready:
        st.sidebar.success("Model trained ✔")
    else:
        st.sidebar.warning("Model not trained yet")
        if st.sidebar.button("▶ Train Model Now", type="primary"):
            with st.sidebar:
                with st.spinner("Training… this may take a few minutes."):
                    result = subprocess.run(
                        [sys.executable, "train.py", "--epochs", "100"],
                        capture_output=True, text=True,
                        cwd=os.path.dirname(__file__),
                    )
                if result.returncode == 0:
                    st.success("Training complete!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Training failed.")
                    st.code(result.stderr[-3000:])

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Dataset** : {len(df):,} days  \n"
        f"**Period**  : {df['date'].min().date()} → {df['date'].max().date()}  \n"
        f"**Look-back**: {ENC_LEN} days  \n"
        f"**Horizon** : {PRED_LEN} days  \n"
        f"**Quantiles**: 10 % / 50 % / 90 %"
    )

    if os.path.exists(HISTORY_PATH):
        st.sidebar.divider()
        hist = pd.read_csv(HISTORY_PATH)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["epoch"], y=hist["train_loss"], name="Train", mode="lines"
        ))
        fig.add_trace(go.Scatter(
            x=hist["epoch"], y=hist["val_loss"], name="Val", mode="lines"
        ))
        fig.update_layout(
            title="Training History", height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h"),
        )
        st.sidebar.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Historical Overview
# ---------------------------------------------------------------------------

def page_historical(df: pd.DataFrame):
    st.header("Historical Inflow — DAS Grindulu")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days",  f"{len(df):,}")
    col2.metric("Mean Q (m³/s)", f"{df['q_total'].mean():.2f}")
    col3.metric("Max Q (m³/s)",  f"{df['q_total'].max():.2f}")
    col4.metric("Min Q (m³/s)",  f"{df['q_total'].min():.3f}")

    # Date range filter
    c1, c2 = st.columns(2)
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    start = c1.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
    end   = c2.date_input("To",   value=date_max, min_value=date_min, max_value=date_max)

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    view = df[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["date"], y=view["q_total"],
        mode="lines", name="Q Total (m³/s)",
        line=dict(color="#2ca02c", width=1),
    ))
    fig.add_trace(go.Bar(
        x=view["date"], y=view["p_das"],
        name="P_DAS (mm)", yaxis="y2",
        marker_color="rgba(31,119,180,0.4)",
    ))
    fig.update_layout(
        yaxis=dict(title="Q Total (m³/s)"),
        yaxis2=dict(title="Rainfall P_DAS (mm)", overlaying="y", side="right", autorange="reversed"),
        legend=dict(orientation="h"),
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(view[["date", "p_das", "pe", "q_runoff", "q_baseflow", "q_total"]], use_container_width=True)


# ---------------------------------------------------------------------------
# Page: 7-Day Forecast
# ---------------------------------------------------------------------------

def page_forecast(df: pd.DataFrame, model: Seq2SeqQuantile | None, q_scaler):
    st.header("7-Day Inflow Forecast")

    if model is None:
        st.error("No trained model found.  Train the model first (sidebar).")
        return

    if q_scaler is None:
        st.error("Scalers not found. Re-train the model.")
        return

    # ------------------------------------------------------------------
    # Date picker — any date, including future (beyond dataset)
    # ------------------------------------------------------------------
    import calendar
    data_end   = df["date"].max()
    data_start = df["date"].min() + pd.Timedelta(days=ENC_LEN)

    st.markdown(
        "Pilih **tanggal mulai forecast** — bisa tanggal historis (dalam dataset) "
        "maupun tanggal masa depan (2025, 2026, dst)."
    )

    col_y, col_m, col_d = st.columns(3)
    # Allow years up to current year + 5 for future forecasting
    years  = list(range(data_start.year, pd.Timestamp.now().year + 6))
    months = list(range(1, 13))

    # Default to today or dataset end, whichever is later
    default_ref = max(pd.Timestamp.now().normalize(), data_end + pd.Timedelta(days=1))
    default_year_idx  = years.index(default_ref.year) if default_ref.year in years else len(years) - 1

    sel_year  = col_y.selectbox("Tahun", years, index=default_year_idx)
    sel_month = col_m.selectbox(
        "Bulan", months,
        index=default_ref.month - 1,
        format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
    )

    max_day     = calendar.monthrange(sel_year, sel_month)[1]
    default_day = min(default_ref.day, max_day)
    sel_day = col_d.selectbox("Hari", list(range(1, max_day + 1)), index=default_day - 1)

    try:
        ref_date = pd.Timestamp(sel_year, sel_month, sel_day)
    except Exception:
        st.error("Tanggal tidak valid.")
        return

    # Enforce minimum (need ENC_LEN days of look-back in dataset)
    if ref_date < data_start:
        st.error(f"Tanggal minimal adalah {data_start.date()} (butuh 30 hari look-back dari dataset).")
        return

    is_future   = ref_date > data_end          # forecast start is beyond dataset
    has_actuals = (ref_date + pd.Timedelta(days=PRED_LEN - 1)) <= data_end  # actuals available

    # Info banner
    if is_future:
        st.info(
            f"Tanggal **{ref_date.date()}** berada di luar dataset (dataset berakhir {data_end.date()}). "
            f"Look-back otomatis menggunakan 30 hari terakhir dataset. "
            f"Input curah hujan forecast secara manual."
        )
    else:
        st.caption(
            f"Forecast: **{ref_date.date()}** s.d. **{(ref_date + pd.Timedelta(days=PRED_LEN-1)).date()}** | "
            f"Look-back: {(ref_date - pd.Timedelta(days=ENC_LEN)).date()} s.d. {(ref_date - pd.Timedelta(days=1)).date()}"
        )

    # ------------------------------------------------------------------
    # Look-back window
    # ------------------------------------------------------------------
    if is_future:
        # Use last 30 days of dataset
        lb_end = df.tail(ENC_LEN).reset_index(drop=True)
    else:
        lb_end = df[df["date"] < ref_date].tail(ENC_LEN).reset_index(drop=True)

    if len(lb_end) < ENC_LEN:
        st.error(f"Data look-back tidak cukup (hanya {len(lb_end)} hari tersedia, butuh {ENC_LEN}).")
        return

    st.subheader("Look-back window (30 hari sebelum tanggal referensi — bisa diedit)")
    look_back_df = lb_end[["date", "p_das", "q_total"]].copy()
    look_back_df["date"] = look_back_df["date"].dt.strftime("%Y-%m-%d")
    edited_lb = st.data_editor(look_back_df, use_container_width=True, num_rows="fixed")

    # ------------------------------------------------------------------
    # Forecast input — auto-fill from dataset if available, else 0
    # ------------------------------------------------------------------
    st.subheader("Curah Hujan Forecast — 7 hari ke depan")
    forecast_dates = [ref_date + pd.Timedelta(days=i) for i in range(PRED_LEN)]

    # Pre-fill P_DAS from dataset if available, else 0 (user inputs manually)
    fc_rows  = []
    actual_q = []
    for fd in forecast_dates:
        row   = df[df["date"] == fd]
        p_val = float(row["p_das"].values[0])   if len(row) else 0.0
        q_val = float(row["q_total"].values[0]) if len(row) else None
        fc_rows.append({"Tanggal": fd.strftime("%Y-%m-%d"), "P_DAS forecast (mm)": p_val})
        actual_q.append(q_val)

    forecast_df = pd.DataFrame(fc_rows)
    edited_fc = st.data_editor(forecast_df, use_container_width=True, num_rows="fixed")

    if is_future:
        st.caption("Masukkan perkiraan curah hujan harian (mm) untuk 7 hari ke depan secara manual.")
    elif has_actuals:
        st.info("P_DAS otomatis diisi dari data historis. Setelah prediksi, debit aktual ditampilkan sebagai pembanding.")

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    if st.button("▶ Run Forecast", type="primary"):
        try:
            p_hist = edited_lb["p_das"].astype(float).values
            q_hist = edited_lb["q_total"].astype(float).values
            p_fore = edited_fc["P_DAS forecast (mm)"].astype(float).values

            if len(p_hist) != ENC_LEN or len(p_fore) != PRED_LEN:
                st.error(f"Need exactly {ENC_LEN} look-back rows and {PRED_LEN} forecast rows.")
                return

            preds = predict(model, p_hist, q_hist, p_fore, q_scaler)
            # preds: (7, 3) — [q10, q50, q90]

            _show_forecast_results(preds, forecast_dates, p_fore, actual_q)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise


def _show_forecast_results(preds: np.ndarray, dates, p_fore: np.ndarray, actual_q=None):
    """Render the forecast chart and summary table."""
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # ------------------------------------------------------------------
    # Plotly chart
    # ------------------------------------------------------------------
    fig = go.Figure()

    # Confidence band q10–q90
    fig.add_trace(go.Scatter(
        x=date_strs + date_strs[::-1],
        y=list(preds[:, 2]) + list(preds[:, 0])[::-1],
        fill="toself",
        fillcolor="rgba(31,119,180,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Q10–Q90 band",
        hoverinfo="skip",
    ))
    # Q50 (median)
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 1],
        mode="lines+markers", name="Q50 — Normal",
        line=dict(color=QUANTILE_COLORS["q50"], width=2.5),
        marker=dict(size=7),
    ))
    # Q90 (wet)
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 2],
        mode="lines+markers", name="Q90 — Wet",
        line=dict(color=QUANTILE_COLORS["q90"], dash="dot", width=1.5),
        marker=dict(size=5),
    ))
    # Q10 (dry)
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 0],
        mode="lines+markers", name="Q10 — Dry",
        line=dict(color=QUANTILE_COLORS["q10"], dash="dot", width=1.5),
        marker=dict(size=5),
    ))
    # Rainfall bar on secondary axis
    fig.add_trace(go.Bar(
        x=date_strs, y=p_fore,
        name="P_DAS forecast (mm)", yaxis="y2",
        marker_color="rgba(31,119,180,0.35)",
    ))

    # Actual observed Q (if available from historical data)
    if actual_q and any(v is not None for v in actual_q):
        actual_vals = [v if v is not None else None for v in actual_q]
        fig.add_trace(go.Scatter(
            x=date_strs, y=actual_vals,
            mode="lines+markers", name="Aktual (observed)",
            line=dict(color="black", width=2, dash="dash"),
            marker=dict(size=8, symbol="circle-open"),
        ))

    fig.update_layout(
        title="7-Day Inflow Forecast — PLTA Grindulu",
        xaxis_title="Tanggal",
        yaxis=dict(title="Inflow Q (m³/s)", rangemode="tozero"),
        yaxis2=dict(title="Curah Hujan (mm)", overlaying="y", side="right", autorange="reversed"),
        legend=dict(orientation="h", y=-0.2),
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    actual_col = [f"{v:.3f}" if v is not None else "—"
                  for v in (actual_q or [None] * len(date_strs))]

    result_df = pd.DataFrame({
        "Tanggal":              date_strs,
        "P_DAS (mm)":           p_fore.round(1),
        "Q10 — Kering (m³/s)":  preds[:, 0].round(3),
        "Q50 — Normal (m³/s)":  preds[:, 1].round(3),
        "Q90 — Basah (m³/s)":   preds[:, 2].round(3),
        "Aktual (m³/s)":        actual_col,
    })
    st.subheader("Tabel Prediksi")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Error metrics if actuals available
    if actual_q and any(v is not None for v in actual_q):
        obs_arr  = np.array([v for v in actual_q if v is not None], dtype=float)
        q50_arr  = preds[[i for i, v in enumerate(actual_q) if v is not None], 1]
        mae  = float(np.mean(np.abs(obs_arr - q50_arr)))
        rmse = float(np.sqrt(np.mean((obs_arr - q50_arr) ** 2)))
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE vs Aktual",  f"{mae:.3f} m³/s")
        c2.metric("RMSE vs Aktual", f"{rmse:.3f} m³/s")
        covered = np.sum(
            (obs_arr >= preds[[i for i, v in enumerate(actual_q) if v is not None], 0]) &
            (obs_arr <= preds[[i for i, v in enumerate(actual_q) if v is not None], 2])
        )
        c3.metric("Aktual dalam [Q10,Q90]", f"{covered}/{len(obs_arr)} hari")

    # Download button
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download forecast CSV",
        data=csv,
        file_name="grindulu_forecast.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page: Analysis & Turbine Performance
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running test-set predictions...")
def _get_test_predictions(_model_state_path):
    """Cache-safe function: load model + test predictions once."""
    device = torch.device("cpu")
    model = Seq2SeqQuantile()
    model.load_state_dict(torch.load(_model_state_path, map_location=device))

    _, _, test_loader, q_scaler, _ = load_data(save_scalers=False)
    preds_scaled, tgts_scaled = collect_predictions(model, test_loader, device)

    N, P, Q = preds_scaled.shape
    with open(Q_SCALER_PATH, "rb") as f:
        import pickle as _pk
        q_sc = _pk.load(f)

    preds_orig = q_sc.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(N, P, Q)
    preds_orig = np.sort(preds_orig, axis=-1)

    tgts_orig = q_sc.inverse_transform(
        tgts_scaled.reshape(-1, 1)
    ).reshape(N, P)

    return preds_orig, tgts_orig


def page_analysis(model, q_scaler):
    st.header("Analysis & Turbine Performance")

    # ---------------------------------------------------------------
    # Section A — Model prediction charts (1, 2, 3)
    # ---------------------------------------------------------------
    st.subheader("A. Model Prediction Evaluation")

    if model is None or not os.path.exists(MODEL_PATH):
        st.warning("Train the model first to see prediction charts.")
    else:
        preds_orig, tgts_orig = _get_test_predictions(MODEL_PATH)

        obs_flat  = tgts_orig.flatten()
        q10_flat  = preds_orig[:, :, 0].flatten()
        q50_flat  = preds_orig[:, :, 1].flatten()
        q90_flat  = preds_orig[:, :, 2].flatten()

        # Metrics summary
        crps  = compute_crps(preds_orig, tgts_orig)
        picp  = compute_picp(preds_orig, tgts_orig)
        pinaw = compute_pinaw(preds_orig, tgts_orig)

        c1, c2, c3 = st.columns(3)
        c1.metric("CRPS (m³/s)",  f"{crps:.4f}",   help="Lower is better")
        c2.metric("PICP (%)",      f"{picp*100:.2f}", help="Target >= 80%")
        c3.metric("PINAW",         f"{pinaw:.4f}",   help="Lower is better")

        n_show = st.slider("Points to display in time-series charts", 50, 500, 200, 50)

        with st.expander("Chart 1 — Observed vs Predicted (Q50)", expanded=True):
            st.plotly_chart(
                chart_observed_vs_predicted(obs_flat, q50_flat),
                use_container_width=True,
            )

        with st.expander("Chart 2 — Quantile Prediction (Q10 / Q50 / Q90)", expanded=True):
            st.plotly_chart(
                chart_quantile_prediction(obs_flat, q10_flat, q50_flat, q90_flat, n_show=n_show),
                use_container_width=True,
            )

        with st.expander("Chart 3 — Prediction Interval Coverage", expanded=True):
            st.plotly_chart(
                chart_prediction_interval(obs_flat, q10_flat, q90_flat, n_show=n_show),
                use_container_width=True,
            )

    st.divider()

    # ---------------------------------------------------------------
    # Section B — Turbine performance charts (4, 5, 6)
    # ---------------------------------------------------------------
    st.subheader("B. Turbine Performance — PLTA Grindulu 1000 MW")
    st.caption(
        "Francis Reversible Pumped | H_net = 486.5 m | "
        "Q_design = 60.5 m³/s/unit | 4 units | ηT = 0.90 | ηG = 0.96"
    )

    with st.expander("Chart 4 — Turbine Efficiency vs Flow Rate", expanded=True):
        st.plotly_chart(chart_turbine_efficiency(), use_container_width=True)
        st.caption(
            "Efficiency curve modelled using standard Francis turbine polynomial "
            "(IEC 60193). Peak efficiency at design flow Q = 60.5 m³/s."
        )

    with st.expander("Chart 5 — Fixed Speed vs Variable Speed Operation", expanded=True):
        st.plotly_chart(chart_fixed_vs_variable_speed(), use_container_width=True)
        st.caption(
            "Variable speed range: 450–550 RPM. Variable speed maintains higher "
            "efficiency at partial load, improving annual energy yield."
        )

    with st.expander("Chart 6 — Power Output at Various Flow Conditions", expanded=True):
        st.plotly_chart(chart_power_output(), use_container_width=True)
        st.caption(
            "Shows achievable power for 1–4 active generating units. "
            "Each unit requires Q >= 18.2 m³/s (30% of design) to operate."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df      = load_df()
    model   = load_model()
    q_scaler = load_q_scaler()

    render_sidebar(df)

    tab1, tab2, tab3 = st.tabs([
        "📈 Historical Data",
        "🔮 7-Day Forecast",
        "📊 Analysis & Turbine Performance",
    ])
    with tab1:
        page_historical(df)
    with tab2:
        page_forecast(df, model, q_scaler)
    with tab3:
        page_analysis(model, q_scaler)


if __name__ == "__main__":
    main()
