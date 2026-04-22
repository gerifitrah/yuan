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
    chart_all_quantiles,
    chart_fixed_vs_variable_speed,
    chart_observed_vs_predicted,
    chart_power_output,
    chart_prediction_interval,
    chart_probabilistic_forecast,
    chart_quantile_prediction,
    chart_turbine_efficiency,
    chart_uncertainty_width,
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
# PLTA Grindulu PS — Pumped Storage specs (OPSI-1A)
# ---------------------------------------------------------------------------

H_NET         = 486.5        # m  net head
Q_DESIGN      = 60.5         # m³/s per unit (turbine mode)
N_UNITS       = 4
ETA_T_MAX     = 0.90         # max turbine efficiency
ETA_G         = 0.96         # generator efficiency
ETA_PUMP      = 0.85         # pump mode efficiency (Francis reversible)
RHO           = 1000.0
G_GRAV        = 9.81

# Reservoir capacities — OPSI-1A
VOL_UPPER_MAX = 7_900_000    # m³  upper reservoir (7.90 juta m³)
VOL_LOWER_MAX = 8_990_000    # m³  lower reservoir (8.99 juta m³)

# Derived operational parameters
Q_TURBINE     = N_UNITS * Q_DESIGN   # 242 m³/s  total turbine flow
P_RATED_MW    = N_UNITS * 250.0      # 1000 MW   total rated power
HOURS_GEN     = 8                    # h/day  peak generation
HOURS_PUMP    = 16                   # h/day  off-peak pumping
ETA_LOSS_FRAC = 0.03                 # 3% reservoir losses per cycle (evap+seepage)


def simulate_ps_day(
    q_inflow: float,
    vol_upper: float,
    vol_lower: float,
    hours_gen: int = HOURS_GEN,
) -> dict:
    """Simulate one day of pumped storage operation.

    Cycle order:
      1. Night pump (off-peak, 16 h): lower reservoir -> upper reservoir
      2. Day generation (peak, hours_gen h): upper -> lower
      3. River inflow: adds to lower reservoir
      4. Losses: evaporation + seepage (ETA_LOSS_FRAC of generated volume)
    """
    # 1. Pumping phase
    vol_needed   = VOL_UPPER_MAX - vol_upper
    vol_pump_cap = Q_TURBINE * HOURS_PUMP * 3600
    vol_pump     = min(vol_lower, vol_needed, vol_pump_cap)
    vol_upper    = vol_upper + vol_pump
    vol_lower    = vol_lower - vol_pump

    # 2. Generation phase
    vol_max_gen = Q_TURBINE * hours_gen * 3600
    vol_gen     = min(vol_upper, vol_max_gen)
    actual_h    = vol_gen / (Q_TURBINE * 3600)
    energy_mwh  = P_RATED_MW * actual_h
    vol_upper   = vol_upper - vol_gen
    vol_lower   = min(vol_lower + vol_gen, VOL_LOWER_MAX)

    # 3. River inflow
    vol_lower = min(vol_lower + q_inflow * 86400, VOL_LOWER_MAX)

    # 4. Losses
    vol_lower = max(vol_lower - vol_gen * ETA_LOSS_FRAC, 0.0)

    return {
        "vol_upper":  vol_upper,
        "vol_lower":  vol_lower,
        "hours_gen":  round(actual_h, 2),
        "energy_mwh": round(energy_mwh, 1),
        "power_mw":   round(P_RATED_MW if actual_h > 0 else 0.0, 1),
        "upper_pct":  round(vol_upper / VOL_UPPER_MAX * 100, 1),
        "lower_pct":  round(vol_lower / VOL_LOWER_MAX * 100, 1),
    }


def simulate_ps_series(
    q_array,
    vol_upper_init_pct: float = 90.0,
    hours_gen: int = HOURS_GEN,
) -> pd.DataFrame:
    """Run daily pumped storage simulation over a Q inflow time series."""
    vol_upper = VOL_UPPER_MAX * vol_upper_init_pct / 100
    vol_lower = VOL_LOWER_MAX * 0.50

    rows = []
    for q in q_array:
        result    = simulate_ps_day(float(q), vol_upper, vol_lower, hours_gen)
        vol_upper = result["vol_upper"]
        vol_lower = result["vol_lower"]
        rows.append(result)

    return pd.DataFrame(rows)


def render_ps_section(
    date_strs: list,
    q10: list,
    q50: list,
    q90: list,
    label: str = "",
    key_prefix: str = "ps",
):
    """Render pumped storage performance chart + summary for a forecast."""
    from plotly.subplots import make_subplots as _msp

    with st.expander("Pengaturan Awal Reservoir", expanded=False):
        init_pct = st.slider(
            "Level awal upper reservoir (%)",
            min_value=10, max_value=100, value=90, step=5,
            key=f"{key_prefix}_init",
        )

    ps10 = simulate_ps_series(q10, init_pct)
    ps50 = simulate_ps_series(q50, init_pct)
    ps90 = simulate_ps_series(q90, init_pct)

    rated_energy = P_RATED_MW * HOURS_GEN

    fig = _msp(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Energi Terbangkit (MWh) {label}",
            "Level Upper Reservoir (%)",
        ),
    )

    # Energy output
    fig.add_trace(go.Bar(
        x=date_strs, y=ps50["energy_mwh"],
        name="Energi Q50 — Normal",
        marker_color="#2E7D32",
        marker_line_color="#1B5E20",
        marker_line_width=1,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=date_strs, y=ps90["energy_mwh"],
        mode="lines", name="Energi Q90 — Basah",
        line=dict(color="#1565C0", width=1.5, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=date_strs, y=ps10["energy_mwh"],
        mode="lines", name="Energi Q10 — Kering",
        line=dict(color="#B71C1C", width=1.5, dash="dot"),
    ), row=1, col=1)

    fig.add_hline(
        y=rated_energy, line_dash="dot", line_color="gray",
        annotation_text=f"Kapasitas penuh {rated_energy:,.0f} MWh/hari",
        annotation_position="top left",
        row=1, col=1,
    )

    # Reservoir level
    fig.add_trace(go.Scatter(
        x=date_strs, y=ps50["upper_pct"],
        mode="lines", name="Level Reservoir (Q50)",
        line=dict(color="#2ca02c", width=2),
        fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=date_strs, y=ps90["upper_pct"],
        mode="lines", name="Level Q90",
        line=dict(color="#1565C0", width=1.2, dash="dash"),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=date_strs, y=ps10["upper_pct"],
        mode="lines", name="Level Q10",
        line=dict(color="#B71C1C", width=1.2, dash="dot"),
    ), row=2, col=1)

    fig.add_hline(
        y=20, line_dash="dot", line_color="#FF6600",
        annotation_text="Batas aman 20%",
        annotation_position="top right",
        row=2, col=1,
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(title_text="Energi (MWh)", gridcolor="#eeeeee",
                     rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Level (%)", gridcolor="#eeeeee",
                     range=[0, 105], row=2, col=1)
    fig.update_xaxes(showgrid=False, tickangle=-30, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    total_energy = ps50["energy_mwh"].sum()
    avg_hours    = ps50["hours_gen"].mean()
    days_full    = (ps50["energy_mwh"] >= rated_energy * 0.99).sum()
    min_level    = ps50["upper_pct"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Energi (Q50)",         f"{total_energy:,.0f} MWh")
    c2.metric("Rata-rata Jam Bangkit/Hari",  f"{avg_hours:.1f} jam")
    c3.metric("Hari Kapasitas Penuh",        f"{days_full} hari")
    c4.metric("Level Reservoir Min",         f"{min_level:.1f}%")

    with st.expander("Tabel Detail Pumped Storage per Hari", expanded=False):
        tbl = pd.DataFrame({
            "Tanggal":             date_strs,
            "Q50 Inflow (m³/s)":   np.round(q50, 3),
            "Jam Bangkit (Q50)":   ps50["hours_gen"],
            "Energi Q50 (MWh)":    ps50["energy_mwh"],
            "Energi Q10 (MWh)":    ps10["energy_mwh"],
            "Energi Q90 (MWh)":    ps90["energy_mwh"],
            "Upper Reservoir (%)": ps50["upper_pct"],
            "Lower Reservoir (%)": ps50["lower_pct"],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)


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
    # Forecast input — auto-fill from dataset if available, else climatology
    # ------------------------------------------------------------------
    st.subheader("Curah Hujan Forecast — 7 hari ke depan")
    forecast_dates = [ref_date + pd.Timedelta(days=i) for i in range(PRED_LEN)]

    # Monthly climatology from historical data (same approach as annual/30-day pages)
    df["_month"] = df["date"].dt.month
    monthly_avg  = df.groupby("_month")["p_das"].mean().to_dict()

    # Pre-fill P_DAS: from dataset if date exists, else from monthly climatology
    fc_rows  = []
    actual_q = []
    for fd in forecast_dates:
        row   = df[df["date"] == fd]
        p_val = float(row["p_das"].values[0])   if len(row) else round(monthly_avg.get(fd.month, 0.0), 2)
        q_val = float(row["q_total"].values[0]) if len(row) else None
        fc_rows.append({"Tanggal": fd.strftime("%Y-%m-%d"), "P_DAS forecast (mm)": p_val})
        actual_q.append(q_val)

    forecast_df = pd.DataFrame(fc_rows)
    edited_fc = st.data_editor(forecast_df, use_container_width=True, num_rows="fixed")

    if is_future:
        st.caption(
            "P_DAS diisi otomatis dari **klimatologi historis per bulan** (rata-rata 2014–2024). "
            "Edit jika ada prakiraan curah hujan lebih akurat (mis. dari BMKG)."
        )
    elif has_actuals:
        st.info("P_DAS otomatis diisi dari data historis. Setelah prediksi, debit aktual ditampilkan sebagai pembanding.")

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    _fc7_key = f"fc7_{ref_date}"

    if st.button("▶ Run Forecast", type="primary"):
        try:
            p_hist = edited_lb["p_das"].astype(float).values
            q_hist = edited_lb["q_total"].astype(float).values
            p_fore = edited_fc["P_DAS forecast (mm)"].astype(float).values

            if len(p_hist) != ENC_LEN or len(p_fore) != PRED_LEN:
                st.error(f"Need exactly {ENC_LEN} look-back rows and {PRED_LEN} forecast rows.")
                return

            preds = predict(model, p_hist, q_hist, p_fore, q_scaler)
            st.session_state[_fc7_key] = dict(preds=preds, p_fore=p_fore, actual_q=actual_q)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

    if _fc7_key in st.session_state:
        s = st.session_state[_fc7_key]
        _show_forecast_results(s["preds"], forecast_dates, s["p_fore"], s["actual_q"], df=df)


def _show_forecast_results(preds: np.ndarray, dates, p_fore: np.ndarray, actual_q=None, df=None):
    """Render the forecast chart and summary table."""
    from plotly.subplots import make_subplots

    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    has_actual = actual_q and any(v is not None for v in actual_q)

    # ------------------------------------------------------------------
    # Subplot: atas = curah hujan, bawah = inflow Q
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.06,
        subplot_titles=("Curah Hujan Forecast P_DAS (mm)", "Prediksi Inflow Q (m³/s)"),
    )

    # --- Baris 1: Curah Hujan ---
    fig.add_trace(go.Bar(
        x=date_strs,
        y=p_fore,
        name="P_DAS (mm)",
        marker_color="#4C9BE8",
        marker_line_color="#1a6bbf",
        marker_line_width=1.2,
        showlegend=True,
    ), row=1, col=1)

    # --- Baris 2: Confidence band Q10–Q90 ---
    fig.add_trace(go.Scatter(
        x=date_strs + date_strs[::-1],
        y=list(preds[:, 2]) + list(preds[:, 0])[::-1],
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Interval Q10–Q90",
        hoverinfo="skip",
        showlegend=True,
    ), row=2, col=1)

    # Q90 — Basah
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 2],
        mode="lines+markers",
        name="Q90 — Basah",
        line=dict(color="#1565C0", width=2, dash="dash"),
        marker=dict(size=8, symbol="triangle-up", color="#1565C0"),
    ), row=2, col=1)

    # Q50 — Normal
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 1],
        mode="lines+markers",
        name="Q50 — Normal",
        line=dict(color="#2E7D32", width=3),
        marker=dict(size=10, symbol="circle", color="#2E7D32",
                    line=dict(color="white", width=1.5)),
    ), row=2, col=1)

    # Q10 — Kering
    fig.add_trace(go.Scatter(
        x=date_strs, y=preds[:, 0],
        mode="lines+markers",
        name="Q10 — Kering",
        line=dict(color="#B71C1C", width=2, dash="dot"),
        marker=dict(size=8, symbol="triangle-down", color="#B71C1C"),
    ), row=2, col=1)

    # Aktual (jika ada)
    if has_actual:
        actual_vals = [v if v is not None else None for v in actual_q]
        fig.add_trace(go.Scatter(
            x=date_strs, y=actual_vals,
            mode="lines+markers",
            name="Aktual (observed)",
            line=dict(color="#000000", width=2.5),
            marker=dict(size=10, symbol="circle-open",
                        color="#000000", line=dict(width=2.5)),
        ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text="7-Day Inflow Forecast — PLTA Grindulu",
            font=dict(size=16),
        ),
        height=560,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Axis styling
    fig.update_yaxes(
        title_text="P_DAS (mm)", row=1, col=1,
        gridcolor="#eeeeee", showline=True, linecolor="#cccccc",
    )
    fig.update_yaxes(
        title_text="Inflow Q (m³/s)", row=2, col=1,
        rangemode="tozero",
        gridcolor="#eeeeee", showline=True, linecolor="#cccccc",
    )
    fig.update_xaxes(
        showgrid=False, showline=True,
        linecolor="#cccccc", tickformat="%d %b",
        row=2, col=1,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Metric cards ringkasan
    # ------------------------------------------------------------------
    total_vol  = float(preds[:, 1].sum()) * 86400 / 1_000_000  # juta m³
    peak_q50   = float(preds[:, 1].max())
    avg_q50    = float(preds[:, 1].mean())
    avg_width  = float((preds[:, 2] - preds[:, 0]).mean())
    peak_day   = date_strs[int(preds[:, 1].argmax())]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Volume Inflow (Q50)", f"{total_vol:.3f} juta m³")
    c2.metric("Q50 Puncak", f"{peak_q50:.2f} m³/s", help=f"Hari: {peak_day}")
    c3.metric("Rata-rata Q50", f"{avg_q50:.3f} m³/s")
    c4.metric("Rata-rata Lebar Interval", f"{avg_width:.3f} m³/s",
              help="Q90 - Q10: makin lebar = makin tidak pasti")

    # ------------------------------------------------------------------
    # Konteks historis
    # ------------------------------------------------------------------
    if df is not None:
        forecast_months = set(d.month for d in dates)
        hist_same_month = df[df["date"].dt.month.isin(forecast_months)]["q_total"]
        hist_avg  = float(hist_same_month.mean())
        hist_p10  = float(hist_same_month.quantile(0.10))
        hist_p90  = float(hist_same_month.quantile(0.90))
        delta_pct = (avg_q50 - hist_avg) / hist_avg * 100
        arah      = "di atas" if delta_pct >= 0 else "di bawah"
        st.caption(
            f"Rata-rata Q50 forecast **{avg_q50:.3f} m³/s** — "
            f"**{abs(delta_pct):.1f}% {arah}** rata-rata historis bulan yang sama "
            f"({hist_avg:.3f} m³/s | rentang normal: {hist_p10:.3f}–{hist_p90:.3f} m³/s, 2014–2024)."
        )

    # ------------------------------------------------------------------
    # Summary table + kolom Kondisi
    # ------------------------------------------------------------------
    if df is not None:
        df_tmp     = df.copy()
        df_tmp["_m"] = df_tmp["date"].dt.month
        p10_m = df_tmp.groupby("_m")["q_total"].quantile(0.10)
        p90_m = df_tmp.groupby("_m")["q_total"].quantile(0.90)
        kondisi = []
        for i, d in enumerate(dates):
            q   = float(preds[i, 1])
            p10 = float(p10_m.get(d.month, 0.0))
            p90 = float(p90_m.get(d.month, 1e9))
            if q < p10:
                kondisi.append("Kering")
            elif q > p90:
                kondisi.append("Basah")
            else:
                kondisi.append("Normal")
    else:
        kondisi = ["—"] * len(date_strs)

    actual_col = [f"{v:.3f}" if v is not None else "—"
                  for v in (actual_q or [None] * len(date_strs))]

    result_df = pd.DataFrame({
        "Tanggal":              date_strs,
        "P_DAS (mm)":           p_fore.round(1),
        "Q10 — Kering (m³/s)":  preds[:, 0].round(3),
        "Q50 — Normal (m³/s)":  preds[:, 1].round(3),
        "Q90 — Basah (m³/s)":   preds[:, 2].round(3),
        "Kondisi":              kondisi,
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

    # ---------------------------------------------------------------
    # Pumped storage performance section
    # ---------------------------------------------------------------
    st.divider()
    st.subheader("Simulasi Pumped Storage — 7 Hari")
    render_ps_section(
        date_strs,
        q10=preds[:, 0].tolist(),
        q50=preds[:, 1].tolist(),
        q90=preds[:, 2].tolist(),
        label="(7 Hari ke Depan)",
        key_prefix="fc7",
    )


# ---------------------------------------------------------------------------
# Page: Annual Simulation
# ---------------------------------------------------------------------------

def page_annual(df: pd.DataFrame):
    st.header("Simulasi Inflow Tahunan — Klimatologi")
    st.markdown(
        "Distribusi Q10/Q50/Q90 dihitung langsung dari **data historis 2014–2024** per bulan. "
        "Tidak ada error akumulasi — setiap bulan independen berdasarkan pola historis aktual."
    )

    col1, col2 = st.columns(2)
    data_years  = sorted(df["date"].dt.year.unique().tolist())
    first_year  = data_years[0]
    all_years   = list(range(first_year, pd.Timestamp.now().year + 6))
    default_idx = all_years.index(2025) if 2025 in all_years else len(all_years) - 1
    target_year = col1.selectbox(
        "Tahun simulasi",
        all_years,
        index=default_idx,
    )
    scenario = col2.selectbox(
        "Skenario curah hujan",
        ["normal", "dry", "wet"],
        format_func=lambda s: {
            "normal": "Normal (100% klimatologi)",
            "dry":    "Kering — 70% dari rata-rata",
            "wet":    "Basah — 130% dari rata-rata",
        }[s],
    )

    multiplier = {"dry": 0.70, "normal": 1.00, "wet": 1.30}[scenario]

    # ------------------------------------------------------------------
    # Select data source: single year (if in dataset) or all-year climatology
    # ------------------------------------------------------------------
    df_c = df.copy()
    df_c["month"] = df_c["date"].dt.month
    df_c["year"]  = df_c["date"].dt.year

    is_historical = target_year in data_years
    df_src = df_c[df_c["year"] == target_year] if is_historical else df_c

    if is_historical:
        st.info(
            f"Tahun **{target_year}** ada dalam dataset — menampilkan distribusi "
            f"Q10/Q50/Q90 dari data aktual tahun {target_year}."
        )
    else:
        st.caption("Data klimatologi berdasarkan rata-rata historis 2014–2024.")

    q10_hist = df_src.groupby("month")["q_total"].quantile(0.10)
    q50_hist = df_src.groupby("month")["q_total"].quantile(0.50)
    q90_hist = df_src.groupby("month")["q_total"].quantile(0.90)
    p_avg    = df_src.groupby("month")["p_das"].mean()

    q10_sc = q10_hist * multiplier
    q50_sc = q50_hist * multiplier
    q90_sc = q90_hist * multiplier
    p_sc_m = p_avg * multiplier

    # Operable days per month from the same source
    df_src = df_src.copy()
    df_src["inflow_vol"] = df_src["q_total"] * multiplier * 86400 / 1_000_000  # juta m³/hari
    if is_historical:
        avg_op = df_src.groupby("month")["inflow_vol"].mean().rename(target_year).round(3)
    else:
        monthly_vol = df_src.groupby(["year", "month"])["inflow_vol"].mean().reset_index()
        avg_op = monthly_vol.groupby("month")["inflow_vol"].mean().round(3)

    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]

    # ------------------------------------------------------------------
    # Chart: rainfall (top) + Q climatology bar (bottom)
    # ------------------------------------------------------------------
    from plotly.subplots import make_subplots as _msp
    fig = _msp(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.05,
        subplot_titles=(
            "Rata-rata P_DAS per Bulan (mm)",
            f"{'Data Aktual' if is_historical else 'Klimatologi'} Inflow Q per Bulan — Skenario {scenario.upper()}",
        ),
    )

    fig.add_trace(go.Bar(
        x=month_names, y=p_sc_m.values,
        name="P_DAS (mm)", marker_color="#4C9BE8",
        marker_line_color="#1a6bbf", marker_line_width=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=month_names, y=q50_sc.values,
        name="Q50 — Median",
        marker_color="#2ca02c",
        error_y=dict(
            type="data", symmetric=False,
            array=(q90_sc - q50_sc).values,
            arrayminus=(q50_sc - q10_sc).values,
            visible=True, color="#555555",
        ),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=month_names, y=q10_sc.values,
        mode="lines+markers", name="Q10 — Kering",
        line=dict(color="#B71C1C", width=1.5, dash="dot"),
        marker=dict(size=6),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=month_names, y=q90_sc.values,
        mode="lines+markers", name="Q90 — Basah",
        line=dict(color="#1565C0", width=1.5, dash="dash"),
        marker=dict(size=6),
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{'Data Aktual' if is_historical else 'Klimatologi'} Inflow {target_year} — Skenario {scenario.upper()}",
            font=dict(size=15),
        ),
        height=560,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc", borderwidth=1, font=dict(size=12),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="P_DAS (mm)", gridcolor="#eeeeee", row=1, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", rangemode="tozero", gridcolor="#eeeeee", row=2, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Monthly summary table
    # ------------------------------------------------------------------
    tbl = pd.DataFrame({
        "Bulan":                        month_names,
        "P_DAS (mm)":                   p_sc_m.values.round(1),
        "Q10 (m³/s)":                   q10_sc.values.round(3),
        "Q50 (m³/s)":                   q50_sc.values.round(3),
        "Q90 (m³/s)":                   q90_sc.values.round(3),
        "Rata-rata Inflow (juta m³/hr)": avg_op.values,
    })
    st.subheader("Ringkasan per Bulan")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Annual metrics
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Inflow Tahunan (Q50)", f"{avg_op.sum():.2f} juta m³")
    c2.metric("Rata-rata Q50 Tahunan", f"{q50_sc.mean():.3f} m³/s")
    c3.metric("Q50 Puncak", f"{q50_sc.max():.2f} m³/s",
              help=f"Bulan {month_names[int(q50_sc.values.argmax())]}")

    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV Klimatologi",
        data=csv,
        file_name=f"klimatologi_{target_year}_{scenario}.csv",
        mime="text/csv",
    )

    st.caption(
        "Sumber: distribusi historis q_total 2014–2024 per bulan. "
        "Skenario kering/basah mengalikan Q dengan faktor 0.70 / 1.30."
    )

    # ------------------------------------------------------------------
    # Expand to daily for turbine section
    # Historical years: use actual daily q_total so turbine counts match
    # Future years: expand monthly quantiles to daily
    # ------------------------------------------------------------------
    if is_historical:
        df_yr = df_c[df_c["year"] == target_year].sort_values("date").reset_index(drop=True)
        df_ann = df_yr[["date"]].copy()
        df_ann["q10"] = df_yr["q_total"].values * multiplier
        df_ann["q50"] = df_yr["q_total"].values * multiplier
        df_ann["q90"] = df_yr["q_total"].values * multiplier
    else:
        all_dates = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")
        df_ann = pd.DataFrame({"date": all_dates})
        df_ann["month"] = df_ann["date"].dt.month
        df_ann["q10"] = df_ann["month"].map(q10_sc)
        df_ann["q50"] = df_ann["month"].map(q50_sc)
        df_ann["q90"] = df_ann["month"].map(q90_sc)

    st.divider()
    st.subheader(f"Simulasi Pumped Storage — Tahun {target_year}")
    render_ps_section(
        df_ann["date"].dt.strftime("%Y-%m-%d").tolist(),
        q10=df_ann["q10"].tolist(),
        q50=df_ann["q50"].tolist(),
        q90=df_ann["q90"].tolist(),
        label=f"(Tahun {target_year} — Skenario {scenario.upper()})",
        key_prefix="ann",
    )


# ---------------------------------------------------------------------------
# Page: 30-Day Forecast
# ---------------------------------------------------------------------------

def _show_monthly_results(date_strs, p_fore_all, all_q10, all_q50, all_q90, actual_q, ref_date, end_date):
    """Render 30-day forecast results. Called outside button block so slider persists."""
    from plotly.subplots import make_subplots as _msp2
    has_act = any(v is not None for v in actual_q)

    fig = _msp2(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.05,
        subplot_titles=(
            "Curah Hujan P_DAS (mm)",
            "Prediksi Inflow Q (m³/s) — 30 Hari",
        ),
    )
    fig.add_trace(go.Bar(
        x=date_strs, y=p_fore_all.tolist(),
        name="P_DAS (mm)", marker_color="#4C9BE8",
        marker_line_color="#1a6bbf", marker_line_width=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=date_strs + date_strs[::-1],
        y=all_q90 + all_q10[::-1],
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Interval Q10–Q90", hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=date_strs, y=all_q90,
        mode="lines", name="Q90 — Basah",
        line=dict(color="#1565C0", width=2, dash="dash"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=date_strs, y=all_q50,
        mode="lines+markers", name="Q50 — Normal",
        line=dict(color="#2E7D32", width=3),
        marker=dict(size=6, color="#2E7D32", line=dict(color="white", width=1)),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=date_strs, y=all_q10,
        mode="lines", name="Q10 — Kering",
        line=dict(color="#B71C1C", width=2, dash="dot"),
    ), row=2, col=1)
    if has_act:
        fig.add_trace(go.Scatter(
            x=date_strs,
            y=[v if v is not None else None for v in actual_q],
            mode="lines+markers", name="Aktual (observed)",
            line=dict(color="#000000", width=2.5),
            marker=dict(size=7, symbol="circle-open", color="#000000", line=dict(width=2)),
        ), row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"Prediksi Inflow 30 Hari — {ref_date.strftime('%d %B %Y')} s.d. {end_date.strftime('%d %B %Y')}",
            font=dict(size=15),
        ),
        height=580, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#cccccc", borderwidth=1, font=dict(size=12),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_yaxes(title_text="P_DAS (mm)", gridcolor="#eeeeee", row=1, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", rangemode="tozero", gridcolor="#eeeeee", row=2, col=1)
    fig.update_xaxes(showgrid=False, tickformat="%d %b", tickangle=-30, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata Q50 Inflow", f"{np.mean(all_q50):.3f} m³/s")
    c2.metric("Q50 Maksimum", f"{max(all_q50):.2f} m³/s",
              help=f"Hari ke-{all_q50.index(max(all_q50))+1} ({date_strs[all_q50.index(max(all_q50))]})")
    c3.metric("Q50 Minimum", f"{min(all_q50):.3f} m³/s")
    total_vol = sum(q * 86400 / 1_000_000 for q in all_q50)
    c4.metric("Total Volume Inflow (Q50)", f"{total_vol:.3f} juta m³")

    if has_act:
        obs = np.array([v for v in actual_q if v is not None])
        pred_q50_match = np.array([all_q50[i] for i, v in enumerate(actual_q) if v is not None])
        mae  = np.mean(np.abs(obs - pred_q50_match))
        rmse = np.sqrt(np.mean((obs - pred_q50_match) ** 2))
        st.info(f"Vs Aktual — MAE: **{mae:.3f} m³/s** | RMSE: **{rmse:.3f} m³/s**")

    with st.expander("Tabel Prediksi Lengkap (30 hari)", expanded=False):
        tbl = pd.DataFrame({
            "Tanggal":             date_strs,
            "P_DAS (mm)":          p_fore_all.round(1),
            "Q10 — Kering (m³/s)": np.round(all_q10, 4),
            "Q50 — Normal (m³/s)": np.round(all_q50, 4),
            "Q90 — Basah (m³/s)":  np.round(all_q90, 4),
            "Aktual (m³/s)":       [f"{v:.3f}" if v is not None else "—" for v in actual_q],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", data=csv,
            file_name=f"forecast_30hari_{ref_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    st.caption(
        "Prediksi rolling: akurasi tertinggi di 7 hari pertama, "
        "menurun di hari 8–30 karena prediksi sebelumnya dipakai sebagai look-back."
    )
    st.divider()
    st.subheader("Simulasi Pumped Storage — 30 Hari")
    render_ps_section(
        date_strs,
        q10=all_q10, q50=all_q50, q90=all_q90,
        label="(30 Hari ke Depan)",
        key_prefix="fc30",
    )


def page_monthly(df: pd.DataFrame, model, q_scaler):
    st.header("Prediksi Inflow 30 Hari ke Depan")
    st.markdown(
        "Prediksi debit inflow selama **30 hari** menggunakan rolling forecast — "
        "model berjalan 7 hari per langkah sebanyak 4–5 kali iterasi."
    )

    if model is None or q_scaler is None:
        st.error("Train model terlebih dahulu.")
        return

    import calendar as _cal

    # ------------------------------------------------------------------
    # Pilih tanggal mulai
    # ------------------------------------------------------------------
    data_end   = df["date"].max()
    data_start = df["date"].min() + pd.Timedelta(days=ENC_LEN)

    st.markdown("Pilih **tanggal mulai** prediksi 30 hari:")
    col_y, col_m, col_d = st.columns(3)

    years   = list(range(data_start.year, pd.Timestamp.now().year + 6))
    default_ref = max(pd.Timestamp.now().normalize(), data_end + pd.Timedelta(days=1))
    default_yi  = years.index(default_ref.year) if default_ref.year in years else len(years)-1

    sel_year  = col_y.selectbox("Tahun", years, index=default_yi, key="m_year")
    sel_month = col_m.selectbox(
        "Bulan", list(range(1, 13)),
        index=default_ref.month - 1,
        format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
        key="m_month",
    )
    max_day   = _cal.monthrange(sel_year, sel_month)[1]
    sel_day   = col_d.selectbox("Hari", list(range(1, max_day + 1)),
                                index=min(default_ref.day, max_day) - 1, key="m_day")

    try:
        ref_date = pd.Timestamp(sel_year, sel_month, sel_day)
    except Exception:
        st.error("Tanggal tidak valid.")
        return

    is_future = ref_date > data_end
    end_date  = ref_date + pd.Timedelta(days=29)

    st.caption(
        f"Periode prediksi: **{ref_date.date()}** s.d. **{end_date.date()}** (30 hari)"
    )

    # ------------------------------------------------------------------
    # Look-back
    # ------------------------------------------------------------------
    if is_future:
        lb_df = df.tail(ENC_LEN).reset_index(drop=True)
        st.info(f"Tanggal di luar dataset — look-back otomatis dari 30 hari terakhir dataset ({data_end.date()}).")
    else:
        lb_df = df[df["date"] < ref_date].tail(ENC_LEN).reset_index(drop=True)

    if len(lb_df) < ENC_LEN:
        st.error("Data look-back tidak cukup.")
        return

    with st.expander("Look-back window (30 hari sebelum tanggal referensi)", expanded=False):
        lb_edit = lb_df[["date", "p_das", "q_total"]].copy()
        lb_edit["date"] = lb_edit["date"].dt.strftime("%Y-%m-%d")
        lb_edit = st.data_editor(lb_edit, use_container_width=True, num_rows="fixed", key="lb_30")

    # ------------------------------------------------------------------
    # Input curah hujan 30 hari
    # ------------------------------------------------------------------
    st.subheader("Input Curah Hujan 30 Hari (P_DAS mm/hari)")

    # Auto-fill: dari dataset jika ada, kalau tidak dari klimatologi
    df["month"] = df["date"].dt.month
    monthly_avg = df.groupby("month")["p_das"].mean().to_dict()

    fc_rows  = []
    actual_q = []
    for i in range(30):
        d    = ref_date + pd.Timedelta(days=i)
        row  = df[df["date"] == d]
        if len(row):
            p_val = float(row["p_das"].values[0])
            q_val = float(row["q_total"].values[0])
        else:
            p_val = round(monthly_avg.get(d.month, 0.0), 2)
            q_val = None
        fc_rows.append({
            "Tanggal": d.strftime("%Y-%m-%d"),
            "P_DAS (mm)": p_val,
        })
        actual_q.append(q_val)

    fc_df   = pd.DataFrame(fc_rows)
    fc_edit = st.data_editor(fc_df, use_container_width=True, num_rows="fixed", key="fc_30")

    if is_future:
        st.caption("P_DAS diisi dari klimatologi historis per bulan. Edit sesuai prakiraan BMKG.")
    else:
        st.caption("P_DAS diisi dari data historis (jika tersedia) atau klimatologi.")

    # ------------------------------------------------------------------
    # Rolling forecast 30 hari
    # ------------------------------------------------------------------
    _fc30_key = f"fc30_{ref_date}"

    if st.button("▶ Prediksi 30 Hari", type="primary"):
        with st.spinner("Menghitung prediksi 30 hari..."):
            p_hist_arr = lb_edit["p_das"].astype(float).values
            q_hist_arr = lb_edit["q_total"].astype(float).values
            p_fore_all = fc_edit["P_DAS (mm)"].astype(float).values  # (30,)

            import pickle as _pk
            with open(FEATURE_SCALER, "rb") as f:
                scalers = _pk.load(f)

            p_buf = list(p_hist_arr)
            q_buf = list(q_hist_arr)
            all_q10, all_q50, all_q90 = [], [], []
            all_dates = [ref_date + pd.Timedelta(days=i) for i in range(30)]

            for step_start in range(0, 30, PRED_LEN):
                step_end = min(step_start + PRED_LEN, 30)
                p_fore_step = p_fore_all[step_start:step_end]
                pad = PRED_LEN - len(p_fore_step)
                p_fore_padded = np.pad(p_fore_step.astype(np.float32), (0, pad), mode="edge")

                enc_t, dec_t = make_inference_sequence(
                    np.array(p_buf[-ENC_LEN:], dtype=np.float32),
                    np.array(q_buf[-ENC_LEN:], dtype=np.float32),
                    p_fore_padded,
                )
                with torch.no_grad():
                    out = model(enc_t, dec_t).squeeze(0).numpy()

                n_q = out.shape[1]
                out_orig = q_scaler.inverse_transform(
                    out.reshape(-1, 1)
                ).reshape(PRED_LEN, n_q)
                out_orig = np.sort(out_orig, axis=-1)

                actual_step = step_end - step_start
                for j in range(actual_step):
                    all_q10.append(float(out_orig[j, 0]))
                    all_q50.append(float(out_orig[j, 1]))
                    all_q90.append(float(out_orig[j, 2]))
                    p_buf.append(float(p_fore_step[j] if j < len(p_fore_step) else p_fore_step[-1]))
                    q_buf.append(float(out_orig[j, 1]))

        date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]
        st.session_state[_fc30_key] = dict(
            date_strs=date_strs, p_fore_all=p_fore_all,
            all_q10=all_q10, all_q50=all_q50, all_q90=all_q90,
            actual_q=actual_q,
        )

    if _fc30_key in st.session_state:
        s = st.session_state[_fc30_key]
        _show_monthly_results(
            s["date_strs"], s["p_fore_all"],
            s["all_q10"], s["all_q50"], s["all_q90"],
            s["actual_q"], ref_date, end_date,
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

        # ---------------------------------------------------------------
        # Reference-paper style charts (7, 8, 9) — top row
        # ---------------------------------------------------------------
        st.markdown("#### BiLSTM Quantile — Paper-Style Charts")

        with st.expander("Chart 7 — Probabilistic Forecast (Confidence Band)", expanded=True):
            st.plotly_chart(
                chart_probabilistic_forecast(obs_flat, q10_flat, q50_flat, q90_flat, n_show=n_show),
                use_container_width=True,
            )

        col_left, col_right = st.columns(2)
        with col_left:
            with st.expander("Chart 8 — All Quantile Predictions", expanded=True):
                st.plotly_chart(
                    chart_all_quantiles(obs_flat, q10_flat, q50_flat, q90_flat, n_show=n_show),
                    use_container_width=True,
                )
        with col_right:
            with st.expander("Chart 9 — Prediction Uncertainty Width", expanded=True):
                st.plotly_chart(
                    chart_uncertainty_width(q10_flat, q90_flat, n_show=n_show),
                    use_container_width=True,
                )

        st.divider()
        st.markdown("#### Standard Evaluation Charts")

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
    st.subheader("B. Turbine Performance — PLTA Grindulu Pumped Storage 1000 MW")
    st.caption(
        "Francis Reversible Pumped Storage | OPSI-1A | H_net = 486.5 m | "
        "Q_turbine = 60.5 m³/s/unit × 4 unit = 242 m³/s | P_rated = 1000 MW | "
        "Upper reservoir 7.90 juta m³ | Generasi 8 jam/hari (peak)"
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Historical Data",
        "🔮 7-Day Forecast",
        "📆 30-Day Forecast",
        "📅 Simulasi Tahunan",
        "📊 Analysis & Turbine Performance",
    ])
    with tab1:
        page_historical(df)
    with tab2:
        page_forecast(df, model, q_scaler)
    with tab3:
        page_monthly(df, model, q_scaler)
    with tab4:
        page_annual(df)
    with tab5:
        page_analysis(model, q_scaler)


if __name__ == "__main__":
    main()
