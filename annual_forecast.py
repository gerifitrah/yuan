"""
annual_forecast.py — Simulasi prediksi inflow 1 tahun penuh (rolling forecast)

Cara kerja:
    1. Mulai dari look-back 30 hari terakhir dataset
    2. Prediksi 7 hari ke depan (Q10/Q50/Q90)
    3. Gunakan Q50 prediksi sebagai Q_total look-back berikutnya
    4. Untuk P_DAS forecast → pakai rata-rata historis bulan tersebut
    5. Ulangi sampai 365 hari

P_DAS yang digunakan adalah rata-rata per bulan dari data historis 2014-2024
sebagai "klimatologi" — pola musiman representatif.

Catatan: error bisa akumulasi di step jauh ke depan.
         Hasil paling akurat di 1-4 minggu pertama.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from model import Seq2SeqQuantile
from preprocess import ENC_LEN, PRED_LEN, SAVE_DIR

SAVE_DIR      = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH    = os.path.join(SAVE_DIR, "best_model.pt")
SCALER_PATH   = os.path.join(SAVE_DIR, "feature_scaler.pkl")
Q_SCALER_PATH = os.path.join(SAVE_DIR, "q_scaler.pkl")
DATA_PATH     = os.path.join(os.path.dirname(__file__), "data_grindulu.csv")


def load_monthly_avg_p_das(df: pd.DataFrame) -> dict:
    """Hitung rata-rata P_DAS per bulan dari data historis (klimatologi)."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")["p_das"].mean().to_dict()
    return monthly  # {1: 15.2, 2: 18.3, ..., 12: 20.1}


@torch.no_grad()
def run_annual_forecast(
    target_year: int = 2025,
    scenario: str = "normal",   # "dry" | "normal" | "wet"
    p_das_override: dict = None,  # optional: {month: p_das_mm} override per bulan
) -> pd.DataFrame:
    """
    Jalankan rolling forecast untuk 1 tahun penuh.

    Args:
        target_year    : tahun yang ingin diprediksi
        scenario       : "dry" (75% avg), "normal" (100%), "wet" (125%)
        p_das_override : dict opsional untuk override P_DAS per bulan

    Returns:
        DataFrame dengan kolom:
            date, p_das_used, q10, q50, q90, step
    """
    # ------------------------------------------------------------------
    # Load model & scalers
    # ------------------------------------------------------------------
    model = Seq2SeqQuantile()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
    p_sc = scalers["p_scaler"]
    q_sc = scalers["q_scaler"]

    with open(Q_SCALER_PATH, "rb") as f:
        q_scaler = pickle.load(f)

    # ------------------------------------------------------------------
    # Load historical data & compute monthly P_DAS climatology
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    monthly_avg = load_monthly_avg_p_das(df)

    # Scenario multiplier
    multiplier = {"dry": 0.70, "normal": 1.00, "wet": 1.30}[scenario]

    # ------------------------------------------------------------------
    # Initialise look-back from last ENC_LEN days of dataset
    # ------------------------------------------------------------------
    recent       = df.tail(ENC_LEN).reset_index(drop=True)
    p_das_buffer = list(recent["p_das"].values)    # rolling list of ENC_LEN values
    q_tot_buffer = list(recent["q_total"].values)

    # ------------------------------------------------------------------
    # Build list of all dates in target_year
    # ------------------------------------------------------------------
    all_dates = pd.date_range(
        start=f"{target_year}-01-01",
        end=f"{target_year}-12-31",
        freq="D",
    )

    results = []
    i = 0

    while i < len(all_dates):
        # 7-day window starting at position i
        window_dates = all_dates[i : i + PRED_LEN]
        if len(window_dates) == 0:
            break

        # P_DAS forecast for this window — use monthly climatology
        p_fore = np.array([
            (p_das_override or {}).get(d.month, monthly_avg[d.month]) * multiplier
            for d in window_dates
        ], dtype=np.float32)

        # Pad to PRED_LEN if window is shorter (last few days of year)
        if len(p_fore) < PRED_LEN:
            p_fore = np.pad(p_fore, (0, PRED_LEN - len(p_fore)), mode="edge")

        # Scale inputs
        p_hist_arr = np.array(p_das_buffer[-ENC_LEN:], dtype=np.float32)
        q_hist_arr = np.array(q_tot_buffer[-ENC_LEN:], dtype=np.float32)

        p_hist_s = p_sc.transform(p_hist_arr.reshape(-1, 1)).flatten()
        q_hist_s = q_scaler.transform(q_hist_arr.reshape(-1, 1)).flatten()
        p_fore_s = p_sc.transform(p_fore.reshape(-1, 1)).flatten()

        enc = torch.from_numpy(
            np.stack([p_hist_s, q_hist_s], axis=-1).astype(np.float32)
        ).unsqueeze(0)  # (1, ENC_LEN, 2)
        dec = torch.from_numpy(
            p_fore_s[:, None].astype(np.float32)
        ).unsqueeze(0)  # (1, PRED_LEN, 1)

        # Predict
        out_scaled = model(enc, dec).squeeze(0).numpy()  # (PRED_LEN, 3)
        n_q = out_scaled.shape[1]
        out_orig = q_scaler.inverse_transform(
            out_scaled.reshape(-1, 1)
        ).reshape(PRED_LEN, n_q)
        out_orig = np.sort(out_orig, axis=-1)  # enforce q10 <= q50 <= q90

        # Store results for actual window days
        for j, d in enumerate(window_dates):
            results.append({
                "date":       d,
                "p_das_used": round(float(p_fore[j]), 2),
                "q10":        round(float(out_orig[j, 0]), 4),
                "q50":        round(float(out_orig[j, 1]), 4),
                "q90":        round(float(out_orig[j, 2]), 4),
                "step":       i // PRED_LEN + 1,
            })

        # Update rolling buffer with Q50 predictions
        for j in range(len(window_dates)):
            p_das_buffer.append(float(p_fore[j]))
            q_tot_buffer.append(float(out_orig[j, 1]))  # use Q50 as next look-back

        i += PRED_LEN

    result_df = pd.DataFrame(results)
    return result_df


if __name__ == "__main__":
    print("Menjalankan rolling forecast untuk tahun 2025...")
    df_result = run_annual_forecast(target_year=2025, scenario="normal")

    print(f"\nHasil: {len(df_result)} hari diprediksi")
    print(f"Periode: {df_result['date'].min().date()} s.d. {df_result['date'].max().date()}")
    print(f"\nRingkasan Q50 per bulan (m3/s):")
    df_result["month"] = df_result["date"].dt.month
    monthly = df_result.groupby("month")[["q10","q50","q90"]].mean().round(3)
    monthly.index = [pd.Timestamp(2000,m,1).strftime("%B") for m in monthly.index]
    print(monthly.to_string())

    df_result.to_csv("annual_forecast_2025.csv", index=False)
    print("\nDisimpan ke annual_forecast_2025.csv")
