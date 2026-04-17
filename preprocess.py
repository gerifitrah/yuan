"""
preprocess.py — Data loading, normalisation, and sequence creation.

Produces three (encoder_input, decoder_input, target) PyTorch DataLoaders:
    train  (first 70 % of time series)
    val    (next  15 %)
    test   (last  15 %)

Sequence design
---------------
Encoder input  : past enc_len=30 days   of [p_das (scaled), q_total (scaled)]
Decoder input  : next pred_len=7 days   of [p_das (scaled)]   ← "known future rainfall"
Target         : next pred_len=7 days   of  q_total (original scale, de-normalised later)

The target is kept in SCALED units during training so the Pinball loss
operates on a normalised range.  De-normalisation for evaluation is done
via the saved q_scaler object.

Saved artefacts (written to saved_model/)
-----------------------------------------
    saved_model/feature_scaler.pkl   — MinMaxScaler for [p_das, q_total]
    saved_model/q_scaler.pkl         — MinMaxScaler for q_total only (for inverse transform)
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENC_LEN    = 30   # look-back window
PRED_LEN   = 7    # forecast horizon
BATCH_SIZE = 32

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# Test fraction = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ---------------------------------------------------------------------------
# Dataset switch — change DATASET here to swap everything at once
#
#   "v1" : data_grindulu.csv      (original, constant baseflow)  → saved_model/
#   "v2" : data_grindulu_raw.csv  (raw PDF, dynamic baseflow)    → saved_model_v2/
# ---------------------------------------------------------------------------

DATASET = "v1"

_BASE = os.path.dirname(__file__)

if DATASET == "v1":
    DATA_PATH = os.path.join(_BASE, "data_grindulu.csv")
    SAVE_DIR  = os.path.join(_BASE, "saved_model")
elif DATASET == "v2":
    DATA_PATH = os.path.join(_BASE, "data_grindulu_raw.csv")
    SAVE_DIR  = os.path.join(_BASE, "saved_model_v2")
else:
    raise ValueError(f"Unknown DATASET '{DATASET}'. Choose 'v1' or 'v2'.")


# ---------------------------------------------------------------------------
# Helper: build sliding-window sequences
# ---------------------------------------------------------------------------

def _make_sequences(
    p_das_scaled: np.ndarray,
    q_total_scaled: np.ndarray,
    enc_len: int = ENC_LEN,
    pred_len: int = PRED_LEN,
):
    """Slide a window over 1-D arrays to build (enc_in, dec_in, target).

    Args:
        p_das_scaled    : shape (T,)
        q_total_scaled  : shape (T,)

    Returns:
        enc_inputs  : float32 array (N, enc_len, 2)   — [p_das, q_total]
        dec_inputs  : float32 array (N, pred_len, 1)  — [p_das_forecast]
        targets     : float32 array (N, pred_len)     — q_total
    """
    T = len(p_das_scaled)
    total_len = enc_len + pred_len

    enc_inputs, dec_inputs, targets = [], [], []

    for i in range(T - total_len + 1):
        enc_p   = p_das_scaled[i : i + enc_len]          # (enc_len,)
        enc_q   = q_total_scaled[i : i + enc_len]        # (enc_len,)
        dec_p   = p_das_scaled[i + enc_len : i + total_len]     # (pred_len,)
        tgt_q   = q_total_scaled[i + enc_len : i + total_len]   # (pred_len,)

        enc_inputs.append(np.stack([enc_p, enc_q], axis=-1))  # (enc_len, 2)
        dec_inputs.append(dec_p[:, None])                      # (pred_len, 1)
        targets.append(tgt_q)                                  # (pred_len,)

    enc_inputs = np.array(enc_inputs, dtype=np.float32)
    dec_inputs = np.array(dec_inputs, dtype=np.float32)
    targets    = np.array(targets,    dtype=np.float32)

    return enc_inputs, dec_inputs, targets


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_data(
    data_path: str = DATA_PATH,
    enc_len: int   = ENC_LEN,
    pred_len: int  = PRED_LEN,
    batch_size: int = BATCH_SIZE,
    save_scalers: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, pd.DataFrame]:
    """Load CSV, fit scalers, build sequences, return DataLoaders.

    Returns:
        train_loader  : DataLoader
        val_loader    : DataLoader
        test_loader   : DataLoader
        q_scaler      : MinMaxScaler fitted on q_total (for inverse transform)
        df            : original DataFrame (useful for the Streamlit app)
    """
    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    p_das   = df["p_das"].values.reshape(-1, 1)    # (T, 1)
    q_total = df["q_total"].values.reshape(-1, 1)  # (T, 1)

    T = len(df)

    # ------------------------------------------------------------------
    # 2. Chronological split indices (on raw series, before windowing)
    # ------------------------------------------------------------------
    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * VAL_FRAC)
    # n_test  = T - n_train - n_val

    # ------------------------------------------------------------------
    # 3. Fit scalers on training portion only
    # ------------------------------------------------------------------
    p_scaler = MinMaxScaler()
    q_scaler = MinMaxScaler()

    p_scaler.fit(p_das[:n_train])
    q_scaler.fit(q_total[:n_train])

    p_scaled = p_scaler.transform(p_das).flatten()      # (T,)
    q_scaled = q_scaler.transform(q_total).flatten()    # (T,)

    # ------------------------------------------------------------------
    # 4. Build sequences for each split
    # ------------------------------------------------------------------
    # We build sequences independently per split to avoid look-ahead bias.
    # Each split gets its own contiguous slice of the scaled series, with
    # a small overlap of enc_len at the boundary so no data is wasted.

    def _split_sequences(start: int, end: int):
        return _make_sequences(
            p_scaled[start:end],
            q_scaled[start:end],
            enc_len,
            pred_len,
        )

    # Train: [0, n_train]
    enc_tr, dec_tr, tgt_tr = _split_sequences(0, n_train)

    # Val: include enc_len overlap from training tail for context
    val_start = n_train - enc_len
    enc_va, dec_va, tgt_va = _split_sequences(val_start, n_train + n_val)

    # Test: include enc_len overlap from val tail
    test_start = n_train + n_val - enc_len
    enc_te, dec_te, tgt_te = _split_sequences(test_start, T)

    # ------------------------------------------------------------------
    # 5. Convert to TensorDatasets
    # ------------------------------------------------------------------
    def _make_loader(enc, dec, tgt, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(enc),
            torch.from_numpy(dec),
            torch.from_numpy(tgt),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _make_loader(enc_tr, dec_tr, tgt_tr, shuffle=True)
    val_loader   = _make_loader(enc_va, dec_va, tgt_va, shuffle=False)
    test_loader  = _make_loader(enc_te, dec_te, tgt_te, shuffle=False)

    # ------------------------------------------------------------------
    # 6. Save scalers
    # ------------------------------------------------------------------
    if save_scalers:
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(os.path.join(SAVE_DIR, "feature_scaler.pkl"), "wb") as f:
            pickle.dump({"p_scaler": p_scaler, "q_scaler": q_scaler}, f)
        with open(os.path.join(SAVE_DIR, "q_scaler.pkl"), "wb") as f:
            pickle.dump(q_scaler, f)

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    print(f"Dataset    : {T} days  ({df['date'].min().date()} — {df['date'].max().date()})")
    print(f"Train      : {len(enc_tr):>5} sequences  (rows 0–{n_train-1})")
    print(f"Validation : {len(enc_va):>5} sequences  (rows {val_start}–{n_train+n_val-1})")
    print(f"Test       : {len(enc_te):>5} sequences  (rows {test_start}–{T-1})")

    return train_loader, val_loader, test_loader, q_scaler, df


# ---------------------------------------------------------------------------
# Utility: build a single inference sequence from raw arrays
# ---------------------------------------------------------------------------

def make_inference_sequence(
    p_das_history: np.ndarray,
    q_total_history: np.ndarray,
    p_das_forecast: np.ndarray,
    scaler_path: str = os.path.join(SAVE_DIR, "feature_scaler.pkl"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one (enc_input, dec_input) pair for inference.

    Args:
        p_das_history    : (enc_len,) raw p_das values for the past enc_len days
        q_total_history  : (enc_len,) raw q_total for the past enc_len days
        p_das_forecast   : (pred_len,) raw p_das forecast for the next pred_len days
        scaler_path      : path to saved feature_scaler.pkl

    Returns:
        enc_input : (1, enc_len, 2)   FloatTensor on CPU
        dec_input : (1, pred_len, 1)  FloatTensor on CPU
    """
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    p_sc = scalers["p_scaler"]
    q_sc = scalers["q_scaler"]

    p_hist_s = p_sc.transform(p_das_history.reshape(-1, 1)).flatten()
    q_hist_s = q_sc.transform(q_total_history.reshape(-1, 1)).flatten()
    p_fore_s = p_sc.transform(p_das_forecast.reshape(-1, 1)).flatten()

    enc = np.stack([p_hist_s, q_hist_s], axis=-1).astype(np.float32)  # (enc_len, 2)
    dec = p_fore_s[:, None].astype(np.float32)                         # (pred_len, 1)

    return torch.from_numpy(enc).unsqueeze(0), torch.from_numpy(dec).unsqueeze(0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tr, va, te, q_sc, df = load_data()
    enc_b, dec_b, tgt_b = next(iter(tr))
    print(f"Enc batch : {enc_b.shape}")
    print(f"Dec batch : {dec_b.shape}")
    print(f"Tgt batch : {tgt_b.shape}")
    print("preprocess.py sanity check passed.")
