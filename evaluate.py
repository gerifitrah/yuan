"""
evaluate.py — Probabilistic evaluation of quantile forecasts.

Metrics
-------
CRPS   (Continuous Ranked Probability Score) — approximated from quantile
        predictions at τ = 0.1 / 0.5 / 0.9 using the quantile-score sum.
        Lower is better.

PICP   (Prediction Interval Coverage Probability) — fraction of observed
        values that fall inside the [q10, q90] prediction interval.
        Target ≥ 80 % (80 % nominal interval).

PINAW  (Prediction Interval Normalised Average Width) — average interval
        width normalised by the observed range.
        Lower is better.

Usage (standalone)
------------------
    python evaluate.py

This will load saved_model/best_model.pt, run inference on the test split,
de-normalise predictions and targets, and print a results table.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from model import Seq2SeqQuantile, pinball_loss
from preprocess import load_data, PRED_LEN, SAVE_DIR, ENC_LEN

QUANTILES   = [0.1, 0.5, 0.9]
MODEL_PATH  = os.path.join(SAVE_DIR, "best_model.pt")
Q_SCALER_PATH = os.path.join(SAVE_DIR, "q_scaler.pkl")


# ---------------------------------------------------------------------------
# Metric functions (operate on numpy arrays, original scale)
# ---------------------------------------------------------------------------

def compute_crps(
    preds_original: np.ndarray,
    targets_original: np.ndarray,
    quantiles: list[float] = QUANTILES,
) -> float:
    """Approximate CRPS via quantile-score sum.

    CRPS ≈ 2 * mean_τ  E_τ(y, ŷ_τ)
    where E_τ = τ*(y - ŷ_τ) if y > ŷ_τ else (1-τ)*(ŷ_τ - y)

    Args:
        preds_original  : (N, pred_len, n_quantiles)
        targets_original: (N, pred_len)

    Returns:
        scalar CRPS
    """
    target_exp = targets_original[..., np.newaxis]  # (N, pred_len, 1)
    q = np.array(quantiles)                          # (n_quantiles,)
    errors = target_exp - preds_original             # (N, pred_len, n_quantiles)
    scores = np.where(errors >= 0, q * errors, (q - 1.0) * errors)
    return float(2.0 * scores.mean())


def compute_picp(
    preds_original: np.ndarray,
    targets_original: np.ndarray,
    lower_idx: int = 0,
    upper_idx: int = 2,
) -> float:
    """Fraction of observations within [q10, q90].

    Args:
        preds_original  : (N, pred_len, n_quantiles)
        targets_original: (N, pred_len)
        lower_idx       : column index for lower bound (q10 = 0)
        upper_idx       : column index for upper bound (q90 = 2)

    Returns:
        PICP as a float in [0, 1]
    """
    q_lo = preds_original[:, :, lower_idx]  # (N, pred_len)
    q_hi = preds_original[:, :, upper_idx]  # (N, pred_len)
    inside = (targets_original >= q_lo) & (targets_original <= q_hi)
    return float(inside.mean())


def compute_pinaw(
    preds_original: np.ndarray,
    targets_original: np.ndarray,
    lower_idx: int = 0,
    upper_idx: int = 2,
) -> float:
    """Normalised average prediction interval width.

    PINAW = mean(q90 - q10) / (max(y) - min(y))

    Args:
        preds_original  : (N, pred_len, n_quantiles)
        targets_original: (N, pred_len)

    Returns:
        PINAW as a float
    """
    q_lo  = preds_original[:, :, lower_idx]
    q_hi  = preds_original[:, :, upper_idx]
    width = (q_hi - q_lo).mean()
    obs_range = targets_original.max() - targets_original.min()
    if obs_range == 0:
        return float("nan")
    return float(width / obs_range)


# ---------------------------------------------------------------------------
# Collect predictions from a DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: Seq2SeqQuantile,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on an entire DataLoader.

    Returns:
        all_preds   : (N, pred_len, 3)  — scaled quantile predictions
        all_targets : (N, pred_len)     — scaled targets
    """
    model.eval()
    preds_list, tgt_list = [], []

    for enc_b, dec_b, tgt_b in loader:
        enc_b = enc_b.to(device)
        dec_b = dec_b.to(device)
        out   = model(enc_b, dec_b).cpu().numpy()  # (batch, pred_len, 3)
        preds_list.append(out)
        tgt_list.append(tgt_b.numpy())             # (batch, pred_len)

    return np.concatenate(preds_list, axis=0), np.concatenate(tgt_list, axis=0)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str = MODEL_PATH,
    device_str: str = "auto",
    split: str = "test",
) -> dict:
    """Load best model, run on test set, compute all metrics.

    Args:
        model_path : path to saved state dict
        device_str : 'auto' / 'cuda' / 'cpu'
        split      : 'test' | 'val'

    Returns:
        dict with keys: crps, picp, pinaw, pinball_loss_scaled
    """
    # ------------------------------------------------------------------
    # Device & model
    # ------------------------------------------------------------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model = Seq2SeqQuantile().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, q_scaler, df = load_data(save_scalers=False)
    loader = test_loader if split == "test" else val_loader

    # ------------------------------------------------------------------
    # Predictions in scaled space
    # ------------------------------------------------------------------
    preds_scaled, tgts_scaled = collect_predictions(model, loader, device)
    # preds_scaled : (N, pred_len, 3),  tgts_scaled : (N, pred_len)

    # Pinball loss on scaled data (matches training objective)
    pb_loss = float(pinball_loss(
        torch.from_numpy(preds_scaled),
        torch.from_numpy(tgts_scaled),
        QUANTILES,
    ).item())

    # ------------------------------------------------------------------
    # De-normalise to original scale (m³/s)
    # ------------------------------------------------------------------
    N, P, Q = preds_scaled.shape
    preds_flat = preds_scaled.reshape(-1, 1)
    preds_orig = q_scaler.inverse_transform(preds_flat).reshape(N, P, Q)

    tgts_flat  = tgts_scaled.reshape(-1, 1)
    tgts_orig  = q_scaler.inverse_transform(tgts_flat).reshape(N, P)

    # Enforce quantile ordering (monotonicity): q10 ≤ q50 ≤ q90
    preds_orig = np.sort(preds_orig, axis=-1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    crps  = compute_crps(preds_orig, tgts_orig)
    picp  = compute_picp(preds_orig, tgts_orig)
    pinaw = compute_pinaw(preds_orig, tgts_orig)

    results = {
        "split":              split,
        "n_sequences":        N,
        "crps_m3s":           crps,
        "picp_%":             picp * 100,
        "pinaw":              pinaw,
        "pinball_loss_scaled": pb_loss,
    }

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS  ({split.upper()} SET)")
    print("=" * 50)
    print(f"  Sequences evaluated : {N}")
    print(f"  CRPS  (m3/s)         : {crps:.4f}   (lower is better)")
    print(f"  PICP  (%)            : {picp*100:.2f}   (target >= 80 %)")
    print(f"  PINAW                : {pinaw:.4f}   (lower is better)")
    print(f"  Pinball Loss (scaled): {pb_loss:.6f}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Per-step metrics (averaged across samples)
    # ------------------------------------------------------------------
    step_rows = []
    for step in range(P):
        p_s = preds_orig[:, step, :]  # (N, 3)
        t_s = tgts_orig[:, step]      # (N,)
        step_rows.append({
            "forecast_day": step + 1,
            "crps":   compute_crps(p_s[:, np.newaxis, :], t_s[:, np.newaxis]),
            "picp_%": compute_picp(p_s[:, np.newaxis, :], t_s[:, np.newaxis]) * 100,
            "mean_q10": p_s[:, 0].mean(),
            "mean_q50": p_s[:, 1].mean(),
            "mean_q90": p_s[:, 2].mean(),
            "mean_obs": t_s.mean(),
        })

    step_df = pd.DataFrame(step_rows)
    print("\nPer-forecast-day breakdown:")
    print(step_df.to_string(index=False, float_format="{:.3f}".format))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
