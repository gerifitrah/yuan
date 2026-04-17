"""
train.py — Training loop for Bi-LSTM Seq2Seq Quantile Regression.

Usage:
    python train.py [--epochs N] [--lr LR] [--hidden H] [--layers L]
                    [--dropout D] [--batch B] [--device cuda|cpu]

Saved artefacts (written to saved_model/)
-----------------------------------------
    saved_model/best_model.pt      — state dict of best val-loss model
    saved_model/last_model.pt      — state dict after final epoch
    saved_model/feature_scaler.pkl — MinMaxScaler for [p_das, q_total]
    saved_model/q_scaler.pkl       — MinMaxScaler for q_total (inverse transform)
    saved_model/train_history.csv  — epoch / train_loss / val_loss
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model import Seq2SeqQuantile, pinball_loss
from preprocess import load_data, ENC_LEN, PRED_LEN, SAVE_DIR
QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for enc_b, dec_b, tgt_b in loader:
        enc_b = enc_b.to(device)
        dec_b = dec_b.to(device)
        tgt_b = tgt_b.to(device)

        optimizer.zero_grad()
        preds = model(enc_b, dec_b)                  # (batch, pred_len, 3)
        loss  = pinball_loss(preds, tgt_b, QUANTILES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * enc_b.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for enc_b, dec_b, tgt_b in loader:
        enc_b = enc_b.to(device)
        dec_b = dec_b.to(device)
        tgt_b = tgt_b.to(device)

        preds = model(enc_b, dec_b)
        loss  = pinball_loss(preds, tgt_b, QUANTILES)
        total_loss += loss.item() * enc_b.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    epochs:     int   = 100,
    lr:         float = 1e-3,
    hidden:     int   = 128,
    num_layers: int   = 2,
    dropout:    float = 0.2,
    batch_size: int   = 32,
    device_str: str   = "auto",
    patience:   int   = 15,
):
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, q_scaler, _ = load_data(
        batch_size=batch_size
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = Seq2SeqQuantile(
        enc_input_size=2,
        dec_input_size=1,
        hidden_size=hidden,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=PRED_LEN,
        n_quantiles=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + LR scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time (s)':>10}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {elapsed:>10.1f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    # ------------------------------------------------------------------
    # Save last model + history
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pt"))
    pd.DataFrame(history).to_csv(
        os.path.join(SAVE_DIR, "train_history.csv"), index=False
    )

    print(f"\nBest validation loss: {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"Saved to: {SAVE_DIR}")

    # ------------------------------------------------------------------
    # Quick test-set evaluation
    # ------------------------------------------------------------------
    print("\nLoading best model for test evaluation…")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=device))
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Pinball Loss: {test_loss:.6f}")

    return model, q_scaler


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train Bi-LSTM Seq2Seq Quantile model")
    p.add_argument("--epochs",  type=int,   default=100,   help="Max training epochs")
    p.add_argument("--lr",      type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--hidden",  type=int,   default=128,   help="LSTM hidden size")
    p.add_argument("--layers",  type=int,   default=2,     help="Number of LSTM layers")
    p.add_argument("--dropout", type=float, default=0.2,   help="Dropout probability")
    p.add_argument("--batch",   type=int,   default=32,    help="Batch size")
    p.add_argument("--device",  type=str,   default="auto",help="cuda / cpu / auto")
    p.add_argument("--patience",type=int,   default=15,    help="Early-stopping patience")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        batch_size=args.batch,
        device_str=args.device,
        patience=args.patience,
    )
