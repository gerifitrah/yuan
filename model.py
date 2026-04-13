"""
model.py — Bi-LSTM Seq2Seq Quantile Regression
Architecture: Bidirectional LSTM encoder + LSTM decoder
Predicts q10, q50, q90 inflow for each of pred_len forecast days.

Encoder input : (batch, enc_len=30,  2)  — [p_das, q_total] for past 30 days
Decoder input : (batch, pred_len=7,  1)  — [p_das_forecast] for next 7 days
Output        : (batch, pred_len=7,  3)  — [q10, q50, q90] per day
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder.

    Projects the final (forward + backward) hidden & cell states into the
    decoder's hidden size so the decoder can be initialised directly.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project bidirectional state (hidden_size*2) → decoder hidden_size
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, enc_len, input_size)

        Returns:
            enc_outputs : (batch, enc_len, hidden_size*2)
            h_dec       : (num_layers, batch, hidden_size)  — initial decoder h
            c_dec       : (num_layers, batch, hidden_size)  — initial decoder c
        """
        enc_outputs, (h_n, c_n) = self.lstm(x)
        # h_n shape: (num_layers*2, batch, hidden_size)

        # Concatenate forward and backward final hidden/cell per layer
        # Then project to decoder hidden_size
        batch = x.size(0)
        h_dec_layers = []
        c_dec_layers = []

        for layer in range(self.num_layers):
            # Forward layer idx: 2*layer,  Backward: 2*layer+1
            h_fwd = h_n[2 * layer]       # (batch, hidden_size)
            h_bwd = h_n[2 * layer + 1]   # (batch, hidden_size)
            c_fwd = c_n[2 * layer]
            c_bwd = c_n[2 * layer + 1]

            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, hidden_size*2)
            c_cat = torch.cat([c_fwd, c_bwd], dim=-1)

            h_dec_layers.append(torch.tanh(self.fc_h(h_cat)))  # (batch, hidden_size)
            c_dec_layers.append(torch.tanh(self.fc_c(c_cat)))

        # Stack back to (num_layers, batch, hidden_size)
        h_dec = torch.stack(h_dec_layers, dim=0)
        c_dec = torch.stack(c_dec_layers, dim=0)

        return enc_outputs, (h_dec, c_dec)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """Unidirectional LSTM decoder.

    Steps through forecast horizon one time-step at a time, producing
    three quantile predictions per step.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_quantiles)

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        """Single decoder step.

        Args:
            x      : (batch, 1, input_size) — one forecast timestep
            hidden : (h, c) each (num_layers, batch, hidden_size)

        Returns:
            out    : (batch, n_quantiles)
            hidden : updated (h, c)
        """
        output, hidden = self.lstm(x, hidden)  # output: (batch, 1, hidden_size)
        out = self.fc(output.squeeze(1))        # (batch, n_quantiles)
        return out, hidden


# ---------------------------------------------------------------------------
# Seq2Seq wrapper
# ---------------------------------------------------------------------------

class Seq2SeqQuantile(nn.Module):
    """Full Seq2Seq model for quantile inflow forecasting.

    Default hyperparameters match the PLTA Grindulu specification:
        enc_input_size = 2   [p_das, q_total]
        dec_input_size = 1   [p_das_forecast]
        hidden_size    = 128
        num_layers     = 2
        dropout        = 0.2
        enc_len        = 30  (look-back window)
        pred_len       = 7   (forecast horizon)
        n_quantiles    = 3   (q10, q50, q90)
    """

    def __init__(
        self,
        enc_input_size: int = 2,
        dec_input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pred_len: int = 7,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles

        self.encoder = BiLSTMEncoder(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            input_size=dec_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_quantiles=n_quantiles,
        )

    def forward(
        self,
        enc_input: torch.Tensor,
        dec_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enc_input : (batch, enc_len, 2)     — past [p_das, q_total]
            dec_input : (batch, pred_len, 1)    — future [p_das_forecast]

        Returns:
            predictions : (batch, pred_len, 3)  — [q10, q50, q90] each day
        """
        _, hidden = self.encoder(enc_input)

        outputs = []
        for t in range(self.pred_len):
            x_t = dec_input[:, t : t + 1, :]   # (batch, 1, dec_input_size)
            out_t, hidden = self.decoder.forward_step(x_t, hidden)
            outputs.append(out_t)               # (batch, 3)

        predictions = torch.stack(outputs, dim=1)  # (batch, pred_len, 3)
        return predictions


# ---------------------------------------------------------------------------
# Pinball (Quantile) Loss
# ---------------------------------------------------------------------------

def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float] | tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Mean pinball loss across all quantiles and forecast steps.

    Args:
        pred     : (batch, pred_len, n_quantiles) — model output
        target   : (batch, pred_len)              — ground-truth q_total
        quantiles: sequence of tau values matching n_quantiles

    Returns:
        scalar loss tensor
    """
    target = target.unsqueeze(-1)  # (batch, pred_len, 1) — broadcast over quantiles
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)  # (n_quantiles,)

    errors = target - pred                          # (batch, pred_len, n_quantiles)
    loss = torch.where(
        errors >= 0,
        q * errors,
        (q - 1.0) * errors,
    )
    return loss.mean()


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    batch, enc_len, pred_len = 8, 30, 7

    model = Seq2SeqQuantile()
    enc_x = torch.randn(batch, enc_len, 2)
    dec_x = torch.randn(batch, pred_len, 1)
    preds = model(enc_x, dec_x)

    assert preds.shape == (batch, pred_len, 3), f"Unexpected shape: {preds.shape}"

    targets = torch.randn(batch, pred_len)
    loss = pinball_loss(preds, targets)
    print(f"Model output shape : {preds.shape}")
    print(f"Pinball loss       : {loss.item():.6f}")
    print("model.py sanity check passed.")
