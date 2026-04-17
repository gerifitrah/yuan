# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run Streamlit dashboard
streamlit run app.py

# Train model (saves to saved_model/)
python train.py --epochs 100 --lr 0.001

# Evaluate model on test set
python evaluate.py

# Run 1-year rolling forecast standalone
python annual_forecast.py

# Add new rainfall data to dataset
# (edit RAINFALL_DATA dict in update_data.py first)
python update_data.py

# Sanity checks
python model.py
python preprocess.py
```

## Architecture

The system is a **Bi-LSTM Seq2Seq Quantile Regression** pipeline for daily streamflow forecasting.

### Data Flow

```
data_grindulu.csv  →  preprocess.py  →  train.py  →  saved_model/best_model.pt
                                                     saved_model/feature_scaler.pkl
                                                     saved_model/q_scaler.pkl
```

At inference time (in `app.py` and `annual_forecast.py`):
1. Load `best_model.pt` + both scalers
2. Assemble encoder input: last 30 days of `[p_das, q_total]` (scaled)
3. Assemble decoder input: next 7 days of `[p_das_forecast]` (scaled)
4. Model outputs `(1, 7, 3)` → inverse-transform → `[Q10, Q50, Q90]` m³/s per day

### Key Constants (defined in `preprocess.py`, imported everywhere)

| Constant | Value | Meaning |
|---|---|---|
| `ENC_LEN` | 30 | Look-back window (days) |
| `PRED_LEN` | 7 | Forecast horizon (days) |
| `SAVE_DIR` | `./saved_model/` | Model artifacts directory |

### Files

- **`model.py`** — `BiLSTMEncoder` + `LSTMDecoder` + `Seq2SeqQuantile` + `pinball_loss()`
- **`preprocess.py`** — `load_data()` for training; `make_inference_sequence()` for single inference
- **`train.py`** — Training loop: AdamW + ReduceLROnPlateau + early stopping (patience=15)
- **`evaluate.py`** — `run_evaluation()` computing CRPS, PICP, PINAW on test set
- **`charts.py`** — Six Plotly chart functions (all return `go.Figure`): `chart_observed_vs_predicted`, `chart_quantile_prediction`, `chart_prediction_interval`, `chart_turbine_efficiency`, `chart_fixed_vs_variable_speed`, `chart_power_output`
- **`annual_forecast.py`** — `run_annual_forecast(year, scenario, p_das_override)` — rolling 52×7-day forecast; returns DataFrame with `date, p_das_used, q10, q50, q90, step`
- **`app.py`** — Streamlit dashboard with 5 tabs

### Streamlit App Structure (`app.py`)

Five tabs:
1. **Historical Data** — raw CSV explorer + 6 charts from `charts.py`
2. **7-Day Forecast** — date picker → `_show_forecast_results()` → turbine section
3. **30-Day Forecast** — rolling 5×7-day inference with P_DAS input table
4. **Simulasi Tahunan** — calls `run_annual_forecast()` + monthly bar chart + turbine section
5. **Analysis & Turbine Performance** — standalone charts

Shared turbine functions (defined in `app.py`, before helpers):
- `_eta_turbine(q_per_unit)` — Francis turbine efficiency curve
- `calc_turbine(q_total)` → dict with `n_units, q_unit, eta_t, power_mw, energy_mwh`
- `turbine_series(q_array)` → DataFrame
- `render_turbine_section(date_strs, q10, q50, q90, label)` — 2-row subplot (power bars + unit count)

### PLTA Grindulu Turbine Specs (hardcoded in `app.py`)

```python
H_NET = 486.5       # m
Q_DESIGN = 60.5     # m³/s per unit
N_UNITS = 4
ETA_T_MAX = 0.90
ETA_G = 0.96
Q_MIN_UNIT = 18.15  # m³/s minimum per unit
```

### Forecast for Future Dates

When the selected date is after `data_end` (2024-12-31), the app uses the **last 30 days of the dataset** as look-back, not real future data. This is a design choice — results are climatologically plausible but not data-grounded.

### Scaler Notes

Two separate scalers are saved:
- `feature_scaler.pkl` → dict `{"p_scaler": ..., "q_scaler": ...}` — used in `make_inference_sequence()`
- `q_scaler.pkl` → bare `MinMaxScaler` for `q_total` only — used in `evaluate.py` and `annual_forecast.py`

Both are fit on **training data only** (first 70% of the time series).
