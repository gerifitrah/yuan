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

# Regenerate data_grindulu.csv from PERHITUNGAN_v2.xlsx (one-shot)
python extract_v2_data.py

# Regenerate data from raw PDFs (6 stations)
python build_raw_dataset.py

# Add new rainfall data to dataset
# (edit RAINFALL_DATA dict in update_data.py first)
python update_data.py

# Sanity checks
python model.py
python preprocess.py
```

## Architecture

The system is a **Bi-LSTM Seq2Seq Quantile Regression** pipeline for daily streamflow (inflow) forecasting, coupled with a **Pumped Storage hydropower simulation**.

### Data Flow

```
PERHITUNGAN_v2.xlsx  →  extract_v2_data.py  →  data_grindulu.csv
raw PDFs (6 stations) →  build_raw_dataset.py →  data_grindulu_raw.csv

data_grindulu.csv  →  preprocess.py  →  train.py  →  saved_model/best_model.pt
                                                     saved_model/feature_scaler.pkl
                                                     saved_model/q_scaler.pkl
```

At inference time (in `app.py`):
1. Load `best_model.pt` + both scalers
2. Assemble encoder input: last 30 days of `[p_das, q_total]` (scaled)
3. Assemble decoder input: next 7 days of `[p_das_forecast]` (scaled)
4. Model outputs `(1, 7, 3)` → inverse-transform → `[Q10, Q50, Q90]` m³/s per day

### Hydrology Parameters (from PERHITUNGAN_v2.xlsx)

| Parameter | Value | Source sheet |
|---|---|---|
| CN | 76.46 | SCS-CN |
| S | 78.1998 mm | derived |
| Ia | 15.64 mm | derived (0.2 × S) |
| DAS_KM2 (runoff area) | 127.86 km² | HUJAN WILAYAH DAS |
| Thiessen total area | 754.24 km² | LUAS DAERAH |
| Q_baseflow | 1.02111226 m³/s (constant) | HUJAN WILAYAH DAS |

Thiessen weights (Wi = Ai / 754.24):
- pacitan: 77.11, nawangan: 124.06, kebonagung: 124.85
- bandar: 117.34, tegalombo: 149.26, tulakan: 161.62

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
- **`charts.py`** — Plotly chart functions (return `go.Figure`): `chart_observed_vs_predicted`, `chart_quantile_prediction`, `chart_prediction_interval`, `chart_turbine_efficiency`, `chart_fixed_vs_variable_speed`, `chart_power_output`, `chart_probabilistic_forecast`, `chart_all_quantiles`, `chart_uncertainty_width`
- **`extract_v2_data.py`** — One-shot script: reads PERHITUNGAN_v2.xlsx, recalculates hydrology, writes data_grindulu.csv
- **`build_raw_dataset.py`** — Extracts rainfall from 6 raw PDFs, applies SCS-CN + dynamic baseflow routing
- **`update_data.py`** — Appends new daily rainfall rows to data_grindulu.csv
- **`app.py`** — Streamlit dashboard with 5 tabs

### Streamlit App Structure (`app.py`)

Five tabs:
1. **Historical Data** — raw CSV explorer + 6 charts from `charts.py`
2. **7-Day Forecast** — date picker → `_show_forecast_results()` → pumped storage section
3. **30-Day Forecast** — rolling 5×7-day inference with P_DAS input table → pumped storage section
4. **Simulasi Tahunan** — climatological Q10/Q50/Q90 per month → pumped storage section
5. **Analysis & Turbine Performance** — model eval charts + turbine characteristic charts

Pumped storage simulation functions (defined in `app.py`, before helpers):
- `simulate_ps_day(q_inflow, vol_upper, vol_lower, hours_gen)` → dict with energy/reservoir state
- `simulate_ps_series(q_array, vol_upper_init_pct, hours_gen)` → DataFrame
- `render_ps_section(date_strs, q10, q50, q90, label, key_prefix)` — 2-row chart: energy (MWh) + upper reservoir level (%)

### PLTA Grindulu Pumped Storage Specs — OPSI-1A (hardcoded in `app.py`)

```python
H_NET         = 486.5       # m  net head
Q_DESIGN      = 60.5        # m³/s per unit (turbine mode)
N_UNITS       = 4
ETA_T_MAX     = 0.90        # max turbine efficiency
ETA_G         = 0.96        # generator efficiency
ETA_PUMP      = 0.85        # pump mode efficiency
VOL_UPPER_MAX = 7_900_000   # m³  upper reservoir (7.90 juta m³)
VOL_LOWER_MAX = 8_990_000   # m³  lower reservoir (8.99 juta m³)
Q_TURBINE     = 242         # m³/s total (4 × 60.5)
P_RATED_MW    = 1000        # MW total rated power
HOURS_GEN     = 8           # h/day peak generation
HOURS_PUMP    = 16          # h/day off-peak pumping
ETA_LOSS_FRAC = 0.03        # 3% reservoir losses per cycle
```

Daily operation cycle: Night pump (16h, lower→upper) → Day generation (8h peak, upper→lower) → River inflow (adds to lower) → Losses (3% of generated volume).

River inflow (q_total from model) supplements the lower reservoir — the system is water-recycling, not run-of-river. Average inflow ~1.65 m³/s; design flow 242 m³/s comes from pumped storage cycling.

### Forecast for Future Dates

When the selected date is after `data_end` (2024-12-31), the app uses the **last 30 days of the dataset** as look-back, not real future data. This is a design choice — results are climatologically plausible but not data-grounded.

### Scaler Notes

Two separate scalers are saved:
- `feature_scaler.pkl` → dict `{"p_scaler": ..., "q_scaler": ...}` — used in `make_inference_sequence()`
- `q_scaler.pkl` → bare `MinMaxScaler` for `q_total` only — used in `evaluate.py`

Both are fit on **training data only** (first 70% of the time series).
