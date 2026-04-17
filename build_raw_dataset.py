"""
build_raw_dataset.py
Extract rainfall directly from the 6 raw PDFs in raw data/,
apply SCS-CN surface runoff + dynamic linear reservoir baseflow routing,
and save to data_grindulu_raw.csv.

Hydrology pipeline (implements paper methodology section 3.4 step 2):
  a. SCS Curve Number  -> surface runoff Q_runoff
  b. Rainfall-runoff transformation
  c. Linear reservoir routing -> dynamic Q_baseflow  *** improved from constant ***
  d. Q_total = Q_runoff + Q_baseflow

data_grindulu.csv is NOT touched.

Usage:
    python build_raw_dataset.py
"""

import re
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR  = Path("raw data")
OUT_FILE = Path("data_grindulu_raw.csv")

PDF_STATIONS = {
    "pacitan":    "Pos Curah Hujan Pacitan - Pacitan 2014 - 2024.pdf",
    "nawangan":   "Pos Curah Hujan Nawangan Grindulu - Pacitan 2014 - 2024.pdf",
    "kebonagung": "Curah Hujan Kebongagung - Pacitan 2014 - 2024.pdf",
    "bandar":     "DATA CURAH HUJAN Bandar.pdf",
    "tegalombo":  "DATA CURAH HUJAN Tegalombo.pdf",
    "tulakan":    "DATA CURAH HUJAN Tulakan.pdf",
}

# ---------------------------------------------------------------------------
# SCS-CN constants (from PERHITUNGAN.xlsx)
# Composite CN = 0.0272*55 + 0.1733*78 + 0.4069*72 + 0.3926*90 = 79.644 → 80
# ---------------------------------------------------------------------------

CN      = 80.0
S       = (25400 / CN) - 254   # 63.5 mm
Ia      = 0.2 * S              # 12.7 mm  (initial abstraction)
DAS_KM2 = 700.0                # km²  catchment area

# ---------------------------------------------------------------------------
# Dynamic baseflow — linear reservoir routing
# Implements "Routing hidrologi" (paper methodology 3.4 step 2c)
#
# Q_base(t) = max(Q_BASE_MIN,  K * Q_base(t-1)  +  ALPHA * Q_runoff(t))
#                               ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^
#                               groundwater          recharge from today's
#                               recession            surface runoff
#
# Parameters calibrated for tropical Java DAS (literature-based):
#   K     = 0.92 → baseflow halves in ~8 days without rain (typical Java)
#   ALPHA = 0.15 → 15% of surface runoff recharges groundwater
#   Q_BASE_MIN  = physical lower bound (same as original constant estimate)
# ---------------------------------------------------------------------------

K_RECESSION   = 0.92     # groundwater recession coefficient
ALPHA_RECHARGE = 0.15    # fraction of Q_runoff recharging groundwater
Q_BASE_MIN    = 3.92918  # m³/s  minimum baseflow (physical lower bound)
Q_BASE_INIT   = 3.92918  # m³/s  initial condition (2014-01-01)

# ---------------------------------------------------------------------------
# Month-name → month-number map (handles both BBWS and Hidrologi formats)
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Mei": 5, "Jun": 6,
    "Jul": 7, "Agst": 8, "Ags": 8, "Sep": 9, "Okt": 10, "Nop": 11, "Des": 12,
}

# Words that mark the end of the data table
STOP_WORDS = {
    "total", "hari", "rerata", "maksimum", "periode",
    "perbandingan", "debit", "rata-rata",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_year(text: str) -> int | None:
    """Find the data year on a PDF page."""
    for pattern in [
        r"Tahun\s+Data\s*:\s*(\d{4})",   # BBWS format
        r"Tahun\s*:\s*(\d{4})",           # Hidrologi format
    ]:
        m = re.search(pattern, text)
        if m:
            yr = int(m.group(1))
            if 2014 <= yr <= 2024:
                return yr
    return None


def parse_value(tok: str) -> float:
    """Convert a table cell to mm; dashes / blanks → 0."""
    tok = tok.strip()
    if tok in ("-", "", "nan", "#DIV/0!"):
        return 0.0
    try:
        return max(0.0, float(tok))
    except ValueError:
        return 0.0


def parse_page(text: str, year: int) -> dict:
    """
    Parse one page of a daily rainfall table.

    Returns {(month, day): mm}  for every cell found.

    Works for both PDF formats:
      - BBWS  : header row contains Tgl + 12 month names, dashes = no rain
      - Hidrologi: header row contains Tanggal + 12 month names, zeros = no rain
    """
    lines = text.splitlines()

    # Locate the month-name header row
    header_idx   = None
    months_order = []

    for i, line in enumerate(lines):
        tokens = line.strip().split()
        found_months = [t for t in tokens if t in MONTH_MAP]
        if len(found_months) >= 10:                  # at least 10 of 12
            months_order = [MONTH_MAP[t] for t in tokens if t in MONTH_MAP]
            header_idx   = i
            break

    if header_idx is None or len(months_order) < 10:
        return {}

    data = {}
    for line in lines[header_idx + 1:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        # Stop at summary rows
        if tokens[0].lower() in STOP_WORDS:
            break
        # First token must be a day number (1-31)
        try:
            day = int(tokens[0])
        except ValueError:
            continue
        if not (1 <= day <= 31):
            break

        for col_i, val in enumerate(tokens[1:]):
            if col_i >= len(months_order):
                break
            month = months_order[col_i]
            try:
                max_day = calendar.monthrange(year, month)[1]
            except Exception:
                continue
            if day > max_day:
                continue
            data[(month, day)] = parse_value(val)

    return data


# ---------------------------------------------------------------------------
# Per-station extraction
# ---------------------------------------------------------------------------

def extract_station(pdf_path: Path, station: str) -> pd.Series:
    """
    Parse all pages of a station PDF.
    Returns a daily pd.Series indexed 2014-01-01 → 2024-12-31.
    """
    date_idx = pd.date_range("2014-01-01", "2024-12-31", freq="D")
    daily = pd.Series(0.0, index=date_idx, name=station)

    years_found = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            year = extract_year(text)
            if year is None:
                continue
            page_data = parse_page(text, year)
            if not page_data:
                continue
            years_found.append(year)
            for (month, day), mm in page_data.items():
                try:
                    date = pd.Timestamp(year, month, day)
                    if date in daily.index:
                        daily[date] = mm
                except Exception:
                    pass

    rainy = int((daily > 0).sum())
    print(f"  {station:<12}: years {sorted(set(years_found))} | {rainy} rainy days")
    return daily


# ---------------------------------------------------------------------------
# Hydrology functions
# ---------------------------------------------------------------------------

def scs_runoff_mm(p_das: pd.Series) -> pd.Series:
    """Rainfall excess Pe (mm) via SCS Curve Number method."""
    pe = np.where(
        p_das > Ia,
        (p_das - Ia) ** 2 / (p_das - Ia + S),
        0.0,
    )
    return pd.Series(pe, index=p_das.index, name="pe")


def runoff_to_flow(pe_mm: pd.Series) -> pd.Series:
    """Convert runoff depth (mm) over catchment area to m³/s."""
    v_m3 = pe_mm * DAS_KM2 * 1_000   # mm * km² * 1000 = m³
    return v_m3 / 86_400              # daily volume -> m³/s


def dynamic_baseflow(q_runoff: pd.Series) -> pd.Series:
    """
    Linear reservoir baseflow routing (implements 'Routing hidrologi').

    Q_base(t) = max(Q_BASE_MIN,  K * Q_base(t-1)  +  ALPHA * Q_runoff(t))

    Parameters (literature-based for tropical Java DAS):
      K_RECESSION    = 0.92  (baseflow halves in ~8 days without rain)
      ALPHA_RECHARGE = 0.15  (15% of surface runoff recharges groundwater)
      Q_BASE_MIN     = 3.929 m³/s  (physical minimum, same as original estimate)
    """
    arr = q_runoff.values
    q_base = np.empty(len(arr))
    q_base[0] = Q_BASE_INIT
    for t in range(1, len(arr)):
        q_base[t] = max(
            Q_BASE_MIN,
            K_RECESSION * q_base[t - 1] + ALPHA_RECHARGE * arr[t],
        )
    return pd.Series(q_base, index=q_runoff.index, name="q_baseflow")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Building data_grindulu_raw.csv from raw PDFs")
    print("=" * 60)

    # --- Extract rainfall from each station PDF ---
    station_series = {}
    for station, filename in PDF_STATIONS.items():
        path = RAW_DIR / filename
        print(f"\n[{station}]  {filename}")
        station_series[station] = extract_station(path, station)

    # --- Assemble DataFrame ---
    df = pd.DataFrame(station_series)
    df.index.name = "date"
    df = df.reset_index()

    # P_DAS = simple mean of 6 stations
    # (confirmed from XLSX comment: "PDAS = jumlah hujan tiap pos / jumlah pos")
    df["p_das"] = df[list(PDF_STATIONS.keys())].mean(axis=1)

    # --- Hydrology pipeline ---
    df["pe"]         = scs_runoff_mm(df["p_das"]).values
    df["q_runoff"]   = runoff_to_flow(df["pe"]).values
    df["q_baseflow"] = dynamic_baseflow(df["q_runoff"]).values
    df["q_total"]    = df["q_runoff"] + df["q_baseflow"]

    # --- Column order matching data_grindulu.csv ---
    cols = [
        "date", "pacitan", "nawangan", "kebonagung",
        "bandar", "tegalombo", "tulakan",
        "p_das", "pe", "q_runoff", "q_baseflow", "q_total",
    ]
    df = df[cols]

    df.to_csv(OUT_FILE, index=False)

    # --- Summary ---
    Q_MIN_1UNIT = 0.30 * 60.5   # 18.15 m³/s
    operable    = (df["q_total"] >= Q_MIN_1UNIT).sum()
    total_days  = len(df)

    print(f"\n{'=' * 60}")
    print(f"Saved -> {OUT_FILE}  ({total_days:,} rows)")
    print(f"Period : {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    print(f"\nQ_total stats:")
    print(f"  mean       = {df['q_total'].mean():.3f} m³/s")
    print(f"  min        = {df['q_total'].min():.3f} m³/s")
    print(f"  max        = {df['q_total'].max():.3f} m³/s")
    print(f"  Q_baseflow mean = {df['q_baseflow'].mean():.3f} m³/s  (was 3.929 constant)")
    print(f"\nTurbine operability (Q >= 18.15 m³/s = 1 unit min):")
    print(f"  {operable} / {total_days} days  ({operable/total_days*100:.1f}%)")

    print(f"\nBaseflow routing parameters used:")
    print(f"  K_RECESSION    = {K_RECESSION}  (recession coefficient)")
    print(f"  ALPHA_RECHARGE = {ALPHA_RECHARGE}  (recharge fraction)")
    print(f"  Q_BASE_MIN     = {Q_BASE_MIN} m³/s")

    # --- Compare P_DAS vs original (surface runoff source is same) ---
    orig_path = Path("data_grindulu.csv")
    if orig_path.exists():
        ref = pd.read_csv(orig_path, parse_dates=["date"])
        merged = df.merge(ref, on="date", suffixes=("_raw", "_orig"))
        mae_p   = (merged["p_das_raw"] - merged["p_das_orig"]).abs().mean()
        match_p = (merged["p_das_raw"].round(2) == merged["p_das_orig"].round(2)).mean() * 100
        orig_op = (ref["q_total"] >= Q_MIN_1UNIT).sum()
        print(f"\nComparison vs data_grindulu.csv (original):")
        print(f"  P_DAS match      : {match_p:.1f}%  (MAE={mae_p:.4f} mm)")
        print(f"  Operable days    : {orig_op} ({orig_op/total_days*100:.1f}%)  ->  {operable} ({operable/total_days*100:.1f}%)")


if __name__ == "__main__":
    main()
