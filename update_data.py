"""
update_data.py — Tambah data hujan baru ke data_grindulu.csv

Cara pakai:
    1. Isi tabel RAINFALL_DATA di bawah dengan data hujan baru
    2. Jalankan: python update_data.py
    3. Script otomatis hitung Pe, Q_runoff, Q_total dan append ke CSV

Format input:
    "YYYY-MM-DD": [pacitan, nawangan, kebonagung, bandar, tegalombo, tulakan]
    Satuan: mm/hari. Gunakan 0.0 jika tidak ada hujan.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# EDIT BAGIAN INI — isi data hujan per stasiun per hari
# ---------------------------------------------------------------------------
RAINFALL_DATA = {
    # Format: "YYYY-MM-DD": [pacitan, nawangan, kebonagung, bandar, tegalombo, tulakan]
    # Contoh data — ganti dengan data nyata Anda:
    "2025-01-01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "2025-01-02": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # tambahkan baris berikutnya...
}
# ---------------------------------------------------------------------------

# SCS-CN parameter (dari PERHITUNGAN_v2.xlsx — SCS-CN sheet & LUAS DAERAH sheet)
CN         = 76.46
S          = (25400 / CN) - 254   # 78.1998 mm
Ia         = 0.2 * S              # 15.64 mm
LUAS_DAS   = 127.86e6             # 127.86 km² dalam m² (luas efektif DAS)
Q_BASEFLOW = 1.02111226           # m³/s (konstan, dari HUJAN WILAYAH DAS sheet)

# Bobot Thiessen (dari LUAS DAERAH sheet — Wi = Ai / A_total)
# Urutan: [pacitan, nawangan, kebonagung, bandar, tegalombo, tulakan]
_THIESSEN_WEIGHTS = [
    77.11  / 754.24,   # pacitan
    124.06 / 754.24,   # nawangan
    124.85 / 754.24,   # kebonagung
    117.34 / 754.24,   # bandar
    149.26 / 754.24,   # tegalombo
    161.62 / 754.24,   # tulakan
]

CSV_PATH = os.path.join(os.path.dirname(__file__), "data_grindulu.csv")


def scs_cn(p_das):
    """Hitung hujan efektif Pe (mm) dari P_DAS menggunakan SCS-CN."""
    if p_das <= Ia:
        return 0.0
    return (p_das - Ia) ** 2 / (p_das - Ia + S)


def compute_row(date_str, stations):
    """Hitung semua kolom dari data hujan stasiun."""
    p_das     = np.dot(stations, _THIESSEN_WEIGHTS)
    pe        = scs_cn(p_das)
    q_runoff  = (pe / 1000) * LUAS_DAS / 86400   # m³/s
    q_total   = q_runoff + Q_BASEFLOW

    return {
        "date":       date_str,
        "pacitan":    stations[0],
        "nawangan":   stations[1],
        "kebonagung": stations[2],
        "bandar":     stations[3],
        "tegalombo":  stations[4],
        "tulakan":    stations[5],
        "p_das":      round(p_das, 4),
        "pe":         round(pe, 6),
        "q_runoff":   round(q_runoff, 6),
        "q_baseflow": Q_BASEFLOW,
        "q_total":    round(q_total, 6),
    }


def main():
    # Load existing CSV
    df_existing = pd.read_csv(CSV_PATH, parse_dates=["date"])
    last_date   = df_existing["date"].max()
    print(f"Data existing: {len(df_existing)} baris, terakhir: {last_date.date()}")

    # Build new rows
    new_rows = []
    skipped  = []
    for date_str, stations in sorted(RAINFALL_DATA.items()):
        d = pd.Timestamp(date_str)
        if d <= last_date:
            skipped.append(date_str)
            continue
        if len(stations) != 6:
            print(f"  SKIP {date_str}: butuh 6 stasiun, dapat {len(stations)}")
            continue
        new_rows.append(compute_row(date_str, stations))

    if skipped:
        print(f"Dilewati (sudah ada di CSV): {skipped}")

    if not new_rows:
        print("Tidak ada data baru yang ditambahkan.")
        return

    # Append
    df_new = pd.DataFrame(new_rows)
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out = df_out.sort_values("date").reset_index(drop=True)
    df_out.to_csv(CSV_PATH, index=False)

    print(f"\nBerhasil menambahkan {len(new_rows)} baris baru:")
    for r in new_rows:
        print(f"  {r['date']} | P_DAS={r['p_das']:.2f}mm | "
              f"Q_runoff={r['q_runoff']:.3f} | Q_total={r['q_total']:.3f} m3/s")

    print(f"\nTotal data sekarang: {len(df_out)} baris")
    print(f"Periode: {df_out['date'].min().date()} s.d. {df_out['date'].max().date()}")
    print("\nJangan lupa retrain model setelah data diperbarui:")
    print("  python train.py --epochs 100")


if __name__ == "__main__":
    main()
