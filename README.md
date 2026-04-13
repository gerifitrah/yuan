# PLTA Grindulu — Bi-LSTM Seq2Seq Inflow Prediction

Sistem prediksi debit inflow harian untuk PLTA Grindulu (Pumped Storage, 1000 MW)
di DAS Grindulu, Pacitan, Jawa Timur.
Karena PLTA ini belum dibangun (*greenfield*), debit inflow tidak diukur langsung —
melainkan dihitung dari data curah hujan historis menggunakan metode SCS Curve Number.

---

## Struktur File

```
new version/
├── data_grindulu.csv      ← Dataset utama (4.018 hari, 2014–2024)
├── model.py               ← Arsitektur Bi-LSTM Seq2Seq (PyTorch)
├── preprocess.py          ← Load data, normalisasi, buat sequence
├── train.py               ← Training loop (Pinball Loss)
├── evaluate.py            ← Metrik probabilistik (CRPS, PICP, PINAW)
├── charts.py              ← Semua fungsi chart (Plotly)
├── app.py                 ← Streamlit dashboard
├── requirements.txt
└── saved_model/
    ├── best_model.pt      ← Model terbaik (val loss terendah)
    ├── feature_scaler.pkl ← MinMaxScaler untuk input fitur
    ├── q_scaler.pkl       ← MinMaxScaler untuk q_total
    └── train_history.csv  ← Loss per epoch
```

---

## Dataset: `data_grindulu.csv`

**Periode:** 1 Januari 2014 – 31 Desember 2024  
**Jumlah baris:** 4.018 hari  
**Jumlah kolom:** 12

### Penjelasan Setiap Kolom

---

#### `date`
Tanggal pengamatan dalam format `YYYY-MM-DD`.
- Rentang: 2014-01-01 s.d. 2024-12-31
- Frekuensi: harian (tidak ada tanggal yang hilang)

---

#### `pacitan`, `nawangan`, `kebonagung`, `bandar`, `tegalombo`, `tulakan`
**Curah hujan harian (mm) di masing-masing stasiun penakar hujan** dalam DAS Grindulu.

| Kolom | Nama Stasiun | Elevasi | Operasi |
|-------|-------------|---------|---------|
| `pacitan` | Stasiun Pacitan | — | 2014–2024 |
| `nawangan` | Stasiun Nawangan | — | 2014–2024 |
| `kebonagung` | Stasiun Kebon Agung | — | 2014–2024 |
| `bandar` | Stasiun Bandar | 957 m | 2014–2024 |
| `tegalombo` | Stasiun Tegalombo | 200 m | 2014–2024 |
| `tulakan` | Stasiun Tulakan | 350 m | 2014–2024 |

> **Satuan:** mm/hari  
> **Nilai 0:** tidak ada hujan pada hari tersebut  
> **Sumber:** BBWS Bengawan Solo (data PDF per stasiun)

Statistik ringkas:

| Stasiun | Rata-rata | Maks | Catatan kualitas data |
|---------|-----------|------|-----------------------|
| pacitan | 6.35 mm | 245 mm | — |
| nawangan | 7.32 mm | 195 mm | Apr 2018: 1 hari data invalid |
| kebonagung | 8.40 mm | 304 mm | Nov 2017: 2 hari diduga salah baca 10× |
| bandar | 7.10 mm | 206 mm | Okt–Nov 2022: ~8 hari kosong |
| tegalombo | 5.89 mm | 189 mm | Mei–Des 2015: 8 bulan semua nol (alat rusak) |
| tulakan | 7.23 mm | 293 mm | 2017 Jan & 2020: total ~47 hari kosong |

---

#### `p_das`
**Curah hujan wilayah DAS (mm/hari)** — rata-rata aritmatika dari 6 stasiun.

```
P_DAS = (Pacitan + Nawangan + Kebonagung + Bandar + Tegalombo + Tulakan) / 6
```

> Meskipun sheet Excel bernama "Thiessen Polygon", perhitungan aktual menggunakan
> **rata-rata aritmatika** (bobot seragam = 1/6 untuk setiap stasiun).

| Statistik | Nilai |
|-----------|-------|
| Rata-rata | 7.05 mm/hari |
| Nilai maks | 138.83 mm (28 Nov 2017) |
| Nilai min | 0.00 mm |
| Hari dengan hujan (P_DAS > 0) | 2.695 hari (67.1%) |

---

#### `pe`
**Curah hujan efektif (mm/hari)** — bagian dari curah hujan yang menjadi limpasan permukaan,
dihitung menggunakan metode **SCS Curve Number (CN)**.

**Formula:**

```
S  = (25400 / CN) - 254        → S  = 63.5 mm  (dengan CN = 80)
Ia = 0.2 × S                   → Ia = 12.7 mm  (abstraksi awal)

Jika P_DAS > Ia:
    Pe = (P_DAS - Ia)² / (P_DAS - Ia + S)
Jika P_DAS ≤ Ia:
    Pe = 0
```

**Parameter DAS Grindulu:**

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| CN | 80 | Tata guna lahan campuran (hutan + pertanian) |
| S | 63.5 mm | Retensi potensial maksimum |
| Ia | 12.7 mm | Abstraksi awal (kehilangan sebelum limpasan mulai) |

> Artinya: hujan baru mulai menghasilkan limpasan setelah melebihi **12.7 mm**.
> Itulah mengapa banyak hari bernilai Pe = 0 meskipun ada hujan kecil.

| Statistik | Nilai |
|-----------|-------|
| Rata-rata | 0.63 mm/hari |
| Nilai maks | 83.90 mm |
| Hari dengan Pe > 0 | 784 hari (19.5%) |

---

#### `q_runoff`
**Debit limpasan permukaan (m³/s)** — konversi dari volume limpasan SCS-CN ke debit harian.

**Formula:**

```
Luas DAS  = 700 km² = 700 × 10⁶ m²
Volume    = Pe (m) × Luas DAS (m²)           [dalam m³]
Q_runoff  = Volume / 86.400                  [dibagi detik dalam 1 hari]
          = (Pe / 1000) × 700×10⁶ / 86400
```

> **Asumsi:** limpasan tersebar merata selama 24 jam penuh.
> Ini adalah penyederhanaan untuk skala feasibility study.

| Statistik | Nilai |
|-----------|-------|
| Rata-rata | 5.11 m³/s |
| Nilai maks | 679.72 m³/s |
| Hari dengan Q_runoff > 0 | 784 hari (19.5%) |

---

#### `q_baseflow`
**Debit aliran dasar / baseflow (m³/s)** — aliran sungai minimum yang berasal dari
air tanah (groundwater), bukan dari hujan langsung.

```
Q_baseflow = konstan = 3.9292 m³/s  (untuk semua hari)
```

> Nilai konstan ini diambil dari analisis debit minimum historis DAS Grindulu.
> Pada hari kering (tidak ada hujan / limpasan = 0), sungai tetap mengalir
> dengan debit dasar ini.

---

#### `q_total`
**Debit total (m³/s)** — variabel TARGET model prediksi.

```
Q_total = Q_runoff + Q_baseflow
```

Ini merepresentasikan **total debit inflow ke reservoir PLTA Grindulu** setiap hari.

| Statistik | Nilai |
|-----------|-------|
| Rata-rata | 9.04 m³/s |
| Nilai maks | 683.65 m³/s (28 Nov 2017) |
| Nilai min | 3.93 m³/s (hari kering, baseflow saja) |
| Hari dengan Q_total = baseflow (kering) | 3.234 hari (80.5%) |

> **Catatan:** 80.5% hari memiliki Q_total = Q_baseflow = 3.93 m³/s karena
> tidak ada limpasan (P_DAS ≤ Ia = 12.7 mm). Distribusi ini **sangat right-skewed**
> (ekor panjang ke kanan) — kondisi normal adalah kering, banjir adalah outlier.

---

### Ringkasan Alur Perhitungan Data

```
Hujan Stasiun (6 titik)
        ↓  rata-rata aritmatika
    P_DAS (mm/hari)
        ↓  SCS-CN  [CN=80, S=63.5mm, Ia=12.7mm]
      Pe (mm/hari)  ← hujan efektif
        ↓  × Luas DAS / 86400
   Q_runoff (m³/s)
        ↓  + baseflow konstan
   Q_total (m³/s)  ← TARGET PREDIKSI
```

---

## Model Arsitektur

```
Input (Encoder)         30 hari ke belakang:  [P_DAS, Q_total]
                                 ↓
                    Bi-LSTM Encoder (128 hidden, 2 layer)
                    membaca urutan maju DAN mundur
                                 ↓
                    Proyeksi hidden state → Decoder
                                 ↓
Input (Decoder)          7 hari ke depan:  [P_DAS_forecast]
                                 ↓
                    LSTM Decoder (128 hidden, 2 layer)
                    step-by-step untuk setiap hari
                                 ↓
Output               [Q10, Q50, Q90] × 7 hari
```

**Loss Function:** Pinball Loss (Quantile Loss) pada τ = 0.1, 0.5, 0.9

---

## Pembagian Data

| Split | Proporsi | Jumlah Hari | Periode (approx) |
|-------|----------|-------------|-----------------|
| Training | 70% | 2.812 hari | Jan 2014 – Sep 2021 |
| Validasi | 15% | 603 hari | Sep 2021 – Apr 2023 |
| Test | 15% | 603 hari | Apr 2023 – Des 2024 |

> Pembagian **kronologis** (bukan acak) untuk menghindari *data leakage* —
> model tidak pernah melihat data masa depan saat training.

---

## Hasil Evaluasi (Test Set)

| Metrik | Nilai | Keterangan |
|--------|-------|-----------|
| **CRPS** | 0.0978 m³/s | Skor probabilistik gabungan, lebih kecil lebih baik |
| **PICP** | 94.48% | 94.48% observasi masuk dalam interval [Q10, Q90] — target ≥ 80% ✅ |
| **PINAW** | 0.0037 | Lebar interval relatif terhadap range observasi, lebih kecil lebih baik |

---

## Spesifikasi PLTA Grindulu

| Parameter | Nilai |
|-----------|-------|
| Kapasitas total | 1.000 MW (4 × 250 MW) |
| Gross head | 500 m |
| Net head | 486,5 m |
| Debit desain per unit | 60,5 m³/s |
| Jumlah unit | 4 unit |
| Tipe turbin | Francis Reversible Pumped |
| Efisiensi turbin (ηT) | 90% |
| Efisiensi generator (ηG) | 96% |
| Kecepatan | 500 RPM |
| Tegangan generator | 18 kV |
| Daya generator | 344 MVA / 275 MW |

---

## Cara Penggunaan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Training model
python train.py --epochs 100 --lr 0.001

# 3. Evaluasi model
python evaluate.py

# 4. Jalankan dashboard
streamlit run app.py
```

---

## Catatan Kualitas Data

| Masalah | Dampak | Penanganan |
|---------|--------|-----------|
| Tegalombo Mei–Des 2015 semua nol | ~245 hari P_DAS sedikit lebih rendah | Diterima — data dari Excel sumber |
| Tulakan 2020 ~47 hari kosong | P_DAS sedikit lebih rendah | Diterima |
| Kebonagung Nov 2017: 304+301mm (diduga 30.4+30.1mm) | 2 hari Q_total sangat tinggi | Tidak dikoreksi — menggunakan data Excel sumber |
| Tegalombo 2017 Feb: 6 hari kosong | Minor | Diterima |
| Bandar 2018 & 2022: beberapa hari kosong | Minor | Diterima |

Semua nilai kosong ("-") pada sumber PDF ditangani sebagai **0** di file Excel sumber,
sehingga `data_grindulu.csv` mencerminkan tepat apa yang ada di `PERHITUNGAN.xlsx`.
