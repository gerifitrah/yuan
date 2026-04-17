# PLTA Grindulu — Bi-LSTM Seq2Seq Inflow Prediction

Sistem prediksi debit inflow harian untuk **PLTA Grindulu (Pumped Storage, 1000 MW)**
di DAS Grindulu, Pacitan, Jawa Timur menggunakan model **Bi-LSTM Seq2Seq Quantile Regression**.

Karena PLTA ini belum dibangun (*greenfield*), debit inflow tidak diukur langsung —
melainkan direkonstruksi dari data curah hujan historis menggunakan pipeline hidrologi
SCS Curve Number + Linear Reservoir Routing.

---

## Struktur File

```
new version/
├── data_grindulu.csv          ← Dataset v1 (baseflow konstan, dari PERHITUNGAN.xlsx)
├── data_grindulu_raw.csv      ← Dataset v2 (baseflow dinamis, dari raw PDF langsung)
├── PERHITUNGAN.xlsx           ← Workbook Excel sumber (QC + SCS-CN pipeline)
├── raw data/                  ← PDF data hujan harian dari BBWS / Hidrologi PUPR
│   ├── Pos Curah Hujan Pacitan - Pacitan 2014 - 2024.pdf
│   ├── Pos Curah Hujan Nawangan Grindulu - Pacitan 2014 - 2024.pdf
│   ├── Curah Hujan Kebongagung - Pacitan 2014 - 2024.pdf
│   ├── DATA CURAH HUJAN Bandar.pdf
│   ├── DATA CURAH HUJAN Tegalombo.pdf
│   └── DATA CURAH HUJAN Tulakan.pdf
├── referansi/
│   ├── PENELITIAN.pdf         ← Metodologi penelitian
│   └── Tugas Paper_Kelompok 2_Rafsanjani_Arham.pdf  ← Referensi model Poso PLTA
├── build_raw_dataset.py       ← Build data_grindulu_raw.csv dari raw PDF
├── model.py                   ← Arsitektur Bi-LSTM Seq2Seq (PyTorch)
├── preprocess.py              ← Load data, normalisasi, buat sequence
├── train.py                   ← Training loop (Pinball Loss + early stopping)
├── evaluate.py                ← Metrik probabilistik (CRPS, PICP, PINAW)
├── annual_forecast.py         ← Rolling forecast 365 hari
├── charts.py                  ← Semua fungsi chart (Plotly)
├── app.py                     ← Streamlit dashboard (5 tab)
├── requirements.txt
└── saved_model/
    ├── best_model.pt          ← Model terbaik (val loss terendah)
    ├── feature_scaler.pkl     ← MinMaxScaler untuk fitur input
    ├── q_scaler.pkl           ← MinMaxScaler untuk q_total
    └── train_history.csv      ← Loss per epoch
```

---

## Dataset

### v1 — `data_grindulu.csv` (original)

Diekspor dari `PERHITUNGAN.xlsx`, menggunakan **baseflow konstan**.

| Info | Nilai |
|------|-------|
| Periode | 1 Jan 2014 – 31 Des 2024 |
| Jumlah baris | 4.018 hari |
| Q_baseflow | konstan = 3.929 m³/s |
| Hari turbin bisa beroperasi (Q ≥ 18.15 m³/s) | 324 hari (8.1%) |

---

### v2 — `data_grindulu_raw.csv` (improved)

Dibangun langsung dari 6 PDF raw, menggunakan **dynamic baseflow (linear reservoir routing)**.
Mengimplementasikan *Routing hidrologi* sesuai metodologi penelitian (PENELITIAN.pdf, 3.4.2.c).

| Info | Nilai |
|------|-------|
| Periode | 1 Jan 2014 – 31 Des 2024 |
| Jumlah baris | 4.018 hari |
| Q_baseflow | dinamis, rata-rata 11.1 m³/s (seasonal) |
| Hari turbin bisa beroperasi (Q ≥ 18.15 m³/s) | 833 hari (20.7%) |

Untuk meregenerasi file ini dari PDF raw:

```bash
python build_raw_dataset.py
```

---

### Penjelasan Kolom

| Kolom | Satuan | Deskripsi |
|-------|--------|-----------|
| `date` | — | Tanggal (YYYY-MM-DD), harian tanpa gap |
| `pacitan` | mm/hari | Curah hujan Stasiun Pacitan |
| `nawangan` | mm/hari | Curah hujan Stasiun Nawangan |
| `kebonagung` | mm/hari | Curah hujan Stasiun Kebon Agung |
| `bandar` | mm/hari | Curah hujan Stasiun Bandar (elv. 957 m) |
| `tegalombo` | mm/hari | Curah hujan Stasiun Tegalombo (elv. 200 m) |
| `tulakan` | mm/hari | Curah hujan Stasiun Tulakan (elv. 350 m) |
| `p_das` | mm/hari | Rata-rata aritmatika 6 stasiun |
| `pe` | mm/hari | Curah hujan efektif SCS-CN |
| `q_runoff` | m³/s | Debit limpasan permukaan |
| `q_baseflow` | m³/s | Debit aliran dasar (konstan v1 / dinamis v2) |
| `q_total` | m³/s | **Target prediksi** = q_runoff + q_baseflow |

---

## Pipeline Hidrologi

### v1 — Baseflow Konstan

```
Hujan 6 Stasiun
      ↓  rata-rata aritmatika
  P_DAS (mm/hari)
      ↓  SCS-CN  [CN=80, S=63.5mm, Ia=12.7mm]
    Pe (mm/hari)
      ↓  × 700 km² × 1000 / 86400
  Q_runoff (m³/s)
      ↓  + 3.929 (konstan)
  Q_total (m³/s)  ← TARGET
```

### v2 — Dynamic Baseflow (Linear Reservoir Routing)

```
Hujan 6 Stasiun
      ↓  rata-rata aritmatika
  P_DAS (mm/hari)
      ↓  SCS-CN  [CN=80, S=63.5mm, Ia=12.7mm]
    Pe (mm/hari)
      ↓  × 700 km² × 1000 / 86400
  Q_runoff (m³/s)
      ↓  linear reservoir routing
  Q_baseflow(t) = max(Q_min,  K × Q_base(t-1)  +  α × Q_runoff(t))
      ↓  +
  Q_total (m³/s)  ← TARGET
```

**Parameter routing (tipikal DAS tropis Jawa, berbasis literatur):**

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| K | 0.92 | Koefisien resesi; aliran dasar turun setengahnya dalam ~8 hari tanpa hujan |
| α | 0.15 | Fraksi recharge; 15% Q_runoff masuk ke groundwater |
| Q_min | 3.929 m³/s | Batas minimum fisik (sama dengan estimasi original) |

**Untuk paper:**
> *"Routing hidrologi dilakukan menggunakan model linear reservoir:*
> *Q_base(t) = max(Q_min, K · Q_base(t−1) + α · Q_runoff(t))*
> *dengan K = 0,92, α = 0,15, dan Q_min = 3,929 m³/s."*

---

## SCS-CN Parameters

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| CN | 80 | Tata guna lahan campuran (hutan 2.7% + sawah 17.3% + tegalan 40.7% + pemukiman 39.3%) |
| S | 63.5 mm | (25400/CN) − 254 |
| Ia | 12.7 mm | 0.2 × S (abstraksi awal) |
| Luas DAS | 700 km² | Delineasi GIS dari DEM |

---

## Arsitektur Model

```
Encoder Input   : 30 hari × [P_DAS, Q_total]
                        ↓
             Bi-LSTM Encoder
             hidden=128, layers=2
             (baca maju + mundur)
                        ↓
             Proyeksi state → Decoder
                        ↓
Decoder Input   : 7 hari × [P_DAS_forecast]
                        ↓
             LSTM Decoder
             hidden=128, layers=2
                        ↓
Output          : 7 hari × [Q10, Q50, Q90]
```

| Komponen | Detail |
|----------|--------|
| Loss function | Pinball Loss (τ = 0.10, 0.50, 0.90) |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Early stopping | patience = 15 epoch |
| Look-back window | 30 hari |
| Forecast horizon | 7 hari |

---

## Pembagian Data

| Split | Proporsi | Hari | Periode |
|-------|----------|------|---------|
| Training | 70% | 2.812 | Jan 2014 – Sep 2021 |
| Validasi | 15% | 603 | Sep 2021 – Mei 2023 |
| Test | 15% | 603 | Mei 2023 – Des 2024 |

> Pembagian **kronologis** — tidak ada data leakage.

---

## Hasil Evaluasi (Test Set)

| Metrik | Nilai | Target | Status |
|--------|-------|--------|--------|
| CRPS | 0.0978 m³/s | lebih kecil = lebih baik | — |
| PICP | 94.48% | ≥ 80% | ✅ |
| PINAW | 0.0037 | lebih kecil = lebih baik | — |

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
| Efisiensi turbin ηT | 90% |
| Efisiensi generator ηG | 96% |
| Kecepatan | 500 RPM (12 kutub, jaringan 50 Hz) |
| Tegangan generator | 18 kV |
| Rating generator | 344 MVA / 275 MW (pf = 0.80) |
| Q minimum per unit | 18,15 m³/s (30% × Q_design) |

**Formula daya:**
```
P (MW) = ρ × g × Q × H_net × ηT × ηG / 10⁶
       = 1000 × 9.81 × 60.5 × 486.5 × 0.90 × 0.96 / 10⁶
       ≈ 249.5 MW per unit  ≈ 250 MW  ✓
```

---

## Cara Penggunaan

```bash
# Install dependencies
pip install -r requirements.txt

# (Opsional) Rebuild dataset dari raw PDF
python build_raw_dataset.py

# Training model
python train.py --epochs 100 --lr 0.001

# Evaluasi model
python evaluate.py

# Jalankan dashboard
streamlit run app.py
```

### Ganti Dataset

Untuk melatih model menggunakan `data_grindulu_raw.csv` (v2 dengan dynamic baseflow),
ubah satu baris di `preprocess.py`:

```python
# Sebelum (v1 — baseflow konstan):
DATA_PATH = "data_grindulu.csv"

# Sesudah (v2 — dynamic baseflow routing):
DATA_PATH = "data_grindulu_raw.csv"
```

Kemudian retrain: `python train.py --epochs 100`

---

## Rencana Peningkatan Model

Untuk meningkatkan akurasi dan representasi hidrologi model,
berikut roadmap berdasarkan prioritas:

---

### Tier 1 — Perubahan Besar (butuh data baru)

#### Data Debit Aktual (AWLR)
Data terukur langsung dari pos AWLR (Automatic Water Level Recorder) Sungai Grindulu
akan menggantikan seluruh proxy inflow yang bersifat estimatif.

| Item | Detail |
|------|--------|
| Sumber | BBWS Bengawan Solo — pos AWLR terdekat di Sungai Grindulu |
| Format | Level air harian → konversi via rating curve → m³/s |
| Dampak | Mengganti semua q_total sintetis dengan data terukur |
| Prioritas | **Tertinggi** — mengubah penelitian dari estimasi ke kalibrasi nyata |

#### Indeks Iklim ENSO / IOD
El Niño/La Niña sangat mempengaruhi curah hujan Jawa.
Model saat ini tidak mengetahui kondisi iklim inter-annual.

| Item | Detail |
|------|--------|
| Sumber | NOAA — Oceanic Niño Index (ONI), bulanan |
| Format | Nilai indeks bulanan (−3 s.d. +3) |
| Dampak | Model belajar pola musim kering ekstrem (El Niño 2015, 2019, 2023) |
| Cara integrasi | Tambahkan sebagai fitur encoder bulanan |

---

### Tier 2 — Gratis, Sudah Ada di Dataset (implementasi segera)

#### Curah Hujan Per Stasiun sebagai Fitur Encoder
Model saat ini hanya menerima `p_das` (rata-rata). Informasi spasial dari
6 stasiun terpisah dapat meningkatkan akurasi signifikan.

```python
# Encoder saat ini:
[p_das, q_total]       # 2 fitur

# Encoder yang ditingkatkan:
[pacitan, nawangan, kebonagung, bandar, tegalombo, tulakan, q_total]  # 7 fitur
```

| Stasiun | Korelasi dengan q_total | Keterangan |
|---------|------------------------|-----------|
| kebonagung | r = 0.69 | Terbaik |
| pacitan | r = 0.68 | Terbaik kedua |
| nawangan | r = 0.37 | Hulu DAS |

#### Seasonal Encoding (Sin/Cos Bulan)
Model tidak mengetahui posisi musim. Pola musim hujan (Nov–Mar) vs
musim kemarau (Apr–Okt) sangat kuat di data.

```python
# Tambahkan ke encoder setiap hari:
sin_month = sin(2π × month / 12)
cos_month = cos(2π × month / 12)
```

---

### Tier 3 — Data Satelit Gratis (perlu download)

| Data | Sumber | Cakupan | Manfaat |
|------|--------|---------|---------|
| **CHIRPS Rainfall** | chirps.ucsb.edu | 1981–kini, harian, 0.05° | Curah hujan spasial lebih baik (50+ grid vs 6 stasiun) |
| **SMAP Soil Moisture** | NASA EarthData | 2015–kini, harian | Kondisi kelembaban tanah awal → akurasi runoff |
| **ERA5 Temperature** | Copernicus CDS | 1940–kini, harian | Evapotranspirasi nyata (gantikan Pe konstan) |

---

### Tier 4 — Peningkatan Arsitektur (tanpa data baru)

| Perubahan | Cara | Dampak Estimasi |
|-----------|------|----------------|
| Tambah fitur stasiun (Tier 2) | Update `preprocess.py` encoder | Sedang–Tinggi |
| Seasonal encoding | Tambah sin/cos ke encoder | Sedang |
| Look-back lebih panjang | `ENC_LEN` 30 → 60 hari | Kecil–Sedang |
| Attention mechanism | Tambah layer attention pada encoder | Sedang |
| Tambah quantile | Q25, Q75 selain Q10/Q50/Q90 | Kecil |

---

### Prioritas Implementasi

```
Jangka pendek (tanpa data baru):
  → Tier 2: Tambah 6 fitur stasiun + seasonal encoding ke encoder
    File yang diubah: preprocess.py (make_inference_sequence, load_data)
    Kemudian: retrain model

Jangka menengah (download gratis):
  → Tier 1: ENSO ONI index (NOAA, bulanan)
  → Tier 3: ERA5 temperature untuk ET nyata

Jangka panjang (akses institusional):
  → Tier 1: Data AWLR debit aktual dari BBWS Bengawan Solo
            → penelitian berubah dari proxy ke kalibrasi nyata
```

---

## Catatan Kualitas Data

| Masalah | Dampak | Penanganan |
|---------|--------|-----------|
| Tegalombo Mei–Des 2015 semua nol | P_DAS sedikit lebih rendah ~245 hari | Diterima — mengikuti data sumber |
| Tulakan 2020 ~47 hari kosong | P_DAS sedikit lebih rendah | Diterima |
| Kebonagung Nov 2017: 304+301mm | 2 hari Q_total sangat tinggi | Tidak dikoreksi — menggunakan data sumber |
| Bandar 2018 & 2022: beberapa hari kosong | Minor | Diterima |

> Semua nilai "-" pada PDF sumber ditangani sebagai **0 mm**.
> `data_grindulu.csv` mencerminkan tepat apa yang ada di `PERHITUNGAN.xlsx`.
> `data_grindulu_raw.csv` dibangun langsung dari PDF menggunakan `build_raw_dataset.py`.

---

## Referensi

- Rafsanjani B. Muhammadi & Arham (ITPLN) — *Inflow Forecasting System and Production Optimization Poso Hydropower Plant Using Bi-LSTM Seq2Seq Quantile Regression Model* (referensi metodologi)
- BBWS Bengawan Solo — Data Hujan Harian 6 Stasiun, 2014–2024
- Dokumen Pra-FS PLTA Grindulu (spesifikasi turbin dan DAS)
- SCS National Engineering Handbook Section 4 — Hydrology (CN method)
- IEC 60193 — Hydraulic turbines, storage pumps and pump-turbines (turbine efficiency)
