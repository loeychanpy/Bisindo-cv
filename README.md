# Pengenalan Alfabet BISINDO (A–Z) dari Citra Gestur Tangan

Proyek Akhir Visi Komputer (IF0PP006, Kelas 5PITA, Semester Ganjil 2026/2027) — **Kelompok 4**.

## Ringkasan Proyek
Proyek ini mengimplementasikan pipeline visi komputer *end-to-end* untuk mengenali 26 huruf alfabet Bahasa Isyarat Indonesia (BISINDO) A–Z secara otomatis dari citra gestur tangan. Arsitektur model mengadaptasi **MobileNetV3-Small** pra-latih ImageNet menggunakan transfer learning, dilengkapi optimasi kuantisasi/TorchScript untuk inferensi berlatensi rendah, serta antarmuka web interaktif berbasis **Streamlit**.

### Hasil Utama (Benchmark Evaluasi pada `split_v3` Zero-Leakage)
- **Model Terbaik:** `v3_full_cos30_ls_strong` (Full Fine-Tuning 30 Epoch + Cosine Annealing LR + Label Smoothing 0.1 + Strong Augmentation).
- **Test Accuracy:** **90,0%** (0,9000 pada 690 citra uji independen).
- **Test Macro-F1:** **0,8979** (Macro Precision: 0,897 | Macro Recall: 0,899).
- **Val Macro-F1:** 0,8083 (meningkat signifikan dari baseline 0,4925).
- **Latensi Inferensi Forward (TorchScript Channels-Last):**
  - **GPU (NVIDIA RTX 3050 Laptop):** **6,5 ms** (total *end-to-end* 59,4 ms).
  - **CPU (8-thread / 2-thread opt):** **11,2 ms** (total *end-to-end* 38,2 ms).

---

## Dataset
- **Sumber:** Mendeley Data — *BISINDO Indonesian Sign Language Alphabet Image Data* (Sanjaya, Universitas Multimedia Nusantara, 2024).
- **DOI:** [10.17632/ywnjpbcz8m.1](https://doi.org/10.17632/ywnjpbcz8m.1)
- **Lisensi:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Struktur:** Pipeline memproses varian `01. Original Images` (4.462 citra valid setelah pembersihan). Dataset mentah tidak di-commit ke repositori (lihat `.gitignore`).
- **Penempatan Lokal:** Ekstrak dataset ke folder `data/` sehingga jalur berikut tersedia:  
  `data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images/<HURUF>/`

---

## Pembagian Data & Jaminan Bebas Kebocoran (`split_v3`)
Agar evaluasi model mencerminkan performa dunia nyata, repositori ini menggunakan **`split_v3`** sebagai satu-satunya tolok ukur resmi:
1. **Pengelompokan Berbasis Ponsel/Kontributor:** Sebanyak 4.462 citra dikelompokkan ke dalam 44 grup ponsel kontributor independen berdasarkan metadata nama file (`<Huruf>_<Latar>_<Merek>_<Model>.jpg`).
2. **Zero-Leakage (Irisan = 0):** Seluruh citra dari satu kontributor ponsel dialokasikan eksklusif ke salah satu split (*Train*, *Validation*, atau *Test*). Tidak ada kontributor, perangkat, atau latar yang bocor antar-split.
3. **Statistik Split:**
   - **Train:** 3.052 citra (30 grup ponsel)
   - **Validation:** 720 citra (8 grup ponsel)
   - **Test:** 690 citra (6 grup ponsel)
   - Status audit kebocoran data: `splits/leakage_report_v3.txt` berakhir dengan status **`HASIL: BERSIH`**.

---

## Setup Lingkungan Pengembangan
Semua perintah dijalankan dari root repositori (Windows PowerShell).

```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Pasang dependensi utama
pip install -r requirements.txt
```

### Akselerasi GPU NVIDIA (CUDA)
Secara bawaan `pip install torch` di Windows sering memasang versi CPU. Untuk mengaktifkan akselerasi GPU:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Pastikan `python -c "import torch; print(torch.cuda.is_available())"` menghasilkan output `True`.

---

## Panduan Langkah demi Langkah

### Langkah 1: Sanitasi Citra & Partisi Data (Sudah Beku di Repo)
Sanitasi dan pembentukan split sudah dibekukan di direktori `splits/`. Jika ingin memverifikasi ulang:
```powershell
# Verifikasi sanitasi data
python src/sanitary_check.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data"

# Verifikasi audit kebocoran split_v3
python src/make_split.py --config configs/config_v3.yaml --version v3 --check-only
```

### Langkah 2: Pelatihan Model (Transfer Learning)
Pelatihan dijalankan dengan mengacu pada `configs/config_v3.yaml`.

```powershell
# A. Baseline: Head-Only (Hanya melatih classification head, backbone freeze)
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_head --mode head --epochs 20

# B. Partial Fine-Tuning (Melatih head + 6 blok konvolusi teratas)
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_partial6 --mode partial --unfreeze-blocks 6 --epochs 12

# C. Model SOTA Terbaik (Full Fine-Tuning + Cosine Annealing + Label Smoothing 0.1 + Strong Aug)
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_full_cos30_ls_strong --mode full --epochs 30 --scheduler cosine --warmup-epochs 2 --label-smoothing 0.1 --aug strong
```

### Langkah 3: Evaluasi & Perbandingan Hasil
```powershell
# Evaluasi model pada test set beku (sekali saja)
python src/train_transfer.py --config configs/config_v3.yaml --eval-only --run v3_full_cos30_ls_strong

# Rekapitulasi perbandingan seluruh model ke results/transfer/comparison.csv
python src/train_transfer.py --compare
```

### Langkah 4: Uji Latensi Inferensi
Mengukur waktu pemrosesan per tahap (*decode*, *preprocessing*, *forward pass eager*, *forward pass TorchScript*, dan total *end-to-end*):
```powershell
# Benchmark GPU
python src/benchmark_latency.py --config configs/config_v3.yaml --run v3_full_cos30_ls_strong --device cuda

# Benchmark CPU
python src/benchmark_latency.py --config configs/config_v3.yaml --run v3_full_cos30_ls_strong --device cpu
```

### Langkah 5: Analisis Kesalahan (Error Analysis)
```powershell
python src/error_analysis.py --run v3_full_cos30_ls_strong --split test
```
Menghasilkan matriks kesalahan per latar belakang (`error_by_background.png`), kelompok kontributor (`error_worst_group.png`), pasangan huruf paling membingungkan (M/N, D/P), dan kalibrasi confidence (`error_reliability.csv`).

---

## Tabel Perbandingan Eksperimen Model (`split_v3`)

| Model | Strategi | Parameter Dilatih | Val Macro-F1 | Test Acc | Test Macro-F1 | Forward Latency | Keterangan |
|---|---|---|---|---|---|---|---|
| `v3_head` | Head-Only | 26.650 (1,7%) | 0,4925 | 53,8% | 0,5346 | 20,8 ms (CPU) | Baseline awal, representasi ImageNet kurang spesifik |
| `v3_partial6` | Partial 6 Blok | 1.405.754 (91,0%) | 0,7327 | — | — | 61,7 ms (CPU) | Peningkatan adaptasi representasi tangan |
| `v3_full_e30` | Full Fine-Tune | 1.544.506 (100%) | 0,7765 | — | — | 20,8 ms (CPU) | 30 epoch konstan LR |
| **`v3_full_cos30_ls_strong`** | **Full + Cosine + LS + Strong** | **1.544.506 (100%)** | **0,8083** | **90,0%** | **0,8979** | **6,5 ms (GPU) / 11,2 ms (CPU)** | **Model Terbaik (SOTA)** |

---

## Aplikasi GUI Interaktif (Streamlit)
Aplikasi pengenalan BISINDO berbasis web lokal dapat dijalankan tanpa memerlukan dependensi eksternal cloud:

```powershell
# Pasang dependensi GUI
pip install -r requirements-app.txt

# Jalankan server aplikasi
streamlit run app/streamlit_app.py
```

### Fitur Utama Aplikasi:
1. **Unggah Gambar (Batch Inference):** Mengunggah banyak citra secara bersamaan, menampilkan bounding box prediksi, top-3 confidence, dan latensi pemrosesan.
2. **Kamera & Susun Kata:** Mengambil foto gestur secara langsung dari webcam lokal dan merangkai huruf demi huruf menjadi kata dengan tombol spasi, koreksi, dan reset.
3. **Contoh Data Uji:** Melakukan simulasi pengujian pada citra uji acak tanpa memerlukan kamera fisik, dilengkapi filter untuk mengevaluasi kasus salah (*false predictions*).
4. **Hasil Model & Visualisasi:** Menampilkan kurva pelatihan, confusion matrix interaktif, tabel metrik per kelas, serta kalibrasi threshold.
5. **Potong ke Tangan (MediaPipe Crop):** Opsi pra-pemrosesan opsional menggunakan `MediaPipe HandLandmarker` untuk memotong area tangan sebelum diklasifikasikan oleh MobileNetV3.

---

## Struktur Repositori
```
Bisindo-cv/
├── configs/              # Konfigurasi eksperimen (config_v3.yaml, transfer.yaml)
├── data/                 # Dataset mentah (diabaikan oleh git)
├── models/               # Model detektor tangan MediaPipe (hand_landmarker.task)
├── splits/               # Partisi split_v3.csv, groups_v3.csv, label_map.json, leakage_report_v3.txt
├── src/                  # Kode sumber modul inti:
│   ├── sanitary_check.py # Audit citra rusak & duplikasi
│   ├── make_split.py     # Pembuat partisi zero-leakage
│   ├── group_keys.py     # Normalisasi kelompok kontributor ponsel
│   ├── dataset.py        # PyTorch Dataset & Image Transforms
│   ├── train_transfer.py # Engine pelatihan transfer learning & evaluasi
│   ├── benchmark_latency.py # Skrip benchmark performa CPU & GPU
│   ├── error_analysis.py # Analisis diagnostik error & kalibrasi
│   ├── hand_bbox.py      # Ekstraksi bounding box tangan via MediaPipe
│   └── demo.py           # Engine inferensi dan optimasi TorchScript
├── app/                  # Antarmuka Streamlit:
│   ├── streamlit_app.py  # Entrypoint antarmuka web
│   ├── inference.py      # Wrapper inferensi model & cropper
│   └── ui_parts.py       # Komponen visual & CSS
├── results/              # Artefak hasil eksperimen:
│   └── transfer/         # Log pelatihan, bobot model, kurva, dan confusion matrix
└── tests/                # Unit test (pytest)
```

---

## Kontribusi Anggota (Kelompok 4 - Kelas 5PITA)

| Nama Anggota | NIM | Akun GitHub / Identitas | Peran & Kontribusi Utama |
|---|---|---|---|
| **Felix Wong** | 412024004 | `NoluckDv` | **Project Lead & Integrator**<br>• Manajemen repositori, branch, dan integrasi pull request.<br>• Perancangan struktur konfigurasi `configs/config_v3.yaml` dan skema partisi dataset.<br>• Penyusunan laporan master dan dokumentasi teknis. |
| **Janisha Jaya** | 412024033 | `loeychanpy` | **Data & Leakage Engineer**<br>• Pengembangan modul sanitasi citra (`src/sanitary_check.py`).<br>• Implementasi pemuatan data dan transformasi (`src/dataset.py`).<br>• Formulasi identifikasi grup ponsel kontributor (`src/group_keys.py`) dan pemetaan label (`label_map.json`).<br>• Pengujian integritas dan audit *zero-leakage* pada `split_v3`. |
| **Christian Melhan** | 412024003 | `Vetians` / `Pongo` | **Model & Application Engineer**<br>• Implementasi arsitektur transfer learning dan engine pelatihan (`src/train_transfer.py`).<br>• Eksperimen optimasi GPU: Cosine Annealing, Label Smoothing, dan Regularisasi Augmentasi.<br>• Pengembangan modul akselerasi inferensi TorchScript Channels-Last dan benchmark latensi (`src/benchmark_latency.py`).<br>• Perancangan antarmuka aplikasi interaktif (`app/streamlit_app.py`, `app/inference.py`). |

---

## Referensi & Dependensi
- **Arsitektur Backbone:** MobileNetV3-Small (*Searching for MobileNetV3*, Howard et al., ICCV 2019).
- **Framework & Pustaka Utama:** PyTorch, Torchvision, Scikit-Learn, Pandas, NumPy, Pillow, Streamlit, MediaPipe, Matplotlib.
- **Dataset:** Sanjaya, R. (2024). *BISINDO Indonesian Sign Language Alphabet Image Data*, Mendeley Data, V1, doi: 10.17632/ywnjpbcz8m.1.
