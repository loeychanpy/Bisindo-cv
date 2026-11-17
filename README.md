# Pengenalan Alfabet BISINDO (A-Z) dari Gambar Tangan

Proyek Akhir Visi Komputer (IF0PP006, kelas 5PITA, Semester Ganjil 2026/2027) - Kelompok 4.  
Dokumentasi lengkap proyek, pemilahan berkas, analisis kegagalan, dan panduan rubrik tersedia di [DOKUMENTASI_APLIKASI.md](DOKUMENTASI_APLIKASI.md).

## Pertanyaan proyek
Seberapa baik MobileNetV3-Small pra-latih (ImageNet) mengenali 26 huruf BISINDO dari foto tangan, dan strategi transfer learning mana yang paling efektif: hanya melatih kepala klasifikasi (`head`, baseline), membuka beberapa blok terakhir (`partial`, metode utama), atau melatih seluruh jaringan (`full`). Dataset: BISINDO Alphabet (Sanjaya, UMN 2024), varian Original. Ukuran keberhasilan: macro-F1 dan akurasi pada test set beku (`split_v3` bebas kebocoran perangkat dan `split_v2`), plus kelas yang paling sering tertukar dan latensi inferensi.

## Ringkasan Hasil Eksperimen Utama (Split Beku v3)
Seluruh eksperimen dievaluasi pada split beku `split_v3.csv` (pengelompokan 44 grup ponsel kontributor; zero-leakage):

| Run Name | Mode Transfer Learning | Parameter Dilatih | Val Macro-F1 | Test Akurasi | Test Macro-F1 | Median Latensi (CPU) | Status & Catatan |
|---|---|---|---|---|---|---|---|
| `v3_full` | **Full Fine-Tuning** | 1.544.506 ($100\%$) | **0.7538** | **84.1%** | **0.8369** | 98.3 ms | Akurasi tertinggi pada data ponsel baru |
| `v3_partial4` | **Partial (4 Blok)** | 1.353.986 ($87.7\%$) | **0.7281** | **78.6%** | **0.7815** | **64.4 ms** | **Titik efisiensi optimal (*Sweet Spot*)** |
| `v3_partial6` | Partial (6 Blok) | 1.405.754 ($91.0\%$) | 0.7327 | 79.1% | 0.7864 | 122.5 ms | Kenaikan F1 marjinal, latensi $2\times$ lebih lama |
| `v3_partial2` | Partial (2 Blok) | 968.042 ($62.7\%$) | 0.6398 | 71.4% | 0.7092 | 62.1 ms | Pembukaan blok minimal |
| `v3_head` | Baseline (Head Only) | 26.650 ($1.7\%$) | 0.4925 | 53.8% | 0.5346 | 61.7 ms | Baseline ImageNet (hanya Linear akhir) |

*(Pembanding pada `split_v2` acak tanpa grouping ponsel: `mnv3s_full` akurasi 81.1%, macro-F1 0.8082; `mnv3s_head2` akurasi 52.8%, macro-F1 0.5248).*

## Dataset
- Sanjaya, UMN 2024, Mendeley Data. DOI: 10.17632/ywnjpbcz8m.1
- Lisensi: CC BY 4.0
- Cara memperoleh: unduh dari Mendeley Data, ekstrak ke `data/`, sehingga folder `data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images/<HURUF>/` ada. Pipeline hanya memakai varian Original.
- Dataset tidak di-commit (lihat `.gitignore`). Folder varian lain (`02. Resized Images`, `03. Binary Images`, `04. Code Chain Data`) tidak digunakan dan aman dihapus untuk menghemat ruang disk.

## Setup
Semua perintah dijalankan dari root repo (Windows, PowerShell).
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-app.txt
```

### Memakai GPU NVIDIA (opsional, jauh lebih cepat)
`pip install torch` di Windows biasanya memasang build CPU. Cek dulu:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Jika hasilnya `...+cpu False`, ganti dengan build CUDA (driver NVIDIA yang sudah terpasang cukup):
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Ulangi cek di atas; harus tercetak `True`. Training memilih GPU otomatis (`device: auto` di `configs/transfer.yaml`); paksa dengan `--device cuda` atau `--device cpu`.

## Panduan Langkah demi Langkah
Urutan lengkap dari data sampai demo. Langkah 1-2 cukup sekali; hasilnya (`results/sanitary/`, `splits/`) sudah dibekukan di repo.

**Langkah 0 - Siapkan data.** Unduh dataset (lihat bagian Dataset), ekstrak ke `data/` sehingga folder `data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images/<HURUF>/` ada.

**Langkah 1 - Sanity check data** (opsional; hasilnya sudah di repo).
```powershell
python src/sanitary_check.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data"
```
Menulis `results/sanitary/` termasuk `excluded_files.txt` (membuang 337 file rusak, HEIC bernama `.jpg`, duplikat). Notebook `notebooks/bisindo_cp07_data_pipeline.ipynb` (Run All) menghasilkan `results/sanity_summary.json` dan `results/figures/`.

**Langkah 2 - Split data** (dibuat sekali, lalu dibekukan; skrip menolak menimpa split yang sudah ada).
- `split_v3` (**Disarankan; Zero-Leakage per ponsel**) dibuat dengan dua perintah:
```powershell
python src/group_keys.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images"
python src/make_split.py --config configs/config_v3.yaml --version v3
python src/make_split.py --config configs/config_v3.yaml --version v3 --check-only    # cek ulang leakage
```
`group_keys.py` menulis `splits/groups_v3.csv` (ID ponsel dari nama file, bucket latar, dan koreksi 5 file yang salah folder). `make_split.py` menulis `splits/split_v3.csv` dan `splits/leakage_report_v3.txt` (laporan berakhir `HASIL: BERSIH`).

**Langkah 3 - Latih model.** Gunakan `--config configs/config_v3.yaml` agar memakai `split_v3`. Hyperparameter dasar ada di `configs/transfer.yaml` dan bisa ditimpa lewat CLI.
```powershell
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_head --mode head --epochs 20
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_partial4 --mode partial --unfreeze-blocks 4 --no-test
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_full --mode full --epochs 12
```
- `--no-test`: tuning memakai val saja; test tidak disentuh.
- Skrip dilengkapi fitur **RAM Caching & Multi-Thread Preload** (`cache_ram: true`), membaca 4.462 citra ke RAM dalam $\sim 30\text{ detik}$ sehingga disk I/O berikutnya 0 detik.
- Run yang sudah ada tidak ditimpa tanpa `--overwrite`. Satu run, satu perubahan dari run pembanding.
- Atau lewat `notebooks/02_transfer_learning.ipynb` (notebook tipis; logikanya di `src/train_transfer.py`).

**Langkah 4 - Evaluasi test tanpa melatih ulang.**
```powershell
python src/train_transfer.py --config configs/config_v3.yaml --eval-only --run v3_partial4
```

**Langkah 5 - Bandingkan semua run.**
```powershell
python src/train_transfer.py --compare
```
Menulis tabel ringkas ke `results/transfer/comparison.csv`.

**Langkah 6 - Analisis kesalahan** (butuh `predictions_test.csv`):
```powershell
python src/error_analysis.py --run v3_full --split test
```
Keluaran di `results/transfer/<run>/`: `error_analysis.md`, pasangan huruf yang paling sering tertukar (P vs D, M vs N, U vs C, I vs J), dan tabel reliability vs threshold.

**Langkah 7 - Kotak tangan MediaPipe** (opsional untuk fitur pemotongan tangan):
```powershell
pip install mediapipe
python src/hand_bbox.py --config configs/config_v3.yaml
```
Model tangan (`models/hand_landmarker.task`) diunduh otomatis.

**Langkah 8 - Demo terminal:**
```powershell
python src/demo.py --run v3_full --image foto1.jpg foto2.jpg    # demo offline
python src/demo.py --run v3_full --webcam                       # bonus webcam terminal
python src/demo.py --run v3_full --crop --image foto1.jpg       # dengan deteksi MediaPipe
```

---

## Aplikasi Antarmuka (Streamlit, Lokal)
Aplikasi antarmuka demo interaktif yang berjalan lokal di browser. Memuat otomatis 9 model yang sudah terlatih dari `results/transfer/`:
```powershell
streamlit run app/streamlit_app.py
```
Akses di browser: `http://localhost:8501`.

| Tab | Fungsi & Fitur |
|---|---|
| **Live** | Video streaming real-time WebRTC + penstabil huruf (`LetterStabilizer`) dan tombol kontrol kata (Spasi, Hapus, Reset). |
| **Unggah Gambar** | Prediksi banyak foto sekaligus (batch upload): huruf, probabilitas, top-$k$ bar chart, latensi, dan deteksi tangan. |
| **Kamera** | Snapshot foto kamera browser + fitur interaktif "Susun Kata". |
| **Contoh Data Uji** | 9 sampel acak dari data test beku dengan filter huruf dan checkbox *"Hanya tampilkan yang salah"* untuk analisis kegagalan langsung. |
| **Hasil Model** | Metrik akurasi/F1, kurva loss/akurasi, Confusion Matrix resolusi tinggi, tabel per kelas, dan tabel perbandingan run. |
| **Tentang** | Latar belakang akademis, atribusi dataset UMN, dan batasan sistem. |

Unit test aplikasi: `pytest tests/test_app.py`.

### Skenario Demo UAS (5 Menit)
1. **Hasil Model (30 dtk):** Perlihatkan kurva konvergensi, Confusion Matrix, dan tabel `comparison.csv` (pembuktian `v3_partial4` vs `v3_full`).
2. **Unggah Gambar (1,5 mnt):** Uji 3 foto (foto jelas, foto dengan latar belakang ramai, dan foto bukan tangan untuk menunjukkan proteksi tanda tanya `?`).
3. **Kamera & Eja Kata (1,5 mnt):** Ambil foto langsung dari kamera, demonstrasikan penyusunan satu kata (misal "K-A-M-I").
4. **Contoh Data Uji & Analisis Kegagalan (1 mnt):** Centang "Hanya tampilkan yang salah", jelaskan kebingungan morfologis antara huruf M vs N atau P vs D.
5. **Penutup & Video Cadangan (30 dtk):** Sampaikan kesimpulan performa dan siapkan video cadangan 2–3 menit jika koneksi/kamera terkendala.

---

## Struktur Folder Repositori
```
Bisindo-cv/
├── app/                        # Aplikasi Streamlit (streamlit_app.py, inference.py, live.py, ui_parts.py)
├── configs/                    # Konfigurasi YAML (config.yaml, config_v3.yaml, transfer.yaml)
├── data/                       # Dataset mentah 01. Original Images (di-.gitignore)
├── models/                     # Model MediaPipe hand_landmarker.task
├── notebooks/                  # Notebook interaktif (CP07 data pipeline & CP08 transfer learning)
├── results/                    # Hasil transfer learning, kurva, metrics.json, dan comparison.csv
├── splits/                     # File split beku (split_v2.csv, split_v3.csv, groups_v3.csv, label_map.json)
├── src/                        # Modul Python (train_transfer, dataset, demo, hand_bbox, error_analysis)
├── tests/                      # Automated tests (test_app.py)
├── tidak_digunakan/            # Arsip berkas tidak terpakai (split v1, run kosong, folder sisa venv)
├── DOKUMENTASI_APLIKASI.md     # Dokumen komprehensif proyek (rubrik lengkap, 5 rujukan, failure analysis)
├── README.md                   # Panduan ringkas repositori ini
├── requirements.txt            # Dependensi dasar
└── requirements-app.txt        # Dependensi aplikasi Streamlit & WebRTC
```

---

## Kontribusi Anggota
| Anggota (Akun GitHub) | Kontribusi Utama |
|---|---|
| **loeychanpy** | Struktur repositori, konfigurasi `config.yaml`, integrasi PR & merge coordinator, dokumentasi awal. |
| **NoluckDv** | `src/make_split.py`: split stratified, deteksi duplikat (sha256), verifikasi leakage, penyusunan `split_v3.csv` & `group_keys.py`. |
| **Vetians** | `src/sanitary_check.py`, `src/dataset.py` (loader, transform augmentasi tanpa flip), hasil sanity check CP07. |
| **Pongo** | CP08–CP12: `train_transfer.py` (optimasi RAM cache & GPU), `demo.py`, `02_transfer_learning.ipynb`, arsitektur aplikasi GUI `app/` (Streamlit, WebRTC Live, Inference, UI Parts), serta penyusunan [DOKUMENTASI_APLIKASI.md](DOKUMENTASI_APLIKASI.md). |

## Bantuan Eksternal & Pustaka
- **PyTorch & torchvision:** Arsitektur MobileNetV3-Small pra-latih ImageNet, DataLoader, dan transformasi tensor.
- **MediaPipe:** Hand Landmarker untuk deteksi dan pemotongan telapak tangan (*Hand BBox*).
- **Streamlit & streamlit-webrtc:** Antarmuka web lokal dan pemrosesan video streaming *real-time*.
- **scikit-learn, pandas, NumPy, Altair, Pillow, Matplotlib, PyYAML:** Kalkulasi metrik (Macro-F1, Confusion Matrix), pengolahan data, visualisasi kurva, dan parsing konfigurasi.
- **Dataset:** Sanjaya, UMN 2024, Mendeley Data (CC BY 4.0), DOI: [10.17632/ywnjpbcz8m.1](https://doi.org/10.17632/ywnjpbcz8m.1).
