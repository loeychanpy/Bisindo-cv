# Pengenalan Alfabet BISINDO (A-Z) dari Gambar Tangan

Proyek Akhir Visi Komputer (IF0PP006, kelas 5PITA, Semester Ganjil 2026/2027) - Kelompok 4.

## Pertanyaan proyek
Seberapa baik MobileNetV3-Small pra-latih (ImageNet) mengenali 26 huruf BISINDO dari foto tangan, dan strategi transfer learning mana yang paling efektif: hanya melatih kepala klasifikasi (`head`, baseline), membuka beberapa blok terakhir (`partial`, metode utama), atau melatih seluruh jaringan (`full`). Dataset: BISINDO Alphabet (Sanjaya, UMN 2024), varian Original. Ukuran keberhasilan: macro-F1 dan akurasi pada test set beku (`split_v2`), plus kelas yang paling sering tertukar dan latensi inferensi.

## Dataset
- Sanjaya, UMN 2024, Mendeley Data. DOI: 10.17632/ywnjpbcz8m.1
- Lisensi: CC BY 4.0
- Cara memperoleh: unduh dari Mendeley Data, ekstrak ke `data/`, sehingga folder `data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images/<HURUF>/` ada. Pipeline hanya memakai varian Original.
- Dataset tidak di-commit (lihat `.gitignore`).

## Setup
Semua perintah dijalankan dari root repo (Windows, PowerShell).
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Memakai GPU NVIDIA (opsional, jauh lebih cepat)
`pip install torch` di Windows biasanya memasang build CPU. Cek dulu:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Jika hasilnya `...+cpu False`, ganti dengan build CUDA (driver NVIDIA yang sudah terpasang cukup):
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Ulangi cek di atas; harus tercetak `True`. Training memilih GPU otomatis (`device: auto` di `configs/transfer.yaml`); paksa dengan `--device cuda` atau `--device cpu`. Baris pertama log run mencetak `device=cuda` atau `device=cpu`. Pada GPU 4 GB, turunkan `--batch-size` bila memori habis.

## Panduan langkah demi langkah
Urutan lengkap dari data sampai demo. Langkah 1-2 cukup sekali; hasilnya (`results/sanitary/`, `splits/`) sudah ada di repo.

**Langkah 0 - Siapkan data.** Unduh dataset (lihat bagian Dataset), ekstrak ke `data/` sehingga folder `data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images/<HURUF>/` ada.

**Langkah 1 - Sanity check data** (opsional; hasilnya sudah di repo).
```
python src/sanitary_check.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data"
```
Menulis `results/sanitary/` termasuk `excluded_files.txt` (file rusak, HEIC bernama `.jpg`, duplikat). Notebook `notebooks/bisindo_cp07_data_pipeline.ipynb` (Run All) menghasilkan `results/sanity_summary.json` dan `results/figures/`.

**Langkah 2 - Split data** (dibuat sekali, lalu dibekukan; skrip menolak menimpa split yang sudah ada).
- `split_v2` (lama, stratified per kelas) sudah ada di `splits/` dan menjadi default `configs/config.yaml`.
- `split_v3` (disarankan; tanpa kebocoran ponsel antar split) dibuat dengan dua perintah:
```
python src/group_keys.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data/01. Original Images"
python src/make_split.py --config configs/config_v3.yaml --version v3
python src/make_split.py --config configs/config_v3.yaml --version v3 --check-only    # cek ulang leakage
```
`group_keys.py` menulis `splits/groups_v3.csv` (ID ponsel dari nama file, bucket latar, dan koreksi 5 file yang salah folder) beserta daftar pemetaan nama mentah ke grup untuk diperiksa mata. `make_split.py` menulis `splits/split_v3.csv` dan `splits/leakage_report_v3.txt`; laporan harus berakhir `HASIL: BERSIH`.

**Langkah 3 - Latih model.** Gunakan `--config configs/config_v3.yaml` agar memakai `split_v3` (tanpa opsi itu dipakai `split_v2`). Hyperparameter dasar ada di `configs/transfer.yaml` dan bisa ditimpa lewat CLI.
```
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_head --mode head --epochs 20
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_partial4 --mode partial --unfreeze-blocks 4 --no-test
python src/train_transfer.py --config configs/config_v3.yaml --run-name v3_full --mode full --epochs 12
```
- `--no-test`: tuning memakai val saja; test tidak disentuh.
- Smoke test sebelum run panjang: `--epochs 1 --no-test --run-name smoke_x`, lalu hapus folder `results/transfer/smoke_x`.
- Run yang sudah ada tidak ditimpa tanpa `--overwrite`. Satu run, satu perubahan dari run pembanding.
- Opsi kecepatan: `amp` (mixed precision, hanya di GPU), `fast_cudnn`, `cache_images`, dan `num_workers` di `configs/transfer.yaml`; matikan AMP dengan `--no-amp`. Untuk eksperimen terkontrol yang harus persis sama antar run, set `amp: false` dan `fast_cudnn: false`.
- Atau lewat `notebooks/02_transfer_learning.ipynb` (notebook tipis; logikanya di `src/train_transfer.py`).
- Run panjang di background, simpan log: `python src/train_transfer.py ... > results/transfer/_logs/<run>.txt 2>&1`.

**Langkah 4 - Evaluasi test tanpa melatih ulang.** Setelah memilih run terbaik dari nilai val, evaluasi `best.pth`-nya pada test sekali saja:
```
python src/train_transfer.py --config configs/config_v3.yaml --eval-only --run v3_partial4
```
Perintah ini menolak berjalan bila split berbeda dari saat run dilatih.

**Langkah 5 - Bandingkan semua run.**
```
python src/train_transfer.py --compare
```
Menulis `results/transfer/comparison.csv`.

**Langkah 6 - Analisis kesalahan** (butuh `predictions_test.csv`, ada setelah run dievaluasi di test).
```
python src/error_analysis.py --run v3_partial4 --split test
```
Keluaran di `results/transfer/<run>/`: `error_analysis.md` (ringkasan), akurasi per latar dan per grup ponsel, pasangan huruf yang paling tertukar (M/N, D/P, J/I) beserta contoh gambar, dan tabel akurasi vs confidence dengan saran `--threshold` demo.

**Langkah 7 - Kotak tangan MediaPipe** (opsional; dasar fitur "potong ke tangan"). MediaPipe dipasang di environment terpisah agar tidak mengganggu environment utama:
```
python -m venv .venv-mp
.venv-mp\Scripts\pip install mediapipe pandas pillow pyyaml
.venv-mp\Scripts\python src/hand_bbox.py --config configs/config_v3.yaml
```
Model tangan (`models/hand_landmarker.task`) diunduh otomatis. Keluaran: `results/crops/bboxes_v3.csv`, `detection_rate.csv`, dan `crop_grid.png` (periksa visual). Opsi training `crop=bbox` belum tersedia di branch ini; bbox saat ini dipakai oleh `demo.py --crop`.

**Langkah 8 - Demo inferensi** (butuh `best.pth` dari run yang dipilih).
```
python src/demo.py --run v3_partial4 --image foto1.jpg foto2.jpg    # demo offline
python src/demo.py --run v3_partial4 --webcam                       # bonus; perlu pip install opencv-python
python src/demo.py --run v3_partial4 --crop --image foto1.jpg       # deteksi tangan dulu; perlu pip install mediapipe
```
Tanpa `--crop` model tidak punya deteksi tangan: tanpa tangan di bingkai ia tetap menebak satu huruf, sehingga prediksi di bawah `--threshold` ditampilkan sebagai `?`. Dengan `--crop`, bila tidak ada tangan demo menampilkan "tangan tidak terdeteksi" dan tidak menebak. Jalankan mode webcam di terminal, bukan di Jupyter.

**Langkah 9 - Tampilan Streamlit.** Lihat bagian berikutnya.

Keluaran tiap run ada di `results/transfer/<run_name>/`: `config_used.json`, `log.csv`, `curves.png`, `metrics.json`, per-kelas, confusion matrix, pasangan huruf yang paling tertukar, dan prediksi contoh. `best.pth` tidak di-commit (`.gitignore`), jadi hanya ada di laptop yang melatihnya.

Aturan yang dijaga: augmentasi hanya di train (tanpa horizontal flip karena bisa mengubah makna isyarat), BatchNorm pada layer beku tetap mode eval, model terbaik dipilih dari val macro-F1, test dievaluasi sekali di akhir, dan run yang sudah ada tidak ditimpa tanpa `--overwrite`.

### Masalah umum
| Gejala | Penyebab dan solusi |
|---|---|
| Log mencetak `device=cpu` padahal ada GPU | torch build CPU; ikuti "Memakai GPU NVIDIA" di atas. |
| `CUDA out of memory` | Turunkan `--batch-size` (mis. 16) atau `--num-workers`. |
| `FileExistsError ... sudah berisi hasil` | Ganti `--run-name` atau tambahkan `--overwrite`. |
| `File contoh tidak ada di ...` | `data_dir` di `configs/config*.yaml` salah; harus folder `01. Original Images`. |
| `make_split.py` menolak menulis | Split itu sudah dibekukan; buat versi baru (`--version v4`) dan catat alasannya. |
| Error saat `num_workers` > 0 di Windows | Set `num_workers: 0` di `configs/transfer.yaml`. |

## Tampilan (Streamlit, lokal)
Aplikasi demo yang hanya berjalan di laptop sendiri (tanpa deployment, tanpa training di dalam aplikasi). Butuh `best.pth` hasil `src/train_transfer.py` (tidak di-commit, jadi hanya run yang dilatih di laptop ini yang muncul).

```
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```
Fitur "Potong ke tangan" butuh `pip install mediapipe`; tanpa itu kotak centangnya nonaktif dan aplikasi tetap jalan.

| Tab | Fungsi |
|---|---|
| Unggah Gambar | Prediksi banyak foto sekaligus: huruf, keyakinan, top-k, latensi. File rusak/HEIC/terlalu kecil hanya menampilkan pesan di kartunya. |
| Kamera | Foto dari kamera + "Susun kata" (tambah huruf, spasi, hapus, reset). Video real-time tetap lewat `python src/demo.py --webcam`. |
| Contoh Data Uji | Demo tanpa kamera: 9 gambar acak dari split test, border hijau/merah, opsi "Hanya tampilkan yang salah". |
| Hasil Model | Metrik, kurva, confusion matrix, tabel per kelas, perbandingan semua run. |
| Tentang | Masalah, dataset, model, dan keterbatasan. |

Tes: `pytest tests/test_app.py` (tes yang butuh `best.pth` di-skip bila tidak ada).

### Skrip demo UAS (5 menit)
1. **Hasil Model** (30 dtk): metrik, confusion matrix, perbandingan run.
2. **Unggah Gambar** (1,5 mnt): 3 foto yang sudah disiapkan, termasuk satu yang gagal (mis. `.heic` atau foto tanpa tangan); jelaskan "?" dan threshold.
3. **Kamera** (2 mnt): ambil foto langsung, susun satu kata pendek.
4. **Contoh Data Uji** (1 mnt): centang "Hanya tampilkan yang salah", jelaskan satu kegagalan.

Ingat: tetap rekam **video cadangan 2-3 menit** yang diwajibkan mata kuliah.

## Struktur folder
```
configs/    config.yaml (split_v2), config_v3.yaml (split_v3), transfer.yaml (hyperparameter training)
data/       data mentah (tidak di-commit)
models/     hand_landmarker.task untuk MediaPipe (diunduh otomatis, tidak di-commit)
splits/     split_v2.csv, split_v3.csv, groups_v3.csv, label_map.json, laporan leakage; split_v1.csv hanya arsip
app/        aplikasi Streamlit lokal (streamlit_app.py, inference.py, ui_parts.py)
src/        sanitary_check.py, make_split.py, group_keys.py, dataset.py, train_transfer.py, hand_bbox.py,
            error_analysis.py, demo.py
notebooks/  bisindo_cp07_data_pipeline.ipynb (CP07), 02_transfer_learning.ipynb (CP08)
results/    sanitary/, figures/, transfer/<run_name>/, sanity_summary.json
```

## Split dan leakage
- Split dibuat sekali (`split_v2.csv`) dengan seed 42, lalu dibekukan. Hasilnya identik (byte per byte) saat dijalankan dua kali.
- Mode split: stratified per kelas (`per_class`), 4.462 gambar: train 3.114, val 675, test 673.
- Sebelum split, `sanitary_check.py` membuang 337 file (HEIC bernama `.jpg`, file rusak atau kosong, file `._*` macOS, dan duplikat lintas kelas). File identik (sha256) digabung ke satu grup sehingga tidak terpisah antar split.
- Alasan `split_v1` diganti `split_v2`: `split_v1` masih memuat 297 file yang tidak bisa dibuka dan 1 hash yang ada di dua split. Pemeriksaan sekarang: `python src/make_split.py --version v2 --check-only`.
- **Keterbatasan:** `group_regex` kosong, jadi perangkat dan latar yang sama bisa muncul di train dan test. Akibatnya skor test bisa lebih optimis daripada kondisi nyata. Nama file memuat latar dan perangkat, sehingga ini bisa diperbaiki pada split berikutnya.
- `splits/split_v1.csv` dipertahankan sebagai arsip dan tidak dipakai oleh kode mana pun.
- Perubahan split harus menjadi versi baru dengan alasan tercatat di sini.
- **`split_v3` (split berbasis ponsel):** nama file mengikuti `<Huruf>_<Latar>_<Merek>_<Model>.jpg` dan satu ponsel memuat sekitar 104 gambar, sehingga ponsel dianggap satu kontributor (asumsi). `src/group_keys.py` menormalkan nama ponsel (alias merek digabung, penghitung duplikat seperti `(2)` dibuang) menjadi 44 grup; satu grup selalu masuk tepat satu split. Hasil: train 3.052, val 720, test 690 gambar, 26 huruf di tiap split, irisan path, sha256, dan grup antar split semuanya 0, dan hasilnya identik byte per byte saat dijalankan dua kali. Lima file yang namanya menunjuk huruf lain dari foldernya dikoreksi labelnya lewat `groups_v3.csv` (file tidak dipindah): `C/D_Bebas_Redmi_Note 12 Pro.jpg`, `L/E_Baju_Apple_Iphone 11.jpg`, `L/E_Bebas_Apple_Iphone 11.jpg`, `V/Y_Polos Gelap_Iphone_14 Pro.jpg`, `Z/S_Baju_Iphone_14 Pro.jpg`.
- **Keterbatasan `split_v3`:** ponsel dianggap setara dengan kontributor, padahal satu orang bisa memakai beberapa ponsel. Skor `split_v2` dan `split_v3` tidak boleh dibandingkan tanpa menyebut perbedaan ini.
- Dampak ke CP08: karena split tidak dikelompokkan per perangkat atau latar, skor test bisa lebih tinggi daripada performa pada foto dari HP atau latar baru. Laporkan angka test dengan catatan ini.

## Kontribusi anggota
| Anggota (akun GitHub) | Kontribusi |
|---|---|
| loeychanpy | Struktur proyek, `config.yaml`, README, integrasi dan merge |
| NoluckDv | `src/make_split.py`: split stratified, deteksi duplikat (sha256), laporan leakage, `label_map.json` |
| Vetians | `src/sanitary_check.py`, `src/dataset.py` (loader, transform), hasil sanity check |
| (isi) | CP08: `train_transfer.py`, `demo.py`, `02_transfer_learning.ipynb` |

## Bantuan eksternal
- **PyTorch dan torchvision:** model MobileNetV3-Small dengan bobot ImageNet pra-latih, transform, DataLoader.
- **scikit-learn:** metrik (accuracy, macro-F1, confusion matrix); **pandas, NumPy, matplotlib, Pillow, PyYAML:** pengolahan data, grafik, dan pembacaan gambar dan config.
- **Dataset:** Sanjaya, UMN 2024, Mendeley Data (CC BY 4.0), DOI 10.17632/ywnjpbcz8m.1.
