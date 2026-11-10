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
```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Cara menjalankan (CP07)
Dari root repo, setelah `pip install -r requirements.txt`:
1. `python src/sanitary_check.py --root "data/BISINDO Indonesian Sign Language Alphabet Image Data"` menulis `results/sanitary/` (termasuk `excluded_files.txt`).
2. `python src/make_split.py --version v2` menulis `splits/split_v2.csv`, `splits/label_map.json`, dan `splits/leakage_report_v2.txt`. Skrip menolak menimpa split yang sudah ada.
3. Buka `notebooks/bisindo_cp07_data_pipeline.ipynb`, lalu Run All. Keluarannya `results/sanity_summary.json`, `results/class_counts_per_split.csv`, dan `results/figures/`.

Path data, seed, rasio split, dan daftar file buangan dibaca dari `configs/config.yaml`.

## Cara menjalankan (CP08: transfer learning dan demo)
Model: MobileNetV3-Small dengan bobot ImageNet (unduhan pertama butuh internet). Skrip training tidak membuat split; ia memakai `split_v2.csv` yang beku. Hyperparameter ada di `configs/transfer.yaml` dan bisa ditimpa lewat argumen CLI.
```
python src/train_transfer.py                                  # baseline: mode head
python src/train_transfer.py --mode partial --unfreeze-blocks 3 --run-name mnv3s_partial3
python src/train_transfer.py --run-name tuning1 --no-test     # tuning tanpa menyentuh test
python src/train_transfer.py --compare                        # tabel perbandingan semua run
```
Atau lewat `notebooks/02_transfer_learning.ipynb` (notebook tipis; logikanya di `src/train_transfer.py`).

Keluaran tiap run ada di `results/transfer/<run_name>/`: `config_used.json`, `log.csv`, `curves.png`, `metrics.json`, per-kelas, confusion matrix, pasangan huruf yang paling tertukar, dan prediksi contoh. `best.pth` tidak di-commit (`.gitignore`), jadi hanya ada di laptop yang melatihnya.

Aturan yang dijaga: augmentasi hanya di train (tanpa horizontal flip karena bisa mengubah makna isyarat), BatchNorm pada layer beku tetap mode eval, model terbaik dipilih dari val macro-F1, test dievaluasi sekali di akhir, dan run yang sudah ada tidak ditimpa tanpa `--overwrite`.

Demo inferensi (butuh `best.pth` dari run yang dipilih):
```
python src/demo.py --run mnv3s_partial3 --image foto1.jpg foto2.jpg    # demo offline
python src/demo.py --run mnv3s_partial3 --webcam                       # bonus; perlu pip install opencv-python
```
Model tidak punya deteksi tangan: tanpa tangan di bingkai ia tetap menebak satu huruf, sehingga prediksi di bawah `--threshold` ditampilkan sebagai `?`. Jalankan mode webcam di terminal, bukan di Jupyter.

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
configs/    config.yaml (data, split, seed) dan transfer.yaml (hyperparameter training)
data/       data mentah (tidak di-commit)
splits/     split_v2.csv (dipakai), label_map.json, laporan leakage; split_v1.csv hanya arsip
app/        aplikasi Streamlit lokal (streamlit_app.py, inference.py, ui_parts.py)
src/        sanitary_check.py, make_split.py, dataset.py, train_transfer.py, demo.py
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
- Perubahan split harus menjadi `split_v3` dengan alasan tercatat di sini.
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
