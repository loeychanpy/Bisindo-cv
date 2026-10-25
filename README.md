# Pengenalan Alfabet BISINDO (A-Z) dari Gambar Tangan

Proyek Akhir Visi Komputer (IF0PP006, kelas 5PITA, Semester Ganjil 2026/2027) - Kelompok 4.

## Pertanyaan proyek
(Isi 2-3 kalimat: apa yang dibandingkan, pada dataset apa, dan ukuran keberhasilannya.)

## Dataset
- Sanjaya, UMN 2024, Mendeley Data. DOI: 10.17632/ywnjpbcz8m.1
- Lisensi: CC BY 4.0
- Cara memperoleh: unduh dari Mendeley Data, ekstrak ke `data/bisindo/<HURUF>/gambar.jpg`.
- Dataset tidak di-commit (lihat `.gitignore`).

## Setup
```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Cara menjalankan (CP07)
1. Letakkan data di `data/bisindo/`.
2. Sesuaikan `GROUP_REGEX` di notebook jika ada info penanda/latar.
3. Buka `bisindo_cp07_data_pipeline.ipynb`, lalu Run All.
4. Keluaran: `splits/split_v1.csv`, `splits/label_map.json`, dan `results/*`.

## Struktur folder
```
configs/    konfigurasi (seed, path, rasio split)
data/       data mentah (tidak di-commit)
splits/     split tetap (di-commit)
src/        kode yang dipakai ulang (mulai CP08)
notebooks/  notebook eksperimen berikutnya
results/    tabel dan figur hasil
```

## Split dan leakage
- Split dibuat sekali (`split_v1.csv`) dengan seed 42, lalu dibekukan.
- Mode split: (isi: per grup atau stratified) dan keterbatasannya.
- Perubahan split harus menjadi `split_v2` dengan alasan tercatat di sini.

## Kontribusi anggota
| Anggota | Kontribusi |
|---|---|
| | |

## Bantuan eksternal
(Catat pustaka, model pra-latih, kode rujukan, tutorial, dan alat bantu lain beserta bagian yang dipakai.)
