# Analisis kesalahan: v3_full_cos30_ls_strong (test)

- Gambar: 690 | akurasi 0.900 | macro-F1 0.898 | salah: 69
- Grup ponsel di split ini: 6

## Kategori kegagalan

### 1. Huruf yang bentuk tangannya mirip
Pasangan tertukar terbanyak: M->N (7), D->P (6), P->D (4), N->M (3), U->C (3).
Contoh gambar (salah paling yakin) untuk M/N, D/P, J/I: `error_pairs.png`. Hipotesis: perbedaan jari (mis. posisi ibu jari pada M vs N) kecil pada foto 3000 px yang diperkecil ke 224 px, sehingga detail hilang. Crop ke tangan (Phase 3) menargetkan kategori ini.

### 2. Bergantung pada latar
Akurasi per latar (rendah ke tinggi): baju 0.88 (n=173), bebas 0.88 (n=173), polos_gelap 0.91 (n=171), polos_terang 0.93 (n=173).
Grafik: `error_by_background.png`. Latar ramai atau mirip warna kulit/baju menyulitkan model yang melihat seluruh gambar; hipotesis ini juga ditargetkan crop.

### 3. Bergantung pada kontributor atau perangkat
Lima grup terburuk (n>=10): iphone15 0.85 (n=172), iphonexr 0.86 (n=102), samsunga51 0.90 (n=104), oppoa53 0.92 (n=104), realme7i 0.93 (n=104).
Contoh: `error_worst_group.png`. Hipotesis: gaya tangan, jarak, dan kamera tiap kontributor berbeda dari data latih. Ini tidak ditangani crop; butuh lebih banyak ragam kontributor atau augmentasi yang lebih kuat.

### 4. Salah dengan keyakinan tinggi
6 dari 69 kesalahan punya confidence >= 0.8. Model tidak bisa mengenali sendiri kapan ia salah, sehingga `--threshold` demo hanya menyaring sebagian.

## Confidence vs akurasi

| bin confidence | n | akurasi | rata-rata conf |
|---|---|---|---|
| [0.0, 0.5) | 98.0 | 0.612 | 0.371 |
| [0.5, 0.6) | 60.0 | 0.817 | 0.550 |
| [0.6, 0.7) | 71.0 | 0.887 | 0.652 |
| [0.7, 0.8) | 95.0 | 0.937 | 0.749 |
| [0.8, 0.9) | 142.0 | 0.965 | 0.854 |
| [0.9, 1.0) | 224.0 | 0.996 | 0.953 |

| threshold | akurasi di atas threshold | cakupan |
|---|---|---|
| 0.5 | 0.948 | 0.86 |
| 0.55 | 0.957 | 0.82 |
| 0.6 | 0.962 | 0.77 |
| 0.65 | 0.970 | 0.72 |
| 0.7 | 0.974 | 0.67 |
| 0.75 | 0.980 | 0.59 |
| 0.8 | 0.984 | 0.53 |
| 0.85 | 0.987 | 0.44 |
| 0.9 | 0.996 | 0.32 |
| 0.95 | 0.992 | 0.18 |

**Saran `--threshold` demo:** 0.5 (akurasi >= 0.9 dengan cakupan >= 50%).
