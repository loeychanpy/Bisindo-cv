# LAPORAN PROYEK AKHIR VISI KOMPUTER
**Pengenalan Alfabet BISINDO (A–Z) Menggunakan Transfer Learning MobileNetV3-Small dan Antarmuka Streamlit**

**Disusun Oleh:**
- **412024003 - CHRISTIAN MELHAN**
- **412024004 - FELIX WONG**
- **412024033 - JANISHA JAYA**

**PROGRAM STUDI INFORMATIKA**  
**FAKULTAS TEKNOLOGI CERDAS**  
**UNIVERSITAS KRISTEN KRIDA WACANA**  
**JAKARTA**  
**2026**

---

## DAFTAR ISI

- **BAB I PENDAHULUAN & PEMAHAMAN MASALAH**
  - 1.1 Identifikasi Permasalahan
  - 1.2 Pengumpulan dan Analisis Data
  - 1.3 Kategorisasi Permasalahan dan Matriks Perencanaan
  - 1.4 Identifikasi Pengguna Sistem
  - 1.5 Formulasi Tujuan
  - 1.6 Rujukan Akademik dan Relevansinya
- **BAB II PERANCANGAN (DESIGN)**
  - 2.1 Model Permasalahan
  - 2.2 Kandidat Strategi Transfer Learning
  - 2.3 Kriteria Pemilihan Model dan Metrik Evaluasi
  - 2.4 Konfigurasi Inti dan Kontrol Eksperimen
  - 2.5 Rancangan Validasi dan Strategi Pembagian Data (split_v3)
  - 2.6 Arsitektur Aplikasi dan Struktur Repositori
- **BAB III BUKTI HASIL EKSPERIMEN & PEMAHAMAN MODEL**
  - 3.1 Verifikasi Data dan Loader Determinisme
  - 3.2 Hasil Evaluasi Strategi Transfer Learning pada split_v3
  - 3.3 Optimasi Model Lanjutan (Cosine LR, Label Smoothing, dan Strong Augmentation)
  - 3.4 Perbandingan dan Pemilihan Model
  - 3.5 Analisis Kegagalan (Failure Analysis)
  - 3.6 Perbaikan yang Telah Diuji
  - 3.7 Interpretasi Hasil
- **BAB IV APLIKASI STREAMLIT & SKENARIO DEMO**
  - 4.1 Antarmuka Aplikasi
  - 4.2 Pengukuran Latensi dan Throughput
  - 4.3 Strategi Cadangan Demo UAS
- **BAB V MANAJEMEN TIM & TATA KELOLA**
  - 5.1 Anggota dan Pembagian Peran
  - 5.2 Kanal Komunikasi dan Aturan Rapat
  - 5.3 Resolusi Konflik
- **DAFTAR PUSTAKA**

---

## BAB I PENDAHULUAN & PEMAHAMAN MASALAH

### 1.1 Identifikasi Permasalahan
**Konteks.** Bahasa Isyarat Indonesia (BISINDO) memiliki alfabet statis yang terdiri atas 26 gestur tangan, satu gestur untuk setiap huruf A sampai Z. Pengguna yang mempelajari dasar alfabet ini secara mandiri membutuhkan umpan balik atas ketepatan gestur yang diperagakan. Sistem pengenalan citra dapat memberikan umpan balik tersebut dengan menebak huruf secara langsung dari foto tunggal atau tayangan kamera.

**Ketegangan utama.** Masalahnya bukan sekadar memilih model dengan akurasi tertinggi, melainkan menimbang tiga hal yang saling menarik. Pertama, akurasi dan kecepatan: model harus cukup akurat, tetapi juga cukup ringan untuk berjalan secara real-time pada CPU laptop. Kedua, kedalaman adaptasi: melatih lebih banyak bobot jaringan memberi ruang adaptasi yang lebih besar terhadap gestur tangan, tetapi menambah jumlah parameter yang dilatih dan latensi komputasi (Bab III). Ketiga, kejujuran evaluasi: citra dalam dataset diambil dengan ponsel dan latar yang berbeda-beda, sehingga pembagian data yang tidak memisahkan perangkat berisiko membuat skor uji tampak lebih baik daripada kondisi sebenarnya. Karena itu, tidak ada satu konfigurasi yang otomatis terbaik; pilihan bergantung pada kompromi antara ketiga hal tersebut.

**Kondisi data.** Dataset yang digunakan memuat 4.799 berkas mentah yang, setelah sanitasi, menyisakan 4.462 citra valid untuk 26 kelas. Nama berkas memuat huruf, kondisi latar, serta merek dan model ponsel, sehingga keragaman perangkat dan latar dapat ditelusuri langsung dari data (rincian pada subbab 1.2). Perlu dicatat bahwa bukti ini bersifat deskriptif: data menunjukkan keragaman perangkat dan latar, bukan bahwa model telah terbukti tahan terhadap keragaman tersebut. Hal itu diuji pada Bab III.

**Rumusan masalah.** Seberapa efektif model konvolusi ringan MobileNetV3-Small pra-latih ImageNet dalam mengklasifikasikan 26 huruf statis alfabet BISINDO dari citra tangan alami, dan strategi transfer learning manakah yang paling optimal: melatih kepala klasifikasi saja (head, baseline), membuka sebagian blok konvolusi terakhir (partial), atau membuka seluruh bobot jaringan (full), pada dataset yang telah diisolasi dari kebocoran perangkat dan latar belakang?

**Alasan perlunya solusi.** Jumlah citra yang tersedia (4.462) relatif kecil untuk melatih jaringan konvolusi dari awal, sedangkan aplikasi harus berjalan pada laptop. Transfer learning dari bobot ImageNet menjawab keduanya karena fitur visual umum, seperti tepi dan sudut, sudah dipelajari dan hanya sebagian jaringan yang perlu diadaptasi (Yosinski dkk., 2014). Dengan membandingkan beberapa tingkat adaptasi pada pembagian data yang bebas kebocoran, kelompok memperoleh dasar empiris untuk memilih konfigurasi, bukan sekadar asumsi.

### 1.2 Pengumpulan dan Analisis Data
**Sumber dan lisensi.** Data yang digunakan adalah data sekunder dari Mendeley Data (Universitas Multimedia Nusantara; Sanjaya, 2024) dengan lisensi Creative Commons Attribution 4.0 International (CC BY 4.0). Data tidak dikumpulkan langsung oleh kelompok. Hanya varian *01. Original Images* yang dipakai sesuai spesifikasi proyek; varian Resized, Binary, dan Code Chain tidak dipakai.

**Jenis dan ukuran data.** Dataset berbentuk citra foto statis untuk 26 kelas huruf A–Z. Format penamaan berkas memuat pola `[Huruf]_[Kondisi Latar]_[Merek Ponsel]_[Model Ponsel].jpg`, misalnya `A_Polos Terang_Xiaomi _Note 8 Pro.jpg`. Ringkasan karakteristik dataset disajikan sebagai berikut.

| Atribut | Nilai |
|---|---|
| Total berkas mentah | 4.799 citra |
| Berkas bermasalah yang dibuang | 337 berkas |
| Citra valid | 4.462 citra |
| Jumlah kelas | 26 (huruf A–Z) |
| Rata-rata citra per kelas | 171 citra |
| Resolusi asli | sekitar 3000 piksel |
| Jumlah grup ponsel kontributor | 44 grup |

**Kualitas data.** Pemeriksaan kualitas dilakukan melalui skrip `src/sanitary_check.py` yang membuang 337 berkas bermasalah dengan rincian jenis sebagai berikut:
1. Berkas Apple HEIC yang salah dinamai berekstensi .jpg.
2. Berkas korup berukuran 0 byte.
3. Berkas metadata tersembunyi macOS (berawalan `._`).
4. Berkas duplikat identik lintas kelas berdasarkan hash SHA-256.

Hasil akhirnya adalah 4.462 citra valid untuk 26 kelas alfabet, dengan rata-rata 171 citra per kelas.

**Potensi bias dan kebocoran data.** Karena nama berkas memuat model ponsel dan kondisi latar, citra dari satu ponsel dan latar yang sama cenderung mirip satu sama lain. Apabila citra semacam itu tersebar di data latih dan data uji, model dapat tampak berkinerja baik hanya karena mengenali kekhasan perangkat atau latar, bukan bentuk gestur. Risiko ini ditangani secara tuntas melalui pembagian berbasis grup ponsel (`split_v3`, subbab 2.5).

**Keterbatasan data.** Pertama, citra bersifat statis sehingga gestur dinamis, seperti huruf J yang berupa gerakan melengkung di udara, hanya terwakili oleh satu posisi diam. Kedua, hanya varian Original Images yang digunakan. Ketiga, dataset berasal dari satu sumber (Sanjaya, 2024), sehingga kemampuan generalisasi ke perangkat, latar, dan pengguna di luar dataset belum diuji.

### 1.3 Kategorisasi Permasalahan dan Matriks Perencanaan
Perencanaan proyek dirangkum dalam matriks berikut, yang mengkategorikan permasalahan berdasarkan masalah, data, pendekatan, kebutuhan komputasi, dan risiko.

| Dimensi | Spesifikasi proyek | Penjelasan |
|---|---|---|
| **Masalah** | Klasifikasi citra alfabet BISINDO A–Z | Mengenali 26 gestur tangan statis alfabet Bahasa Isyarat Indonesia dari citra foto tunggal. |
| **Data** | BISINDO Mendeley (Sanjaya, 2024) | 4.462 citra varian Original, 26 kelas huruf, resolusi asli sekitar 3000 piksel. |
| **Baseline** | MobileNetV3-Small pra-latih (mode head) | Hanya melatih classifier akhir (26.650 parameter); backbone ImageNet dibekukan penuh. |
| **Metrik evaluasi** | Macro-F1 (utama), akurasi, latensi (ms) | Macro-F1 mencegah bias kelas dominan; latensi memastikan kelayakan interaksi real-time. |
| **Kebutuhan komputasi** | CPU 16-thread dan GPU NVIDIA RTX Laptop | Pelatihan GPU ~15 menit; inferensi real-time 11–64 ms per citra. |
| **Risiko utama** | Kebocoran perangkat (device leakage) | Satu ponsel muncul di data latih dan data uji; dimitigasi tuntas dengan split_v3 berbasis 44 grup ponsel. |
| **Risiko sekunder** | False positive tanpa tangan di bingkai | Model CNN selalu menebak satu kelas; dimitigasi dengan confidence threshold dan deteksi tangan MediaPipe. |

### 1.4 Identifikasi Pengguna Sistem
- **Pengguna akhir:** Siswa, mahasiswa, dan masyarakat umum yang ingin mempelajari dasar alfabet BISINDO secara mandiri.
- **Pengguna sekunder (potensial):** Pengembang aplikasi teknologi aksesibilitas.
- **Skenario penggunaan:** Aplikasi edukasi interaktif berbasis web lokal. Pengguna memperagakan gestur tangan di depan webcam atau mengunggah foto isyarat untuk memperoleh prediksi huruf, visualisasi keyakinan, dan perangkaian kata secara langsung.
- **Kebutuhan pengguna:**
  1. Mendapat prediksi huruf dari webcam atau foto yang diunggah.
  2. Melihat tingkat keyakinan prediksi beserta alternatif teratas (top-k).
  3. Merangkai huruf menjadi kata secara langsung.
  4. Mengakses sistem melalui aplikasi web lokal yang ringan dan responsif.

### 1.5 Formulasi Tujuan
- **Tujuan umum:** Membangun sistem pengenalan alfabet BISINDO A–Z dari citra tangan berbasis transfer learning MobileNetV3-Small yang akurat, ringan, dan dapat dioperasikan melalui antarmuka web Streamlit.
- **Tujuan khusus:**
  1. Menyiapkan dataset yang bersih dan bebas kebocoran data melalui sanitasi berkas dan pembagian berbasis grup ponsel (`split_v3`).
  2. Membandingkan strategi transfer learning *head*, *partial*, dan *full* menggunakan Macro-F1 sebagai metrik utama pada split beku yang sama.
  3. Menganalisis pola kegagalan model dan menguji perbaikan berupa *confidence threshold* serta deteksi tangan MediaPipe.
  4. Membangun aplikasi Streamlit yang menyediakan prediksi langsung, unggah gambar, snapshot kamera, dan dashboard hasil model.
- **Kriteria keberhasilan:**
  1. Baseline (head): Macro-F1 >= 0,50 (50%).
  2. Metode utama (partial atau full): Macro-F1 >= 0,80 (80%) pada split_v3.
  3. Latensi inferensi <= 100 ms per citra pada CPU laptop (setara >= 10 FPS).

### 1.6 Rujukan Akademik dan Relevansinya
1. **Sanjaya, A. (2024).** *BISINDO Indonesian Sign Language Alphabet Image Data*. Mendeley Data. DOI: 10.17632/ywnjpbcz8m.1.  
   *Relevansi:* Rujukan dataset utama 4.462 citra varian Original untuk 26 alfabet BISINDO dengan variasi ponsel dan latar.
2. **Howard, A., dkk. (2019).** *Searching for MobileNetV3*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 1314–1324. DOI: 10.1109/ICCV.2019.00140.  
   *Relevansi:* Rujukan arsitektur CNN efisien dengan inverted bottleneck dan aktivasi Hard-Swish yang dioptimalkan untuk perangkat bergerak.
3. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** *How transferable are features in deep neural networks?*. Advances in Neural Information Processing Systems (NeurIPS), 27.  
   *Relevansi:* Dasar teori transfer learning: lapisan awal mengekstrak fitur visual umum (tepi/sudut), sedangkan lapisan atas menangani fitur spesifik domain.
4. **Lugaresi, C., dkk. (2019).** *MediaPipe: A framework for building perception pipelines*. arXiv preprint arXiv:1906.08172.  
   *Relevansi:* Landasan modul deteksi tangan (Hand Landmarker) untuk memotong area telapak tangan secara otomatis sebelum klasifikasi.
5. **Sutjiadi, R. (2023).** *Android-based application for real-time Indonesian Sign Language recognition using convolutional neural network*. TEM Journal, 12(3), 1541–1549. DOI: 10.18421/TEM123-35.  
   *Relevansi:* Rujukan studi implementasi pengenalan bahasa isyarat Indonesia secara real-time pada perangkat berdaya komputasi terbatas.

---

## BAB II PERANCANGAN (DESIGN)

### 2.1 Model Permasalahan
Sistem dirancang untuk mengenali 26 huruf statis alfabet BISINDO dari citra tangan. Variabel masukan berupa citra berwarna yang diubah ke resolusi 224 x 224 piksel; pada aplikasi, citra dapat dipotong terlebih dahulu pada area tangan menggunakan deteksi MediaPipe dengan margin 20%. Keluaran berupa distribusi probabilitas pada 26 kelas, dari mana sistem menampilkan huruf prediksi, tingkat keyakinan, dan alternatif top-k. Apabila probabilitas tertinggi berada di bawah ambang keyakinan, sistem menampilkan tanda tanya (?) alih-alih huruf.

Alur sistem terdiri atas empat tahap berikut:
1. **Persiapan data:** Citra mentah Mendeley disanitasi (`sanitary_check.py`), dikelompokkan berdasarkan ponsel (`group_keys.py`), lalu dibagi menjadi split beku bebas bocor (`split_v3.csv`).
2. **Mesin pelatihan:** `FastBisindoDataset` dengan RAM caching multi-thread, augmentasi citra tanpa flip, transfer learning MobileNetV3-Small, evaluasi Val Macro-F1, dan penyimpanan bobot terbaik (`best.pth`).
3. **Evaluasi dan diagnostik:** Pengujian pada test set beku `split_v3`, perhitungan confusion matrix, analisis pasangan huruf tertukar (`error_analysis.py`), dan tabel perbandingan (`comparison.csv`).
4. **Aplikasi antarmuka:** Streamlit GUI (`app/streamlit_app.py`) memuat model dan menyediakan tab Live WebRTC, Unggah Gambar, Kamera, dan Contoh Data Uji dengan integrasi deteksi tangan MediaPipe.

### 2.2 Kandidat Strategi Transfer Learning
Seluruh kandidat memakai MobileNetV3-Small pra-latih ImageNet (13 blok backbone, 1,54 juta parameter) yang diuji secara terkontrol pada `split_v3`:
1. **Head only (baseline):** Seluruh backbone dibekukan (`requires_grad = False`) dan hanya lapisan `Linear(1024, 26)` yang dilatih (26.650 parameter atau 1,73% dari total).
2. **Partial unfreezing:** Membuka sebagian blok konvolusi terakhir bersama classifier:
   - Partial 2 blok: Blok 12–13 + classifier (968.042 parameter / 62,7%).
   - Partial 4 blok: Blok 10–13 + classifier (1.353.986 parameter / 87,7%).
   - Partial 6 blok: Blok 8–13 + classifier (1.405.754 parameter / 91,0%).
3. **Full fine-tuning:** Seluruh 13 blok konvolusi dan classifier dilatih (1.544.506 parameter / 100%).
4. **Full fine-tuning dengan optimasi lanjutan:** Melatih seluruh jaringan dengan integrasi Cosine Annealing LR Scheduler, Label Smoothing (0.1), dan Strong Data Augmentation selama 30 epoch.

### 2.3 Kriteria Pemilihan Model dan Metrik Evaluasi
Pemilihan model didasarkan pada kriteria berikut:

| Kriteria | Penjelasan |
|---|---|
| **Macro-F1 (utama)** | Rata-rata F1 per kelas tanpa pembobotan di seluruh 26 huruf; mencegah dominasi kelas tertentu dan memastikan model handal di seluruh alfabet. |
| **Akurasi** | Rasio tebakan benar terhadap seluruh sampel data uji. |
| **Weighted-F1** | Rata-rata F1 yang dibobot menurut jumlah sampel masing-masing kelas. |
| **Latensi inferensi (ms)** | Waktu median untuk dekode, praproses, dan forward pass per citra. |

Macro-F1 dihitung dengan rumus:
- F1_c = 2 x (P_c x R_c) / (P_c + R_c)
- Macro-F1 = (1 / C) x sum_c F1_c (dengan C = 26)

### 2.4 Konfigurasi Inti dan Kontrol Eksperimen
Konfigurasi inti pelatihan (`transfer.yaml` dan `config_v3.yaml`):

| Parameter | Nilai Dasar | Nilai Optimasi Lanjutan |
|---|---|---|
| Resolusi input citra | 224 x 224 piksel | 224 x 224 piksel |
| Batch size | 64 | 32 |
| Epoch | 12 (early stopping patience 4) | 30 (patience 8) |
| Optimizer | AdamW (weight decay 0,0001) | AdamW (weight decay 0,0001) |
| Learning rate head | 0,001 | 0,001 |
| Learning rate backbone | 0,0001 | 0,0001 |
| LR Scheduler | Konstan | Cosine Annealing (warmup 2 epoch, min_lr 0.01) |
| Label Smoothing | 0.0 | 0.1 |
| Augmentasi | Standar (crop, rotasi 10°, color, blur) | Strong (perspektif, color, blur, rotasi) |
| RAM cache | Aktif (multi-threaded preload) | Aktif |

**Kontrol eksperimen.** Setiap pengujian dijalankan dengan kontrol ketat: seluruh model diuji pada split beku yang sama (`split_v3.csv`, MD5: `2769336b...`), seed dikunci penuh pada 42, dan lapisan BatchNorm pada layer beku dipaksa tetap dalam mode eval.

### 2.5 Rancangan Validasi dan Strategi Pembagian Data (split_v3)
Seluruh evaluasi eksperimental dalam proyek ini **hanya menggunakan `split_v3`**. Strategi `split_v3` dirancang khusus untuk mengatasi cacat metodologis kebocoran data (*data leakage*):
- Citra dikelompokkan secara ketat berdasarkan **44 grup ponsel kontributor** (`src/group_keys.py`).
- Seluruh citra dari satu ponsel dialokasikan tepat ke satu split saja:
  - **Data Latih (Train):** 3.052 citra.
  - **Data Validasi (Val):** 720 citra.
  - **Data Uji (Test):** 690 citra.
- **Audit Kebocoran Data (`splits/leakage_report_v3.txt`):**
  - Irisan path antar split = 0.
  - Irisan hash SHA-256 antar split = 0.
  - Irisan grup ponsel antar split = 0 (`HASIL: BERSIH`).

Dengan prosedur ini, data uji benar-benar mengevaluasi ketahanan model terhadap model kamera ponsel dan latar belakang baru yang belum pernah dilihat saat pelatihan.

### 2.6 Arsitektur Aplikasi dan Struktur Repositori
Berkas yang digunakan langsung oleh aplikasi Streamlit:

| Berkas | Fungsi |
|---|---|
| `app/streamlit_app.py` | Entry point utama aplikasi antarmuka Streamlit. |
| `app/inference.py` | Backend inferensi aman: pemuat model, penanganan error, dan kalkulasi top-k. |
| `app/live.py` | Pemroses video streaming real-time WebRTC untuk Tab Live dengan worker asinkron. |
| `app/ui_parts.py` | Komponen kartu prediksi, visualisasi bounding box, CSS kustom, dan chart Altair. |
| `models/hand_landmarker.task` | Model bobot MediaPipe untuk deteksi kotak tangan. |
| `src/demo.py` | Pemuat model, optimasi TorchScript JIT, dan algoritma penstabil huruf (LetterStabilizer). |
| `src/dataset.py` | Definisi kelas huruf (CLASSES A–Z) dan fungsi transformasi citra. |
| `src/train_transfer.py` | Definisi arsitektur build_model dan pembaca citra cepat load_rgb. |
| `src/hand_bbox.py` | Modul pemotong telapak tangan MediaPipe. |
| `src/benchmark_latency.py`| Skrip benchmark latensi inferensi per tahap. |
| `splits/split_v3.csv` | File split data beku untuk Tab Contoh Data Uji. |
| `results/transfer/<run>/` | Checkpoint model terlatih (best.pth, metrics.json, curves.png, confusion matrix). |

---

## BAB III BUKTI HASIL EKSPERIMEN & PEMAHAMAN MODEL

### 3.1 Verifikasi Data dan Loader Determinisme
- **Integritas citra:** Verifikasi pemuatan berhasil memuat seluruh 4.462 berkas citra tanpa `FileNotFoundError`.
- **Kecepatan preload:** Seluruh 4.462 citra berhasil didekode ke skala 256 piksel dan disimpan di RAM dalam waktu 30,0 detik menggunakan 16 thread paralel, memangkas waktu I/O disk menjadi 0 detik pada epoch-epoch berikutnya.
- **Determinisme:** Penguncian seed 42 pada Python `random`, NumPy, PyTorch CPU, dan CUDA menghasilkan inisialisasi bobot dan urutan batch yang identik saat dijalankan ulang.

### 3.2 Hasil Evaluasi Strategi Transfer Learning pada split_v3
Tabel berikut menyajikan hasil evaluasi perbandingan strategi transfer learning MobileNetV3-Small pada data uji beku `split_v3` (690 citra uji, isolasi 44 grup ponsel, seed 42):

| Run | Strategi Transfer Learning | Layer Dilatih | Parameter Dilatih | Epoch Terbaik | Val Macro-F1 | Test Akurasi | Test Macro-F1 | Latensi CPU (ms) |
|---|---|---|---|---|---|---|---|---|
| `v3_head` | Baseline (Head Only) | Linear Akhir | 26.650 (1,7%) | 15 | 0,4925 | 53,8% | 0,5346 | 61,7 ms |
| `v3_partial2` | Partial 2 blok | Blok 12–13 + classifier | 968.042 (62,7%) | 10 | 0,6398 | 71,4% | 0,7092 | 62,1 ms |
| **`v3_partial4`** | **Partial 4 blok (Sweet Spot)** | **Blok 10–13 + classifier** | **1.353.986 (87,7%)** | **11** | **0,7281** | **78,7%** | **0,7845** | **64,4 ms** |
| `v3_partial6` | Partial 6 blok | Blok 8–13 + classifier | 1.405.754 (91,0%) | 11 | 0,7327 | 79,1% | 0,7864 | 122,5 ms |
| `v3_full` | Full Fine-Tuning (12 epoch) | Seluruh Jaringan (13 blok) | 1.544.506 (100%) | 11 | 0,7538 | 84,1% | 0,8369 | 98,3 ms |

### 3.3 Optimasi Model Lanjutan (Cosine LR, Label Smoothing, dan Strong Augmentation)
Untuk mengeksplorasi batas kemampuan MobileNetV3-Small pada `split_v3`, dilakukan optimasi lanjutan berbasis akselerasi GPU NVIDIA selama 30 epoch dengan menambahkan penjadwal *Cosine Annealing*, *Label Smoothing*, dan *Strong Augmentation*:

| Run Optimasi Lanjutan | Konfigurasi Tambahan | Parameter Dilatih | Epoch Terbaik | Val Macro-F1 | Test Akurasi | Test Macro-F1 | Latensi GPU Forward |
|---|---|---|---|---|---|---|---|
| `v3_full_e30` | Full Tuning 30 Epoch | 1.544.506 | 20 | 0,7601 | — | — | ~6,5 ms |
| `v3_full_cos30` | + Cosine Annealing LR | 1.544.506 | 20 | 0,7712 | — | — | ~6,2 ms |
| `v3_full_cos30_ls` | + Label Smoothing (0.1) | 1.544.506 | 29 | 0,8083 | — | — | ~6,0 ms |
| **`v3_full_cos30_ls_strong`** | **+ Strong Data Augmentation** | **1.544.506** | **30** | **0,8083** | **90,0%** | **0,8979** | **5,78 ms** |

Model **`v3_full_cos30_ls_strong`** berhasil mencatatkan rekor performa tertinggi pada proyek ini dengan **Akurasi 90,0%** dan **Macro-F1 0,8979** pada data uji beku `split_v3`.

### 3.4 Perbandingan dan Pemilihan Model
1. **Peningkatan dari Baseline:** Membuka 4 blok konvolusi (`v3_partial4`) meningkatkan Macro-F1 dari 0,5346 menjadi 0,7845 (+25,0 poin persentase) dibanding baseline `v3_head`. Hal ini membuktikan bahwa adaptasi representasi gestur tangan pada layer konvolusi atas sangat krusial.
2. **Efisiensi Partial vs Full:** Membuka 6 blok (`v3_partial6`) hanya menaikkan F1 sekitar 0,002 dibanding `v3_partial4`, tetapi melipatgandakan latensi (122,5 ms vs 64,4 ms). Oleh karena itu, **`v3_partial4` dipilih sebagai model paling optimal (*Sweet Spot*) untuk perangkat berdaya komputasi rendah**.
3. **Model Akurasi Maksimal:** Varian **`v3_full_cos30_ls_strong`** menjadi pilihan utama untuk skenario di mana akurasi menjadi prioritas tertinggi, mencapai akurasi 90,0% dengan latensi forward pass hanya 5,78 ms di GPU atau 11,23 ms di CPU teroptimasi.

Pencapaian terhadap target keberhasilan (subbab 1.5):
- Baseline (head): Macro-F1 >= 0,50 -> **Tercapai (0,5346)**.
- Metode utama: Macro-F1 >= 0,80 pada split_v3 -> **Tercapai oleh `v3_full_cos30_ls_strong` (0,8979) dan `v3_full` (0,8369)**.
- Latensi inferensi <= 100 ms per citra (CPU) -> **Tercapai (38,2–64,4 ms)**.

### 3.5 Analisis Kegagalan (Failure Analysis)
Analisis dilakukan berdasarkan audit confusion matrix pada berkas `top_confusions_test.csv` model terbaik:

| Pasangan Huruf Tertukar | Frekuensi Kesalahan | Penyebab Morfologis & Visual |
|---|:---:|---|
| **M tertukar N; N tertukar M** | 7 kasus M->N; 3 kasus N->M | Huruf M melipat tiga jari ke bawah, sedangkan huruf N melipat dua jari. Pada resolusi 224 piksel atau pencahayaan minim, batas pemisah antar celah jari manis dan kelingking mengalami peleburan piksel (*blur*). |
| **D tertukar P; P tertukar D** | 6 kasus D->P; 4 kasus P->D | Huruf P dan D membentuk lingkaran antara ibu jari dan telunjuk. Pada sudut pandang kamera miring (*oblique*), orientasi lingkaran tampak serupa. |
| **U tertukar C** | 3 kasus | Lengkungan jari telunjuk dan tengah yang merapat pada sudut tertentu menyerupai lengkungan huruf C. |
| **J tertukar I** | 2 kasus | Huruf J adalah gerakan dinamis melengkung di udara dengan jari kelingking, sedangkan foto 2D statis hanya menangkap satu posisi diam yang mirip huruf I. |

**Pengaruh Kondisi Latar Belakang:**
Akurasi model `v3_full_cos30_ls_strong` bervariasi menurut kondisi latar:
- Latar Baju: 88%
- Latar Bebas: 88%
- Latar Polos Gelap: 91%
- Latar Polos Terang: 93% (paling mudah dikenali)

### 3.6 Perbaikan yang Telah Diuji
1. **Confidence Thresholding:** Sistem menolak prediksi jika probabilitas tertinggi berada di bawah ambang keyakinan (rekomendasi: 0.50) dan menampilkan tanda tanya (?). Berdasarkan data keandalan (`error_reliability.csv`), ambang batas 0.50 menghasilkan akurasi 94,8% pada 86% data uji, efektif menyaring tebakan palsu saat tidak ada tangan.
2. **Integrasi Deteksi Tangan MediaPipe (`hand_bbox.py`):** Citra dipotong pada area telapak tangan dengan margin 20%, mengurangi distraksi dari latar belakang ruangan dan pakaian bermotif.

### 3.7 Interpretasi Hasil
- Seluruh angka yang dilaporkan mencerminkan performa pada perangkat ponsel baru karena evaluasi menggunakan set data uji `split_v3` dengan isolasi 44 grup ponsel kontributor secara penuh (*zero-leakage*).
- Macro-F1 dijadikan acuan utama karena menimbang seluruh 26 huruf secara setara, memastikan model tidak hanya mengandalkan huruf-huruf yang mudah dikenali (seperti A, L, V).
- Penerapan teknik *regularization* (Label Smoothing dan Strong Augmentation) terbukti menjadi faktor penentu kenaikan akurasi dari 84,1% menjadi 90,0% pada evaluasi ponsel baru.

---

## BAB IV APLIKASI STREAMLIT & SKENARIO DEMO

### 4.1 Antarmuka Aplikasi
Aplikasi web lokal dijalankan melalui perintah `streamlit run app/streamlit_app.py` (port 8501) dan terdiri atas enam tab:

| Tab | Fungsi |
|---|---|
| **Live** | Video streaming real-time WebRTC dengan LetterStabilizer (jendela 15 frame) dan tombol eja kata (Spasi, Hapus, Reset). Dilengkapi penguncian bitrate 4 Mbps agar video tidak patah-patah. |
| **Unggah Gambar** | Unggah multi-foto JPG/PNG sekaligus dengan kartu prediksi, probabilitas, diagram batang top-k (Altair), dan latensi pemrosesan. |
| **Kamera** | Snapshot foto dari webcam browser dengan tombol interaktif Tambah Huruf untuk menyusun kata secara bertahap. |
| **Contoh Data Uji** | Menampilkan 9 sampel acak dari split test beku dengan filter huruf A–Z dan kotak centang *"Hanya tampilkan yang salah"* untuk memperlihatkan kelemahan model secara transparan. |
| **Hasil Model** | Dashboard diagnostik: metrik (akurasi, F1, latensi), kurva loss dan akurasi pelatihan, confusion matrix resolusi tinggi, dan tabel komparasi seluruh run. |
| **Tentang** | Informasi ilmiah dataset, lisensi CC BY 4.0, dan batasan operasional model. |

### 4.2 Pengukuran Latensi dan Throughput
Pengukuran latensi resmi per tahap menggunakan `src/benchmark_latency.py` pada model `v3_full_cos30_ls_strong`:

| Tahap Pemrosesan | CPU Intel Core i7 (ms) | GPU NVIDIA RTX (ms) | Keterangan |
|---|:---:|:---:|---|
| Decode Citra (JPEG -> RGB) | 28,17 ms | 26,16 ms | Dekode draft skala cepat |
| Preprocessing (Resize + Norm) | 1,91 ms | 4,11 ms | Transformasi tensor |
| Forward Pass (Eager) | 22,18 ms | 10,90 ms | Model standar |
| **Forward Pass (Optimized TorchScript)** | **11,23 ms** | **6,48 ms** | **Kernel Conv+BN terfusi (Channels-Last)** |
| **Total Pipeline Inferensi** | **38,24 ms (~26 FPS)** | **59,41 ms (~17 FPS)** | **Mendukung interaksi real-time tanpa jeda** |

### 4.3 Strategi Cadangan Demo UAS
Untuk mengantisipasi kendala saat presentasi langsung di depan penguji, disiapkan tiga skenario cadangan:

| Skenario Kendala | Tindakan Cadangan |
|---|---|
| **Fallback 1: Kamera browser terblokir** | Beralih dari Tab Live ke Tab Unggah Gambar menggunakan berkas foto uji yang telah disiapkan di laptop. |
| **Fallback 2: Kendala berkas input** | Menggunakan Tab Contoh Data Uji yang membaca data uji beku langsung dari harddisk tanpa memerlukan kamera. |
| **Fallback 3: Kendala teknis laptop** | Memutar rekaman video demonstrasi cadangan (durasi 2–3 menit, resolusi 1080p) yang memperlihatkan jalannya seluruh fitur aplikasi. |

---

## BAB V MANAJEMEN TIM & TATA KELOLA

### 5.1 Anggota dan Pembagian Peran
Kelompok 4 terdiri atas tiga mahasiswa dengan pembagian peran terstruktur sebagai berikut:

| Anggota & NIM | Peran Utama | Tanggung Jawab Deliverable |
|---|---|---|
| **FELIX WONG**<br>(412024004) | **Project Lead & Integrator** | Tata kelola repositori, struktur folder, konfigurasi `config_v3.yaml`, manajemen integrasi pull request dan branch, serta penyusunan dokumen laporan master. |
| **JANISHA JAYA**<br>(412024033) | **Data & Leakage Engineer** | Skrip pembagian data `src/make_split.py`, audit kebocoran data zero-leakage (`leakage_report_v3.txt`), sanitasi citra `src/sanitary_check.py`, pemetaan label `label_map.json`, dan formulasi grup ponsel `src/group_keys.py`. |
| **CHRISTIAN MELHAN**<br>(412024003) | **Model & Application Engineer** | Implementasi transfer learning `src/train_transfer.py` (optimasi GPU, scheduler, label smoothing), inferensi `src/demo.py`, benchmark latensi, serta arsitektur antarmuka GUI `app/streamlit_app.py` (Live WebRTC, Batch Upload, Camera). |

### 5.2 Kanal Komunikasi dan Aturan Rapat
- **Komunikasi Harian:** WhatsApp Group Kelompok 4 digunakan untuk koordinasi instan dan pelaporan kendala.
- **Rapat Daring & Screen-Sharing:** Discord Server Kelompok 4 digunakan untuk sesi pemrograman berpasangan (*pair-programming*), evaluasi model mingguan, dan latihan presentasi.
- **Manajemen Repositori:** GitHub Issues & Pull Requests pada branch terpisah (`feat/model-gpu`, `feat/optimize`).
- **Jadwal dan Format Rapat:** Pertemuan mingguan dilaksanakan setiap hari Selasa pukul 19.30 WIB. Rapat diawali sesi standup 15 menit (progres, kendala, target), dilanjutkan peninjauan teknis kode.
- **Pencatatan:** Setiap keputusan hyperparameter dan modifikasi dataset dicatat pada log commit Git.

### 5.3 Resolusi Konflik
- **Perbedaan Pendapat Teknis:** Diputuskan berbasis data empiris (*data-driven decision*), seperti pengujian metrik Macro-F1 dan laporan audit kebocoran data.
- **Keterlambatan Tugas:** Anggota yang memiliki kendala akademik wajib melapor kepada Project Lead minimal 24 jam sebelum tenggat agar beban tugas dapat dialihkan sementara.
- **Keputusan Akhir:** Apabila hasil voting berimbang, keputusan akhir dipegang oleh Project Lead dengan bimbingan dosen pengampu atau asisten laboratorium.

---

## DAFTAR PUSTAKA

1. Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1314–1324. https://doi.org/10.1109/ICCV.2019.00140
2. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.
3. Sanjaya, A. (2024). *BISINDO Indonesian Sign Language Alphabet Image Data*. Mendeley Data. https://doi.org/10.17632/ywnjpbcz8m.1
4. Sutjiadi, R. (2023). Android-based application for real-time Indonesian Sign Language recognition using convolutional neural network. *TEM Journal*, 12(3), 1541–1549. https://doi.org/10.18421/TEM123-35
5. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems (NeurIPS)*, 27.
