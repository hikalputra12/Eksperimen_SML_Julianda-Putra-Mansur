# Eksperimen Supervised Machine Learning - Loan Approval

Proyek ini bertujuan untuk membangun sistem pemrosesan data otomatis dan pemodelan prediksi persetujuan pinjaman (*Loan Approval*) menggunakan algoritma **Random Forest Classifier**. Pelacakan eksperimen (*experiment tracking*) diintegrasikan menggunakan **MLflow** yang terhubung secara ganda (dual-tracking) ke repositori **DagsHub** dan server lokal (**Localhost**).

---

## 📁 Struktur Repositori

```text
Eksperimen_SML_Julianda-Putra-Mansur/
│
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # Pipeline otomatisasi CI/CD (GitHub Actions)
│
├── Eksperimen_SML_Julianda-Putra-Mansur/
│   ├── loan_approval_raw/
│   │   └── dataset_raw.csv            # Dataset mentah awal
│   └── preprocessing/
│       ├── automate_Julianda-Putra-Mansur.py   # Script pemrosesan/pembersihan data
│       └── loan_approval_preprocessing/
│           └── dataset_clean.csv      # Dataset bersih hasil olahan otomatis
│
├── Membangun_model/
│   ├── DagsHub.txt                    # Tautan repositori DagsHub resmi
│   ├── requirements.txt               # Kebutuhan library Python untuk pemodelan
│   ├── modelling.py                   # Pemodelan baseline (Random Forest)
│   ├── modelling_tuning.py            # Hyperparameter tuning (Bayesian Optimization)
│   ├── mlruns/                        # Database lokal run/experiment MLflow (diabaikan Git)
│   └── mlartifacts/                   # Penyimpanan lokal artefak model MLflow (diabaikan Git)
│
├── .gitignore                         # Konfigurasi file yang diabaikan Git (mlruns & mlartifacts)
└── README.md                          # Dokumentasi proyek
```

---

## 🛠️ Persyaratan & Instalasi

Pastikan Anda telah menginstal **Python 3.12+** atau environment **Miniconda/Anaconda**.

1. Clone repositori ini ke komputer Anda:
   ```bash
   git clone https://github.com/hikalputra12/Eksperimen_SML_Julianda-Putra-Mansur.git
   cd Eksperimen_SML_Julianda-Putra-Mansur
   ```

2. Instal dependensi library yang dibutuhkan:
   ```bash
   pip install -r Membangun_model/requirements.txt
   ```
   *(Library utama: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `mlflow`, `dagshub`, `scikit-optimize`)*

---

## 🔄 1. Otomatisasi Preprocessing Data (Kriteria 1)

Proses pembersihan data mentah dijalankan secara otomatis melalui script [automate_Julianda-Putra-Mansur.py](file:///c:/Users/Julianda/Eksperimen_SML_Julianda-Putra-Mansur/Eksperimen_SML_Julianda-Putra-Mansur/preprocessing/automate_Julianda-Putra-Mansur.py). 

### Fitur Preprocessing:
* Penanganan *missing values* secara otomatis (mengisi kolom numerik dengan median, kategorikal dengan modus).
* Penghapusan kolom tidak prediktif (`loan_id`) dan baris duplikat.
* Encoding variabel kategorikal menggunakan `LabelEncoder` dan pemetaan target biner (`loan_status`: `Approved` -> 1, `Rejected` -> 0).
* Standarisasi skala fitur numerik menggunakan `StandardScaler`.

### Otomatisasi CI/CD (GitHub Actions):
Pipeline diatur di [.github/workflows/preprocessing.yml](file:///c:/Users/Julianda/Eksperimen_SML_Julianda-Putra-Mansur/.github/workflows/preprocessing.yml). Ketika ada `push` kode baru ke branch `main`/`master`, server GitHub Actions secara otomatis akan:
1. Menyiapkan environment Python.
2. Mengeksekusi script pembersihan data.
3. Melakukan commit dan push balik file `dataset_clean.csv` terbaru ke repositori jika terdapat perubahan isi data (dan mencegah loop pemicu menggunakan flag `[skip ci]`).

---

## 🧪 2. Pemodelan & Tracking Eksperimen (Kriteria 2)

Proyek ini mencakup dua tahap pemodelan yang mencatat parameter, metrik, artefak gambar, serta registrasi model secara otomatis dan manual.

### Integrasi Dual-Tracking (DagsHub & Localhost):
Kedua script pemodelan diprogram untuk mengirimkan log eksperimen ke dua server sekaligus:
1. **DagsHub Remote**: Untuk pelacakan online secara kolaboratif.
2. **Localhost Server (`http://127.0.0.1:5000`)**: Untuk pelacakan offline berkecepatan tinggi di komputer lokal.

### A. Model Baseline (`modelling.py`)
Script [modelling.py](file:///c:/Users/Julianda/Eksperimen_SML_Julianda-Putra-Mansur/Membangun_model/modelling.py) melatih Random Forest Classifier standar dengan parameter bawaan (`n_estimators=100`, `max_depth=None`).
* **Automatic Logging**: Memanggil `mlflow.autolog()` untuk merekam metrik bawaan scikit-learn.
* **Manual Logging**: Merekam parameter inisialisasi, model biner sklearn (`baseline_model`), metrik akurasi (`accuracy`, `precision`, `recall`, `f1_score`), serta 2 artefak gambar tambahan (`confusion_matrix.png` & `feature_importance.png`).

### B. Hyperparameter Tuning (`modelling_tuning.py`)
Script [modelling_tuning.py](file:///c:/Users/Julianda/Eksperimen_SML_Julianda-Putra-Mansur/Membangun_model/modelling_tuning.py) mencari kombinasi parameter terbaik menggunakan **Bayesian Optimization** (`BayesSearchCV` dari `scikit-optimize`).
* **Autologging Safe Configuration**: Menggunakan `mlflow.sklearn.autolog(max_tuning_runs=0, log_post_training_metrics=False)` guna mencegah penimbunan data bertipe kompleks dan menghindari error *concurrent-lock* SQLite (HTTP 500) di server lokal.
* **Manual Logging**: Merekam kombinasi parameter terbaik (`best_n_estimators`, `best_max_depth`, `best_min_samples_split`), model terbaik (`model`), visualisasi pohon keputusan (`estimator.html`), file konfigurasi metrik (`metric_info.json`), plot confusion matrix, serta grafik feature importance.

---

## 🏃 Cara Menjalankan Eksekusi Secara Lokal

Ikuti langkah-langkah berikut untuk melatih model dan mencatat eksperimen Anda secara lokal:

1. **Jalankan MLflow Tracking Server Lokal** dari dalam direktori `Membangun_model`:
   ```bash
   cd Membangun_model
   mlflow server --host 127.0.0.1 --port 5000
   ```
   *(Biarkan terminal ini tetap terbuka dan berjalan di latar belakang)*

2. **Jalankan Pemodelan Baseline** di jendela/tab terminal baru:
   ```bash
   python modelling.py
   ```

3. **Jalankan Hyperparameter Tuning**:
   ```bash
   python modelling_tuning.py
   ```

4. **Visualisasikan Hasil Eksperimen**:
   * Akses Dashboard MLflow Lokal Anda di [http://127.0.0.1:5000](http://127.0.0.1:5000)
   * Akses Dashboard Repositori Online Anda di [DagsHub MLflow Server](https://dagshub.com/hikalputra12/Eksperimen_SML_Julianda-Putra-Mansur.mlflow)