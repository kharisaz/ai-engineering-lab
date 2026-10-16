# ⚙️ Modul 01: The ML System & Data Pipeline

Selamat datang di Modul 01 track Machine Learning! Modul ini membahas fondasi arsitektur sistem pembelajaran mesin: bagaimana data mentah ditransformasikan menjadi input model secara bersih, tanpa kebocoran data (*data leakage*).

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami taksonomi pembelajaran mesin: **Supervised, Unsupervised, dan Self-Supervised Learning**.
2. Memahami siklus hidup pipeline ML: Data Ingestion -> Cleaning -> Feature Preprocessing -> Modeling -> Evaluation.
3. Memahami bahaya laten **Data Leakage (Kebocoran Data)** dan mengapa ia menjadi penyebab nomor 1 kegagalan model saat dideploy ke produksi.
4. Mampu merancang pipeline Scikit-Learn (`sklearn.pipeline.Pipeline`) yang aman dan modular.
5. Menyelesaikan **The Sabotage Lab**: Membuktikan bagaimana fitting scaler sebelum train-test split memberikan metrik palsu yang menyesatkan.

---

## 🧠 Konsep & Mental Model

### 1. Taksonomi Machine Learning
```text
┌─────────────────────────────────────────────────────────────┐
│                      MACHINE LEARNING                       │
├──────────────────────────────┬──────────────────────────────┤
│ 1. SUPERVISED (Terawasi)     │ 2. UNSUPERVISED (Tak Terawasi)│
│    Data memiliki Label (y)   │    Hanya ada Fitur (X), No y │
│    - Klasifikasi (Kategori)  │    - Clustering (K-Means)    │
│    - Regresi (Angka Kontinu) │    - Dimensionality (PCA)    │
└──────────────────────────────┴──────────────────────────────┘
```

---

### 2. The Silent Killer: Data Leakage
> ⚠️ **Data Leakage** adalah kondisi di mana informasi dari data masa depan (Test/Validation Set) secara tidak sengaja bocor ke dalam data pelatihan (Training Set).

Contoh klasik:
* Melakukan `StandardScaler.fit(X)` pada **seluruh dataset** sebelum membagi data menjadi Train dan Test.
* Saat melakukan `fit()`, nilai rata-rata dan standar deviasi dari Test Set ikut terhitung. Model secara tidak sadar sudah mengintip data pengujian.
* **Akibat di Produksi**: Akurasi di laptop terlihat fantastis (98%), tetapi begitu di-deploy ke server untuk melayani data pelanggan baru, performa hancur berantakan (50%).

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`01_ml_workflow.ipynb`](./01_ml_workflow.ipynb)**

Alur hands-on:
1. **The Working Blackbox**: Membangun end-to-end classification pipeline dengan Scikit-Learn dalam 5 menit.
2. **Anatomical Dissection**: Memeriksa langkah-langkah preprocessing dan transformasi data.
3. **The Sabotage Lab**: Membuktikan efek Data Leakage pada estimasi performa.
4. **The Clean Build**: Mengemas pipeline menggunakan `sklearn.pipeline.Pipeline` dan `ColumnTransformer`.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa kita hanya boleh memanggil `.fit()` atau `.fit_transform()` pada data Train, dan hanya memanggil `.transform()` pada data Test/Validation?*
2. *Jelaskan perbedaan antara Train Set, Validation Set, dan Test Set beserta fungsinya masing-masing!*
3. *Apa risiko bisnis jika seorang data scientist tidak menyadari adanya Data Leakage sebelum model dideploy ke sistem perbankan?*
