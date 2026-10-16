# 🎯 Modul 02: Classical ML Algorithms Deep Dive

Selamat datang di Modul 02! Di sini kita membedah algoritma klasik fundamental: **Linear & Logistic Regression, K-Nearest Neighbors (KNN), dan Naive Bayes**.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami geometri batas keputusan (*Decision Boundary*) dari Logistic Regression vs KNN.
2. Memahami trade-off Bias vs Variance pada algoritma berbasis jarak (KNN).
3. Memahami mengapa Naive Bayes bekerja luar biasa cepat meskipun asumsi independensi bersyaratnya jarang terpenuhi di dunia nyata.
4. Menyelesaikan **The Sabotage Lab**: Membuktikan bagaimana KNN dengan k=1 menyebabkan overfit ekstrem dan batas keputusan fractal.

---

## 🧠 Konsep & Mental Model

### Perbandingan 4 Algoritma Klasik

| Algoritma | Tipe Model | Decision Boundary | Sensitif terhadap Skala? | Keunggulan Utama |
| :--- | :--- | :--- | :---: | :--- |
| **Logistic Regression** | Parametrik | Garis lurus / Hyperplane linier | Ya | Cepat, menghasilkan probabilitas terkalibrasi, interpretasi bobot mudah. |
| **Ridge / Lasso** | Regresi Ter-regularisasi | Linier dengan penalti L1/L2 | Ya | Mencegah multikolinearitas dan melakukan seleksi fitur otomatis (Lasso). |
| **KNN (K-Nearest Neighbors)** | Non-parametrik | Sangat fleksibel / Non-linier | **Sangat Sensitif** | Tidak ada asumsi distribusi data, sangat intuitif. |
| **Naive Bayes** | Probabilistik Generatif | Kuadratis / Linier | Tidak terlalu | Sangat cepat untuk klasifikasi teks dan data berdimensi tinggi. |

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`02_classical_ml.ipynb`](./02_classical_ml.ipynb)**

Alur hands-on:
1. **The Working Blackbox**: Membandingkan 4 model secara berdampingan pada dataset 2D yang sama.
2. **The Sabotage Lab**: Eksperimen k=1 (Overfitting parah) vs k=15 (Generalisasi optimal).
3. **Visualisasi Decision Boundary**: Mengamati bentuk garis pemisah masing-masing algoritma.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa KNN sangat sensitif terhadap skala fitur, sedangkan Naive Bayes relatif tahan terhadap perbedaan skala?*
2. *Apa yang dimaksud dengan 'Curse of Dimensionality' saat kita menggunakan algoritma berbasis jarak seperti KNN pada data dengan 1000 fitur?*
3. *Bagaimana penalti L1 (Lasso) dapat membuat sebagian bobot fitur menjadi persis nol, sedangkan L2 (Ridge) hanya mendekati nol?*
