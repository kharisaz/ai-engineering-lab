# 🛠️ Modul 05: Feature Engineering & Selection

Selamat datang di Modul 05! *"Better data beats fancier algorithms"*. Algoritma tercanggih tidak akan mampu belajar jika fitur yang diberikan berkualitas buruk atau tidak relevan.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Menguasai teknik encoding data kategorikal: **One-Hot Encoding, Label Encoding, dan Target Encoding**.
2. Memahami transformasi numerik (Log Transform, Box-Cox) untuk menormalkan distribusi skewed.
3. Membuat **Interaction & Polynomial Features** untuk menangkap hubungan non-linear.
4. Mampu melakukan **Feature Selection** (Mutual Information, Correlation Filtering, RFE) untuk memangkas dimensi fitur tanpa kehilangan akurasi.
5. Menyelesaikan **The Sabotage Lab**: Membuktikan Target Encoding tanpa Cross-Validation yang menyebabkan kebocoran label fatal.

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`05_feature_engineering.ipynb`](./05_feature_engineering.ipynb)**

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Kapan kita harus menggunakan One-Hot Encoding vs Target Encoding pada fitur berkardinalitas tinggi (misal: 1000 kota)?*
2. *Mengapa transformasi logaritmik `np.log1p(x)` membantu model linier ketika fiturnya memiliki distribusi right-skewed (ekor panjang)?*
3. *Bagaimana metode Recursive Feature Elimination (RFE) bekerja dalam menyeleksi subset fitur paling penting?*
