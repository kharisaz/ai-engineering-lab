# 🏆 Modul 06: Real Project - End-to-End ML System

Selamat datang di modul puncak track Machine Learning! Di sini Anda akan membangun sistem pembelajaran mesin kelas produksi secara lengkap: **Sistem Prediksi Risiko Kredit (Loan Default Prediction System)**.

---

## 🎯 Tujuan Proyek
Membangun pipeline end-to-end yang menerima data aplikasi pinjaman nasabah (fitur numerik & kategorikal), melakukan validasi input, transformasi otomatis menggunakan `ColumnTransformer`, mengevaluasi beberapa model dengan cross-validation, dan mengemas model terbaik ke dalam fungsi inferensi yang siap dipanggil oleh sistem perbankan.

---

## 🛠️ Komponen Sistem
1. **Data Ingestion & EDA**: Memahami karakteristik fitur kredit.
2. **Preprocessing Pipeline**: 
   * Fitur Numerik: Imputasi median + `StandardScaler`.
   * Fitur Kategorikal: Imputasi modus + `OneHotEncoder(handle_unknown='ignore')`.
3. **Model Selection Tournament**: Mengadu Logistic Regression vs Random Forest vs Gradient Boosting.
4. **Final Model Packaging**: Menyimpan pipeline utuh dengan `joblib`.
5. **Production Inference Function**: Menerima JSON dictionary pelamar baru dan mengembalikan skor probabilitas gagal bayar serta status keputusan (Approve / Reject).

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`06_real_project.ipynb`](./06_real_project.ipynb)**

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa penggunaan `ColumnTransformer` di dalam Scikit-Learn Pipeline mutlak diperlukan saat menangani data campuran numerik dan kategorikal?*
2. *Bagaimana parameter `handle_unknown='ignore'` pada OneHotEncoder mencegah crash saat ada kategori baru yang belum pernah muncul di data training?*
3. *Bagaimana cara bank menentukan ambang batas keputusan (threshold) klasifikasi probabilitas default agar menyeimbangkan antara keuntungan bunga dan risiko kredit macet?*
