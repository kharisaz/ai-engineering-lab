# 📊 Modul 04: Model Evaluation & Selection

Selamat datang di Modul 04! Dalam proyek AI di dunia nyata, **Akurasi adalah Jebakan Terbesar**. Jika 99% transaksi adalah normal dan hanya 1% penipuan (fraud), model bodoh yang selalu memprediksi "Normal" akan memiliki akurasi 99%, tetapi gagal total mendeteksi satu pun kejahatan!

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami anatomi **Confusion Matrix**: True Positive, False Positive, True Negative, False Negative.
2. Memahami trade-off **Precision vs Recall** dan metrik harmonik **F1-Score**.
3. Memahami kurva **ROC-AUC** dan kapan harus beralih ke **PR-AUC** untuk data yang sangat tidak seimbang (*imbalanced*).
4. Menguasai **Stratified K-Fold Cross Validation** untuk estimasi performa yang reliabel.
5. Menyelesaikan **The Sabotage Lab**: Membuktikan model fraud dengan akurasi 99.5% yang ternyata mendeteksi 0 transaksi penipuan!

---

## 🧠 Konsep & Mental Model

### Kapan Mengutamakan Precision vs Recall?
* **Utamakan RECALL**: Biaya False Negative sangat mahal!
  * *Contoh*: Deteksi Kanker (lebih baik salah menduga sehat sebagai sakit daripada melewatkan pasien kanker).
  * *Contoh*: Deteksi Fraud Finansial.
* **Utamakan PRECISION**: Biaya False Positive sangat mengganggu!
  * *Contoh*: Filter Spam Email (jangan sampai email penting masuk ke folder Spam).
  * *Contoh*: Rekomendasi Produk YouTube.

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`04_model_evaluation.ipynb`](./04_model_evaluation.ipynb)**

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa metrik Akurasi menyesatkan pada dataset dengan ketidakseimbangan kelas 99:1?*
2. *Jelaskan perbedaan interpretasi antara kurva ROC-AUC dan PR-AUC (Precision-Recall AUC)!*
3. *Mengapa Stratified K-Fold lebih disarankan dibanding K-Fold standar pada kasus klasifikasi?*
