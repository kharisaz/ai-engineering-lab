# 🌳 Modul 03: Tree-Based Models & Ensemble Methods

Selamat datang di Modul 03! Di dunia data tabular industri, **Ensemble Methods (Random Forest & XGBoost)** adalah raja yang memenangkan ribuan kompetisi Kaggle dan menjadi tulang punggung sistem produksi perbankan dan e-commerce.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami cara kerja Decision Tree: **Gini Impurity vs Information Gain (Entropy)**.
2. Memahami prinsip **Bagging (Bootstrap Aggregating)** pada Random Forest untuk memangkas variansi.
3. Memahami prinsip **Boosting (Sequential Error Correction)** pada Gradient Boosting dan XGBoost.
4. Mampu mengekstrak dan menginterpretasi **Feature Importance**.
5. Menyelesaikan **The Sabotage Lab**: Mengamati Decision Tree tanpa batas (`max_depth=None`) yang memorisasi seluruh noise data.

---

## 🧠 Konsep & Mental Model

```text
               [ SINGLE TREE ]                   [ RANDOM FOREST ]               [ GRADIENT BOOSTING ]
                 (High Variance)                   (Bagging / Paralel)             (Boosting / Sekuensial)
                       O                             O   O   O   O                   Tree 1 (Predict)
                      / \                           / \ / \ / \ / \                         |
                     O   O                         Tree 1..N (Vote Rata2)           Tree 2 (Predict Error 1)
                    / \ / \                                                            |
                   O  O O  O                                                       Tree 3 (Predict Error 2)
```

---

## 🧪 Hands-on Lab
Buka dan jalankan:
👉 **[`03_ensemble_methods.ipynb`](./03_ensemble_methods.ipynb)**

Alur hands-on:
1. **The Working Blackbox**: Membandingkan Decision Tree tunggal vs Random Forest vs Gradient Boosting.
2. **The Sabotage Lab**: Decision Tree tanpa pembatasan kedalaman (`max_depth=None`).
3. **Feature Importance Ranking**: Mengidentifikasi fitur mana yang paling menentukan prediksi.
4. **Hyperparameter Tuning**: Menggunakan `GridSearchCV` untuk mengoptimasi performa.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Apa perbedaan mendasar antara cara kerja Bagging (Random Forest) dan Boosting (Gradient Boosting/XGBoost)?*
2. *Mengapa Random Forest memilih subset fitur acak di setiap split dan bukan hanya subset baris data?*
3. *Bagaimana parameter `learning_rate` pada Gradient Boosting mengendalikan kontribusi masing-masing pohon?*
