# 📊 Machine Learning Progress Tracker

Gunakan file ini sebagai papan kendali (*learning dashboard*) pribadi Anda untuk track **Machine Learning**. Setiap kali Anda menyelesaikan konsep teori, hands-on, atau tantangan pada suatu modul, centang kotak statusnya (`[x]`) dan tulis refleksi singkat di kolom catatan.

---

## 📈 Dashboard Kemajuan ML

* **Total Modul**: 6 Modul
* **Status Keseluruhan**: `0 / 6 Selesai`
* **Metode**: *Inverted Engineering & Sabotage Lab*

---

## 📋 Detail Checklist per Modul

- [ ] **Modul 01: The ML System & Data Pipeline**
  - [ ] Paham taksonomi ML: Supervised vs Unsupervised vs Self-Supervised
  - [ ] Paham siklus hidup pipeline ML end-to-end
  - [ ] Paham bahaya Data Leakage saat preprocessing dan standarisasi
  - [ ] Selesaikan Hands-on: `01_ml_workflow.ipynb`
  - [ ] Selesaikan Sabotage Lab: Data Leakage fitting scaler sebelum split
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 01
  - *Catatan / Kendala*: *...*

- [ ] **Modul 02: Classical ML Algorithms Deep Dive**
  - [ ] Paham geometri decision boundary: Logistic Regression vs KNN vs Naive Bayes
  - [ ] Paham Trade-off Bias vs Variance dan Curse of Dimensionality
  - [ ] Paham perbedaan penalti regularisasi L1 (Lasso) vs L2 (Ridge)
  - [ ] Selesaikan Hands-on: `02_classical_ml.ipynb`
  - [ ] Selesaikan Sabotage Lab: The Curse of k=1 in KNN (Extreme Overfitting)
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 02
  - *Catatan / Kendala*: *...*

- [ ] **Modul 03: Tree-Based Models & Ensemble Methods**
  - [ ] Paham cara kerja Decision Tree: Gini Impurity vs Entropy
  - [ ] Paham perbedaan Bagging (Random Forest) vs Boosting (Gradient Boosting / XGBoost)
  - [ ] Paham interpretasi Feature Importance ranking
  - [ ] Selesaikan Hands-on: `03_ensemble_methods.ipynb`
  - [ ] Selesaikan Sabotage Lab: Pohon tanpa batas (`max_depth=None`)
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 03
  - *Catatan / Kendala*: *...*

- [ ] **Modul 04: Model Evaluation & Selection**
  - [ ] Paham anatomi Confusion Matrix (TP, FP, TN, FN)
  - [ ] Paham kapan memprioritaskan Precision vs Recall vs F1-Score
  - [ ] Paham interpretasi ROC-AUC vs PR-AUC dan Stratified K-Fold
  - [ ] Selesaikan Hands-on: `04_model_evaluation.ipynb`
  - [ ] Selesaikan Sabotage Lab: Jebakan Akurasi 99.5% pada data fraud imbalanced
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 04
  - *Catatan / Kendala*: *...*

- [ ] **Modul 05: Feature Engineering & Selection**
  - [ ] Paham One-Hot Encoding vs Target Encoding vs Label Encoding
  - [ ] Paham transformasi logaritmik untuk menormalkan data skewed
  - [ ] Paham seleksi fitur menggunakan Mutual Information dan RFE
  - [ ] Selesaikan Hands-on: `05_feature_engineering.ipynb`
  - [ ] Selesaikan Sabotage Lab: Target Encoding tanpa Cross-Validation
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 05
  - *Catatan / Kendala*: *...*

- [ ] **Modul 06: Real Project - End-to-End Loan Default Prediction**
  - [ ] Bangun pipeline produksi menggunakan `ColumnTransformer` (fitur numerik & kategorikal)
  - [ ] Lakukan turnamen model (Logistic vs RF vs Gradient Boosting) dengan Stratified CV
  - [ ] Simpan pipeline utuh ke format `.joblib`
  - [ ] Bangun fungsi inferensi `evaluate_applicant()` untuk keputusan kredit nasabah baru
  - [ ] Selesaikan Hands-on: `06_real_project.ipynb`
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 06
  - *Catatan / Kendala*: *...*
