# 🧪 AI & Data Science Engineering Lab

Selamat datang di **AI & Data Science Engineering Lab**. Repositori ini dirancang khusus untuk pembelajaran terstruktur, mendalam, dan berbasis proyek nyata (*applied engineering*) untuk mahasiswa Teknologi Informasi.

Setiap track dipisahkan secara modular agar materi tetap terorganisir dan mudah di-track seiring bertambahnya materi baru.

---

## 📂 Struktur Repositori Lab

```text
lab/
├── .gitignore
├── README.md                      <-- Portal Utama Lab
│
├── deep-learning/                 <-- [ACTIVE] Deep Learning Mastery Track
│   ├── README.md                  # Roadmap & panduan track Deep Learning
│   ├── TRACKER.md                 # Progress tracker interaktif per modul
│   ├── modules/                   # Modul belajar (Konsep + Hands-on)
│   ├── src/                       # Reusable Python modules & architectures
│   └── shared/                    # Utility scripts (visualizers, metrics)
│
├── machine-learning/              <-- [PLANNED] Classical ML & Tabular Modeling
│   └── .gitkeep
│
└── computer-vision/               <-- [PLANNED] Advanced CV Systems (YOLO, Seg, 3D)
    └── .gitkeep
```

---

## 🚀 Learning Tracks Status

| Track | Fokus Pembelajaran | Status | Direktori |
| :--- | :--- | :---: | :--- |
| **Deep Learning** | Mental model, PyTorch, CNN, Transformers, MLOps API | 🟢 Active | [`deep-learning/`](./deep-learning/README.md) |
| **Machine Learning** | Supervised/Unsupervised, Feature Eng, Scikit-learn, XGBoost | 🟡 Planned | `machine-learning/` |
| **Computer Vision** | Advanced Detection, Segmentation, Video Stream Processing | 🟡 Planned | `computer-vision/` |

---

## 🛠️ Environment & Prerequisites

Repositori ini terhubung dengan virtual environment lokal yang sudah terpasang di parent folder:
* **Python Path**: `../.venv/bin/python`
* **Jupyter Kernel**: `Python (Data AI .venv)`
* **Hardware Acceleration**: Apple Silicon MPS (`torch.device("mps")`)

Untuk memulai belajar Deep Learning, buka direktori [**`deep-learning/`**](./deep-learning/README.md) dan periksa progress di [**`TRACKER.md`**](./deep-learning/TRACKER.md).
