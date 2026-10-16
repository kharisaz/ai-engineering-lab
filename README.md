# 🧪 AI & Data Science Engineering Lab

Selamat datang di **AI & Data Science Engineering Lab**. Repositori ini dirancang khusus untuk pembelajaran terstruktur, mendalam, dan berbasis proyek nyata (*applied engineering*) untuk mahasiswa Teknologi Informasi.

Seluruh materi di setiap track mengadopsi metodologi **Inverted Engineering & Sabotage Lab**:
1. **The Working Blackbox**: Melihat sistem berjalan end-to-end di 5 menit pertama.
2. **Anatomical Dissection**: Membedah dimensi data, parameter, dan alur komputasi.
3. **The Sabotage Lab ("Break & Fix")**: Sengaja merusak komponen untuk melihat gejalanya dan melatih insting debugging.
4. **The Clean Build & Feynman Test**: Mengemas solusi modular dan menguji pemahaman konsep mandiri.

---

## 📂 Struktur Repositori Lab

```text
Playground/Data AI/lab/
├── .gitignore
├── README.md                      <-- Portal Utama Lab
│
├── deep-learning/                 <-- [ACTIVE] 8 Modul Deep Learning Mastery Track
│   ├── README.md                  # Roadmap & panduan track Deep Learning
│   ├── TRACKER.md                 # Papan kendali progress belajar DL
│   ├── shared/                    # Visualizer decision boundary & tensor helpers
│   └── modules/                   # Modul 01 - 08 (Anatomy, Engine, Tuning, PyTorch, CV, NLP, Detection, MLOps)
│
├── machine-learning/              <-- [ACTIVE] 6 Modul Classical Machine Learning Track
│   ├── README.md                  # Roadmap & panduan track Machine Learning
│   ├── TRACKER.md                 # Papan kendali progress belajar ML
│   ├── shared/                    # Plotter Confusion Matrix & ROC curves
│   └── modules/                   # Modul 01 - 06 (Workflow, Classical, Ensembles, Eval, Feature Eng, Credit Risk)
│
└── computer-vision/               <-- [ACTIVE] 6 Modul Computer Vision Track
    ├── README.md                  # Roadmap & panduan track Computer Vision
    ├── TRACKER.md                 # Papan kendali progress belajar CV
    ├── shared/                    # CV utils (Tensor to image, feature maps, masks)
    └── modules/                   # Modul 01 - 06 (Fundamentals, Classical CV, Architectures, Fine-Tuning, Seg, Production)
```

---

## 🚀 Learning Tracks Status

| Track | Fokus Pembelajaran | Jumlah Modul | Status | Link Track |
| :--- | :--- | :---: | :---: | :--- |
| **Deep Learning** | Mental model, PyTorch, CNN, Transformers, MLOps API | 8 Modul | 🟢 Active | [Track Deep Learning](./deep-learning/README.md) \| [Tracker](./deep-learning/TRACKER.md) |
| **Machine Learning** | Supervised/Unsupervised, Pipeline, XGBoost, Evaluation, Feature Eng | 6 Modul | 🟢 Active | [Track Machine Learning](./machine-learning/README.md) \| [Tracker](./machine-learning/TRACKER.md) |
| **Computer Vision** | Image Tensors, Sobel/Canny, ResNet, Fine-Tuning, U-Net Segmentation | 6 Modul | 🟢 Active | [Track Computer Vision](./computer-vision/README.md) \| [Tracker](./computer-vision/TRACKER.md) |

---

## 🛠️ Environment & Prerequisites

Repositori ini terhubung langsung dengan virtual environment lokal yang sudah terpasang di parent folder:
* **Python Path**: `../.venv/bin/python`
* **Jupyter Kernel**: `Python (Data AI .venv)` (telah terdaftar di sistem)
* **Hardware Acceleration**: Apple Silicon MPS (`torch.device("mps")`) aktif

Pilih track yang ingin Anda pelajari, buka filenya di VS Code / Jupyter Lab, dan pantau kemajuan belajar Anda di masing-masing file `TRACKER.md`!
