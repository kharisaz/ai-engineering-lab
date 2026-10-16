# 📊 Deep Learning Progress Tracker

Gunakan file ini sebagai papan kendali (*learning dashboard*) pribadi Anda. Setiap kali Anda menyelesaikan konsep teori, hands-on, atau tantangan pada suatu modul, centang kotak statusnya (`[x]`) dan tulis refleksi singkat di kolom catatan.

---

## 📈 Dashboard Kemajuan

* **Total Modul**: 8 Modul
* **Status Keseluruhan**: `0 / 8 Selesai`
* **Target Waktu**: Fleksibel, disarankan 1 modul per 2–4 hari.

---

## 📋 Detail Checklist per Modul

### Fase 1: Deep Learning Core & Mental Model

- [ ] **Modul 01: Network Anatomy & Representations**
  - [ ] Paham konsep linear combination ($z = Wx + b$) pada Perceptron
  - [ ] Paham alasan perlunya Multi-Layer Perceptron (mengapa 1 neuron gagal di problem non-linear)
  - [ ] Paham peran fungsi aktivasi (ReLU, LeakyReLU, GELU, Sigmoid, Softmax)
  - [ ] Selesaikan Hands-on: `01_network_anatomy.ipynb`
  - [ ] Selesaikan Sabotage Lab: Mengamati kegagalan model tanpa non-linearitas
  - [ ] Jawab Feynman Checkpoint di README modul 01
  - *Catatan / Kendala*: *...*

- [ ] **Modul 02: The Learning Engine & Optimization**
  - [ ] Paham intuisi Loss Function (Cross-Entropy vs MSE vs BCE)
  - [ ] Paham intuisi Backpropagation & Chain Rule (bagaimana error mengalir mundur)
  - [ ] Paham perbedaan Optimizer: SGD, Momentum, RMSprop, dan Adam/AdamW
  - [ ] Paham Learning Rate Schedules (Cosine Annealing & Warmup)
  - [ ] Selesaikan Hands-on: `02_learning_engine.ipynb`
  - [ ] Selesaikan Sabotage Lab: Meledakkan learning rate vs learning rate terlalu kecil
  - [ ] Jawab Feynman Checkpoint di README modul 02
  - *Catatan / Kendala*: *...*

- [ ] **Modul 03: Generalization, Tuning & Debugging**
  - [ ] Paham diagnosis Overfitting vs Underfitting (Train Loss vs Val Loss)
  - [ ] Paham mekanisme Regularization: Dropout & Weight Decay (L2)
  - [ ] Paham Weight Initialization (kenapa bobot awal tidak boleh nol / Symmetry Problem)
  - [ ] Kuasai The Neural Network Debugging Playbook (mengatasi NaN, vanishing gradient, shape error)
  - [ ] Selesaikan Hands-on: `03_tuning_and_debugging.ipynb`
  - [ ] Selesaikan "Break & Fix" Challenge: Memperbaiki model yang sengaja dirusak
  - *Catatan / Kendala*: *...*

- [ ] **Modul 04: The Production PyTorch Pipeline**
  - [ ] Kuasai PyTorch Core: `torch.Tensor`, `Autograd`, `nn.Module`
  - [ ] Bangun pipeline data standar: `Dataset` & `DataLoader`
  - [ ] Tulis modular train/validation loop dengan Early Stopping & Checkpointing
  - [ ] Selesaikan Real Project: Tabular Churn / Fraud Detection Pipeline
  - *Catatan / Kendala*: *...*

---

### Fase 2: Applied Domains (Real-World Projects)

- [ ] **Modul 05: Applied Computer Vision & Transfer Learning**
  - [ ] Paham operasi Konvolusi, Stride, Padding, Pooling, dan Feature Maps
  - [ ] Paham arsitektur modern (ResNet, ConvNeXt) dan konsep Residual Connection
  - [ ] Kuasai strategi Transfer Learning & Fine-tuning (freeze vs unfreeze layers)
  - [ ] Selesaikan Hands-on: `05_computer_vision.ipynb`
  - [ ] Selesaikan Real Project: Custom Image Classifier (Plant Disease / Medical / Component)
  - *Catatan / Kendala*: *...*

- [ ] **Modul 06: Applied NLP & Modern Transformers**
  - [ ] Paham Tokenization (BPE, WordPiece) & Word Embeddings
  - [ ] Paham prinsip Self-Attention pada Transformers
  - [ ] Kuasai ekosistem Hugging Face (`AutoTokenizer`, `AutoModel`, `Trainer` API)
  - [ ] Selesaikan Hands-on: `06_nlp_transformers.ipynb`
  - [ ] Selesaikan Real Project: Sentiment Analysis & Customer Feedback Tagger
  - *Catatan / Kendala*: *...*

- [ ] **Modul 07: Practical Vision Systems (Object Detection)**
  - [ ] Paham konsep Bounding Box, IoU (Intersection over Union), dan metrik mAP
  - [ ] Paham pipeline Single-Stage Detector (YOLO)
  - [ ] Selesaikan Hands-on: Training YOLO pada format dataset custom
  - [ ] Selesaikan Real Project: Object / Safety Equipment Detector
  - *Catatan / Kendala*: *...*

---

### Fase 3: IT Deployment & Production (MLOps)

- [ ] **Modul 08: Model Serving & Production API Microservice**
  - [ ] Paham siklus hidup model production (Research $\rightarrow$ Serialization $\rightarrow$ Serving)
  - [ ] Export model PyTorch ke format ONNX
  - [ ] Bangun REST API inferensi performa tinggi menggunakan FastAPI & Pydantic
  - [ ] Selesaikan Real Project: Live Containerized AI REST API
  - *Catatan / Kendala*: *...*

---

## 💡 Tips Penggunaan
1. Setiap kali mempelajari modul, buka `README.md` pada modul tersebut untuk memahami **konsep mental model** dan **Feynman Checkpoint**.
2. Jalankan notebook `.ipynb` untuk melakukan verifikasi empiris dan menyelesaikan **Sabotage Lab**.
3. Update status `[ ]` menjadi `[x]` di file ini setelah menyelesaikan modul.
