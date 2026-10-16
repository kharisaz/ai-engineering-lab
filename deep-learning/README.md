# 🧠 Deep Learning Mastery Track (Applied & Mental Model)

Track ini dirancang khusus bagi mahasiswa Teknologi Informasi untuk menguasai Deep Learning secara mendalam: **memahami prinsip teori dan arsitektur sistem di balik neural network tanpa terjebak rumus kalkulus manual di kertas, dan memvalidasinya langsung lewat hands-on serta real-world project**.

---

## 🎯 Metode Belajar: *Inverted Engineering & Sabotage Lab*

Setiap modul menggunakan pola 4 langkah terstruktur:
1. **The Working Blackbox**: Menjalankan sistem end-to-end yang bekerja di 5 menit pertama untuk menangkap gambaran utuh (*big picture*).
2. **Anatomical Dissection**: Membedah komponen dari sisi software/sistem (dimensi tensor, aliran gradien, fungsi aktivasi, loss function).
3. **The Sabotage Lab ("Break & Fix")**: Sengaja merusak satu bagian (misal: mematikan non-linearitas, meledakkan learning rate, inisialisasi bobot nol) untuk melihat gejalanya pada loss curve dan mempelajari cara mengatasinya.
4. **The Clean Build**: Mengemas solusi menjadi kode Python modular berstandar industri.

---

## 🗺️ Roadmap Modul

| No | Modul | Fokus Materi | Deliverable / Hands-on |
|:---:|:---|:---|:---|
| **01** | [**01. Network Anatomy**](./modules/01_network_anatomy/README.md) | Perceptron, Multi-Layer Networks, Non-Linearity, Activation Functions | Interactive Non-Linear Boundary Demo |
| **02** | [**02. The Learning Engine**](./modules/02_learning_engine/README.md) | Loss Functions, Backprop Intuition, Optimizers (SGD vs AdamW), LR Schedules | Optimizer Convergence Benchmark |
| **03** | [**03. Tuning & Debugging**](./modules/03_tuning_and_debugging/README.md) | Overfitting, Regularization (Dropout, L2), Weight Init, Debugging Playbook | "Break & Fix" Neural Net Challenge |
| **04** | [**04. Production PyTorch Pipeline**](./modules/04_pytorch_workflow/README.md) | Tensors, Autograd, Dataset, DataLoader, Modular Train/Val Loop, Checkpointing | Tabular Churn Prediction Pipeline |
| **05** | [**05. Applied Computer Vision**](./modules/05_computer_vision/README.md) | Convolutions, Pooling, Transfer Learning (ResNet, ConvNeXt), Fine-tuning | Custom Image Classifier Project |
| **06** | [**06. Applied NLP & Transformers**](./modules/06_nlp_transformers/README.md) | Tokenizers, Self-Attention, Hugging Face, BERT/RoBERTa Fine-tuning | Sentiment & Feedback Classification |
| **07** | [**07. Practical Vision Systems**](./modules/07_object_detection/README.md) | Object Detection, Bounding Boxes, IoU, mAP, YOLO Pipeline | Custom Safety/Object Detector |
| **08** | [**08. Model Serving & MLOps**](./modules/08_deployment_mlops/README.md) | ONNX Export, Quantization, FastAPI REST API Microservice, Benchmarking | Production AI REST API Endpoint |

👉 **Untuk memantau progress belajar Anda, gunakan file [`TRACKER.md`](./TRACKER.md).**
