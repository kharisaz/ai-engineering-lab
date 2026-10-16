# 📊 Computer Vision Progress Tracker

Gunakan file ini sebagai papan kendali (*learning dashboard*) pribadi Anda untuk track **Computer Vision**. Setiap kali Anda menyelesaikan konsep teori, hands-on, atau tantangan pada suatu modul, centang kotak statusnya (`[x]`) dan tulis refleksi singkat di kolom catatan.

---

## 📈 Dashboard Kemajuan CV

* **Total Modul**: 6 Modul
* **Status Keseluruhan**: `0 / 6 Selesai`
* **Metode**: *Inverted Engineering & Sabotage Lab*

---

## 📋 Detail Checklist per Modul

- [ ] **Modul 01: Image Fundamentals & Representation**
  - [ ] Paham representasi gambar dalam memori sebagai tensor `[C, H, W]`
  - [ ] Paham alasan matematis normalisasi ImageNet (`mean` & `std`)
  - [ ] Selesaikan Hands-on: `01_image_fundamentals.ipynb`
  - [ ] Selesaikan Sabotage Lab: Memprediksi gambar tanpa normalisasi ImageNet
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 01
  - *Catatan / Kendala*: *...*

- [ ] **Modul 02: Feature Extraction & Classical Computer Vision**
  - [ ] Paham operasi konvolusi spasial 2D manual dengan Kernel Sobel ($G_x, G_y$)
  - [ ] Paham mengapa Gaussian Blur mutlak diperlukan sebelum deteksi tepi Canny
  - [ ] Paham operasi Morfologi: Erosi & Dilasi
  - [ ] Selesaikan Hands-on: `02_classical_cv.ipynb`
  - [ ] Selesaikan Sabotage Lab: Deteksi tepi pada citra ber-noise tanpa vs dengan blur
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 02
  - *Catatan / Kendala*: *...*

- [ ] **Modul 03: CNN Architecture Deep Dive**
  - [ ] Paham hierarki fitur konvolusi (Edges $\rightarrow$ Textures $\rightarrow$ Objects)
  - [ ] Paham Degradation Problem pada jaringan dalam dan solusi Skip Connection ($y = F(x) + x$)
  - [ ] Paham efisiensi Depthwise Separable Convolution pada MobileNet
  - [ ] Selesaikan Hands-on: `03_cnn_architectures.ipynb`
  - [ ] Selesaikan Sabotage Lab: Plain Net 10-layer vs ResNet 10-layer
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 03
  - *Catatan / Kendala*: *...*

- [ ] **Modul 04: Advanced Transfer Learning & Fine-Tuning**
  - [ ] Paham 4 Strategi Fine-Tuning (Feature Extractor, Top Layers, Full Tuning, Progressive)
  - [ ] Paham bahaya *Catastrophic Forgetting* dan solusi *Discriminative Learning Rates*
  - [ ] Selesaikan Hands-on: `04_advanced_transfer_learning.ipynb`
  - [ ] Selesaikan Sabotage Lab: Uniform High LR vs Discriminative LR
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 04
  - *Catatan / Kendala*: *...*

- [ ] **Modul 05: Image Segmentation & The U-Net Architecture**
  - [ ] Paham arsitektur U-Net (Encoder, Bottleneck, Decoder dengan Skip Connections)
  - [ ] Paham kelemahan Cross-Entropy pada data tidak seimbang dan keunggulan Dice Loss
  - [ ] Selesaikan Hands-on: `05_segmentation.ipynb`
  - [ ] Selesaikan Sabotage Lab: BCE vs Dice Loss pada segmentasi objek tipis/kecil
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 05
  - *Catatan / Kendala*: *...*

- [ ] **Modul 06: Production Computer Vision System**
  - [ ] Bangun pipeline klasifikasi citra end-to-end
  - [ ] Export model ke TorchScript (`torch.jit.trace`) untuk eksekusi C++ tanpa Python
  - [ ] Ukur benchmark latency milidetik dan throughput FPS
  - [ ] Selesaikan Hands-on: `06_real_project.ipynb`
  - [ ] Jawab 3 Pertanyaan Feynman Checkpoint di README modul 06
  - *Catatan / Kendala*: *...*
