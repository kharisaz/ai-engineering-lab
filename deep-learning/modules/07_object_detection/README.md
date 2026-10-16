# 🎯 Modul 07: Practical Vision Systems (Object Detection)

Selamat datang di Modul 07! Klasifikasi gambar (Modul 05) hanya memberi tahu kita *apa objek di dalam gambar*, tetapi tidak memberi tahu *di mana letaknya*.

Di industri nyata—seperti sistem mobil otonom (*self-driving cars*), pengawasan CCTV keamanan, atau inspeksi manufaktur pabrik—kita butuh **Object Detection**: menemukan koordinat kotak pembatas (*bounding box*) dari banyak objek sekaligus dalam satu frame gambar atau video secara *real-time*.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami perbedaan mendasar: **Classification vs Object Detection vs Instance Segmentation**.
2. Memahami format koordinat bounding box: **Pascal VOC** ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$) vs **YOLO** ($x_{\text{center}}, y_{\text{center}}, w, h$).
3. Memahami konsep dan perhitungan **Intersection over Union (IoU)**.
4. Memahami algoritma **Non-Maximum Suppression (NMS)** untuk mengeliminasi kotak duplikat yang tumpang tindih.
5. Memahami cara kerja arsitektur **Single-Stage Detector (YOLO)** dan metrik evaluasi industri: **Mean Average Precision (mAP)**.
6. Menyelesaikan **Hands-on**: Menjalankan detektor objek multi-kelas secara langsung pada gambar dan memvisualisasikan bounding box beranotasi.

---

## 🧠 Konsep & Mental Model

### 1. Spektrum Visi Komputer

```text
[ Classification ]         [ Object Detection ]          [ Segmentation ]
"Ada kucing di gambar"   "Kucing ada di kotak [x,y,w,h]"  "Tiap piksel bertanda kucing"
(1 Label Global)         (Multiple Bounding Boxes)       (Pixel Mask)
```

Untuk mendeteksi objek, neural network harus memprediksi 2 hal sekaligus:
1. **Regresi Koordinat**: Memprediksi 4 angka koordinat bounding box:
   $$[x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$$
2. **Klasifikasi Kelas**: Memprediksi probabilitas objek di dalam kotak tersebut (misal: Mobil, Pejalan Kaki, Sepeda).

---

### 2. Intersection over Union (IoU): Mengukur Ketepatan Kotak
Bagaimana kita tahu tebakan kotak model itu presisi? Kita bandingkan kotak prediksi dengan kotak label asli (*ground truth*):

$$\text{IoU} = \frac{\text{Luas Irisan (Intersection)}}{\text{Luas Gabungan (Union)}}$$

```text
       ┌───────────┐
       │ Ground    │
       │   Truth   │
   ┌───┼───────┐   │       Irisan (Intersection)
   │   │///////│   │  ===> ---------------------
   │   └───────┼───┘        Gabungan (Union)
   │ Prediksi  │
   └───────────┘
```

* $\text{IoU} = 1.0$: Kotak prediksi menempel sempurna 100% pada target.
* $\text{IoU} \ge 0.5$: Standar ambang batas (*threshold*) deteksi dianggap valid/berhasil.

---

### 3. Non-Maximum Suppression (NMS): Membersihkan Kotak Duplikat
Model deteksi objek sering mengeluarkan puluhan kotak pembatas untuk satu objek yang sama.

**Algoritma NMS**:
1. Urutkan semua kotak berdasarkan skor keyakinan (*confidence score*) tertinggi.
2. Ambil kotak dengan skor tertinggi sebagai pemenang.
3. Hapus semua kotak lain yang memiliki overlap ($\text{IoU} > 0.5$) tinggi dengan kotak pemenang tersebut.
4. Ulangi proses ini untuk kotak yang tersisa.
*Hasil*: Hanya tersisa 1 kotak paling rapi untuk setiap objek unik!

---

### 4. Dua Aliran Detektor: Two-Stage vs Single-Stage

| Kategori | Contoh Model | Cara Kerja | Kecepatan vs Akurasi |
| :--- | :--- | :--- | :--- |
| **Two-Stage** | Faster R-CNN | Tahap 1: Region Proposal Network (mencari kandidat kotak). Tahap 2: Mengklasifikasikan isi tiap kotak. | Akurasi sangat tinggi, tetapi lambat (5–15 FPS). Cocok untuk analisis medis. |
| **Single-Stage** | **YOLO (You Only Look Once)**, SSD | Gambar dibagi menjadi grid; prediksi kotak dan kelas dilakukan sekaligus dalam 1 kali forward pass! | **Sangat cepat (30–120 FPS)**. Standar mutlak untuk aplikasi real-time / video streaming. |

---

## 🧪 Hands-on Lab & Real Project

Buka dan jalankan:
👉 **[`07_object_detection.ipynb`](./07_object_detection.ipynb)**

Alur hands-on:
1. **IoU & NMS from Scratch**: Menulis dan memvisualisasikan bagaimana IoU mengukur overlap dan bagaimana NMS menyaring ratusan kotak duplikat.
2. **Running Pretrained Object Detector**: Menggunakan model deteksi objek modern (`torchvision.models.detection.ssdlite320_mobilenet_v3_large`) yang dioptimasi untuk inferensi cepat di CPU/MPS.
3. **Bounding Box Visualizer**: Menggambar bounding box berwarna, nama kelas, dan persentase keyakinan langsung pada gambar foto nyata.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa perhitungan Intersection over Union (IoU) lebih adil untuk mengevaluasi bounding box dibanding menghitung selisih koordinat biasa?*
2. *Jelaskan logika di balik algoritma Non-Maximum Suppression (NMS) dan apa akibatnya jika NMS tidak diaktifkan pada sistem deteksi objek!*
3. *Apa keunggulan mendasar arsitektur YOLO (Single-Stage) dibanding Faster R-CNN (Two-Stage) ketika di-deploy pada perangkat edge seperti kamera pengawas atau robot?*
