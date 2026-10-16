# 👁️ Modul 05: Applied Computer Vision & Transfer Learning

Selamat datang di Modul 05! Di modul ini kita melompat ke domain persepsi visual: bagaimana komputer memahami piksel, mendeteksi tekstur, dan mengenali objek di dunia nyata.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami mengapa Multi-Layer Perceptron (MLP) biasa gagal menangani data gambar (*spatial collapse* & *parameter explosion*).
2. Memahami mekanisme dasar **Convolution, Stride, Padding, Pooling**, dan hierarki pembentukan fitur.
3. Memahami konsep **Residual Connection ($y = F(x) + x$)** pada ResNet yang memecahkan masalah degradasi layer yang dalam.
4. Menguasai teknik **Transfer Learning & Fine-Tuning**: Memanfaatkan model yang telah dilatih pada jutaan gambar ImageNet untuk menyelesaikan tugas visual kustom kita hanya dalam beberapa menit.
5. Membangun pipeline **Data Augmentation** menggunakan `torchvision.transforms`.
6. Menyelesaikan **Real Project**: Custom Image Classifier dengan pre-trained ResNet18.

---

## 🧠 Konsep & Mental Model

### 1. Mengapa Gambar Membutuhkan Konvolusi (Bukan MLP)?
Sebuah gambar berwarna berukuran $224 \times 224$ piksel memiliki $224 \times 224 \times 3 = 150.528$ angka.
* Jika di-flatten ke MLP dengan 512 neuron tersembunyi, layer pertama saja butuh **77 juta parameter bobot!** Ini menyebabkan overfitting instan dan memakan memori komputasi secara masif.
* Lebih fatal lagi: flattening **menghancurkan hubungan spasial 2D** (piksel tetangga atas-bawah-kiri-kanan hilang maknanya).

**Solusi: Convolutional Neural Network (CNN)**
Konvolusi menggunakan filter/kernel kecil (misal $3 \times 3$) yang digeser (*sliding window*) melintasi seluruh gambar.
* **Parameter Sharing**: Filter yang sama dipakai di seluruh gambar.
* **Translation Invariance**: Kucing di pojok kiri atas dikenali dengan fitur yang sama seperti kucing di pojok kanan bawah.

```text
Input Image (5x5)          Kernel (3x3)           Feature Map (3x3)
[ 1  0  1  0  2 ]         [ 1  0 -1 ]
[ 0  3  2  1  0 ]    *    [ 1  0 -1 ]    ===>    [ 6   3  -2 ]
[ 2  1  0  2  1 ]         [ 1  0 -1 ]            [ .   .   . ]
```

Formula ukuran output setelah konvolusi:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
* $W$: Ukuran input
* $K$: Ukuran Kernel ($3$)
* $P$: Padding (menambah border nol agar ukuran tidak menyusut)
* $S$: Stride (langkah pergeseran kernel)

---

### 2. Hierarki Pembelajaran Visual
CNN belajar secara berjenjang dari abstrak rendah ke konkret tinggi:
* **Layer 1–2**: Mendeteksi garis tepi (*edges*), gradasi warna, dan sudut.
* **Layer 3–4**: Menggabungkan garis menjadi tekstur (garis belang, lingkaran, kotak).
* **Layer 5 ke atas**: Mengenali bagian objek (mata, roda, daun, moncong anjing).
* **Classifier Head**: Menghubungkan bagian objek ke label akhir (Mobil, Kucing, Pesawat).

---

### 3. Transfer Learning: Standar Emas Industri

> 💡 **Prinsip IT Engineer**: Jangan pernah melatih model Computer Vision dari nol (*from scratch*) kecuali Anda punya 1 juta gambar dan ribuan jam GPU.

Sebuah model seperti **ResNet18 / ResNet50** telah dilatih oleh peneliti di dataset ImageNet (1,2 juta gambar, 1000 kelas). Model ini sudah sangat mahir mengekstrak fitur visual dari dunia nyata.

Kita cukup melakukan **Transfer Learning**:
1. Ambil model ResNet pre-trained (Backbone).
2. **Bekukan bobot backbone** (`param.requires_grad = False`) agar fitur visual dasarnya tidak rusak.
3. Ganti layer klasifikasi terakhir (`model.fc`) dengan layer baru yang sesuai dengan jumlah kelas kustom kita (misal: 2 kelas: Sehat vs Sakit).
4. Latih hanya layer terakhir tersebut. Hasilnya: Model mencapai akurasi tinggi hanya dengan puluhan/ratusan data dan dalam hitungan menit!

```text
[ Pretrained Backbone (ResNet) ] ---> [ Custom Head (Linear -> N Classes) ]
       (BEKUKAN / FROZEN)                          (DILATIH)
```

---

## 🧪 Hands-on Lab & Real Project

Buka dan jalankan:
👉 **[`05_computer_vision.ipynb`](./05_computer_vision.ipynb)**

Alur hands-on:
1. **Visualizing Convolutions**: Mengaplikasikan kernel Sobel (Edge Detection) secara manual ke gambar untuk melihat bagaimana filter mengekstrak garis tepi.
2. **Data Augmentation Pipeline**: Membuat transformasi augmentasi (rotasi acak, flip horizontal, normalisasi warna ImageNet).
3. **The Transfer Learning Recipe**: Me-load `torchvision.models.resnet18(weights="DEFAULT")`, membekukan backbone, dan memodifikasi `model.fc`.
4. **End-to-End Training**: Melatih image classifier pada dataset visual dan mengevaluasi confusion matrix performanya.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa parameter sharing pada operasi konvolusi jauh lebih efisien dibanding Dense/Linear layer pada gambar?*
2. *Apa fungsi Residual Connection ($y = F(x) + x$) pada ResNet, dan masalah apa yang dipecahkannya?*
3. *Mengapa kita wajib melakukan normalisasi dengan rata-rata dan standar deviasi ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) saat menggunakan model pre-trained?*
