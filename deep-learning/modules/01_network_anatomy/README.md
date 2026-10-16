# 🧬 Modul 01: Network Anatomy & Representations

Selamat datang di modul pertama! Modul ini membangun **mental model paling mendasar** tentang bagaimana jaringan saraf (neural networks) merepresentasikan data dan memecahkan masalah non-linear.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami intuisi di balik operasi linear kombinasi $z = Wx + b$ pada Perceptron.
2. Memahami keterbatasan 1 neuron (masalah XOR) dan mengapa kita harus menumpuk layer menjadi Multi-Layer Perceptron (MLP).
3. **Memahami secara mendalam mengapa fungsi aktivasi non-linear wajib ada** (dan apa yang terjadi jika fungsi aktivasi dihilangkan).
4. Mengetahui karakteristik dan perbandingan fungsi aktivasi modern: **ReLU, LeakyReLU, GELU, Sigmoid, dan Softmax**.
5. Mampu membaca dan mendesain dimensi tensor dari input hingga output (*Shape Contract*).

---

## 🧠 Konsep & Mental Model

### 1. Anatomi Perceptron: Dari Mana Datangnya $z = Wx + b$?
Satu neuron tiruan terinspirasi dari neuron biologis: menerima sinyal masuk, menimbangnya berdasarkan kepentingan, lalu memutuskan apakah akan "menyala" (*fire*).

Secara sistem, neuron melakukan dua tahap:
1. **Linear Transformation (Akumulasi Sinyal)**:
   $$z = \sum_{i=1}^{n} w_i x_i + b = W \cdot x + b$$
   * $x$: Input feature (misal: umur, pendapatan).
   * $W$ (*Weights*): Bobot kepentingan tiap fitur. Bobot positif memperkuat sinyal, bobot negatif melemahkan.
   * $b$ (*Bias*): Nilai ambang batas fleksibel (threshold). Tanpa bias, garis pemisah selalu dipaksa melewati titik $(0,0)$.
2. **Non-Linear Activation**:
   $$a = \sigma(z)$$
   * Mentransformasikan sinyal akumulasi $z$ agar jaringan mampu memodelkan pola non-linear.

```text
Input Features          Weights
    [ x1 ]  -------\   ( w1 )
                    \
    [ x2 ]  ---------> [  Sum: z = Wx + b  ] ---> [ Activation: a = f(z) ] ---> Output
                    /
    [ x3 ]  -------/   ( w3 )
          Bias [ +b ]
```

---

### 2. Keterbatasan 1 Neuron: The XOR Trap
Sebuah neuron tunggal hanya mampu membuat **satu garis lurus (atau hyperplane) pemisah**. 
* Masalah seperti **AND** atau **OR** bisa diselesaikan karena datanya *linearly separable* (bisa dipisah satu garis lurus).
* Namun, untuk masalah **XOR** (Exclusive OR), tidak ada satu pun garis lurus yang bisa memisahkan kelasnya.

**Solusinya**: Kita menumpuk beberapa neuron membentuk **Hidden Layer**. Layer tersembunyi bertindak sebagai *Feature Transformer* yang memutar/membengkokkan ruang data sehingga data yang tadinya tidak bisa dipisah menjadi bisa dipisah.

---

### 3. Misteri Fungsi Aktivasi: "Kenapa Harus Non-Linear?"

> ⚠️ **Pertanyaan Kunci IT**: *"Jika kita menumpuk 100 layer linear tanpa fungsi aktivasi, seberapa kuat model kita?"*
> 
> **Jawabannya**: Kekuatannya sama persis dengan **1 layer linear biasa (regresi linear sederhana)!**

Secara matematis:
* Layer 1: $h_1 = W_1 x + b_1$
* Layer 2: $h_2 = W_2 h_1 + b_2 = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$
* Matriks dikali matriks menghasilkan matriks baru ($W' = W_2 W_1$), dan konstanta dijumlahkan menghasilkan bias baru ($b' = W_2 b_1 + b_2$).
* Hasil akhirnya tetap berupa persamaan garis lurus: $h_2 = W' x + b'$.

**Fungsi aktivasi adalah "bengkel pelengkung dimensi"**. Ia memperkenalkan non-linearitas, memungkinkan jaringan saraf bertindak sebagai *Universal Function Approximator* (mampu mendekati bentuk fungsi matematika serumit apa pun).

---

### 4. Perbandingan Fungsi Aktivasi Modern

| Fungsi | Formula | Rentang Output | Karakteristik & Kapan Digunakan | Kelemahan |
| :--- | :---: | :---: | :--- | :--- |
| **ReLU** *(Rectified Linear Unit)* | $\max(0, z)$ | $[0, \infty)$ | **Default pilihan pertama** untuk hidden layers. Sangat cepat dihitung, tidak jenuh untuk nilai positif. | *Dying ReLU*: Jika input selalu negatif, gradiennya 0 permanen (neuron mati). |
| **LeakyReLU** | $\max(\alpha z, z)$ | $(-\infty, \infty)$ | Mengatasi Dying ReLU dengan memberi sedikit gradien ($\alpha \approx 0.01$) saat $z < 0$. | Memerlukan tuning hyperparameter $\alpha$. |
| **GELU** *(Gaussian Error Linear Unit)* | $z \cdot \Phi(z)$ | $[-0.17, \infty)$ | Standar emas pada model modern (**Transformers, BERT, GPT, ViT**). Halus dan probabilistik. | Sedikit lebih berat secara komputasi dibanding ReLU. |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Bagus untuk **Output Layer pada Binary Classification** (menghasilkan probabilitas 0 s.d 1). | **Vanishing Gradient**: Jika $z$ sangat besar/kecil, gradiennya mendekati 0, membuat training lambat jika dipakai di hidden layer. |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | $(0, 1)$, $\sum=1$ | Wajib untuk **Output Layer pada Multi-Class Classification** (probabilitas total seluruh kelas = 100%). | Hanya untuk output layer, bukan untuk hidden layer. |

---

### 5. Shape Contract (Kaidah Dimensi Tensor)

Sebagai engineer TI, selalu lacak bentuk tensor Anda di setiap tahap:
* **Batch Input ($X$)**: `[Batch_Size, Input_Features]`
* **Weights Matrix ($W$)**: `[Output_Features, Input_Features]` (di PyTorch: $Y = X W^T + b$)
* **Batch Output ($Y$)**: `[Batch_Size, Output_Features]`

Contoh:
Input 32 gambar flatten ukuran 784 fitur masuk ke Layer 1 (128 neuron) lalu ke Layer 2 (10 kelas output):
$$\underbrace{X}_{[32, 784]} \xrightarrow{\text{Linear}(784, 128)} \underbrace{H_1}_{[32, 128]} \xrightarrow{\text{ReLU}} \underbrace{A_1}_{[32, 128]} \xrightarrow{\text{Linear}(128, 10)} \underbrace{Y}_{[32, 10]}$$

---

## 🧪 Hands-on Lab

Silakan buka dan jalankan notebook pendamping:
👉 **[`01_network_anatomy.ipynb`](./01_network_anatomy.ipynb)**

Alur hands-on di notebook:
1. **The Working Blackbox**: Visualisasi klasifikasi data non-linear 2D (spiral/moons dataset).
2. **Anatomical Inspection**: Membedah bobot, bias, dan dimensi tensor layer demi layer.
3. **The Sabotage Lab**: Mematikan fungsi aktivasi dan menyaksikan bagaimana batas keputusan (*decision boundary*) menjadi kaku lurus dan akurasi anjlok.
4. **Activation Showdown**: Membandingkan performa konvergensi ReLU vs LeakyReLU vs Sigmoid pada hidden layer.

---

## 🎯 Feynman Checkpoint (Uji Konsep)
Sebelum mencentang modul ini di `TRACKER.md`, pastikan Anda bisa menjawab pertanyaan berikut:
1. *Mengapa neural network dengan 10 layer linear tanpa fungsi aktivasi tidak bisa membedakan data lingkaran (lingkaran dalam vs lingkaran luar)?*
2. *Apa yang dimaksud dengan fenomena "Dying ReLU", dan dalam kondisi apa neuron bisa mati permanen?*
3. *Jika kita memiliki batch input berukuran `[64, 20]` dan ingin menghasilkan representasi berukuran `[64, 128]`, berapa ukuran matriks bobot yang dibutuhkan?*
