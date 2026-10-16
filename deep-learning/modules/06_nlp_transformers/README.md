# 💬 Modul 06: Applied NLP & Modern Transformers

Selamat datang di Modul 06! Di modul ini kita masuk ke jantung revolusi AI modern: **Pemrosesan Bahasa Alami (Natural Language Processing) dan Arsitektur Transformers** yang mendasari ChatGPT, BERT, Claude, dan Llama.

---

## 🎯 Tujuan Pembelajaran
Setelah menyelesaikan modul ini, Anda akan:
1. Memahami bagaimana teks diubah menjadi angka: **Tokenization (BPE/WordPiece)** dan **Word Embeddings**.
2. Memahami mengapa arsitektur lawas (RNN/LSTM) ditinggalkan dan digantikan oleh **Transformers**.
3. Memahami intuisi **Self-Attention Mechanism ($Q, K, V$)** tanpa tersesat dalam rumus matematika yang rumit.
4. Mengetahui perbedaan mendasar arsitektur **Encoder-only (BERT)** vs **Decoder-only (GPT)**.
5. Menguasai ekosistem industri **Hugging Face (`transformers`)**: `AutoTokenizer` dan `AutoModelForSequenceClassification`.
6. Menyelesaikan **Real Project**: Sistem Analisis Sentimen & Klasifikasi Ulasan Pelanggan End-to-End.

---

## 🧠 Konsep & Mental Model

### 1. Dari Karakter ke Vektor: Pipeline Teks Modern

Komputer hanya paham angka. Untuk memproses kalimat, kita melalui 2 tahap:

```text
Kalimat Asli        : "Saya suka belajar deep learning"
                           |
1. Tokenizer (Subword) : ['saya', 'suka', 'belajar', 'deep', 'learn', '##ing']
                           |  (Map ke Vocabulary ID)
2. Token IDs (input_ids): [1045,  2341,  8821,     4510,  3210,   120]
                           |  (Lookup ke Embedding Matrix)
3. Vectors / Embeddings: Matrix berukuran [Seq_Length, Hidden_Dim] (misal: 6 x 768)
```

> 💡 **Kenapa Subword Tokenization (BPE/WordPiece)?**
> Jika kita pakai kata utuh (*word-level*), kata langka atau typo akan menjadi `<UNK>` (Unknown). Jika pakai karakter tunggal (*char-level*), kalimat menjadi terlalu panjang. Subword memecah kata majemuk menjadi fragmen kecil yang bermakna (`unbelievable` $\rightarrow$ `un` + `believ` + `able`), sehingga tidak pernah ada kata yang "tidak dikenal".

---

### 2. Mengapa Transformers Mengalahkan RNN & LSTM?
Sebelum tahun 2017, NLP dikuasai oleh Recurrent Neural Networks (RNN) dan LSTM:
* **Kelemahan Fatal RNN**: Memproses kata **satu per satu secara berurutan** (Kata 1 $\rightarrow$ Kata 2 $\rightarrow$ Kata 3). Ini membuat training tidak bisa diparalelkan di ribuan core GPU! Selain itu, RNN sering lupa kata pertama jika kalimat memiliki 50+ kata (*vanishing memory*).
* **Terobosan Transformer (2017)**: Kalimat dibaca **sekaligus secara paralel** dalam satu kali komputasi matriks raksasa.

---

### 3. Intuisi Self-Attention: "Melihat Seluruh Konteks Sekaligus"
Perhatikan kalimat:
> *"Hewan itu tidak menyeberang jalan karena **ia** terlalu lelah."*

Bagaimana komputer tahu kata **"ia"** merujuk ke **hewan**, bukan **jalan**?

Mekanisme **Self-Attention** memetakan setiap kata menjadi 3 vektor:
1. **Query ($Q$)**: *"Apa yang sedang saya cari?"* (Pertanyaan dari kata yang bersangkutan).
2. **Key ($K$)**: *"Apa identitas saya?"* (Label dari semua kata lain).
3. **Value ($V$)**: *"Informasi apa yang saya simpan?"*

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* $Q \cdot K^T$ menghitung skor relevansi (*attention score*) antara kata **"ia"** dengan semua kata lain di kalimat.
* Kata **"hewan"** akan menghasilkan skor dot-product tertinggi dengan **"ia"**.
* Outputnya adalah representasi kata "ia" yang telah diperkaya dengan informasi dari kata "hewan"!

---

### 4. Encoder vs Decoder: Jangan Salah Pilih!

| Tipe | Contoh Model | Karakteristik Utama | Use-Case Terbaik di Industri |
| :--- | :--- | :--- | :--- |
| **Encoder-Only** | **BERT**, RoBERTa, DeBERTa | Membaca dua arah (bidirectional context: kiri dan kanan sekaligus). Sangat paham makna teks. | **Klasifikasi Teks, Analisis Sentimen, Named Entity Recognition (NER), Semantic Search**. |
| **Decoder-Only** | **GPT-4**, Llama 3, Mistral | Membaca satu arah (kiri ke kanan saja, autoregressive). Dilatih memprediksi token berikutnya. | **Text Generation, Chatbot, Coding Assistant, Summarization**. |

---

## 🧪 Hands-on Lab & Real Project

Buka dan jalankan:
👉 **[`06_nlp_transformers.ipynb`](./06_nlp_transformers.ipynb)**

Alur hands-on:
1. **Tokenizer Inspection**: Membedah cara `AutoTokenizer` memecah teks, membuat `input_ids`, dan menyusun `attention_mask`.
2. **Fine-Tuning DistilBERT**: Mengambil model Transformer ringan (DistilBERT) dan melatihnya untuk mengklasifikasikan sentimen ulasan produk (Positif vs Negatif).
3. **Interactive Inference**: Membuat fungsi pengujian langsung di mana Anda bisa mengetikkan kalimat baru dari keyboard dan model langsung mengeluarkan prediksi sentimen serta probabilitas keyakinannya!

---

## 🎯 Feynman Checkpoint (Uji Konsep)
1. *Mengapa mekanisme Self-Attention pada Transformers memungkinkan training berjalan jauh lebih cepat dibanding RNN/LSTM di GPU?*
2. *Apa fungsi `attention_mask` pada batch token teks yang memiliki panjang kalimat berbeda-beda?*
3. *Jika tugas di kantor Anda adalah mengklasifikasikan email komplain pelanggan ke dalam 5 kategori departemen, apakah Anda memilih arsitektur berbasis Encoder (BERT) atau Decoder (GPT)? Mengapa?*
