# Grokking Deep Learning: Scratch Reproduction & Theoretical Deep-Dive

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.13-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Pure%20Scratch-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/Academic-Assignment%20Tugas%202-green)

### **TUGAS 2 (ENRICHMENT FOR DEEP LEARNING CLASSES) - INDIVIDUAL TASK**
**Code Reproduction + Theoretical Deep-Dive from *Grokking Deep Learning* by Andrew W. Trask (Manning Publications)**

---

**Student Name:** Satrio Arjuna Putra  
**Student ID (NIM):** 101032330178  
**Class:** TK-47-05  
**Institution:** Telkom University  
**Repository:** [github.com/SatrioArjunaPutra/GrokkingDeepLearning](https://github.com/SatrioArjunaPutra/GrokkingDeepLearning)

</div>

---

## 📋 1. Deskripsi Penugasan & Panduan Soal (Assignment Brief)

Tugas ini merupakan tugas individu mata kuliah **Deep Learning (Enrichment Class)** yang bertujuan menguji pemahaman fundamental arsitektur dan algoritma Deep Learning tanpa ketergantungan pada *framework high-level* (seperti TensorFlow, PyTorch, atau Keras). 

### Ketentuan & Spesifikasi Soal:
1. **Zero Black-Box Abstraction:** Semua komponen jaringan—mulai dari propagasi maju (*forward propagation*), fungsi *loss* (*squared error* & *cross-entropy*), penurunan gradien kalkulus (*gradient descent*), hingga propagasi balik bertingkat (*backpropagation*) dan lapisan konvolusi—wajib dibangun dari nol (**pure Python & NumPy**).
2. **Kajian Lengkap 10 Bab Buku Panduan:**
   * **Part 1 (Bab 01 – 06):** Fondasi Jaringan Syaraf Tiruan:
     * *Bab 01:* Pengantar Deep Learning, otomatisasi kecerdasan, dan persiapan *environment*.
     * *Bab 02:* Konsep dasar ML (Supervised vs Unsupervised, Parametric vs Nonparametric, siklus 3 langkah Predict-Compare-Learn).
     * *Bab 03:* Propagasi Maju (*Forward Propagation*), analogi kenop sensitivitas (*volume knob*), dan aljabar matriks/vektor.
     * *Bab 04:* Penurunan Gradien (*Gradient Descent*), *Hot & Cold Learning*, perhitungan arah dan besaran (*direction and amount*), serta laju belajar *Alpha*.
     * *Bab 05:* Generalisasi Penurunan Gradien Multivariabel, fenomena kompensasi bobot (*weight freezing*), dan gradien *Outer Product*.
     * *Bab 06:* Jaringan Syaraf Dalam Pertama (*First Deep Network*), bukti kolaps linier, aktivasi non-linier ReLU, dan algoritma *Backpropagation* penuh untuk memecahkan problem non-linier XOR Streetlight.
   * **Part 2 (Bab 07 – 10):** Visi Komputer, Regularisasi, dan Arsitektur Lanjut:
     * *Bab 07:* Visualisasi Bobot sebagai Gambar (*Picture Weights*), *up-weights* vs *down-weights*, *template matching*, dan klasifikasi MNIST.
     * *Bab 08:* Pencegahan *Overfitting*, regularisasi *Inverted Dropout* ($p = 0.5$, skala $\times 2.0$), serta optimasi *Mini-Batch Gradient Descent*.
     * *Bab 09:* Eksplorasi Fungsi Aktivasi Non-linier (Sigmoid, Tanh, ReLU), mitigasi *Vanishing Gradient*, dan pemodelan probabilitas Softmax dengan *Categorical Cross-Entropy*.
     * *Bab 10:* Jaringan Syaraf Konvolusional (CNN), prinsip *weight sharing*, operasi konvolusi 2D, kernel deteksi tepi Sobel, dan *spatial subsampling* Max Pooling.
3. **Struktur Setiap Notebook:**
   * Identitas lengkap mahasiswa & mata kuliah.
   * Judul Bab & Capaian Pembelajaran (*Objectives*).
   * Teori & Formulasi Matematis Terperinci (LaTeX).
   * Implementasi Scratch Murni dengan anotasi dimensi matriks (`# Shape: ...`).
   * Visualisasi Analitis (Grafik konvergensi loss, heatmap bobot, distribusi aktivasi).
   * Kesimpulan & Evaluasi Kritis (*Key Takeaways*).

---

## 🚀 2. Interactive Google Colab Hub & Repository Roadmap

Setiap bab dapat langsung dijalankan secara interaktif di **Google Colab** dengan mengklik tombol badge pada tabel di bawah ini:

| Bab | Topik Inti & Deskripsi | Referensi | Status | Buka di Google Colab | File Notebook |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **01** | **Introducing Deep Learning**<br>Otomatisasi kecerdasan, analogi intuitif, persiapan Jupyter & NumPy | Manning Ch. 1 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2001%20-%20Introducing%20Deep%20Learning/01_introducing_deep_learning.ipynb) | [01_introducing_deep_learning.ipynb](Chapter%2001%20-%20Introducing%20Deep%20Learning/01_introducing_deep_learning.ipynb) |
| **02** | **What is Deep Learning? (Fundamental Concepts)**<br>Supervised vs Unsupervised, Parametric vs Nonparametric, Predict-Compare-Learn | Manning Ch. 2 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2002%20-%20Fundamental%20Prerequisities/02_fundamental_prerequisites.ipynb) | [02_fundamental_prerequisites.ipynb](Chapter%2002%20-%20Fundamental%20Prerequisities/02_fundamental_prerequisites.ipynb) |
| **03** | **Forward Propagation (Making Predictions)**<br>Bobot sebagai kenop volume, dot product aljabar & logika, multi-input/output, stacked layers | Manning Ch. 3 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2003%20-%20Forward%20Propagation/03_forward_propagation.ipynb) | [03_forward_propagation.ipynb](Chapter%2003%20-%20Forward%20Propagation/03_forward_propagation.ipynb) |
| **04** | **Introduction to Gradient Descent (Learning to Reduce Error)**<br>Squared Error, Hot & Cold Learning, Direction & Amount kalkulus, Divergence, Alpha | Manning Ch. 4 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2004%20-%20Gradient%20Descent%20and%20Learning/04_gradient_descent.ipynb) | [04_gradient_descent.ipynb](Chapter%2004%20-%20Gradient%20Descent%20and%20Learning/04_gradient_descent.ipynb) |
| **05** | **Generalizing Gradient Descent (Multi-Variable Optimization)**<br>Penurunan gradien multi-input/output, eksperimen pembekuan bobot, outer product | Manning Ch. 5 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2005%20-%20Generalizing%20Gradient%20Descent/05_generalizing_gradient_descent.ipynb) | [05_generalizing_gradient_descent.ipynb](Chapter%2005%20-%20Generalizing%20Gradient%20Descent/05_generalizing_gradient_descent.ipynb) |
| **06** | **Backpropagation (Building Your First DEEP Network)**<br>Bukti kolaps linier, aktivasi ReLU, propagasi balik rantai, pemecahan Streetlight XOR | Manning Ch. 6 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2006%20-%20Backpropagation/06_backpropagation.ipynb) | [06_backpropagation.ipynb](Chapter%2006%20-%20Backpropagation/06_backpropagation.ipynb) |
| **07** | **Picture Weights and Feature Learning**<br>MNIST 28x28, visualisasi bobot 2D, filter eksitatoris & inhibitoris, batasan model 1-layer | Manning Ch. 7 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2007%20-%20Picture%20Weights%20and%20Feature%20Learning/07_picture_weights.ipynb) | [07_picture_weights.ipynb](Chapter%2007%20-%20Picture%20Weights%20and%20Feature%20Learning/07_picture_weights.ipynb) |
| **08** | **Regularization and Batching**<br>Dilema overfitting, regularisasi Inverted Dropout ($p=0.5$), akselerasi Mini-Batch GD ($B=100$) | Manning Ch. 8 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2008%20-%20Regularization%20and%20Batching/08_regularization_batching.ipynb) | [08_regularization_batching.ipynb](Chapter%2008%20-%20Regularization%20and%20Batching/08_regularization_batching.ipynb) |
| **09** | **Activation Functions**<br>Non-linearitas Sigmoid, Tanh, ReLU, Vanishing Gradient, Softmax & Cross-Entropy Loss | Manning Ch. 9 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2009%20-%20Activation%20Functions/09_activation_functions.ipynb) | [09_activation_functions.ipynb](Chapter%2009%20-%20Activation%20Functions/09_activation_functions.ipynb) |
| **10** | **Convolutional Neural Networks**<br>Konvolusi 2D diskrit, prinsip weight sharing, kernel Sobel & Ridge, Max Pooling ($2\times 2$) | Manning Ch. 10 | Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatrioArjunaPutra/GrokkingDeepLearning/blob/main/Chapter%2010%20-%20Convolutional%20Neural%20Networks/10_convolutional_neural_networks.ipynb) | [10_convolutional_neural_networks.ipynb](Chapter%2010%20-%20Convolutional%20Neural%20Networks/10_convolutional_neural_networks.ipynb) |

> **💡 Catatan Penggunaan Google Colab:**  
> Seluruh tautan Colab di atas telah diformat dengan *URL-encoding* standar (`%20` untuk spasi nama folder). Jika Anda membuka Colab secara manual, pilih tab **GitHub**, masukkan URL repositori `https://github.com/SatrioArjunaPutra/GrokkingDeepLearning`, lalu pilih notebook yang diinginkan.

---

## 🗺️ 3. Struktur Direktori Repositori

```
GrokkingDeepLearning/
├── .gitignore
├── README.md
├── requirements.txt
├── Chapter 01 - Introducing Deep Learning/
│   └── 01_introducing_deep_learning.ipynb
├── Chapter 02 - Fundamental Prerequisities/
│   └── 02_fundamental_prerequisites.ipynb
├── Chapter 03 - Forward Propagation/
│   └── 03_forward_propagation.ipynb
├── Chapter 04 - Gradient Descent and Learning/
│   └── 04_gradient_descent.ipynb
├── Chapter 05 - Generalizing Gradient Descent/
│   └── 05_generalizing_gradient_descent.ipynb
├── Chapter 06 - Backpropagation/
│   └── 06_backpropagation.ipynb
├── Chapter 07 - Picture Weights and Feature Learning/
│   ├── README.md
│   └── 07_picture_weights.ipynb
├── Chapter 08 - Regularization and Batching/
│   ├── README.md
│   └── 08_regularization_batching.ipynb
├── Chapter 09 - Activation Functions/
│   ├── README.md
│   └── 09_activation_functions.ipynb
└── Chapter 10 - Convolutional Neural Networks/
    ├── README.md
    └── 10_convolutional_neural_networks.ipynb
```

---

## 🔬 4. Pembahasan Mendalam Setiap Bab Sesuai Soal Tugas (Detailed Synthesis)

### [Chapter 01 - Introducing Deep Learning](Chapter%2001%20-%20Introducing%20Deep%20Learning/01_introducing_deep_learning.ipynb)
* **Sintesis Konseptual:** Memahami evolusi dari *rule-based expert systems* ke representasi otomatis. Deep Learning menonjol karena menawarkan otomatisasi kecerdasan dan keterampilan teknis secara inkremental tanpa keharusan menguasai matematika kalkulus tingkat lanjut terlebih dahulu.
* **Poin Pembahasan Soal:**
  * Mengapa mempelajari Deep Learning: Dampak signifikan terhadap otomatisasi tenaga kerja terampil dan sifat pemodelan yang kreatif.
  * *Low Barrier to Entry:* Pemahaman matematika didekati menggunakan analogi intuitif dunia nyata.
  * Persyaratan teknis dasar: Pemrograman Python, Jupyter Notebook, dan pustaka matriks NumPy.

---

### [Chapter 02 - What is Deep Learning? (Fundamental Concepts)](Chapter%2002%20-%20Fundamental%20Prerequisities/02_fundamental_prerequisites.ipynb)
* **Sintesis Konseptual:** Mendemistifikasi taksonomi machine learning ke dalam empat kuadran utama.
* **Poin Pembahasan Soal:**
  * **Supervised vs. Unsupervised Learning:** Supervised mentransformasikan data input menjadi target label; unsupervised mengelompokkan data berdasarkan kemiripan pola intrinsik.
  * **Parametric vs. Nonparametric:** Model parametrik menggunakan sejumlah kenop (*knobs*) tetap yang dicari konfigurasinya melalui *trial-and-error*; model non-parametrik melakukan penghitungan frekuensi (*counting*) dengan jumlah parameter yang bertambah dinamis mengikuti data.
  * **Paradigma 3 Langkah Supervised Parametric Learning:**
    1. *Predict:* Data diproses melalui sudut kenop saat ini menjadi prediksi.
    2. *Compare:* Mengukur selisih antara hasil prediksi dan kebenaran faktual (*truth*).
    3. *Learn:* Menyesuaikan arah dan besaran kenop agar kesalahan berkurang pada observasi berikutnya.

---

### [Chapter 03 - Forward Propagation (Intro to Neural Prediction)](Chapter%2003%20-%20Forward%20Propagation/03_forward_propagation.ipynb)
* **Sintesis Konseptual:** Propagasi maju adalah komputasi aliran data sensorik melewati matriks parameter tanpa mengubah nilai bobot.
* **Poin Pembahasan Soal:**
  * **Bobot sebagai Kenop Sensitivitas (*Volume Knobs*):** Bobot positif memperkuat sinyal, bobot negatif membalikkan/melemahkan sinyal, dan bobot nol meredam sinyal irelevan.
  * **Analogi Aljabar & Logika Dot Product:** $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ berfungsi sebagai detektor keselarasan pola (*pattern similarity*) dan operasi logika AND berbobot.
  * **Topologi Jaringan:** Mengimplementasikan single input $\to$ single output, multiple inputs $\to$ single output (`w_sum`), single input $\to$ multiple outputs (`ele_mul`), multiple inputs $\to$ multiple outputs (`vect_mat_mul`), serta *Predicting on Predictions* (jaringan bertumpuk dengan lapisan tersembunyi).

---

### [Chapter 04 - Introduction to Gradient Descent (Learning to Reduce Error)](Chapter%2004%20-%20Gradient%20Descent%20and%20Learning/04_gradient_descent.ipynb)
* **Sintesis Konseptual:** Pembelajaran jaringan syaraf tiruan adalah proses rekonsiliasi kesalahan secara terarah menggunakan kalkulus diferensial.
* **Poin Pembahasan Soal:**
  * **Squared Error ($E = (\hat{y} - y)^2$):** Memastikan nilai error selalu positif, memberikan penalti kuadratik eksponensial pada kesalahan besar, dan membentuk kurva parabola halus yang dapat diturunkan.
  * **Kelemahan Hot and Cold Learning:** Eksplorasi heuristik langkah tetap (*step amount*) lambat dan mengalami ledakan kombinatorik $2^N$ pada multi-variabel.
  * **Penurunan Gradien (Direction & Amount):**
    $$\text{delta} = \hat{y} - y, \quad \text{weight\_delta} = \text{delta} \cdot \text{input}, \quad w \leftarrow w - \text{weight\_delta}$$
  * **Divergence & Alpha ($\alpha$):** Input bernilai besar menyebabkan gradien sangat curam sehingga bobot melompati titik minimum parabola (*overshooting*) dan menyebabkan error meledak. Pengali laju belajar *Alpha* ($w \leftarrow w - \alpha \cdot \text{derivative}$) wajib digunakan untuk meredam pembaruan bobot.

---

### [Chapter 05 - Generalizing Gradient Descent (Multi-Variable Optimization)](Chapter%2005%20-%20Generalizing%20Gradient%20Descent/05_generalizing_gradient_descent.ipynb)
* **Sintesis Konseptual:** Memperluas penurunan gradien ke dimensi tensor multivariat.
* **Poin Pembahasan Soal:**
  * Penurunan gradien multi-input: Setiap kanal fitur memperbarui bobotnya secara independen $\frac{\partial E}{\partial w_j} = \delta \cdot x_j$.
  * **Eksperimen Pembekuan Satu Bobot (*Weight Freezing*):** Membuktikan bahwa saat satu bobot dikunci secara artifisial ($\Delta w_0 = 0$), bobot-bobot lainnya akan mengompensasi penyesuaian untuk mencapai zero error, membuktikan ruang solusi bobot memiliki konfigurasi manifold tak terhingga.
  * Pembaruan multi-output dan **Outer Product**: $\Delta \mathbf{W} = \boldsymbol{\delta} \otimes \mathbf{x} = \boldsymbol{\delta} \cdot \mathbf{x}^T$.

---

### [Chapter 06 - Backpropagation (Building Your First DEEP Neural Network)](Chapter%2006%20-%20Backpropagation/06_backpropagation.ipynb)
* **Sintesis Konseptual:** Mengatasi batasan keterpisahan linier dengan menyusun lapisan tersembunyi yang dilengkapi fungsi aktivasi non-linier.
* **Poin Pembahasan Soal:**
  * **Tantangan Streetlight XOR:** Perseptron satu lapis gagal saat keputusan bergantung pada interaksi non-linier kondisional.
  * **Bukti Matematis Kolaps Linier:** Dua lapisan linier tanpa aktivasi identik dengan satu lapisan linier efektif: $\mathbf{\hat{y}} = (\mathbf{x} \mathbf{W}_1) \mathbf{W}_2 = \mathbf{x} (\mathbf{W}_1 \mathbf{W}_2) = \mathbf{x} \mathbf{W}_{\text{eff}}$.
  * **Fungsi ReLU & Turunannya:** $f(z) = \max(0, z)$ dan $f'(z) = \mathbb{I}(z > 0)$.
  * **Algoritma Backpropagation via Aturan Rantai:**
    $$\boldsymbol{\delta}_2 = \mathbf{a}_2 - \mathbf{y}, \quad \boldsymbol{\delta}_1 = (\boldsymbol{\delta}_2 \mathbf{W}_{12}^T) \odot f'(\mathbf{a}_1)$$
    $$\mathbf{W}_{12} \leftarrow \mathbf{W}_{12} - \alpha (\mathbf{a}_1^T \boldsymbol{\delta}_2), \quad \mathbf{W}_{01} \leftarrow \mathbf{W}_{01} - \alpha (\mathbf{a}_0^T \boldsymbol{\delta}_1)$$

---

### [Chapter 07 - Picture Weights and Feature Learning](Chapter%2007%20-%20Picture%20Weights%20and%20Feature%20Learning/07_picture_weights.ipynb)
* **Sintesis Konseptual:** Bobot pada pengenal gambar merepresentasikan cetak biru visual (*spatial template*).
* **Poin Pembahasan Soal:**
  * **Rekonstruksi Spasial Bobot:** Vektor bobot $\mathbf{w}_c \in \mathbb{R}^{784}$ diubah kembali menjadi matriks citra $28 \times 28$.
  * **Up-weights vs. Down-weights:** Up-weights ($W > 0$, merah) mendeteksi piksel positif goresan angka; down-weights ($W < 0$, biru) menghukum piksel area kosong yang menjadi ciri khas digit lain.
  * **Keterbatasan Single-Layer:** Gagal membedakan variasi tulisan miring, ketebalan tinta, atau bentuk yang bertumpuk (seperti angka 8 dan 0) tanpa lapisan fitur tersembunyi.

---

### [Chapter 08 - Regularization and Batching](Chapter%2008%20-%20Regularization%20and%20Batching/08_regularization_batching.ipynb)
* **Sintesis Konseptual:** Menjembatani kesenjangan antara memorisasi sampel data latih dan kemampuan generalisasi data baru.
* **Poin Pembahasan Soal:**
  * **Overfitting Gap:** Loss data latih terus menurun menuju nol sementara loss data uji berbalik naik (*divergen*).
  * **Inverted Dropout ($p = 0.5$):** Mematikan acak $50\%$ neuron tersembunyi memaksa setiap neuron belajar fitur mandiri tanpa bergantung pada neuron lain. Skalasi $\times 2.0$ menjaga ekspektasi magnitudo aktivasi konstan sehingga saat evaluasi/pengujian dropout cukup dimatikan secara total tanpa penyesuaian bobot.
  * **Mini-Batching Dynamics:** Mengelompokkan data ke dalam batch ($B = 100$) memaksimalkan utilisasi instruksi SIMD prosesor sekaligus mereduksi variansi estimasi gradien.

---

### [Chapter 09 - Activation Functions](Chapter%2009%20-%20Activation%20Functions/09_activation_functions.ipynb)
* **Sintesis Konseptual:** Menelaah dinamika gradien fungsi aktivasi dan memodelkan probabilitas multi-kelas sejati.
* **Poin Pembahasan Soal:**
  * **Analisis Turunan Aktivasi:** Sigmoid ($\sigma' = \sigma(1-\sigma)$ dengan puncak $0.25$) dan Tanh ($\tanh' = 1-\tanh^2$ dengan puncak $1.0$).
  * **Vanishing Gradient:** Mengapa saturasi pada nilai input ekstrim menyebabkan gradien mengecil secara eksponensial di lapisan awal jaringan dalam.
  * **Distribusi Softmax & Keajaiban Gradien Cross-Entropy:**
    $$\hat{y}_k = \frac{e^{z_k - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}, \quad \mathcal{L} = -\sum y_k \ln(\hat{y}_k)$$
    Turunan parsial Cross-Entropy terhadap logit $z_i$ menghasilkan persamaan selisih linier yang sangat sederhana dan stabil: $\boldsymbol{\delta} = \hat{\mathbf{y}} - \mathbf{y}$.

---

### [Chapter 10 - Convolutional Neural Networks](Chapter%2010%20-%20Convolutional%20Neural%20Networks/10_convolutional_neural_networks.ipynb)
* **Sintesis Konseptual:** Memanfaatkan topologi 2D citra melalui pembagian bobot (*weight sharing*) dan invariansi pergeseran (*translation invariance*).
* **Poin Pembahasan Soal:**
  * **Kelemahan Jaringan Dense:** Ledakan parameter dan hilangnya struktur spasial lokal akibat operasi perataan (*flattening*).
  * **Operasi Konvolusi 2D Diskrit:** Meluncurkan kernel filter $3 \times 3$ untuk mengekstrak peta fitur (*feature maps*):
    $$S(i, j) = (\mathbf{I} * \mathbf{K})(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)$$
  * **Kernel Detektor Fitur:** Implementasi filter Sobel Horizontal (garis horizontal), Sobel Vertikal (garis vertikal), dan filter Ridge/Sudut.
  * **Max Pooling:** Reduksi dimensi spasial sebesar $75\%$ (dari $28 \times 28$ ke $14 \times 14$) yang memberikan toleransi terhadap distorsi spasial kecil.

---

## 🛠️ 5. Panduan Instalasi & Eksekusi Lokal

### Prasyarat:
* Python 3.10+ (Diuji pada Python 3.13)
* Git

### Langkah Menjalankan Proyek:
1. **Clone repositori:**
   ```bash
   git clone https://github.com/SatrioArjunaPutra/GrokkingDeepLearning.git
   cd GrokkingDeepLearning
   ```

2. **Buat dan aktifkan virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependensi pustaka:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Buka folder bab yang diinginkan (misalnya `Chapter 06 - Backpropagation/06_backpropagation.ipynb`) dan pilih menu **Cell -> Run All**.

---

## 📚 6. Integritas Akademik & Referensi

Karya ini disusun secara mandiri dengan mematuhi prinsip integritas akademik untuk pemenuhan **Tugas 2: Enrichment Deep Learning Class**.
* **Referensi Utama:** Trask, Andrew W. (2019). *Grokking Deep Learning*. Manning Publications. ISBN: 9781617293702.
* **Referensi Tambahan Aljabar Linear:** Cohen, Mike X. (2021). *Practical Linear Algebra for Data-Science*. O'Reilly Media. ISBN: 9781098120610.

---

<div align="center">
<b>Satrio Arjuna Putra (101032330178) — Kelas TK-47-05 — Telkom University</b>
</div>