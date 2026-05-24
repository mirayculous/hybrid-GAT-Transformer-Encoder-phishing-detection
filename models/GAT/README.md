# 🚀 Phishing URL Detection using Optimized GATv2

Model deteksi phishing berbasis **Graph Neural Network (GNN)** menggunakan arsitektur **Optimized GATv2 (Graph Attention Network v2)** dengan berbagai optimisasi modern untuk meningkatkan performa klasifikasi URL phishing.

---

# 📌 Features

✅ Residual GATv2 Block
✅ Multi-Head Attention
✅ Layer Normalization
✅ GELU Activation
✅ Global Mean + Max Pooling
✅ AdamW Optimizer
✅ Cosine Annealing Learning Rate
✅ Mixed Precision Training (AMP)
✅ Gradient Clipping
✅ Label Smoothing
✅ ROC Curve & Confusion Matrix
✅ Full Classification Report

---

# 📂 Dataset Structure

Dataset menggunakan format CSV:

* `train.csv`
* `val.csv`
* `test.csv`

## Kolom Dataset

| Column       | Description                                      |
| ------------ | ------------------------------------------------ |
| `graph_data` | Representasi graph dalam format JSON             |
| `label`      | Label klasifikasi (0 = Legitimate, 1 = Phishing) |

---

# 🧠 Model Architecture

Model menggunakan pendekatan Graph Neural Network berbasis **GATv2** dengan residual connection.

## Arsitektur Utama

1. Residual GATv2 Block #1
2. Residual GATv2 Block #2
3. Final GATv2 Layer
4. Dual Global Pooling
5. Fully Connected Classifier

---

# 🔄 Flowchart Model

```text
                 +-------------------+
                 |   Input Graph     |
                 | (x, edge_index)   |
                 +---------+---------+
                           |
                           v
             +-------------------------+
             | Residual GATv2 Block 1 |
             | Multi-head Attention   |
             +-----------+-------------+
                         |
                         v
             +-------------------------+
             | Residual GATv2 Block 2 |
             | Multi-head Attention   |
             +-----------+-------------+
                         |
                         v
             +-------------------------+
             |   Final GATv2 Layer    |
             +-----------+-------------+
                         |
                         v
             +-------------------------+
             | LayerNorm + GELU       |
             +-----------+-------------+
                         |
                         v
          +--------------------------------+
          | Global Mean Pooling            |
          | Global Max Pooling             |
          +---------------+----------------+
                          |
                          v
               +----------------------+
               | Feature Concatenate  |
               +----------+-----------+
                          |
                          v
               +----------------------+
               | Fully Connected MLP  |
               +----------+-----------+
                          |
                          v
               +----------------------+
               | Classification       |
               | Legitimate/Phishing  |
               +----------------------+
```

---

# 🏗️ Penjelasan Model

## 1. Residual GATv2 Block

Block utama model terdiri dari:

* GATv2Conv
* LayerNorm
* GELU Activation
* Dropout
* Residual Connection

### Fungsi:

Residual connection membantu:

* mengurangi vanishing gradient
* mempercepat training
* meningkatkan stabilitas model

---

## 2. Multi-Head Attention

Model menggunakan multi-head attention untuk:

* menangkap hubungan antar node
* memahami struktur graph secara lebih kompleks
* meningkatkan representasi fitur

---

## 3. Global Pooling

Menggunakan kombinasi:

* Global Mean Pooling
* Global Max Pooling

### Tujuan:

Mengambil representasi graph secara global agar informasi node lebih kaya.

---

## 4. Fully Connected Classifier

Classifier menggunakan beberapa layer MLP:

* Linear
* GELU
* Dropout
* LayerNorm

Output akhir berupa probabilitas:

* Legitimate URL
* Phishing URL

---

# ⚙️ Hyperparameter

| Hyperparameter    | Value             |
| ----------------- | ----------------- |
| Batch Size        | 64                |
| Epoch             | 50                |
| Learning Rate     | 1e-3              |
| Weight Decay      | 1e-4              |
| Hidden Dimension  | 128               |
| Attention Heads   | 8                 |
| Dropout           | 0.3               |
| Optimizer         | AdamW             |
| Scheduler         | CosineAnnealingLR |
| Label Smoothing   | 0.05              |
| Gradient Clipping | 1.0               |
| Mixed Precision   | Enabled           |

---

# 📈 Evaluation Metrics

Model dievaluasi menggunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

---

# 📊 Output Files

| File                   | Description                 |
| ---------------------- | --------------------------- |
| `best_gatv2_model.pt`  | Best trained model          |
| `loss_curve_gatv2.png` | Training vs validation loss |
| `roc_curve_gatv2.png`  | ROC curve                   |
| `evaluation_gatv2.png` | Confusion matrix + ROC      |

---

# 🛠️ Installation

```bash
pip install torch-geometric
```

---

# ▶️ Run Training

```bash
python train.py
```

---

# 📚 Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

# 📌 Notes

Model ini dioptimalkan untuk:

* stabilitas training
* generalisasi lebih baik
* performa klasifikasi tinggi
* efisiensi GPU melalui Mixed Precision Training

---

# 👨‍💻 Author

Developed for Phishing URL Detection Research using Optimized GATv2 Architecture.
