# Transformer Encoder model

✅ Custom HTML Tokenizer
✅ Transformer Encoder Architecture
✅ Positional Encoding
✅ Attention Pooling
✅ Numerical Feature Fusion
✅ Mixed Precision Training (AMP)
✅ Gradient Accumulation
✅ AdamW Optimizer
✅ Cosine Annealing Scheduler
✅ Weighted Loss
✅ Label Smoothing
✅ ROC Curve & Loss Curve
✅ Complete Evaluation Metrics

---

# 📂 Dataset Structure

Dataset menggunakan format CSV:

* `train.csv`
* `val.csv`
* `test.csv`

## Kolom Dataset

| Column             | Description                                      |
| ------------------ | ------------------------------------------------ |
| `html`             | Source code HTML website                         |
| Numerical Features | Fitur numerik hasil ekstraksi HTML               |
| `label`            | Label klasifikasi (0 = Legitimate, 1 = Phishing) |

---

# 📊 Numerical Features

Model menggunakan 17 fitur numerik:

```text id="4vrh7v"
word_count
num_forms
num_inputs
num_links
num_ext_links
num_scripts
num_ext_scripts
num_iframes
num_images
has_password_input
has_hidden_input
has_external_form
has_iframe
ext_link_ratio
ext_script_ratio
text_length
html_length
```

---

# 🧠 Model Architecture

Arsitektur model terdiri dari:

1. Custom HTML Tokenizer
2. Embedding Layer
3. Positional Encoding
4. Transformer Encoder
5. Attention Pooling
6. Numerical Feature Projection
7. Feature Fusion
8. MLP Classifier

---

# 🔄 Flowchart Model

```text id="z42qz3"
                     +----------------------+
                     |      HTML Text       |
                     +----------+-----------+
                                |
                                v
                 +-----------------------------+
                 | Custom HTML Tokenizer       |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Token Embedding             |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Positional Encoding         |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Transformer Encoder         |
                 | Multi-Head Attention        |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Attention Pooling           |
                 +-------------+---------------+
                               |
                               v
                        HTML Embedding
                               |
                               |
        +----------------------------------------------+
        |                                              |
        |                                              v
        |                           +------------------------+
        |                           | Numerical Features     |
        |                           +-----------+------------+
        |                                       |
        |                                       v
        |                           +------------------------+
        |                           | Feature Projection MLP |
        |                           +-----------+------------+
        |                                       |
        +-------------------+-------------------+
                            |
                            v
                 +-----------------------------+
                 | Feature Concatenation       |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Fully Connected Classifier  |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | Legitimate / Phishing       |
                 +-----------------------------+
```

---

# 🏗️ Penjelasan Model

# 1. Custom HTML Tokenizer

Tokenizer khusus dibuat untuk memproses struktur HTML website.

Tokenizer mampu mengenali:

* tag HTML
* URL
* token teks
* simbol khusus

### Regex Tokenization

```text id="7bqk4f"
r"<[^>]+>|https?://\S+|[a-z0-9]+|[^\w\s]"
```

### Keunggulan:

* memahami struktur HTML lebih baik
* lebih ringan dibanding pretrained tokenizer
* efisien untuk dataset phishing

---

# 2. Embedding Layer

Token hasil tokenizer dikonversi menjadi dense vector menggunakan:

* Embedding Layer
* Linear Projection

### Tujuan:

Mengubah token HTML menjadi representasi numerik yang dapat diproses Transformer.

---

# 3. Positional Encoding

Transformer tidak memahami urutan token secara alami.

Karena itu digunakan sinusoidal positional encoding untuk memberikan informasi posisi token.

### Fungsi:

* mempertahankan urutan struktur HTML
* membantu memahami konteks halaman website

---

# 4. Transformer Encoder

Model menggunakan:

* Multi-Head Self Attention
* Feed Forward Network
* Layer Normalization
* GELU Activation

### Konfigurasi:

* 4 Transformer Layers
* 8 Attention Heads
* Hidden Dimension 256

### Tujuan:

Menangkap hubungan kompleks antar token HTML.

---

# ⚡ Attention Pooling

Attention pooling digunakan untuk mengambil representasi penting dari seluruh sequence token.

### Keunggulan:

* lebih informatif dibanding average pooling
* fokus pada bagian HTML yang relevan

---

# 🔢 Numerical Feature Projection

Fitur numerik diproses menggunakan MLP:

* Linear Layer
* LayerNorm
* GELU
* Dropout

### Tujuan:

Menghasilkan embedding numerik yang stabil sebelum dilakukan fusion.

---

# 🔀 Feature Fusion

Embedding HTML dan numerical features digabungkan:

```text id="tqv4t9"
[Transformer HTML Embedding]
            +
[Numerical Feature Embedding]
            ↓
      Feature Fusion
```

---

# 🧩 MLP Classifier

Classifier akhir terdiri dari:

* Fully Connected Layer
* GELU Activation
* LayerNorm
* Dropout

Output:

* Legitimate Website
* Phishing Website

---

# ⚙️ Hyperparameter

| Hyperparameter          | Value             |
| ----------------------- | ----------------- |
| Max Sequence Length     | 256               |
| Vocabulary Size         | 12000             |
| Embedding Dimension     | 128               |
| Transformer Hidden Size | 256               |
| Attention Heads         | 8                 |
| Transformer Layers      | 4                 |
| Feed Forward Dimension  | 512               |
| Batch Size              | 32                |
| Epoch                   | 50                |
| Learning Rate           | 2e-4              |
| Weight Decay            | 1e-2              |
| Dropout                 | 0.2               |
| Optimizer               | AdamW             |
| Scheduler               | CosineAnnealingLR |
| Label Smoothing         | 0.05              |
| Mixed Precision         | Enabled           |

---

# 📈 Evaluation Metrics

Model dievaluasi menggunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

---

# 📊 Output Files

| File                       | Description                 |
| -------------------------- | --------------------------- |
| `best_transformer_html.pt` | Best trained model          |
| `loss_curve.png`           | Training vs validation loss |
| `roc_curve.png`            | ROC curve                   |

---

# 🛠️ Installation

```bash id="fr4n0u"
pip install torch
```

---

# ▶️ Run Training

```bash id="mqlx8f"
python train.py
```

---

# 📚 Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

# 📌 Notes

Model ini dirancang untuk:

* memahami struktur HTML phishing
* memanfaatkan Transformer Encoder custom
* menggabungkan semantic HTML dan numerical features
* memberikan performa tinggi dengan arsitektur yang lebih ringan dibanding pretrained BERT

---

# 👨‍💻 Author

Developed for Phishing Website Detection Research using Optimized Transformer Encoder Architecture.
