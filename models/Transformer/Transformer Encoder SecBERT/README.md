# Transformer Encoder SecBERT model 

✅ Pretrained SecBERT (`jackaduma/SecBERT`)
✅ Partial Fine-Tuning Transformer Layer
✅ Attention Pooling
✅ Numerical Feature Fusion
✅ Mixed Precision Training (AMP)
✅ Gradient Accumulation
✅ AdamW Optimizer
✅ Cosine Scheduler + Warmup
✅ Weighted Loss
✅ Label Smoothing
✅ ROC Curve & Confusion Matrix
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

```text id="q12qah"
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

Model menggabungkan:

1. Transformer Encoder (SecBERT)
2. Attention Pooling
3. Numerical Feature Projection
4. Feature Fusion
5. MLP Classifier

---

# 🔄 Flowchart Model

```text id="tf2t9u"
                    +----------------------+
                    |      HTML Text       |
                    +----------+-----------+
                               |
                               v
                 +----------------------------+
                 | AutoTokenizer (SecBERT)   |
                 +-------------+--------------+
                               |
                               v
                 +----------------------------+
                 | Pretrained SecBERT         |
                 | Partial Fine-Tuning        |
                 +-------------+--------------+
                               |
                               v
                 +----------------------------+
                 | Attention Pooling          |
                 +-------------+--------------+
                               |
                               v
                        HTML Embedding
                               |
                               |
        +---------------------------------------------+
        |                                             |
        |                                             v
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
                 +----------------------------+
                 | Feature Concatenation      |
                 +-------------+--------------+
                               |
                               v
                 +----------------------------+
                 | Fully Connected Classifier |
                 +-------------+--------------+
                               |
                               v
                 +----------------------------+
                 | Legitimate / Phishing      |
                 +----------------------------+
```

---

# 🏗️ Penjelasan Model

## 1. SecBERT Transformer Encoder

Model menggunakan:

* `jackaduma/SecBERT`
* pretrained khusus domain cybersecurity

### Fungsi:

SecBERT memahami:

* struktur HTML
* pola phishing
* script berbahaya
* token cybersecurity

---

## 2. Partial Fine-Tuning

Model melakukan:

* freeze sebagian besar transformer layer
* hanya unfreeze layer terakhir

### Tujuan:

* mengurangi overfitting
* mempercepat training
* menjaga knowledge pretrained model

---

# ⚡ Attention Pooling

Menggunakan soft-attention pooling pada seluruh token sequence.

### Keunggulan:

* lebih informatif dibanding hanya menggunakan `[CLS] token`
* fokus pada token penting dalam HTML

---

# 🔢 Numerical Feature Projection

Fitur numerik diproses menggunakan MLP:

* Linear Layer
* LayerNorm
* GELU
* Dropout

### Tujuan:

Menghasilkan representasi numerik yang lebih stabil sebelum digabungkan dengan embedding SecBERT.

---

# 🔀 Feature Fusion

Embedding HTML dari SecBERT digabung dengan embedding numerical features:

```text id="c7d1tm"
[HTML Embedding] + [Numerical Embedding]
            ↓
      Feature Fusion
```

---

# 🧩 MLP Classifier

Classifier akhir terdiri dari:

* Fully Connected Layer
* LayerNorm
* GELU Activation
* Dropout

Output:

* Legitimate Website
* Phishing Website

---

# ⚙️ Hyperparameter

| Hyperparameter        | Value               |
| --------------------- | ------------------- |
| Pretrained Model      | `jackaduma/SecBERT` |
| Max Sequence Length   | 256                 |
| BERT Hidden Size      | 768                 |
| Batch Size            | 32                  |
| Gradient Accumulation | 2                   |
| Effective Batch Size  | 64                  |
| Epoch                 | 50                  |
| Learning Rate         | 3e-5                |
| Weight Decay          | 1e-2                |
| Dropout               | 0.2                 |
| Optimizer             | AdamW               |
| Scheduler             | Cosine Warmup       |
| Label Smoothing       | 0.05                |
| Mixed Precision       | Enabled             |

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

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `best_secbert_html.pt`   | Best trained model           |
| `loss_curve_secbert.png` | Training vs validation loss  |
| `evaluation_secbert.png` | Confusion matrix + ROC curve |

---

# 🛠️ Installation

```bash id="cv3dr0"
pip install transformers
```

---

# ▶️ Run Training

```bash id="axdr8r"
python train.py
```

---

# 📚 Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* SecBERT
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

# 📌 Notes

Model ini dirancang untuk:

* memahami struktur HTML phishing
* memanfaatkan pretrained cybersecurity knowledge
* menggabungkan semantic HTML dan numerical features
* meningkatkan generalisasi model phishing detection

---

# 👨‍💻 Author

Developed for Phishing Website Detection Research using SecBERT Transformer Architecture.
