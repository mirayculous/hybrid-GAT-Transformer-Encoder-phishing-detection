# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        35,693 |
| Validation |         1,984 |
| Test       |         1,984 |

---

# ⚙️ Training Configuration

| Configuration            | Value               |
| ------------------------ | ------------------- |
| Total Parameters         | 84,300,227          |
| Trainable Parameters     | 29,791,427          |
| Frozen Parameters        | 54,508,800          |
| Pretrained Model         | `jackaduma/SecBERT` |
| Unfreeze Layers          | 8–11                |
| Vocabulary Size          | 52,000              |
| Epochs                   | 50                  |
| Optimizer                | AdamW               |
| Mixed Precision Training | Enabled             |

---

# 📈 Training Performance

Model menunjukkan performa yang sangat stabil selama proses training dengan peningkatan validation accuracy dan ROC-AUC yang konsisten.

## Best Validation Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 97.98% |
| Validation F1-Score | 97.71% |
| Validation AUC      | 99.01% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.1957 |
| Accuracy  | 97.03% |
| Precision | 98.14% |
| Recall    | 95.14% |
| F1-Score  | 96.62% |
| ROC-AUC   | 98.87% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9618 | 0.9854 |   0.9735 |    1099 |
| Phishing   |    0.9814 | 0.9514 |   0.9662 |     885 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       1083 |       16 |
| Phishing           |         43 |      842 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 97.03% |
| Macro Average F1    | 96.98% |
| Weighted Average F1 | 97.02% |
| ROC-AUC             | 98.87% |

---

# 🔍 Analysis

Model SecBERT menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **97.03%**
* Precision mencapai **98.14%**
* ROC-AUC sangat tinggi yaitu **98.87%**
* Kemampuan generalisasi yang stabil pada validation dan test set

Model mampu:

* memahami struktur HTML phishing secara mendalam
* memanfaatkan pretrained cybersecurity knowledge
* mengurangi false positive dan false negative
* mendeteksi pola phishing dengan akurasi tinggi

---

# 📉 Training Observation

Selama training:

* training loss mengalami penurunan konsisten
* validation accuracy meningkat cepat sejak epoch awal
* validation AUC mencapai lebih dari 99%
* model menunjukkan konvergensi stabil
* tidak terlihat overfitting signifikan

Performa terbaik diperoleh pada sekitar epoch **7–10**.

---

# 🏆 Best Result Summary

```text id="d1km9v"
Accuracy  : 97.03%
Precision : 98.14%
Recall    : 95.14%
F1 Score  : 96.62%
AUC ROC   : 98.87%
```

---

# 📁 Generated Outputs

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `best_secbert_html.pt`   | Best trained model           |
| `loss_curve_secbert.png` | Training & validation loss   |
| `evaluation_secbert.png` | ROC curve + confusion matrix |


<img width="1950" height="750" alt="evaluation_secbert" src="https://github.com/user-attachments/assets/18c027cb-60c4-43eb-8099-171fafcf3224" />
<img width="1200" height="750" alt="loss_curve_secbert" src="https://github.com/user-attachments/assets/f9a2a4bb-9a52-4384-be8d-8759ce0a867f" />

