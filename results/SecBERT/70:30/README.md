# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        27,762 |
| Validation |         5,949 |
| Test       |         5,950 |

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

Model menunjukkan performa yang sangat tinggi selama proses training dengan peningkatan validation accuracy dan ROC-AUC yang stabil.

## Best Validation Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 96.87% |
| Validation F1-Score | 96.44% |
| Validation AUC      | 99.21% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.1947 |
| Accuracy  | 96.79% |
| Precision | 97.64% |
| Recall    | 95.10% |
| F1-Score  | 96.36% |
| ROC-AUC   | 99.15% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9614 | 0.9815 |   0.9713 |    3295 |
| Phishing   |    0.9764 | 0.9510 |   0.9636 |    2655 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       3234 |       61 |
| Phishing           |        130 |     2525 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 96.79% |
| Macro Average F1    | 96.74% |
| Weighted Average F1 | 96.79% |
| ROC-AUC             | 99.15% |

---

# 🔍 Analysis

Model SecBERT menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **96.79%**
* Precision mencapai **97.64%**
* ROC-AUC sangat tinggi yaitu **99.15%**
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
* validation AUC meningkat sangat cepat sejak epoch awal
* model menunjukkan konvergensi stabil
* tidak terlihat overfitting signifikan

Performa terbaik diperoleh pada sekitar epoch **7–8**.

---

# 🏆 Best Result Summary

```text id="u9k2xm"
Accuracy  : 96.79%
Precision : 97.64%
Recall    : 95.10%
F1 Score  : 96.36%
AUC ROC   : 99.15%
```

---

# 📁 Generated Outputs

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `best_secbert_html.pt`   | Best trained model           |
| `loss_curve_secbert.png` | Training & validation loss   |
| `evaluation_secbert.png` | ROC curve + confusion matrix |


<img width="1200" height="750" alt="loss_curve_secbert" src="https://github.com/user-attachments/assets/8f93e539-13c4-458a-930a-dd41b7ea096c" />
<img width="1950" height="750" alt="evaluation_secbert" src="https://github.com/user-attachments/assets/0dc7fbae-27d5-45d3-a8f5-b4a1b8a93d3d" />
