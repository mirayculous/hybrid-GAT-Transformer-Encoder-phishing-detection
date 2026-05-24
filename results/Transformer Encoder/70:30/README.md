
# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        27,762 |
| Validation |         5,949 |
| Test       |         5,950 |

---

# ⚙️ Training Configuration

| Configuration            | Value             |
| ------------------------ | ----------------- |
| Trainable Parameters     | 3,870,659         |
| Epochs                   | 50                |
| Optimizer                | AdamW             |
| Scheduler                | CosineAnnealingLR |
| Mixed Precision Training | Enabled           |
| Vocabulary Size          | 12,000            |

---

# 📈 Training Performance

Model menunjukkan performa yang sangat stabil selama proses training dengan peningkatan akurasi dan AUC yang konsisten.

## Best Validation Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 97.08% |
| Validation F1-Score | 96.69% |
| Validation AUC      | 99.41% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.2083 |
| Accuracy  | 96.99% |
| Precision | 96.23% |
| Recall    | 97.06% |
| F1-Score  | 96.64% |
| ROC-AUC   | 97.83% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9762 | 0.9693 |   0.9727 |    3295 |
| Phishing   |    0.9623 | 0.9706 |   0.9664 |    2655 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       3194 |      101 |
| Phishing           |         78 |     2577 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 96.99% |
| Macro Average F1    | 96.96% |
| Weighted Average F1 | 96.99% |
| ROC-AUC             | 97.83% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **96.99%**
* Precision mencapai **96.23%**
* Recall mencapai **97.06%**
* ROC-AUC mencapai **97.83%**

Model mampu:

* memahami struktur HTML phishing secara efektif
* mengurangi false positive dan false negative
* melakukan generalisasi dengan sangat baik pada unseen data

---

# 📉 Training Observation

Selama training:

* training loss turun secara konsisten
* validation accuracy meningkat cepat sejak epoch awal
* validation AUC mencapai lebih dari 99%
* model menunjukkan konvergensi yang sangat stabil

Performa terbaik diperoleh pada sekitar epoch **3–5**.

---

# 🏆 Best Result Summary

```text id="cw9xmr"
Accuracy  : 96.99%
Precision : 96.23%
Recall    : 97.06%
F1 Score  : 96.64%
AUC ROC   : 97.83%
```

---

# 📁 Generated Outputs

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `best_transformer_html.pt` | Best trained model             |
| `loss_curve.png`           | Training & validation loss     |
| `roc_curve.png`            | ROC curve                      |
| `confusion_matrix.png`     | Confusion matrix visualization |


<img width="666" height="590" alt="download" src="https://github.com/user-attachments/assets/0bfd4bda-2aca-4904-b109-89e054ffe543" />
<img width="700" height="600" alt="roc_curve" src="https://github.com/user-attachments/assets/fa4d36f8-a7c4-4add-814c-7db1e83f7272" />
<img width="800" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/ec3d95c0-a68f-49f8-89ff-8f144d28b736" />
