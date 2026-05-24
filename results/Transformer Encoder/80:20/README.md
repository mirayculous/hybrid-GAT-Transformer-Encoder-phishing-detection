# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        31,727 |
| Validation |         3,967 |
| Test       |         3,967 |

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
| Validation Accuracy | 97.86% |
| Validation F1-Score | 97.58% |
| Validation AUC      | 99.48% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.1957 |
| Accuracy  | 97.58% |
| Precision | 97.72% |
| Recall    | 96.84% |
| F1-Score  | 97.28% |
| ROC-AUC   | 98.43% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9747 | 0.9818 |   0.9782 |    2197 |
| Phishing   |    0.9772 | 0.9684 |   0.9728 |    1770 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       2157 |       40 |
| Phishing           |         56 |     1714 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 97.58% |
| Macro Average F1    | 97.55% |
| Weighted Average F1 | 97.58% |
| ROC-AUC             | 98.43% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **97.58%**
* Precision sangat tinggi (**97.72%**)
* F1-Score mencapai **97.28%**
* ROC-AUC mencapai **98.43%**

Model mampu:

* memahami pola HTML phishing secara efektif
* mengurangi false positive dan false negative
* melakukan generalisasi dengan sangat baik pada unseen data

---

# 📉 Training Observation

Selama training:

* training loss turun secara konsisten
* validation accuracy meningkat cepat sejak epoch awal
* validation AUC mencapai lebih dari 99%
* model menunjukkan konvergensi yang sangat stabil

Performa terbaik diperoleh pada sekitar epoch **6–10**.

---

# 🏆 Best Result Summary

```text id="gq92tm"
Accuracy  : 97.58%
Precision : 97.72%
Recall    : 96.84%
F1 Score  : 97.28%
AUC ROC   : 98.43%
```

---

# 📁 Generated Outputs

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `best_transformer_html.pt` | Best trained model             |
| `loss_curve.png`           | Training & validation loss     |
| `roc_curve.png`            | ROC curve                      |
| `confusion_matrix.png`     | Confusion matrix visualization |


<img width="666" height="590" alt="download" src="https://github.com/user-attachments/assets/0e86695e-6175-407f-ad38-1d8359f42830" />
<img width="700" height="600" alt="roc_curve" src="https://github.com/user-attachments/assets/f608601c-5a73-4536-b6a3-b6ce1af73284" />
<img width="800" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/d3b08807-35b2-47e9-85fa-3945be6890f2" />

