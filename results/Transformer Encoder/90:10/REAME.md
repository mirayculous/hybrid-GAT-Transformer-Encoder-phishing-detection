# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        35,693 |
| Validation |         1,984 |
| Test       |         1,984 |

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
| Validation Accuracy | 97.68% |
| Validation F1-Score | 97.39% |
| Validation AUC      | 99.45% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.1945 |
| Accuracy  | 97.73% |
| Precision | 98.28% |
| Recall    | 96.61% |
| F1-Score  | 97.44% |
| ROC-AUC   | 98.23% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9731 | 0.9864 |   0.9797 |    1099 |
| Phishing   |    0.9828 | 0.9661 |   0.9744 |     885 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       1084 |       15 |
| Phishing           |         30 |      855 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 97.73% |
| Macro Average F1    | 97.70% |
| Weighted Average F1 | 97.73% |
| ROC-AUC             | 98.23% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **97.73%**
* Precision sangat tinggi (**98.28%**)
* F1-Score mencapai **97.44%**
* ROC-AUC mencapai **98.23%**

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

Performa terbaik diperoleh pada sekitar epoch **30–35**.

---

# 🏆 Best Result Summary

```text id="nm98rh"
Accuracy  : 97.73%
Precision : 98.28%
Recall    : 96.61%
F1 Score  : 97.44%
AUC ROC   : 98.23%
```

---

# 📁 Generated Outputs

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `best_transformer_html.pt` | Best trained model             |
| `loss_curve.png`           | Training & validation loss     |
| `roc_curve.png`            | ROC curve                      |
| `confusion_matrix.png`     | Confusion matrix visualization |


<img width="666" height="590" alt="download" src="https://github.com/user-attachments/assets/730a8813-7ffc-4469-8234-c3949bf24b22" />
<img width="800" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/fd504adb-a1aa-4bc7-8873-8e03e11d0268" />
<img width="700" height="600" alt="roc_curve" src="https://github.com/user-attachments/assets/9161c373-e9f0-442a-8027-d07e6703a9d7" />
