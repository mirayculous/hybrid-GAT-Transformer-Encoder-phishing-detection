# 📌 Experimental Results

## 📂 Dataset Split

| Dataset    | Total Samples |
| ---------- | ------------: |
| Train      |        31,727 |
| Validation |         3,967 |
| Test       |         3,967 |

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
| Validation Accuracy | 97.91% |
| Validation F1-Score | 97.63% |
| Validation AUC      | 99.19% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.1966 |
| Accuracy  | 96.70% |
| Precision | 96.96% |
| Recall    | 95.59% |
| F1-Score  | 96.27% |
| ROC-AUC   | 99.03% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9649 | 0.9759 |   0.9704 |    2197 |
| Phishing   |    0.9696 | 0.9559 |   0.9627 |    1770 |

---

# 🔍 Confusion Matrix

| Actual \ Predicted | Legitimate | Phishing |
| ------------------ | ---------: | -------: |
| Legitimate         |       2144 |       53 |
| Phishing           |         78 |     1692 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 96.70% |
| Macro Average F1    | 96.65% |
| Weighted Average F1 | 96.70% |
| ROC-AUC             | 99.03% |

---

# 🔍 Analysis

Model SecBERT menunjukkan performa yang sangat tinggi dalam mendeteksi phishing website dengan:

* Accuracy mencapai **96.70%**
* Precision mencapai **96.96%**
* ROC-AUC sangat tinggi yaitu **99.03%**
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

Performa terbaik diperoleh pada sekitar epoch **6–7**.

---

# 🏆 Best Result Summary

```text id="xv0w8s"
Accuracy  : 96.70%
Precision : 96.96%
Recall    : 95.59%
F1 Score  : 96.27%
AUC ROC   : 99.03%
```

---

# 📁 Generated Outputs

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `best_secbert_html.pt`   | Best trained model           |
| `loss_curve_secbert.png` | Training & validation loss   |
| `evaluation_secbert.png` | ROC curve + confusion matrix |


<img width="1950" height="750" alt="evaluation_secbert" src="https://github.com/user-attachments/assets/16210c98-2248-4af7-b59e-00e640c2849b" />
<img width="1200" height="750" alt="loss_curve_secbert" src="https://github.com/user-attachments/assets/6ac1d8c3-70e8-41c2-95c0-a101231d77cb" />
