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
| Trainable Parameters     | 1,879,426         |
| Epochs                   | 50                |
| Optimizer                | AdamW             |
| Scheduler                | CosineAnnealingLR |
| Mixed Precision Training | Enabled           |
| Label Smoothing          | 0.05              |

---

# 📈 Training Performance

Model menunjukkan peningkatan performa yang stabil selama training.

## Best Validation Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 94.00% |
| Validation F1-Score | 93.18% |
| Validation AUC      | 98.24% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.2330 |
| Accuracy  | 94.50% |
| Precision | 93.89% |
| Recall    | 93.79% |
| F1-Score  | 93.84% |
| ROC-AUC   | 98.49% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9500 | 0.9508 |   0.9504 |    3295 |
| Phishing   |    0.9389 | 0.9379 |   0.9384 |    2655 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 94.50% |
| Macro Average F1    | 94.44% |
| Weighted Average F1 | 94.50% |
| ROC-AUC             | 98.49% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat baik dalam mendeteksi phishing website dengan:

* Accuracy mencapai **94.50%**
* ROC-AUC mencapai **98.49%**
* Precision dan Recall yang seimbang
* Kemampuan generalisasi yang stabil pada validation dan test set

Hasil ini menunjukkan bahwa arsitektur model mampu:

* memahami pola HTML phishing
* menangkap hubungan antar fitur
* membedakan website legitimate dan phishing secara efektif

---

# 📉 Training Observation

Selama training:

* loss mengalami penurunan secara stabil
* validation AUC meningkat konsisten
* model tidak menunjukkan overfitting signifikan
* performa mulai konvergen pada epoch 40–45

---

# 🏆 Best Result Summary

```text id="q3l2vw"
Accuracy  : 94.50%
Precision : 93.89%
Recall    : 93.79%
F1 Score  : 93.84%
AUC ROC   : 98.49%
```

---

# 📁 Generated Outputs

| File             | Description                |
| ---------------- | -------------------------- |
| `best_model.pt`  | Best trained model         |
| `loss_curve.png` | Training & validation loss |
| `roc_curve.png`  | ROC curve                  |
| `evaluation.png` | Evaluation visualization   |

<img width="700" height="600" alt="roc_curve_gatv2" src="https://github.com/user-attachments/assets/098bc77a-ace7-4131-853c-1fbf0b541df8" />
<img width="800" height="500" alt="loss_curve_gatv2" src="https://github.com/user-attachments/assets/0bcb8cec-e3ed-4ca7-bc88-1efb3c1b4d77" />
<img width="1200" height="500" alt="evaluation_gatv2" src="https://github.com/user-attachments/assets/0447aaf1-b847-4163-820e-083e616f544b" />



