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
| Trainable Parameters     | 1,879,426         |
| Epochs                   | 50                |
| Optimizer                | AdamW             |
| Scheduler                | CosineAnnealingLR |
| Mixed Precision Training | Enabled           |
| Label Smoothing          | 0.05              |

---

# 📈 Training Performance

Model menunjukkan peningkatan performa yang stabil selama proses training.

## Best Validation Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 95.01% |
| Validation F1-Score | 94.44% |
| Validation AUC      | 98.86% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.2419 |
| Accuracy  | 93.75% |
| Precision | 93.09% |
| Recall    | 92.88% |
| F1-Score  | 92.99% |
| ROC-AUC   | 98.30% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9428 | 0.9445 |   0.9436 |    1099 |
| Phishing   |    0.9309 | 0.9288 |   0.9299 |     885 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 93.75% |
| Macro Average F1    | 93.68% |
| Weighted Average F1 | 93.75% |
| ROC-AUC             | 98.30% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat baik dalam mendeteksi phishing website dengan:

* Accuracy mencapai **93.75%**
* ROC-AUC mencapai **98.30%**
* Precision dan Recall yang seimbang
* Generalisasi model yang stabil pada validation dan test set

Hasil ini menunjukkan bahwa model mampu:

* memahami pola HTML phishing
* mengekstraksi representasi token secara efektif
* membedakan website legitimate dan phishing dengan akurasi tinggi

---

# 📉 Training Observation

Selama training:

* training loss mengalami penurunan stabil
* validation AUC meningkat secara konsisten
* model menunjukkan konvergensi yang baik
* tidak terlihat overfitting signifikan

Performa mulai stabil pada sekitar epoch **35–40**.

---

# 🏆 Best Result Summary

```text id="z2p8ab"
Accuracy  : 93.75%
Precision : 93.09%
Recall    : 92.88%
F1 Score  : 92.99%
AUC ROC   : 98.30%
```

---

# 📁 Generated Outputs

| File                       | Description                |
| -------------------------- | -------------------------- |
| `best_transformer_html.pt` | Best trained model         |
| `loss_curve.png`           | Training & validation loss |
| `roc_curve.png`            | ROC curve                  |


<img width="1200" height="500" alt="evaluation_gatv2" src="https://github.com/user-attachments/assets/a66c439f-932f-4832-9742-b186c23a479c" />
<img width="700" height="600" alt="roc_curve_gatv2" src="https://github.com/user-attachments/assets/a4b013d1-6635-4c21-9ce8-a8f22e153091" />
<img width="800" height="500" alt="loss_curve_gatv2" src="https://github.com/user-attachments/assets/7a094953-32f6-4f20-a127-12b1819039d9" />
