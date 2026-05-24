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
| Validation Accuracy | 94.56% |
| Validation F1-Score | 93.82% |
| Validation AUC      | 98.37% |

---

# 🧪 Test Evaluation

## Final Test Results

| Metric    |  Score |
| --------- | -----: |
| Loss      | 0.2311 |
| Accuracy  | 94.35% |
| Precision | 94.17% |
| Recall    | 93.11% |
| F1-Score  | 93.64% |
| ROC-AUC   | 98.47% |

---

# 📊 Classification Report

| Class      | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Legitimate |    0.9450 | 0.9536 |   0.9493 |    2197 |
| Phishing   |    0.9417 | 0.9311 |   0.9364 |    1770 |

---

# 📌 Overall Performance

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 94.35% |
| Macro Average F1    | 94.28% |
| Weighted Average F1 | 94.35% |
| ROC-AUC             | 98.47% |

---

# 🔍 Analysis

Model menunjukkan performa yang sangat baik dalam mendeteksi phishing website dengan:

* Accuracy mencapai **94.35%**
* ROC-AUC mencapai **98.47%**
* Precision dan Recall yang seimbang
* Generalisasi model yang stabil pada validation dan test set

Hasil ini menunjukkan bahwa model mampu:

* memahami pola HTML phishing
* mengekstraksi representasi token secara efektif
* membedakan website legitimate dan phishing dengan akurasi tinggi

---

# 📉 Training Observation

Selama training:

* training loss menurun secara konsisten
* validation AUC meningkat stabil
* model menunjukkan konvergensi yang baik
* tidak terlihat overfitting signifikan hingga epoch akhir

Performa mulai stabil pada sekitar epoch **40–45**.

---

# 🏆 Best Result Summary

```text id="r74xcz"
Accuracy  : 94.35%
Precision : 94.17%
Recall    : 93.11%
F1 Score  : 93.64%
AUC ROC   : 98.47%
```

---

# 📁 Generated Outputs

| File                       | Description                |
| -------------------------- | -------------------------- |
| `best_transformer_html.pt` | Best trained model         |
| `loss_curve.png`           | Training & validation loss |
| `roc_curve.png`            | ROC curve                  |


<img width="1200" height="500" alt="evaluation_gatv2" src="https://github.com/user-attachments/assets/b45584dd-0c28-4745-ab7d-8ecf005f22c0" />
<img width="700" height="600" alt="roc_curve_gatv2" src="https://github.com/user-attachments/assets/f050ed1a-2fea-4fb2-8326-ca18606d787c" />
<img width="800" height="500" alt="loss_curve_gatv2" src="https://github.com/user-attachments/assets/f96034b0-f44b-4387-8a07-4dc39f24c349" />

