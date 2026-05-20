Link Drive:....

## Dataset Split Metadata

Dataset ini digunakan untuk penelitian:

> **IMPLEMENTATION OF DEEP LEARNING BASED ON HYBRID GRAPH ATTENTION NETWORK (GAT) ARCHITECTURE AND TRANSFORMER ENCODER FOR PHISHING DETECTION**

### Model Architecture
- **Graph Attention Network (GAT)** → digunakan untuk pemrosesan fitur URL
- **Transformer Encoder** → digunakan untuk pemrosesan konten HTML

---

# Dataset Split Configuration

Total dataset yang digunakan:

- **39,661 samples**

Dataset dibagi menjadi beberapa skenario eksperimen untuk mengevaluasi performa model pada proporsi data training yang berbeda.

---

## Split Scenario: 90 / 5 / 5

| Subset | Jumlah Data | Persentase |
|---|---:|---:|
| Train | 35,693 | 90% |
| Validation | 1,984 | 5% |
| Test | 1,984 | 5% |

### Path
```bash
dataset_splits/split_90_5_5/
```

### File Structure
```bash
train.csv
validation.csv
test.csv
```

---

## Split Scenario: 80 / 10 / 10

| Subset | Jumlah Data | Persentase |
|---|---:|---:|
| Train | 31,727 | 80% |
| Validation | 3,967 | 10% |
| Test | 3,967 | 10% |

### Path
```bash
dataset_splits/split_80_10_10/
```

### File Structure
```bash
train.csv
validation.csv
test.csv
```

---

## Split Scenario: 70 / 15 / 15

| Subset | Jumlah Data | Persentase |
|---|---:|---:|
| Train | 27,762 | 70% |
| Validation | 5,949 | 15% |
| Test | 5,950 | 15% |

### Path
```bash
dataset_splits/split_70_15_15/
```

### File Structure
```bash
train.csv
validation.csv
test.csv
```

---
