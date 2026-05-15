# Hybrid GAT-Transformer Encoder for Phishing Detection

> Deep learning based phishing detection framework using Graph Attention Networks (GAT) and Transformer Encoder for URL structural analysis and HTML semantic understanding.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Status](https://img.shields.io/badge/Status-Research-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Repository

[hybrid-GAT-Transformer-Encoder-phishing-detection](https://github.com/mirayculous/hybrid-GAT-Transformer-Encoder-phishing-detection?utm_source=chatgpt.com)

---

## 📖 Overview

Phishing websites have become increasingly sophisticated, making traditional blacklist and rule-based detection methods less effective. Most existing deep learning approaches focus only on URL patterns or webpage content independently.

This research proposes a **Hybrid Deep Learning Architecture** that combines:

- **Graph Attention Network (GAT)** for URL structural representation
- **Transformer Encoder** for HTML semantic feature extraction

By integrating graph-based learning and transformer-based contextual understanding, the model aims to improve phishing website detection performance and robustness.

---

## 🧠 Proposed Architecture

```text
                    ┌────────────────┐
                    │      URL       │
                    └───────┬────────┘
                            │
                  URL Graph Construction
                            │
                            ▼
                 ┌────────────────────┐
                 │ Graph Attention    │
                 │ Network (GAT)      │
                 └─────────┬──────────┘
                           │
                           │
                           ▼
                    Feature Fusion
                           ▲
                           │
                 ┌─────────┴──────────┐
                 │ Transformer Encoder│
                 │ HTML Representation│
                 └─────────┬──────────┘
                           │
                    HTML Preprocessing
                           │
                           ▼
                    ┌────────────┐
                    │ HTML Page  │
                    └────────────┘

                           ▼
                 ┌──────────────────┐
                 │ Binary Classifier│
                 │ Legit / Phishing │
                 └──────────────────┘
```

---

## ✨ Features

- Hybrid GAT + Transformer architecture
- URL graph representation learning
- HTML semantic analysis
- Attention-based feature extraction
- HTML preprocessing with BeautifulSoup
- Research-oriented modular pipeline
- End-to-end phishing detection workflow
- Scalable deep learning implementation

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| PyTorch Geometric | Graph neural network processing |
| Transformers | Transformer Encoder implementation |
| BeautifulSoup4 | HTML preprocessing |
| Scikit-learn | Evaluation metrics |
| Pandas & NumPy | Data processing |

---

## 📂 Project Structure

```bash
hybrid-GAT-Transformer-Encoder-phishing-detection/
│
├── Dataset/
├── models/
├── preprocessing/
├── training/
├── evaluation/
├── notebooks/
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/mirayculous/hybrid-GAT-Transformer-Encoder-phishing-detection.git
cd hybrid-GAT-Transformer-Encoder-phishing-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

### Prediction

```bash
python predict.py
```

---

## 🔍 Data Preprocessing

### URL Processing
- URL tokenization
- Character segmentation
- Graph generation
- Node embedding preparation

### HTML Processing
HTML content is cleaned using BeautifulSoup by removing irrelevant tags such as:

- `<script>`
- `<style>`
- `<noscript>`
- `<head>`

Then processed for transformer embedding extraction.

---

## 📊 Model Components

### Graph Attention Network (GAT)

Used to:
- Learn URL structural relationships
- Capture contextual node importance
- Generate graph embeddings

### Transformer Encoder

Used to:
- Understand semantic HTML content
- Capture long-range contextual dependencies
- Extract deep textual representations

---

## 🎯 Research Objectives

- Improve phishing detection accuracy
- Reduce false positives
- Combine structural and semantic learning
- Develop a robust phishing detection framework

---

## 📈 Future Improvements

- Browser extension deployment
- Real-time phishing detection
- DistilBERT optimization
- Explainable AI visualization
- Multi-modal phishing analysis
- Large-scale benchmarking

---

## 🧪 Dataset Sources

Potential dataset sources:

- PhishTank
- OpenPhish
- Alexa Top Sites
- Common Crawl
- Custom collected phishing webpages

---

## 🤝 Contribution

Contributions are welcome.

1. Fork this repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

### Raynald Adika Sumarga

Final Project Research:

> *Implementation of Deep Learning Based on Hybrid Graph Attention Network (GAT) Architecture and Transformer Encoder for Phishing Detection*

GitHub:
[@mirayculous](https://github.com/mirayculous?utm_source=chatgpt.com)

---

## ⭐ Support

If you find this project useful, consider giving it a star on GitHub.
